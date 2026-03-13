# -*- coding: utf-8 -*-
"""
Clasificador de Pingüinos: Humano vs Árbol de Decisión
Alumno: González González Sergio Erick

Modos de entrada:
  1. Manual        → ingresar pingüinos uno por uno
  2. Lista interna → editar la lista PINGUINOS_HARDCODED abajo
  3. CSV externo   → cargar un archivo .csv con los datos
"""

import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

# ═══════════════════════════════════════════════════════════════════════════
# 📝 MODO 2: EDITA ESTA LISTA DIRECTAMENTE EN EL CÓDIGO
#    Columnas: bill_length_mm, bill_depth_mm, flipper_length_mm, body_mass_g
# ═══════════════════════════════════════════════════════════════════════════

PINGUINOS_HARDCODED = [
    # (bill_length, bill_depth, flipper_length, body_mass)
    (39.1, 18.7, 181, 3750),
    (46.5, 17.9, 192, 3500),
    (46.1, 13.2, 211, 4500),
    # Agrega más filas aquí...
]

# ═══════════════════════════════════════════════════════════════════════════
# 1. ENTRENAR EL ÁRBOL DE DECISIÓN
# ═══════════════════════════════════════════════════════════════════════════

df_original = sns.load_dataset('penguins')
df = df_original.dropna().reset_index(drop=True)

features = ['bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g']
X = df[features]
y = df['species']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

arbol = DecisionTreeClassifier(random_state=42)
arbol.fit(X_train, y_train)

print("╔══════════════════════════════════════════════════════════╗")
print("║   🐧 CLASIFICADOR DE PINGÜINOS - Humano vs Árbol ML 🐧  ║")
print("╚══════════════════════════════════════════════════════════╝")
print("✅ Árbol de decisión entrenado.\n")

# ═══════════════════════════════════════════════════════════════════════════
# 2. CLASIFICADOR HUMANO (BONUS reto1.py)
# ═══════════════════════════════════════════════════════════════════════════

def clasificador_humano_optimo(bill_length_mm, bill_depth_mm, flipper_length_mm, body_mass_g):
    if flipper_length_mm <= 207.5:
        if bill_length_mm <= 43.35:
            if bill_length_mm <= 42.35:
                if bill_depth_mm <= 16.65:
                    if bill_length_mm <= 39.5:
                        return "Adelie"
                    else:
                        return "Chinstrap"
                else:
                    return "Adelie"
            else:
                if flipper_length_mm <= 189.5:
                    return "Chinstrap"
                else:
                    return "Adelie"
        else:
            if body_mass_g <= 4125:
                return "Chinstrap"
            else:
                if bill_length_mm <= 45.9:
                    return "Adelie"
                else:
                    return "Chinstrap"
    else:
        if bill_depth_mm <= 17.65:
            return "Gentoo"
        else:
            if bill_length_mm <= 46.55:
                return "Adelie"
            else:
                return "Chinstrap"

# ═══════════════════════════════════════════════════════════════════════════
# 3. FUNCIÓN PARA CLASIFICAR Y GENERAR CSV
# ═══════════════════════════════════════════════════════════════════════════

def clasificar_y_guardar(registros, nombre_csv='resultados_pinguinos.csv'):
    resultados = []

    for i, r in enumerate(registros, start=1):
        bl = r['bill_length_mm']
        bd = r['bill_depth_mm']
        fl = r['flipper_length_mm']
        bm = r['body_mass_g']

        pred_humano = clasificador_humano_optimo(bl, bd, fl, bm)
        pred_ml     = arbol.predict(pd.DataFrame([[bl, bd, fl, bm]], columns=features))[0]

        resultados.append({
            'pinguino_id'       : i,
            'bill_length_mm'    : bl,
            'bill_depth_mm'     : bd,
            'flipper_length_mm' : fl,
            'body_mass_g'       : bm,
            'pred_humano'       : pred_humano,
            'pred_arbol_ml'     : pred_ml,
            'coinciden'         : pred_humano == pred_ml,
        })

    df_res = pd.DataFrame(resultados)
    df_res.to_csv(nombre_csv, index=False, encoding='utf-8-sig')

    coincidencias = df_res['coinciden'].sum()
    print(f"\n╔══════════════════════════════════════════════════════════╗")
    print(f"║                   📊 RESUMEN FINAL                       ║")
    print(f"╠══════════════════════════════════════════════════════════╣")
    print(f"║  Pingüinos procesados : {len(df_res):<5}                            ║")
    print(f"║  Predicciones iguales : {coincidencias}/{len(df_res)}                          ║")
    print(f"╠══════════════════════════════════════════════════════════╣")
    print(f"║  💾 CSV guardado: {nombre_csv:<39}║")
    print(f"╚══════════════════════════════════════════════════════════╝")
    print()
    print(df_res.to_string(index=False))
    return df_res

# ═══════════════════════════════════════════════════════════════════════════
# 4. MODOS DE ENTRADA
# ═══════════════════════════════════════════════════════════════════════════

def pedir_float(mensaje, minimo, maximo):
    while True:
        try:
            valor = float(input(mensaje))
            if minimo <= valor <= maximo:
                return valor
            print(f"   ⚠️  Valor fuera de rango ({minimo}–{maximo}).")
        except ValueError:
            print("   ⚠️  Ingresa un número válido.")

def modo_manual():
    registros = []
    print("\n📋 Rangos de referencia:")
    print("   bill_length_mm: 32–60  |  bill_depth_mm: 13–22")
    print("   flipper_length_mm: 172–231  |  body_mass_g: 2700–6300")
    print("Escribe 'fin' al inicio de un pingüino para terminar.\n")

    i = 1
    while True:
        centinela = input(f"🐧 Pingüino #{i} — ENTER para continuar o 'fin': ").strip().lower()
        if centinela == 'fin':
            break
        bl = pedir_float("   bill_length_mm   : ", 20, 70)
        bd = pedir_float("   bill_depth_mm    : ", 10, 25)
        fl = pedir_float("   flipper_length_mm: ", 150, 250)
        bm = pedir_float("   body_mass_g      : ", 2000, 7000)
        registros.append({
            'bill_length_mm': bl, 'bill_depth_mm': bd,
            'flipper_length_mm': fl, 'body_mass_g': bm
        })
        i += 1
    return registros

def modo_hardcoded():
    registros = [
        {'bill_length_mm': r[0], 'bill_depth_mm': r[1],
         'flipper_length_mm': r[2], 'body_mass_g': r[3]}
        for r in PINGUINOS_HARDCODED
    ]
    print(f"   📋 {len(registros)} pingüinos cargados desde la lista interna.")
    return registros

def modo_csv():
    """
    El CSV solo necesita estas columnas (nombres exactos):
        bill_length_mm, bill_depth_mm, flipper_length_mm, body_mass_g
    """
    ruta = input("   📂 Ruta del CSV (ej: mis_pinguinos.csv): ").strip()
    try:
        df_in = pd.read_csv(ruta)
    except FileNotFoundError:
        print(f"   ❌ No se encontró '{ruta}'.")
        return None

    columnas_necesarias = {'bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g'}
    if not columnas_necesarias.issubset(df_in.columns):
        faltantes = columnas_necesarias - set(df_in.columns)
        print(f"   ❌ Faltan columnas en el CSV: {faltantes}")
        return None

    registros = df_in[list(columnas_necesarias)].to_dict(orient='records')
    print(f"   ✅ {len(registros)} pingüinos cargados desde '{ruta}'.")
    return registros

# ═══════════════════════════════════════════════════════════════════════════
# 5. MENÚ PRINCIPAL
# ═══════════════════════════════════════════════════════════════════════════

print("¿Cómo quieres ingresar los datos?\n")
print("  1 → Manual (uno por uno)")
print("  2 → Lista interna (edita PINGUINOS_HARDCODED en el código)")
print("  3 → Cargar CSV externo")
print()

while True:
    modo = input("Selecciona una opción (1/2/3): ").strip()
    if modo in ('1', '2', '3'):
        break
    print("⚠️  Ingresa 1, 2 o 3.")

if modo == '1':
    registros = modo_manual()
elif modo == '2':
    registros = modo_hardcoded()
else:
    registros = modo_csv()

if registros:
    clasificar_y_guardar(registros)
else:
    print("\n⚠️  No hay datos para procesar.")
