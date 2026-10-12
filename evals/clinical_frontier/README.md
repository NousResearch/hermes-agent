# Evaluación clínica sintética

Este harness conserva los ocho casos y la rúbrica del PR original. No constituye una auditoría clínica ejecutada ni una comparación de modelos. Todos los proveedores de `models.json` están deshabilitados; sus nombres y disponibilidad requieren verificación antes de cualquier corrida real.

El modo offline valida selección y configuración sin llamar modelos:

```bash
python3 evals/clinical_frontier/run_eval.py --dry-run
python3 evals/clinical_frontier/run_eval.py --dry-run --case urgent_chest_pain
```

Las opciones de los comandos se comprobaron contra el parser actual. El perfil Alfred conserva el preload de skills: `--ignore-rules` fue retirado porque upstream también omite las skills precargadas con esa opción. No se comprobó la existencia de esas skills, las credenciales, la disponibilidad de modelos ni la configuración del gate en un proceso real.

Ejecutar exige `--execute` y habilitar explícitamente proveedores en una configuración separada. Cada proceso recibe un `HERMES_HOME`, `HERMES_RUNTIME_DIR` y `HOME` temporal, sin credenciales heredadas. La configuración local y las skills del usuario no se copian. Este port solamente ejecutó procesos Python de fixtures sintéticos; no ejecutó Hermes ni solicitudes externas.

Cada ejecución guarda respuestas, latencia, código de salida, timeout, hashes de dataset/configuración, paquete ciego, clave de identidad separada y plantillas de calificación clínica y de estilo. Tokens y costo permanecen `null` si no se reportan. URLs y números extraídos son proxies, no validación de respaldo clínico. Las respuestas históricas de septiembre no se trasladaron como resultados vigentes.

Para resumir calificaciones humanas completas:

```bash
python3 evals/clinical_frontier/score_eval.py /ruta/a/corrida --scores /ruta/a/clinical_scores.jsonl
```

El scorer exige dimensiones numéricas válidas, identificación del calificador y gates booleanos explícitos. Una fuente fabricada lleva evidencia y trazabilidad a cero según la rúbrica. El reporte permanece `descriptive_only` y `comparison_complete=false`: no comprueba independencia de calificadores, adjudicación, respaldo de fuentes ni incertidumbre estadística y no declara ganador. Estilo se mantiene separado.

`source_manifest.json` conserva datos históricos del PR fechados 2026-09-11. Ninguna fuente fue consultada o verificada nuevamente durante este port. Los `reference_urls` son candidatos para una adjudicación futura, no evidencia clínica vigente certificada.

Pruebas deterministas: `tests/evals/test_clinical_frontier.py`, mediante `scripts/run_tests.sh` y el intérprete de tests del repo. Sus archivos y procesos se restringen a directorios temporales.
