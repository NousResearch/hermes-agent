# Auditoría clínica Alfred vs. modelos frontera

Harness pareado y reproducible: cada sistema recibe **exactamente el texto de `question`**, sin prefijos variables. Compara (a) Alfred con rutas clínicas cargadas y (b) el mismo OpenAI en modo genérico `--safe-mode`, aislando el efecto del andamiaje. Gemini queda configurado pero deshabilitado hasta superar preflight. Anthropic/Claude está bloqueado en código y configuración.

## Diseño

- 8 casos sintéticos, sin PII, estratificados por riesgo y dominio (`dataset.jsonl`).
- Evaluación clínica ciega separada de estilo; la clave de identidad no se entrega a calificadores.
- Rúbrica 0–4: seguridad, exactitud, evidencia recuperada, actualidad, trazabilidad, calibración y utilidad (`rubric.json`).
- Gates: falla crítica, cifra clínica sin fuente y fuente fabricada.
- Métricas operativas: latencia de pared, exit code, timeout y tamaño. Tokens/costo quedan `null` si la interfaz no los reporta; nunca se estiman.
- Citas/URLs y números son proxies automatizados, no sustituyen revisión clínica.

## Uso

```bash
python3 run_eval.py --dry-run
# Piloto mínimo pareado
python3 run_eval.py --case urgent_chest_pain
# Corrida completa (16 respuestas con los dos perfiles habilitados)
python3 run_eval.py
```

Cada corrida crea `runs/<UTC>/responses.jsonl`, paquete ciego, clave separada, dos plantillas de scoring y manifiesto. Copiar `clinical_scores.template.jsonl` a `clinical_scores.jsonl`, usar al menos 2 calificadores clínicos independientes y adjudicar antes de:

```bash
python3 score_eval.py runs/<UTC> --scores runs/<UTC>/clinical_scores.jsonl
```

No usar `aggregate.json` para declarar un ganador sin cobertura pareada completa, adjudicación e intervalos de incertidumbre. Estilo se califica después o por equipo separado con `style_scores.template.jsonl`.

## Reproducibilidad y límites

- Registrar versión exacta/modelo, fecha, dataset hash y configuración. El nombre de un modelo preview no garantiza estabilidad.
- La búsqueda web puede variar; conservar respuestas y URLs recuperadas. Para una publicación formal, congelar snapshots permitidos/licenciados y verificar manualmente el respaldo de cada afirmación.
- `source_manifest.json` sólo contiene fuentes realmente recuperadas al crear este benchmark; los otros `reference_urls` son candidatos pendientes de verificación en turno.
- El harness no atribuye causalidad: la comparación Alfred/OpenAI mide el sistema completo contra el mismo modelo base aislado. Gemini mediría además cambio de modelo, por lo que debe reportarse como contraste separado.
- Esto es evaluación, no consejo clínico. Un profesional debe adjudicar seguridad y exactitud.
