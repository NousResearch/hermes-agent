# ADR-001 — Fitness Coach como Agent Core local

## Contexto
Jean quiere que Zeus siga el patrón de cores funcionales invisibles: módulos de negocio/personales operados por WhatsApp/chat, con datos estructurados en el Agent Core DB del agente. La funcionalidad fitness debe cubrir coach personal, nutrición, calorías, alimentos, ejercicios, rutinas, peso corporal, progresión y feedback.

## Decisión
Implementar `Fitness Coach Core` como schema Postgres `fitness` dentro del Agent Core DB compartido del agente.

No se instala wger, SparkyFitness ni otro backend completo dentro de Zeus. Esos proyectos se usan como referencia de producto. El core propio expone herramientas JSON Hermes y mantiene datos canónicos locales.

## Datos externos
- Open Food Facts se integra como lookup/cache por barcode desde `fitness_food_search`. Se conserva atribución/licencia en `fitness.food_sources` y metadata.
- USDA FoodData Central queda modelado como source (`usda_fdc`) para futura integración cuando exista API key.
- La exercise DB inicial se mantiene local, con schema compatible con free-exercise-db, pero seed propio mínimo para evitar copiar media/licencias dudosas.

## Consecuencias
- El coach puede razonar sobre logs estructurados: metas, macros, comida, entrenamientos, sets, RPE/RIR, peso y check-ins.
- Los históricos no cambian cuando cambia una comida de la base, porque `nutrition_logs` guarda snapshot de calorías/macros.
- Las progresiones y reviews se pueden explicar con datos verificables.
- Los adapters futuros (wearables, FDC, GPX/FIT, apps móviles) sincronizan hacia `fitness`, no reemplazan el core.

## Seguridad/licencias
- No se copia código GPL/AGPL de wger/OpenNutriTracker/Food You/LiftLog/FitTrackee/Feeel.
- Open Food Facts requiere User-Agent, atribución y cuidado con ODbL/DbCL/CC BY-SA imágenes.
- Las respuestas de coaching deben ser no médicas y escalar señales de riesgo.
