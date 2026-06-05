# PRD-001 — Fitness Coach Core

## Objetivo
Crear un core funcional local para que Zeus/agents actúen como coach fitness personal por chat: nutrición, calorías/macros, alimentos, ejercicios, rutinas, registros de entrenamiento, peso/medidas, check-ins y revisiones de progreso.

## Principios
- Tracking primero, coaching después: la IA decide sobre datos estructurados, no sobre memoria de chat suelta.
- Local-first/self-hosted: `fitness` vive en el Agent Core DB del agente, igual que `crm`, `sales`, `accounting` y `signature`.
- APIs externas como adaptadores/cache: Open Food Facts y USDA/FDC pueden alimentar alimentos; no son la fuente operacional única.
- No copiar código GPL/AGPL: wger, OpenNutriTracker, Food You, LiftLog, FitTrackee y similares son referencias conceptuales.
- Seguridad: coaching no médico; ante dolor agudo, lesiones, embarazo, enfermedad o señales de trastorno alimentario se escala a profesional.

## Alcance v1 implementado
- Perfil fitness del usuario: altura, preferencias, alergias, equipo, lesiones, timezone.
- Objetivos: calorías, macros, hidratación, peso objetivo y delta semanal.
- Food cache local con fuentes y lookup opcional de Open Food Facts por barcode.
- Registro nutricional con snapshot de calorías/macros para preservar histórico.
- Exercise DB local con taxonomía compatible con free-exercise-db y seed inicial propio/permisivo.
- Rutinas con días y prescripción: sets, rep range, RPE/RIR, descanso, tempo, progresión JSON.
- Workout sessions y sets: peso, reps, RPE/RIR, fallo, descanso, volumen y 1RM estimado.
- Métricas corporales: peso, grasa, medidas, HR reposo, sueño, mood, energía, estrés.
- Check-ins subjetivos y recomendaciones estructuradas.
- Summaries de nutrición diaria y progreso por rango.

## Fuentes estudiadas
- wger: monolito integral fitness/nutrición/progresión/API; AGPL, solo referencia.
- workout.cool: UX/plataforma moderna y estructura de exercise DB; MIT.
- OpenNutriTracker + Food You: food diary mobile privacy-first; GPL, solo referencia.
- Open Food Facts: fuente base de productos; ODbL/DbCL/CC BY-SA imágenes.
- free-exercise-db: seed/taxonomía de ejercicios; Unlicense/public-domain style.
- LiftLog/OpenLift: logging rápido, RPE/RIR, PRs, progresión y 1RM.
- FitTrackee/workout-tracker/OpenTracks: actividad outdoor/GPX/privacidad.
- OpenHIIT/Feeel: intervalos y home workouts guiados.

## Herramientas Hermes
- `fitness_status`
- `fitness_profile_upsert`
- `fitness_goal_upsert`
- `fitness_food_upsert`, `fitness_food_search`
- `fitness_nutrition_log_create`, `fitness_nutrition_day_summary`
- `fitness_exercise_upsert`, `fitness_exercise_search`
- `fitness_routine_create`, `fitness_routine_get`
- `fitness_workout_session_create`, `fitness_workout_set_log`, `fitness_workout_finish`
- `fitness_body_metric_log_create`
- `fitness_checkin_create`
- `fitness_progress_summary`
- `fitness_coach_review`

## Criterios de aceptación
- Migración `db/modules/fitness/000001_fitness_schema.sql` crea schema/tables/seeds/grants.
- Toolset `fitness` resuelve las herramientas.
- Tests unitarios cubren registro de toolset, validaciones, rutina estructurada, cálculo set/1RM y summary.
- `agent_core_sql.py`, migrator, roles y secret sync conocen `FITNESS_*`.
- No se introducen servicios externos obligatorios ni nueva base de datos aislada.
