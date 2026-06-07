# PRD — Factory Runtime Remediation

## 1. Contexto

El Software Factory de Zeus mostró una falla metodológica y operativa: proyectos autónomos podían quedar `blocked` durante horas mientras los cron jobs seguían reportando `ok`, y una remediación interna fue ejecutada sin completar la ruta canónica de Factory: documentos requeridos, página Notion PM y reconciliación documental antes del cierre.

## 2. Problema

El runtime necesitaba tres correctivos técnicos:

1. El detector de blockers debía clasificar y crear acciones, no solo listar bloqueos.
2. Dashboard/cron debía alertar cuando hubiese bloqueos persistentes, `blocked` sin `human_questions`, estados huérfanos y `claimed=null` repetido.
3. El dispatcher debía poder actuar sobre proyectos `blocked` para reparar estados y continuar trabajo independiente.

Además, el propio proyecto de remediación debía cumplir la metodología Factory completa: documentos, Notion, gates y evidencia.

## 3. Objetivos

- Restaurar autonomía operativa del Factory sin depender de Jean para decisiones rutinarias.
- Hacer que los bloqueos se conviertan en acciones clasificadas.
- Asegurar que toda escalación humana se registre en `factory.human_questions` y se notifique.
- Impedir que `blocked` sea un estado absorbente.
- Corregir el incumplimiento metodológico del proyecto `factory-runtime-remediation`.

## 4. No objetivos

- No mover la fuente de verdad a Notion ni Kanban.
- No permitir waivers de Notion/documentos como ruta normal.
- No crear alertas ruidosas en cada cron tick.
- No autoaprobar cambios sin evidencia.

## 5. Requisitos funcionales

| ID | Requisito | Criterio de aceptación |
|---|---|---|
| PRD-F1 | Clasificar blockers | Cada blocker tiene `action_category`, `blocker_category`, `recommended_action`, `requires_human`, `alert_key`. |
| PRD-F2 | Crear acciones | El detector escribe eventos/metadata y crea `human_questions` solo cuando es indispensable. |
| PRD-F3 | Alertar anomalías | Cron/dashboard reportan alertas para blocked>X, blocked sin pregunta humana, huérfanos y claimed-null repetido. |
| PRD-F4 | Dispatch en blocked | Proyectos `blocked` autónomos pueden ser reparados y seguir tareas independientes seguras. |
| PRD-F5 | Ruta Factory completa | El proyecto tiene Notion, PRD, ADRs, sprint plan, task graph, QA/security gates, reports y delivery evidence. |
| PRD-F6 | Índice documental obligatorio | `DOCUMENTATION_INDEX.md` lista todos los documentos requeridos para que builders/reviewers entren por el repo y no por contexto improvisado. |
| PRD-F7 | Checkpoint de commits | El reconciler/critical readiness detecta artifacts project-local modificados o untracked antes de cierre, salvo waiver explícito de Jean. |
| PRD-F8 | Memoria global de proyecto | El repo incluye `PROJECT_GLOBAL_VISION.md` con objetivo, decisiones, fase actual, repo/worktree y próximos incrementos. |

## 6. Requisitos no funcionales

- Fuente de verdad: Agent Core Postgres `factory.*`.
- Notion: capa PM/humana obligatoria para proyectos Factory, no fuente operativa.
- Repo artifacts: deben existir y ser project-local.
- Tests: deben cubrir regresiones del runtime.
- Seguridad: no exponer secrets ni logs crudos en alertas.

## 7. Estado esperado de cierre

El proyecto solo puede marcarse `completed` si:

- Notion project page existe y está linkeada en metadata.
- Todos los documentos requeridos existen en `factory/projects/factory-runtime-remediation/`.
- Gates relevantes pasaron con evidencia.
- Factory DB no reporta `reconciliation_anomalies`.
