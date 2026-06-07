# Sprint Plan — Factory Runtime Remediation

## Sprint 1 — Runtime autonomy repair

**Objetivo:** Corregir el runtime para que blockers no absorban proyectos autónomos y Jean sea notificado sólo cuando una decisión humana sea indispensable.

| Task | Objetivo | Owner | Reviewer | Estado |
|---|---|---|---|---|
| F0 | Intake, método, task graph, Notion/docs | Zeus/factory-reporter | factory-orchestrator | done |
| F1 | Blocker detector con categorías y acciones | Zeus/claude-builder | quality-reviewer | done |
| F2 | Watchdog alerts y notificación | Zeus/claude-builder | devops-release | done |
| F3 | Dispatcher sobre proyectos blocked | Zeus/claude-builder | security-reviewer | done |
| F4 | QA, smoke, delivery | Zeus/qa-verifier | factory-orchestrator | done |
| F5 | Corrección metodológica Notion/docs | Zeus/factory-reporter | factory-orchestrator | done/hold para proyección Notion |
| F6 | Repo-first runtime contract: índice documental + checkpoint git | Zeus/claude-builder | quality-reviewer | implemented |

## Definition of Ready

- Factory DB visible en Agent Core Postgres.
- Project-local artifact dir creado.
- Task graph y acceptance criteria definidos.
- Notion project page creada o en creación durante kickoff.

## Definition of Done

- Código/scripts implementados.
- Tests enfocados pasan.
- Scripts smoke contra Agent Core Postgres pasan.
- `DOCUMENTATION_INDEX.md` lista todos los documentos requeridos.
- Project-local artifacts quedan en checkpoint de git o tienen waiver explícito de Jean.
- Notion project page enlazada en metadata.
- Documentos requeridos existen.
- Reconciler sin anomalías críticas de repo/docs.

## Riesgos del sprint

- Ejecutar rápido y saltar PM/documents.
- Confundir Notion como opcional.
- Cerrar con waivers no aprobados por Jean.
- Cambiar dispatcher sin pruebas de dependencias.

## Cierre esperado

GREEN sólo si no hay `reconciliation_anomalies` y Notion/documents existen.
