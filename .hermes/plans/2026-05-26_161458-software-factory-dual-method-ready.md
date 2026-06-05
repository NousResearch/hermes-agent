# Plan de trabajo — SitioUno Software Factory con dos métodos en paralelo

Fecha: 2026-05-26 16:14:58
Workspace principal inspeccionado: `/home/jean/Projects/hermes-agent-original`
Workspace legado/UX inspeccionado: `/home/jean/Projects/hermes-workspace-su`

## 1. Objetivo

Dejar lista la base operativa de la SitioUno Software Factory antes de recibir el primer proyecto piloto.

La fábrica debe poder operar dos métodos en paralelo sobre el mismo proyecto/spec, con aislamiento, métricas, revisión cruzada y reporte ejecutivo:

1. **Método A — Zeus Factory Native**
   - Zeus como orquestador.
   - Kanban Hermes como workflow operativo.
   - DB de progreso/métricas/auditoría.
   - Jobs determinísticos de monitoreo, revisión, test y reporte.
   - Motores Claude Code, Codex y OpenHands asignados por criterio.

2. **Método B — BMAD/Hybrid Factory**
   - BMAD usado para PRD, arquitectura, epics/stories, sprint/status y revisiones adversariales.
   - Zeus mantiene la capa operativa: Kanban, DB, jobs, métricas, seguridad, benchmarking y entrega.
   - Se instala o aplica solo en worktree/rama aislada para no contaminar el flujo nativo.

El resultado debe permitir que, cuando Jean entregue el primer proyecto, solo falte ejecutar el intake del proyecto y lanzar las dos lanes.

## 2. Contexto descubierto

### 2.1 Hermes Agent original

Repo activo: `/home/jean/Projects/hermes-agent-original`

Elementos útiles existentes:

- Kanban multi-board ya existe en Hermes:
  - Código: `hermes_cli/kanban.py`, `hermes_cli/kanban_db.py`, `tools/kanban_tools.py`
  - DB por defecto compartida: `~/.hermes/kanban.db`
  - Boards aislados: `~/.hermes/kanban/boards/<slug>/kanban.db`
  - Estados actuales: `triage`, `todo`, `scheduled`, `ready`, `running`, `blocked`, `review`, `done`, `archived`
  - Soporta perfiles/trabajadores y workspaces por tarea.
- Plugins/skills Kanban ya existen:
  - `plugins/kanban/`
  - Skills: `kanban-orchestrator`, `kanban-worker`, `kanban-codex-lane`
- AGENTS.md documenta el contrato de desarrollo y la arquitectura de herramientas.

Conclusión: no hay que reescribir Kanban; hay que crear una capa “factory” encima y usar boards por proyecto/lane.

### 2.2 Hermes Workspace SU / legado tipo OpenClaw

Workspace inspeccionado: `/home/jean/Projects/hermes-workspace-su`

Elementos útiles existentes:

- `swarm.yaml` define un roster semántico de agentes:
  - `orchestrator`, `builder`, `reviewer`, `qa`, `researcher`, `ops-watch`, `maintainer`, `strategist`, `km-agent`, `inbox-triage`
- `AGENTS.md` del workspace especifica contrato de operación:
  - Builder implementa.
  - Reviewer gatea.
  - QA verifica.
  - Orchestrator enruta y exige greenlight humano.
- API vieja/intermedia de orquestación:
  - `src/routes/api/swarm-orchestrator-loop.ts`
  - `src/server/orchestration-client.ts`
  - `src/routes/api/orchestration.ts`
- Workflow templates heredados:
  - `src/screens/gateway/lib/workflow-templates.ts`
  - Actualmente menciona `clawsuite:workflow-templates` y templates genéricos.

Conclusión: el roster y partes del loop/checkpoint tienen valor, pero deben alinearse con Hermes Kanban + DB factory. La parte “ClawSuite/OpenClaw” debe renombrarse, aislarse o limpiarse si ya no aporta.

## 3. Decisión arquitectónica propuesta

### 3.1 No revivir la factory OpenClaw

La factory OpenClaw se debe considerar cerrada. No conviene reactivarla como fuente de verdad.

Se reutiliza solo lo que agrega valor:

- Roster semántico de agentes.
- Concepto de checkpoint proof-bearing.
- UI/UX de workflows si sirve como panel visual.
- Swarm loop como referencia para monitor de workers, no como orquestador principal.

Se limpia o migra:

- Nombres `clawsuite`, `openclaw`, `swarm` cuando sean confusos para la nueva factory.
- Workflows viejos que no tengan gates, DB, benchmark ni revisión cruzada.
- Dependencias o rutas que asuman OpenClaw como backend.

### 3.2 Fuente de verdad operativa

- **Workflow visible:** Hermes Kanban board por proyecto/lane.
- **Estado/auditoría/métricas:** Factory DB.
- **Artefactos:** Archivos versionados dentro del proyecto y/o `.factory/<project-id>/`.
- **Decisiones:** Tabla `factory_decisions` y documentos `DECISIONS.md` si aplica.
- **Ejecución:** Claude Code, Codex CLI, OpenHands y subagentes Hermes según tarea.

### 3.3 DB recomendada

Para preparar algo serio y alineado con infraestructura de Jean:

- Crear esquema Postgres en Cloud SQL `fleet_registry` para la factory: `software_factory` o `factory`.
- Mantener compatibilidad local SQLite para pruebas y modo offline si hace falta.

Motivo: Jean ya pidió alinear con la DB del factory y existe Postgres privado GCP con skill `cloud-sql-fleet-registry`. Para producción de fábrica conviene Postgres; para tests unitarios se puede usar SQLite/temp DB.

## 4. Agentes a crear/alinear

### 4.1 Roster mínimo de Factory v1

| Agente | Rol | Motor preferido | Responsabilidad |
|---|---|---|---|
| `factory-orchestrator` | Director de fábrica | Zeus/Hermes | Intake, lanes, gates, asignación, métricas, reporte |
| `product-analyst` | Funcional/Product | Zeus/Claude | PRD, usuarios, flujos, criterios de aceptación |
| `solution-architect` | Arquitectura | Claude Code/Zeus | blueprint, boundaries, DB/API, riesgos |
| `implementation-planner` | Planning | Zeus/Codex | epics, historias, dependencias, task graph |
| `claude-builder` | Implementador principal | Claude Code | features/refactors multiarchivo |
| `codex-builder` | Implementador/fixer | Codex CLI | fixes acotados, tests, QA de diffs |
| `openhands-lab` | Sandbox/lab | OpenHands VM | experimentos, builds pesados, validación aislada |
| `quality-reviewer` | Review independiente | Codex/Claude alternado | revisión de spec/calidad, no self-approval |
| `security-reviewer` | Seguridad | Codex/Claude/OpenHands | auth, pagos, PII, secrets, webhooks, PCI |
| `qa-verifier` | QA | Codex/Hermes/browser | smoke, tests, evidencia, regresión |
| `devops-release` | Entrega | Claude/Codex | CI, env, deploy, runbooks |
| `factory-reporter` | Reportes | Zeus | executive reports, benchmark, métricas |

### 4.2 Reutilización del roster legado

Mapeo desde `hermes-workspace-su/swarm.yaml`:

- `orchestrator` → base para `factory-orchestrator`
- `builder` → dividir en `claude-builder` y `codex-builder`
- `reviewer` → `quality-reviewer` + parte de `security-reviewer`
- `qa` → `qa-verifier`
- `researcher` → `product-analyst` o research lane
- `ops-watch` → deterministic jobs / `devops-release`
- `strategist` → `solution-architect` o advisory lane
- `km-agent` → knowledge/artifacts curator, opcional v2

### 4.3 Perfiles Hermes

Crear o alinear perfiles bajo `~/.hermes/profiles/<agent-id>/` solo en la fase de implementación, no en este plan.

Cada perfil debe tener:

- `config.yaml` con toolsets mínimos.
- Skill core del rol: `<agent-id>-core`.
- Modelo recomendado.
- Límites de permisos.
- Reglas de evidencia obligatoria.
- Greenlight requerido para merge, deploy, destructive, external-send, credential-change.

## 5. Dos métodos paralelos

### 5.1 Lane A: Zeus Factory Native

Flujo:

1. Intake Gate.
2. Functional Gate.
3. Architecture Gate.
4. Planning Gate.
5. Implementación por tareas pequeñas.
6. Review Spec Gate.
7. Review Quality Gate.
8. Test Gate.
9. Security Gate si aplica.
10. Delivery Gate.
11. Benchmark/report.

Artifacts por proyecto:

- `FACTORY_INTAKE.md`
- `FUNCTIONAL_SPEC.md`
- `TECHNICAL_BLUEPRINT.md`
- `IMPLEMENTATION_PLAN.md`
- `KANBAN_TASK_GRAPH.md`
- `QA_REPORT.md`
- `SECURITY_REVIEW.md`
- `DELIVERY_REPORT.md`
- `ENGINE_BENCHMARK.md`

### 5.2 Lane B: BMAD/Hybrid Factory

Flujo:

1. Instalar/aplicar BMAD en worktree dedicado.
2. Generar PRD/Architecture/Epics/Stories/Sprint status.
3. Convertir stories a Kanban tasks.
4. Ejecutar con motores Claude/Codex/OpenHands, igual que lane A, pero siguiendo checkpoints BMAD.
5. Revisiones adversariales estilo BMAD:
   - Blind Hunter.
   - Edge Case Hunter.
   - Acceptance Auditor.
6. Pasar por los mismos gates de Zeus.
7. Comparar contra lane A.

Regla crítica: la lane B no debe leer la implementación de lane A hasta terminar el scoring inicial.

## 6. Modelo de datos Factory

### 6.1 Schema propuesto

Crear schema Postgres `software_factory` con tablas:

- `factory_projects`
- `factory_lanes`
- `factory_tasks`
- `factory_events`
- `factory_metrics`
- `factory_engine_benchmarks`
- `factory_method_benchmarks`
- `factory_gates`
- `factory_decisions`
- `factory_artifacts`
- `factory_agents`
- `factory_runs`
- `factory_risks`
- `factory_blockers`

### 6.2 Campos clave adicionales

Agregar explícitamente:

- `methodology`: `zeus_native`, `bmad_hybrid`, `hybrid_winner`
- `lane_id`: separa lanes del mismo proyecto.
- `engine`: `zeus`, `claude_code`, `codex`, `openhands`, `human`
- `reviewer_agent_id`: para evitar self-approval.
- `evidence_required`: boolean/json.
- `evidence_status`: `missing`, `partial`, `complete`.
- `risk_level`: `low`, `medium`, `high`, `critical`.
- `greenlight_required`: boolean.
- `human_approval_status`: `pending`, `approved`, `rejected`, `waived`.

### 6.3 Relación con Hermes Kanban

Cada `factory_tasks.kanban_id` apunta a la tarea en Hermes Kanban.

Cada proyecto debe tener boards/lane slugs como:

- `<project-slug>-zeus`
- `<project-slug>-bmad`
- `<project-slug>-integration`

O un board por proyecto con campo/metadata `lane`, si la UI lo soporta mejor.

## 7. Jobs determinísticos

Crear scripts/jobs, idealmente no-LLM salvo que necesiten juicio:

1. `factory_status_sync`
   - Sincroniza Kanban ↔ DB.
   - Detecta tareas stale y evidencia faltante.

2. `factory_git_monitor`
   - Inspecciona ramas/worktrees.
   - Registra commits/diffs.
   - Detecta conflictos o dos agentes tocando el mismo archivo.

3. `factory_test_runner`
   - Ejecuta comandos de test/build por proyecto.
   - Guarda logs como artifacts.
   - Actualiza gates y métricas.

4. `factory_reviewer_dispatch`
   - Envía tareas listas a reviewer independiente.
   - Regla: Claude implementa → Codex revisa; Codex implementa → Claude revisa; high-risk → OpenHands valida si aplica.

5. `factory_blocker_detector`
   - Clasifica bloqueos.
   - Resuelve lo obvio y escala lo humano.

6. `factory_engine_benchmark`
   - Compara motores dentro de una misma metodología.

7. `factory_method_benchmark`
   - Compara Zeus Native vs BMAD/Hybrid.

8. `factory_daily_report`
   - Reporte ejecutivo diario o bajo demanda.

9. `factory_orchestrator_tick`
   - Loop principal: lee DB/Kanban, decide próximo paso, invoca agentes o pide aprobación.

## 8. Worktrees y ramas

Cuando llegue el primer proyecto:

- Baseline común: rama original limpia.
- Lane A: `factory/<project-slug>/zeus-native`
- Lane B: `factory/<project-slug>/bmad-hybrid`
- Integración final: `factory/<project-slug>/integration`

Worktrees sugeridos:

- `.factory/worktrees/<project-slug>/zeus-native`
- `.factory/worktrees/<project-slug>/bmad-hybrid`
- `.factory/worktrees/<project-slug>/integration`

Reglas:

- No compartir working tree entre lanes.
- No mezclar commits antes del benchmark.
- No auto-merge sin aprobación humana.
- Registrar checksum/diff/test logs por lane.

## 9. Archivos probablemente a modificar en implementación

### 9.1 En Hermes Agent original

Posibles rutas nuevas:

- `hermes_cli/factory.py`
- `hermes_cli/factory_db.py`
- `hermes_cli/factory_jobs.py`
- `tools/factory_tools.py`
- `plugins/factory/`
- `skills/software-development/software-factory-orchestration/` si se decide actualizar skill in-repo.
- `tests/hermes_cli/test_factory_*.py`
- `tests/tools/test_factory_tools.py`
- `website/docs/user-guide/features/software-factory.md`

Posibles cambios:

- Integración con `hermes_cli/kanban.py` y `hermes_cli/kanban_db.py`.
- Registro de comandos CLI `hermes factory ...`.
- Exposición de tools: `factory_project_create`, `factory_task_sync`, `factory_gate_record`, `factory_report`.

### 9.2 En Hermes Workspace SU

Si se conserva como UI/panel:

- `swarm.yaml` → migrar/renombrar a roster factory o generar `factory-roster.yaml`.
- `src/routes/api/swarm-orchestrator-loop.ts` → adaptar a DB factory/Kanban o archivar.
- `src/server/orchestration-client.ts` → apuntar a factory API real.
- `src/screens/gateway/lib/workflow-templates.ts` → reemplazar `clawsuite` por `sitiouno-factory` y agregar templates Zeus/BMAD.
- `src/routes/workflows.tsx`, `src/screens/workflows/*` → mostrar lanes/gates/evidence si se usa UI.
- Documentar limpieza en `docs/factory/`.

### 9.3 En `~/.hermes`

Con aprobación en implementación:

- Crear perfiles factory bajo `~/.hermes/profiles/`.
- Crear skills core de agentes bajo `~/.hermes/skills/`.
- Crear cronjobs Hermes para jobs determinísticos si conviene.
- Configurar wrappers si se decide usar CLI wrappers por agente.

## 10. Plan de implementación por fases

### Fase 0 — Confirmación de alcance

Objetivo: evitar construir sobre supuestos equivocados.

Tareas:

1. Confirmar que los dos métodos a probar serán:
   - Zeus Factory Native.
   - BMAD/Hybrid Factory.
2. Confirmar si la DB factory debe ser Postgres Cloud SQL desde v1 o SQLite local primero.
3. Confirmar si `hermes-workspace-su` será UI oficial de la factory o solo fuente para rescatar código.

Salida:

- Decisión escrita en `factory_decisions` o documento inicial.

### Fase 1 — Auditoría técnica del legado OpenClaw/Workspace

Objetivo: decidir qué se reutiliza, migra o elimina.

Tareas:

1. Inventariar rutas `openclaw`, `clawsuite`, `swarm`, `workflow`, `orchestration`.
2. Clasificar cada componente:
   - Reutilizar.
   - Migrar/renombrar.
   - Archivar.
   - Eliminar.
3. Validar si el loop `swarm-orchestrator-loop.ts` todavía compila y qué dependencias tiene.
4. Generar `LEGACY_FACTORY_AUDIT.md`.

Criterios de aceptación:

- Lista completa de componentes legado.
- Decisión por componente.
- Sin cambios destructivos todavía.

### Fase 2 — Diseño final DB + migración

Objetivo: tener schema factory idempotente.

Tareas:

1. Crear SQL idempotente para `software_factory`.
2. Crear tablas, índices, constraints y views mínimas.
3. Crear migración local/test si aplica.
4. Aplicar a Postgres Cloud SQL usando scripts del skill `cloud-sql-fleet-registry`.
5. Verificar con queries `information_schema`.

Criterios de aceptación:

- Schema existe.
- Tablas consultables.
- Migración puede correr dos veces sin romper.
- Test de insert/select básico pasa.

### Fase 3 — CLI/Tools Factory

Objetivo: operar factory desde Hermes.

Comandos propuestos:

- `hermes factory init <project>`
- `hermes factory agents list`
- `hermes factory project create`
- `hermes factory lane create --method zeus_native|bmad_hybrid`
- `hermes factory task sync`
- `hermes factory gate record`
- `hermes factory benchmark report`
- `hermes factory status`

Tools propuestos:

- `factory_project_create`
- `factory_lane_create`
- `factory_task_create`
- `factory_event_log`
- `factory_gate_record`
- `factory_artifact_record`
- `factory_status`
- `factory_benchmark_report`

Criterios de aceptación:

- CLI crea proyecto/lane/task en DB.
- Tools pueden ser llamadas por agentes Hermes.
- Tests unitarios cubren create/sync/status.

### Fase 4 — Agentes/perfiles/skills

Objetivo: dejar roster factory listo.

Tareas:

1. Crear skills core para cada agente mínimo.
2. Crear perfiles Hermes con toolsets mínimos.
3. Crear o actualizar roster YAML factory.
4. Crear wrappers opcionales.
5. Validar que cada agente puede cargar su perfil y responder con contrato de evidencia.

Criterios de aceptación:

- Perfiles existen y son cargables.
- Skills existen y contienen reglas de rol/gates/evidencia.
- No hay agente con permisos excesivos.

### Fase 5 — Kanban + lanes paralelas

Objetivo: que el primer proyecto pueda crear boards/lane tasks automáticamente.

Tareas:

1. Crear convención de boards.
2. Implementar sync Kanban ↔ DB.
3. Crear templates de tareas por gate.
4. Asegurar worktree por lane.
5. Asegurar que tareas de una lane no lean artifacts de la otra antes del benchmark.

Criterios de aceptación:

- Se puede crear un proyecto demo con dos lanes.
- Cada lane tiene tareas iniciales.
- DB refleja el estado Kanban.

### Fase 6 — Jobs determinísticos

Objetivo: automatizar observabilidad y control.

Tareas:

1. Implementar scripts de jobs en repo o `~/.hermes/scripts/`.
2. Crear cronjobs Hermes si procede.
3. Jobs deben ser silenciosos si no hay nada que reportar.
4. Jobs deben registrar eventos y artifacts.

Criterios de aceptación:

- `factory_status_sync` funciona.
- `factory_git_monitor` funciona sobre repo demo.
- `factory_test_runner` registra logs.
- `factory_daily_report` produce salida ejecutiva.

### Fase 7 — Integración motores Claude/Codex/OpenHands

Objetivo: poder delegar trabajo real por motor.

Tareas:

1. Smoke checks:
   - `claude auth status --text`
   - `codex login status`
   - OpenHands health check.
2. Crear reglas de routing por tipo de tarea.
3. Crear prompts estándar por agente/motor.
4. Registrar métricas por ejecución.

Criterios de aceptación:

- Cada motor puede ejecutar un dry-run controlado.
- La salida queda registrada como artifact/event.
- Review independiente obligatorio.

### Fase 8 — BMAD/Hybrid lane

Objetivo: preparar método B sin tocar proyectos reales todavía.

Tareas:

1. Clonar/anotar BMAD si no está disponible.
2. Crear instalador o guía para branch/worktree.
3. Mapear artifacts BMAD a DB/Kanban.
4. Crear templates de review adversarial.

Criterios de aceptación:

- Lane BMAD puede inicializarse en un repo demo.
- Produce artifacts esperados.
- Se compara con lane Zeus usando criterios comunes.

### Fase 9 — Limpieza legado OpenClaw

Objetivo: eliminar confusión sin perder componentes útiles.

Tareas:

1. Renombrar claves `clawsuite` a `sitiouno-factory` donde aplique.
2. Archivar docs o rutas obsoletas bajo `docs/legacy-openclaw/` si se conservan.
3. Eliminar dependencias muertas solo después de tests.
4. Actualizar README/AGENTS del workspace.

Criterios de aceptación:

- No quedan referencias activas a OpenClaw como backend de factory.
- Lo reutilizado queda documentado.
- Tests del workspace pasan.

### Fase 10 — Dry run end-to-end

Objetivo: demostrar que la fábrica está lista antes del primer proyecto real.

Tareas:

1. Crear proyecto demo “factory-smoke”.
2. Crear lanes Zeus y BMAD.
3. Crear 2–3 tareas pequeñas ficticias.
4. Ejecutar sync, review dispatch, test runner y report.
5. Generar `FACTORY_READY_REPORT.md`.

Criterios de aceptación:

- Proyecto demo visible en DB y Kanban.
- Jobs corren sin errores.
- Reporte compara lanes aunque sea con tareas dummy.
- Jean puede dar el primer proyecto real.

## 11. Criterios de “factory lista”

La factory estará lista cuando:

1. Exista DB factory operativa.
2. Exista roster de agentes/perfiles/skills.
3. Existan dos métodos/lane templates: Zeus Native y BMAD/Hybrid.
4. Kanban cree y muestre tareas por proyecto/lane.
5. Jobs determinísticos registren estado/evidencia.
6. Claude Code, Codex y OpenHands estén verificados.
7. Exista mecanismo de worktrees/ramas aisladas.
8. Exista reporte ejecutivo y benchmark.
9. El legado OpenClaw esté auditado y limpio/migrado.
10. Se complete un dry-run end-to-end sin proyecto real.

## 12. Riesgos y mitigaciones

| Riesgo | Impacto | Mitigación |
|---|---|---|
| Burocracia excesiva | Lento para tareas pequeñas | Modos Fast Lane, Standard, Critical |
| DB Postgres privada no accesible desde local | Bloquea migración | Usar scripts del bastión o OpenHands VM con acceso VPC |
| OpenClaw legado confunde naming/rutas | Errores operativos | Auditoría y limpieza explícita |
| Dos lanes tocan mismos archivos | Conflictos | Worktrees separados y git monitor |
| Implementador se autoaprueba | Baja calidad | Reviewer independiente obligatorio |
| BMAD demasiado pesado | Overhead | Mantenerlo como lane benchmark/hybrid, no imponerlo siempre |
| Métricas incompletas | Benchmark injusto | Evidence contract + jobs determinísticos |
| Permisos excesivos en agentes | Riesgo operativo | Toolsets mínimos + greenlight humano |

## 13. Validación técnica propuesta

Comandos/checks durante implementación:

- `python -m pytest tests/hermes_cli/test_factory_*.py`
- `python -m pytest tests/tools/test_factory_tools.py`
- `python -m pytest tests/hermes_cli/test_kanban_*.py` según cambios.
- `bash ~/.hermes/skills/devops/cloud-sql-fleet-registry/scripts/health_check.sh`
- Query Postgres: `SELECT 1 AS ok;`
- `claude auth status --text`
- `codex login status`
- `bash ~/.hermes/skills/openhands-gcp/scripts/health_check.sh`
- Dry run factory: `hermes factory init factory-smoke --dry-run`

## 14. Recomendación ejecutiva

Recomiendo implementar una fábrica **híbrida y medible**:

- Base operativa propia de Zeus: Kanban + DB + jobs + gates + benchmarking.
- BMAD solo como segunda lane y fuente de buenas prácticas.
- El legado OpenClaw no debe revivirse; se rescata roster/checkpoints/UI si aportan y se limpia el resto.
- El primer proyecto real debe correr con autonomía nivel 3: ejecución autónoma con aprobación humana final para merge/deploy.

## 15. Próximo paso cuando Jean apruebe

Implementar Fases 1 a 3 primero:

1. Auditoría técnica completa del legado.
2. Schema DB factory en Postgres/Cloud SQL.
3. CLI/tools mínimos `hermes factory`.

Luego Fases 4 a 10:

4. Agentes/perfiles/skills.
5. Kanban/lanes.
6. Jobs.
7. Motores.
8. BMAD.
9. Limpieza OpenClaw.
10. Dry-run end-to-end.
