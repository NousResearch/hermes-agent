# Kanban Task Graph — SMB Agent Business Core

## Lane

`business-core-hybrid` — metodología híbrida, repo local, sin sandbox deploy.

## Tasks

| ID | Fase | Owner | Reviewer | Título | Dependencias | Evidencia requerida |
|---|---|---|---|---|---|---|
| BC-000 | intake | factory-orchestrator | product-analyst | Intake y PRD | - | PRD-001 |
| BC-001 | architecture | solution-architect | quality-reviewer | ADRs de core/adapters/accounting/marketing | BC-000 | ADR-001..004 |
| BC-002 | planning | implementation-planner | quality-reviewer | Sprint plan + task graph + QA gates | BC-001 | SPRINT_PLAN, KANBAN_TASK_GRAPH, QA_GATES |
| BC-003 | architecture | solution-architect | security-reviewer | Repo organization + module boundaries | BC-002 | REPO_ORGANIZATION |
| BC-101 | implementation | claude-builder | quality-reviewer | Commercial/Sales Core schema | BC-003 | migrations + tests |
| BC-102 | implementation | claude-builder | quality-reviewer | Commercial/Sales tools | BC-101 | tools + tests |
| BC-103 | qa | qa-verifier | factory-orchestrator | Sales end-to-end smoke | BC-102 | smoke log |
| BC-201 | implementation | claude-builder | quality-reviewer | Accounting Lite schema/tools | BC-003 | migrations + tests |
| BC-202 | qa | qa-verifier | security-reviewer | Monthly report + fiscal boundary tests | BC-201 | test report |
| BC-301 | implementation | claude-builder | quality-reviewer | Marketing Core schema/tools | BC-003 | migrations + tests |
| BC-302 | integration | codex-builder | quality-reviewer | Media/video skill integration points | BC-301 | adapter contract notes |
| BC-303 | qa | qa-verifier | factory-orchestrator | Marketing campaign smoke | BC-302 | smoke log |
| BC-401 | architecture | solution-architect | security-reviewer | Adapter contracts | BC-103, BC-202, BC-303 | adapter specs |
| BC-501 | docs | factory-reporter | product-analyst | Agent inheritance package | BC-401 | skills/templates/demo prompts |
| BC-999 | delivery | factory-orchestrator | Jean | Final QA and delivery report | all | DELIVERY_REPORT |

## Gate sequencing

1. Intake gate — BC-000.
2. Architecture gate — BC-001/BC-003.
3. Planning gate — BC-002.
4. Implementation gates — per sprint.
5. Quality/test/security gates — per module.
6. Delivery gate — after inheritance package.

## Evidence policy

No task is done only because the implementer says so. Evidence must include files, commands, tests or explicit blockers.
