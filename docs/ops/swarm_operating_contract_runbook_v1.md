# Swarm Operating Contract Runbook (v1)

Date: 2026-03-08  
Status: active (subordinate doctrine module)  
Parent doctrine: `docs/ops/unified_operating_doctrine_v1.md`  
Canonical role contract: `ops/openclaw/architecture/swarm_role_contracts.v1.yaml`

## Goal
Make planner/executor/validator/SRE/librarian collaboration deterministic and successor-operable.

## Role-to-command mapping (control-plane)

### Planner
Inputs:
- `spec_backlog_summary.md`
- continuity latest checkpoint
- research pack notes

Outputs:
- task DAG in continuity DB (`task_dependencies`)
- scoped file targets (`task_file_targets`)

Primary commands:
- `bash ops/openclaw/continuity/queue_sync_from_autopilot_json.sh --json ...`
- `bash ops/openclaw/continuity/queue_arbitrator.sh ready-list --json`
- `bash ops/openclaw/continuity/queue_arbitrator.sh claim --agent <name> --actor-role planner --json`

### Executor
Inputs:
- claimed task + lock targets
- plan artifacts

Outputs:
- mutation + artifacts

Primary commands:
- `bash ops/openclaw/continuity/queue_arbitrator.sh claim --agent <name> --actor-role executor --json`
- execute task work
- `bash ops/openclaw/continuity/queue_arbitrator.sh transition --task-id <id> --to-status REVIEW --actor-role executor --evidence-ref <refs> --json`

### Validator
Inputs:
- executor artifacts
- gate matrix

Outputs:
- APPROVE/BLOCK evidence

Primary commands:
- run quality gates
- `bash ops/openclaw/continuity/queue_arbitrator.sh transition --task-id <id> --to-status DONE|BLOCKED --actor-role validator --evidence-ref <refs> --release-locks --json`
- competitive parity weekly validator path: `bash ops/openclaw/continuity.sh parity-run --json`

### SRE / Watchdog
Inputs:
- runtime status + continuity events

Outputs:
- low-noise blocker events and rollback triggers

Primary commands:
- `bash ops/openclaw/continuity/db_integrity_check.sh --strict --json`
- `bash ops/openclaw/continuity/swarm_runtime_check.sh --strict --json`
- `bash ops/openclaw/continuity/gtc_v2_sync.sh --strict --json` (connector evidence + gateboard readiness)
- `bash ops/openclaw/continuity/queue_arbitrator.sh handoffs --json` (inspect cross-role packets)
- `bash ops/openclaw/continuity/queue_arbitrator.sh locks --active-only --json` (lock contention view)
- `bash ops/openclaw/continuity/queue_arbitrator.sh remediate --expire-overdue-locks --release-terminal-locks --requeue-resolved-blocked --json` (guided recovery dry-run)
- `bash ops/openclaw/continuity/normalize_event_sources.sh --json` (queue-disciplined continuity producer path)

### Librarian
Inputs:
- reports + traces + teardowns

Outputs:
- curated knowledge notes + index updates

Primary outputs:
- `reports/*.md`
- `obsvault_yq_terminal/10_Projects/Shared/Research/*.md`
- `memory/*.md`

## Mandatory handoff packet fields
- `task_id`
- `from_role`
- `to_role`
- `evidence_refs`
- `next_gate`

## Non-negotiable rules
- No executor mutation without declared lock targets.
- No claim/transition that violates task `role_required` boundary (unless explicit manual recovery override via `--allow-any-transition`).
- No cross-role transition without persisted handoff packet (`from_role`, `to_role`, evidence refs, gate metadata, timestamps, task linkage).
- No DONE transition without validator evidence.
- No noisy watchdog spam; blockers only.
- No canonical knowledge promotion without traceable artifact refs.

## Executable operability checks
- Validate role/packet/gating contract + runbook command wiring:
  - `bash ops/openclaw/architecture/check_swarm_operability.sh --json`
- Validate runtime health boundary in one command (operability + DB invariants + continuity-now + queue snapshot):
  - `bash ops/openclaw/continuity/swarm_runtime_check.sh --strict --json`
  - dispatcher: `bash ops/openclaw/continuity.sh swarm-check --strict --json`
  - connector sync: `bash ops/openclaw/continuity.sh gtc-sync --strict --json`
- Validate full architecture pack (contracts + schemas + canonical templates + swarm operability):
  - `bash ops/openclaw/architecture/validate_contracts.sh --json`
- Validate component/frontmatter contract before promoting new component specs:
  - `bash ops/openclaw/architecture/validate_component_spec.sh --spec <path> --json`
