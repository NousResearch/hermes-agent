# Continuity Queue / Dependency / File-Lock State Model (v1)

Date: 2026-03-08  
Status: active (subordinate doctrine module)  
Parent doctrine: `docs/ops/unified_operating_doctrine_v1.md`

## Purpose
Deterministic task orchestration substrate for control-plane autopilot and successor-safe replay.

## Runtime tables
Declared by `ops/openclaw/continuity/init_db.sh`:
- `work_queue`
- `task_transitions`
- `task_handoff_packets`
- `task_dependencies`
- `task_file_targets`
- `file_locks`
- `task_artifacts`
- `continuity_events`

## Task lifecycle
Primary FSM:
- `QUEUED -> RUNNING -> REVIEW -> DONE`
- Failure exits: `BLOCKED | FAILED | ROLLED_BACK`

Role boundary policy:
- `work_queue.role_required` is the runtime boundary owner for current step state.
- Claims/transitions must be executed by matching `--actor-role` unless explicit manual override (`--allow-any-transition`).
- Default role progression in arbitrator:
  - `RUNNING` keeps actor role
  - `REVIEW -> validator`
  - `DONE -> librarian`
  - `BLOCKED|FAILED|ROLLED_BACK -> sre_watchdog`

Policy:
- `REVIEW -> DONE` should be validator-driven.
- File locks should be released on terminal states.
- `queue_arbitrator.sh transition` enforces transition matrix + role boundary by default; bypass requires explicit `--allow-any-transition` (manual recovery only).

## Dependency model
- `task_dependencies(task_id, depends_on_task_id, relation)`
- `relation=blocks` = prerequisite must reach `DONE`
- Ready-list query: queued tasks with no unsatisfied `blocks` dependencies
- Cooldown gate: `work_queue.cooldown_until` suppresses ready/claim eligibility until UTC deadline is reached.

## File-lock model
- `task_file_targets` declares intended mutation paths before execution.
- `queue_arbitrator.sh claim` attempts atomic lock acquisition in `file_locks`.
- Active lock states:
  - `ACTIVE`
  - `RELEASED`
  - `EXPIRED`

## Artifact trace model
- `task_artifacts` stores normalized artifact refs per task.
- `task_transitions.evidence_ref` links gate decisions to evidence bundles.

## Role handoff packet model
- `task_handoff_packets` persists explicit cross-role transitions with:
  - `from_role -> to_role`
  - `transition_event_id` linkage
  - `parent_task_id` task linkage
  - `evidence_refs_json`
  - `gate_metadata_json`
  - `created_at`, `next_gate`, retry/failure metadata
- Packets are emitted on role change in queue arbitration/sync paths and are queryable via `queue_arbitrator.sh handoffs`.

## Operational commands
- Sync from autopilot state:
  - `bash ops/openclaw/continuity/queue_sync_from_autopilot_json.sh`
- List ready tasks:
  - `bash ops/openclaw/continuity/queue_arbitrator.sh ready-list --json`
- Claim one task:
  - `bash ops/openclaw/continuity/queue_arbitrator.sh claim --agent <name> --actor-role <role> --json`
- Replay trace for a task:
  - `bash ops/openclaw/continuity/queue_arbitrator.sh trace --task-id autopilot:apply_fixes --json`
- List handoff packets:
  - `bash ops/openclaw/continuity/queue_arbitrator.sh handoffs --json`
- Guided lock/dependency remediation preview:
  - `bash ops/openclaw/continuity/queue_arbitrator.sh remediate --expire-overdue-locks --release-terminal-locks --requeue-resolved-blocked --json`
- Integrity check:
  - `bash ops/openclaw/continuity/db_integrity_check.sh --strict --json`
- Runtime doctor:
  - `bash ops/openclaw/continuity/swarm_runtime_check.sh --strict --json`

## Successor invariants
- No silent schema drift: use `init_db.sh` as canonical DDL.
- No mutation without declared targets in lock-aware executors.
- No promotion without transition evidence refs.
