# Latest-8 PDF architecture-first program (2026-03-09)

## Scope
Integrated implementation guidance from batch_1_2026-03-09 PDFs into the existing continuity/swarm/web runtime without resetting Phase-1/Phase-2 progress.

Batch sources:
- Native Swarm Runtime for OpenClaw: Production-Ready Local-First Design
- Deterministic Web Interaction Layer on an OpenClaw Local-First Control Plane
- UI Design Copilot Execution OS Runtime Architecture
- YQ Terminal Design Language as an Executable Gate System
- Competitive Parity as a Living Operating Loop for YQ Terminal
- Ground-Truth Connectors v2 for a Local-First OpenClaw Control Plane
- Native Swarm Operating System for a Local-First OpenClaw Assistant
- Turning clawd-architect into a Fully Deterministic Local-First Engineering OS

## Concrete guidance extracted (high-signal)
1. Truth should remain append-only evidence + rebuildable latest pointers (event-sourcing posture, local-first).
2. Connectors must be explicit and typed (`runtime.gateway`, `queue.task`, `validation.gates`, `operator.actions`).
3. Monotonic per-connector ordering (`monotonic_seq`) is required for deterministic replay.
4. Artifact linkage must be hash-addressable; blobs should not be embedded in DB rows.
5. Readiness must be evidence-backed (`verify status`, connector freshness, runtime critical anomalies), not narrative-only status.
6. Low-noise routing should preserve transitions and suppress repeats.
7. Swarm runtime/operator checks should include connector readiness as mutation gate.

## Mapping onto current clawd-architect state
- Already strong: continuity DB + queue arbitration + lock discipline + verify-before-mutate + checkpoint/latest pointers + parity/web wrappers.
- Gap closed in this slice: unified connector substrate (evidence/index/latest surfaces) was not yet first-class.

## Implementation slice shipped
### A) Ground-Truth Connectors v2 architecture contract + schemas
- Added `ground_truth_connectors.v2.yaml`.
- Added machine schemas for evidence and latest surfaces.

### B) Runtime substrate: deterministic connector sync
- Added `ops/openclaw/continuity/gtc_v2_sync.sh`:
  - emits append-only JSONL evidence under `state/gtc-v2/evidence/**`
  - indexes/links evidence in continuity SQLite (`gtc_*` tables)
  - writes latest pointers + gateboard (`state/gtc-v2/latest/*`)
  - computes mutate gate from verify status + connector freshness + runtime critical anomalies

### C) DB schema extension
- Extended `init_db.sh` with `gtc_connector`, `gtc_evidence_index`, `gtc_artifact`, `gtc_evidence_artifact`, `gtc_task_evidence`, `gtc_checkpoint_evidence`, `gtc_latest_pointer`.

### D) Operability and integrity integration
- Added GTC checks into `db_integrity_check.sh`.
- Added `gtc-sync` command to continuity dispatcher.
- Added GTC readiness into `swarm_runtime_check.sh` as hard mutation check.
- Added GTC visibility in `continuity_now.sh` output and readiness reasons.

### E) Contract/docs/runbook updates
- Updated architecture/continuity READMEs and swarm runbook with `gtc_v2_sync.sh` path and dispatcher command.
- Updated `validate_contracts.sh` to include new contract + schemas.

## Next milestone after this slice
1. Add dedicated `gtc.event.v2` low-noise event projection (open/close lifecycle + dedupe cooldown state).
2. Extend web capture/parity runners to emit richer artifact-role manifests directly into GTC refs (HAR/trace/diff bundles).
3. Add replay command that reconstructs incident timelines from GTC evidence only.
4. Add schema-validation gate for generated `state/gtc-v2/latest/*.json` during `swarm_runtime_check`.
