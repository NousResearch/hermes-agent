# OpenClaw Orchestrator API Contract (v1)

Date: 2026-03-30  
Status: active (canonical Program B / Slice B1 contract)  
Owner: Architect  
Scope: System Orchestrator Contract Layer (S2) with evaluation-gated admission hooks (S4)

---

## 1) Purpose

Define one explicit, versioned orchestrator API boundary between OpenClaw control-plane flows and deterministic downstream engines.

This contract makes five runtime surfaces canonical and machine-readable:
1. snapshot resolve
2. artifact plan
3. artifact run
4. ordered event stream
5. replay/resync

This v1 contract is intentionally **contract-first** and **fail-closed**:
- deterministic replay is mandatory,
- admission and degraded/read-only behavior are explicit,
- idempotent retries are mandatory for side-effecting execution.

---

## 2) Authority and scope boundary

### 2.1 Canonical authority
This file is the canonical orchestrator API authority for Program B / B1.

Machine-readable companions:
- `docs/ops/schemas/orchestrator_snapshot_resolve.schema.json`
- `docs/ops/schemas/orchestrator_plan.schema.json`
- `docs/ops/schemas/orchestrator_run.schema.json`
- `docs/ops/schemas/orchestrator_event_stream.schema.json`
- `docs/ops/schemas/orchestrator_replay_resync.schema.json`
- `docs/ops/templates/orchestrator_snapshot_resolve.template.json`
- `docs/ops/templates/orchestrator_plan.template.json`
- `docs/ops/templates/orchestrator_run.template.json`
- `docs/ops/templates/orchestrator_event_stream.template.json`
- `docs/ops/templates/orchestrator_replay_resync.template.json`
- `docs/ops/schemas/orchestrator_contract_bridge_packet.schema.json`
- `docs/ops/templates/orchestrator_contract_bridge_packet.template.json`
- Bounded runtime integration surface: `ops/openclaw/continuity/orchestrator_contract_v1_surface.py`
- Mediator-ready command dispatch seam: `ops/openclaw/continuity/orchestrator_contract_v1_mediator.py`
- Bridge packet builder: `ops/openclaw/continuity/orchestrator_contract_bridge_packet_v1.py`

### 2.2 Scope-in
This contract governs:
- API payload contracts for the five surfaces,
- idempotency semantics,
- admission/degraded/read-only semantics,
- ordered stream and cursor handling,
- replay/resync hook behavior.

### 2.3 Scope-out
This contract does not replace:
- mutation attestation lifecycle contract authority (`mutation_attestation` remains its own schema/contract),
- model routing qualification policy,
- continuity integration policy.

It composes with those layers through explicit references and fail-closed gates.

### 2.4 A4/XE bridge posture (EX-06 bounded MVP)
This contract is canonically registered as a **shared A4/XE boundary**.

Bounded posture in this slice:
- contract surfaces + validation ownership are canonical,
- bridge semantics to current XE runtime packets are explicit,
- broad runtime endpoint-enforcement is intentionally deferred.

No over-claim rule:
- do **not** claim standalone runtime implementation of `/v1/snapshots/resolve`, `/v1/artifacts/plan`, `/v1/artifacts/run`, or `/v1/replay/resync` only from this registration slice.
- the bounded integration surface (`orchestrator_contract_v1_surface.py`) is a packet-level contract runner (plan/run/event/replay semantics), not a full network API service.

---

## 3) Normative terms

- **MUST**: hard requirement.
- **SHOULD**: expected default unless explicitly justified.
- **MAY**: optional behavior.
- **Admission**: runtime gate verdict for an API request (`accepted`, `deferred`, `rejected`).
- **as_known**: bounded historical view; no look-ahead beyond caller ceilings.
- **as_corrected**: best-current reconstructed view, including correction chains.
- **Idempotent retry**: repeat call with same key/hash returns same durable run outcome.

---

## 4) Canonical API surfaces (v1)

| Surface | Method + Path | Primary purpose | Side-effect class |
|---|---|---|---|
| Snapshot resolve | `POST /v1/snapshots/resolve` | Resolve deterministic snapshot/manifest under view bounds | read/materialize |
| Artifact plan | `POST /v1/artifacts/plan` | Build deterministic execution plan from snapshot | non-side-effect planning |
| Artifact run | `POST /v1/artifacts/run` | Execute plan under idempotent run semantics | side-effecting |
| Event stream | `GET /v1/events/stream` (SSE) | Ordered orchestration events/cursors | read stream |
| Replay/resync | `POST /v1/replay/resync` | Gap recovery / cursor repair / replay window negotiation | recovery control |

---

## 5) Global invariants

### I1. Two sequence spaces
Runtime MUST maintain:
- `available_seq` (what was knowable for as-of bounded resolution),
- `event_seq` (strict order of emitted orchestration events).

### I2. View gating
- `mode=as_known` MUST include at least one ceiling (`asof_time` or `asof_seq`).
- `mode=as_known` responses MUST echo effective ceilings used (`effective_asof_*`).
- `mode=as_corrected` MAY omit ceilings and MUST identify corrected reconstruction context.

### I3. Canonical hashing
- Canonical JSON hashing MUST be used for request and output identity (`sha256` baseline).
- plan/run artifacts MUST expose deterministic hashes (`declared_inputs_hash`, `output_hash`).

### I4. Idempotency for run
`/v1/artifacts/run` MUST enforce first-write-wins tuple:
- `idempotency_key`
- `canonical_request_hash`

Retry with identical tuple MUST return same `run_id`/outcome (duplicate suppression). Same key with different canonical hash MUST fail-closed (`idempotency_conflict`).

### I5. Fail-closed admission
Unknown/missing critical fields, stale/mismatched cursor assumptions, or unsatisfied required gates MUST reject request rather than silently degrade semantics.

---

## 6) Endpoint clauses

### C1. Snapshot resolve (`POST /v1/snapshots/resolve`)
**Request MUST include:**
- mode: `as_known|as_corrected`
- optional `asof_time`, `asof_seq`
- optional `feature_flags[]`, `policy_version`

**Response MUST include:**
- `snapshot_id`, `manifest_id`
- `effective_mode`
- `effective_asof_time`, `effective_asof_seq`
- `evidence_refs[]`
- `readiness.status` and `readiness.reasons[]`

**Fail-closed rules:**
- `mode=as_known` without ceiling -> reject.
- malformed/unknown mode -> reject.

### C2. Artifact plan (`POST /v1/artifacts/plan`)
**Request MUST include:**
- `snapshot_id` (+ optional manifest reference)
- requested `artifacts[]`
- determinism policy (`canonical_json=true`, hash algorithm)

**Response MUST include:**
- `plan_id`
- `declared_inputs_hash`
- `expected_outputs[]`
- optional plan TTL

**Fail-closed rules:**
- missing `snapshot_id` -> reject.
- missing or empty `artifacts[]` -> reject.
- missing `artifact_kind` or `artifact_version` in any artifact -> reject.
- invalid determinism policy (non‑canonical JSON or unsupported hash algorithm) -> reject.

Planning is non-side-effecting and SHOULD remain available in `degraded` and bounded `read_only` posture.

### C3. Artifact run (`POST /v1/artifacts/run`)
**Request MUST include:**
- `plan_id`
- `idempotency_key`
- `canonical_request_hash`

**Request MAY include:**
- `dry_run` (default true for safety)
- `dispatch.enabled` (default false)
- `evaluation_gate` packet (attestation refs, canary mode)

**Response MUST include:**
- `run_id`
- `status` (`accepted|running|completed|failed|duplicate_suppressed|rejected`)
- `admission` verdict and reason codes
- `outputs[]` with `output_hash`/provenance when available

**Fail-closed rules:**
- missing idempotency fields -> reject.
- same key + different canonical hash -> reject (`idempotency_conflict`).
- `read_only` mode -> reject side-effecting run admission.
- required evaluation gate missing/stale `attestation_refs` or missing `gate_policy_ref` -> reject.

### C4. Event stream (`GET /v1/events/stream` SSE)
**Query:**
- `since_seq` cursor (optional)
- optional type filters

**Event envelope MUST include:**
- `event_id`, `event_seq`, `emitted_at`
- `available_at`, `available_seq`
- `type`, `severity`
- references (`entity_ref`, `payload_ref`, `correlation_id`, `idempotency_key`, `parent_event_id`)
- `dedupe_key`

**Semantics:**
- stream is ordered by strictly increasing `event_seq`.
- delivery is at-least-once; consumers MUST dedupe by `event_id`/`dedupe_key`.
- cursor and retention metadata MUST be surfaced (`next_after_seq`, `retention.min_seq/max_seq`).

**Canonical event types include:**
- Orchestration events: `artifacts.run.accepted`, `artifacts.run.completed`, `artifacts.run.failed`
- Execution supervisor health events:
  - `execution_supervisor.worker_health.canary_updated`
  - `execution_supervisor.worker_health.status_changed`
  - `execution_supervisor.probe_execution.scheduled`
  - `execution_supervisor.probe_execution.completed`
- Other system events as defined by contract extensions

### C5. Replay/resync (`POST /v1/replay/resync`)
**Purpose:** deterministic recovery when cursor gaps or consumer restarts occur.

**Request MUST include:**
- `reason` (`cursor_gap|consumer_restart|ledger_mismatch|manual_recovery`)
- `last_applied_event_seq`
- optional range preferences (`from_event_seq`, `to_event_seq`, `max_events`)

**Response MUST include:**
- `status` (`ready|partial|snapshot_reseed_required|blocked`)
- negotiated replay window (`start_seq`, `end_seq`, `available_count`)
- resume cursor (`next_since_seq`, `resync_token`, `expires_at`)
- required recovery actions[]

**Fail-closed rules:**
- if retention cannot satisfy requested replay window, response MUST explicitly require snapshot reseed instead of pretending continuity.
- unknown `reason`, invalid replay-window bounds (`from_event_seq > to_event_seq`), or invalid/missing required numeric fields MUST reject replay/resync admission with `status=blocked` and explicit operator action requirements.

### C6. EX-06 bridge semantics to XE-301/302/303 (canonical)

This contract composes with existing XE runtime substrate as follows:

1. **Idempotency retry key family**
   - XE event backbone `idempotency_key` and orchestrator run/event-stream `idempotency_key` are the same logical retry key family.

2. **Conflict detector ownership split (both required)**
   - XE event layer conflict detector: `legacy_parity_fingerprint`.
   - Orchestrator run layer conflict detector: `canonical_request_hash`.
   - These detectors are complementary, not interchangeable.

3. **Ordered event bridge mapping**
   - XE typed event `id` maps to orchestrator `event.event_id`.
   - XE typed event `sequence` maps to orchestrator `event.event_seq`.
   - XE typed event envelope timestamp maps to orchestrator `emitted_at`.
   - XE workflow/event recovery metadata anchors orchestrator cursor fields (`cursor.since_seq`, `cursor.next_after_seq`, retention bounds).

4. **Replay vs reseed boundary**
   - `workflow_state_stale` or stale expected-state mismatches MUST fail closed and drive replay/resync negotiation.
   - DLQ/backpressure or publish failures MAY return replay status `partial` or `blocked` when bounded replay remains possible.
   - Retention-cliff / unavailable replay window MUST return `snapshot_reseed_required` with explicit recovery actions (`resolve_snapshot`, `rebuild_plan`) rather than silent continuity claims.

5. **A4 takeover handoff boundary**
   - When replay/resync cannot safely converge (for example repeated blocked recovery), control must escalate to A4 lease/takeover semantics before mutating resume.
   - This preserves fail-closed authority posture with explicit jeopardy handling.

Bridge evidence packet for this mapping:
- `state/continuity/latest/evidence/ex_06_orchestrator_contract_bridge_packet_2026-04-03.json`

Bridge packet schema/template authority:
- `docs/ops/schemas/orchestrator_contract_bridge_packet.schema.json`
- `docs/ops/templates/orchestrator_contract_bridge_packet.template.json`

### C6. Execution Supervisor Health Event Integration
**Purpose:** integrate execution supervisor health monitoring into orchestrator event stream.

**Event types (canonical):**
- `execution_supervisor.worker_health.canary_updated`
- `execution_supervisor.worker_health.status_changed`
- `execution_supervisor.probe_execution.scheduled`
- `execution_supervisor.probe_execution.completed`

**Event payload MUST include:**
- `entity_ref`: reference to worker/component (e.g., `worker:codex-worker-plus-11`)
- `payload_ref`: optional reference to health artifact (e.g., `state/continuity/latest/execution_supervisor_worker_health_canary_latest.json`)
- `severity`: `info|warn|error` based on health status change
- `dedupe_key`: caller-supplied dedupe token (required; not auto-synthesized)

**Integration contract:**
- Health events flow through same orchestrator event stream as orchestration events
- Same deduplication, ordering, and replay semantics apply
- Fail-closed: invalid health event types/severity or missing required fields (including `dedupe_key`) MUST be rejected

**Bridge to execution supervisor runtime:**
- Execution supervisor health monitoring MAY emit events via orchestrator contract surface
- Events provide contractual integration point for health monitoring visibility

---

## 7) Admission and degraded-state semantics (canonical)

### Runtime modes
- `normal`
- `degraded`
- `read_only`

### Admission matrix

| Surface | normal | degraded | read_only |
|---|---|---|---|
| snapshot resolve | accepted | accepted/deferred (quota constrained) | accepted (bounded) |
| artifact plan | accepted | accepted/deferred (bounded artifact sets) | accepted/deferred (non-side-effect only) |
| artifact run | accepted | accepted for bounded `dry_run`; side-effect run may be deferred/rejected by policy | rejected |
| event stream | accepted | accepted | accepted |
| replay/resync | accepted | accepted (may return partial window) | accepted |

### HTTP posture (recommended)
- validation/policy errors: `4xx` with structured `error` envelope
- throttling/backpressure: `429` + retry hints
- overload/degraded refusal: `503` (+ retry hints)
- idempotency conflict: `409`

---

## 8) Error envelope (canonical)

All non-success responses SHOULD return:
- `error.code` (stable machine string)
- `error.message`
- `error.retryable` (boolean)
- `error.retry_after_ms` (nullable)
- `error.details` (structured object)

---

## 9) Source-input map (archive-mined + roadmap fold-ins)

| Source ID | Source artifact | Classification | Folded into clauses | Status |
|---|---|---|---|---|
| src_pdf_minimal_cp_v1 | `memory/inbound_pdfs/hl_terminal_research_pack_2026-03-03/Minimal v1 Control Plane Contract Between OpenClaw and HL Terminal.txt` | archive_mined_pdf | I1, I2, I4, C3, C4, C5 | source_input_only |
| src_pdf_analyst_orchestrator_blueprint | `memory/inbound_pdfs/hl_terminal_research_pack_2026-03-03/OpenClaw Analyst Orchestrator on Mac mini Integration Blueprint.txt` | archive_mined_pdf | C1, C2, C3, C4, C5, §7 | source_input_only |
| src_rpt_archive_orchestrator_eval | `reports/openclaw_archive_mining_orchestrator_eval_layers_2026-03-30.md` | archive_mined_report | I4, C4, C5, §7 | source_input_only |
| src_rpt_missing_layers_foldin | `reports/openclaw_missing_layers_system_roadmap_foldin_2026-03-30.md` | roadmap_foldin | §2, §4, §7 | source_input_only |
| src_rpt_exec_queue_b1 | `reports/openclaw_system_execution_queue_full_buildout_2026-03-30.md` | canonical_queue_directive | §1, §4, §10 | subordinate_active_contract |
| src_rpt_yq_future_state | `reports/yq_terminal_openclaw_future_state_roadmap_foldin_2026-03-29.md` | roadmap_foldin | I2, I5, C5 | source_input_only |
| src_rpt_yq_archived_unimplemented | `reports/yq_terminal_archived_materials_unimplemented_foldin_2026-03-30.md` | roadmap_foldin | I2, C4, C5, §7 | source_input_only |
| src_schema_mutation_attestation | `docs/ops/schemas/mutation_attestation.schema.json` | active_support_schema | C3 (evaluation_gate linkage) | subordinate_active_contract |

---

## 10) Acceptance criteria for B1

B1 is complete when:
1. A single active orchestrator API contract exists (this file).
2. Schema + template artifacts exist for snapshot resolve, plan, run, event stream, replay/resync.
3. Idempotency and admission/degraded semantics are explicit and fail-closed.
4. Source archive/roadmap fold-ins are mapped and reclassified as source inputs.

---

## 11) Change control

Any change affecting clauses I1–I5 or C1–C5 MUST:
1. update this contract,
2. update impacted schema/template companions,
3. preserve idempotency/admission fail-closed guarantees,
4. record rationale in a dated execution report.
