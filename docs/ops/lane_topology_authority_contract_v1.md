# Lane Topology Authority Contract v1

Date: 2026-03-20  
Status: active (Wave 4 docs-first bounded slice)  
Parent doctrine: `docs/ops/unified_operating_doctrine_v1.md`

## 0) Purpose and boundary

This contract codifies the missing A4 authority surface as machine-readable policy:
- canonical lane-class taxonomy,
- explicit control/workflow lease ownership,
- fencing-term semantics,
- ticketed mutation authority boundaries,
- lease-jeopardy fail-close behavior,
- deterministic takeover sequence for failover/succession.

v1 is **contract-first** and intentionally low-risk:
- no broad runtime refactor,
- no immediate Mutation Gateway hard-enforcement,
- no rewrite of existing lane-boundary packet semantics.

It complements (does not replace):
- `docs/ops/lane_boundary_contract_v1.md`
- `docs/ops/session_topology_contract_v1.md`
- `docs/ops/session_topology_transport_contract_v1.md`

## 1) Canonical authority model

v1 defines lane classes explicitly so authority is inspectable instead of implied.

Required lane classes:
- `CAL` — Control Authority Lane (topology-governance authority)
- `WLL` — Workflow Lead Lane (execution orchestration authority)
- `EXECUTION`
- `RESEARCH`
- `BROWSING`
- `VERIFICATION`
- `ROUTING`
- `SUCCESSION`
- `OPS`

Authority invariants:
1. Exactly one active control lease holder (`CAL`) at a time.
2. Exactly one active workflow lease holder (`WLL`) at a time.
3. All high-risk mutations are bounded by ticket policy + attestation policy.
4. If lease certainty is lost (`jeopardy/unknown`), mutation posture must fail closed.

## 2) Lease and fencing semantics

Two lease families are modeled:
- `control_lease` (governs topology authority and lease issuance)
- `workflow_lease` (governs orchestration/execution leadership)

Required lease fields include:
- stable lease id,
- holder lane + holder epoch,
- fencing term,
- issuance/expiry timestamps,
- status (`active|jeopardy|expired|revoked`),
- jeopardy reason when status is `jeopardy`.

Operational expectation:
- stale term/ticket usage is rejected fail-closed,
- takeover rotates leases/fencing terms before resume.

## 3) Ticketed mutation authority

v1 introduces explicit mutation-ticket policy fields:
- risk tiers requiring tickets,
- maximum ticket TTL,
- required mutation-ticket fields,
- high-risk posture mode (`audit_only|ticket_required`).

Ticket primitives in v1:
- `operation_id`
- `ticket_id`
- `issued_at` / `expires_at`
- `fencing_term`
- `integrity_hash`

Action-intent primitives in v1 (for actuation-boundary override enforcement):
- `schema_version=lane.action_intent.v1`
- `intent_id`
- `operation_id`
- `intent_type`
- `mutation_class` (`none|read_only|drain|mutating`)
- `issued_at` / optional `expires_at`

Attestation artifacts are canonicalized as structured objects (`lane.mutation_attestation.v1`) and consumed by
runtime authority checks alongside legacy attestation names during staged migration.

Invariant:
- high-risk mutation classes must not rely on narrative authority claims.

## 3.5) Explicit override posture at actuation boundaries

v1 now carries a canonical override-governance envelope on the lane-topology authority contract:

- `normal` — default posture; high-risk mutation still requires lease/ticket/attestation checks.
- `stop` — fail-close posture; any action intent with mutating class beyond `none` is blocked.
- `read_only` — only `none` and `read_only` mutation classes are allowed.
- `drain` — allows `none`, `read_only`, and `drain` classes so stale-task cleanup can proceed without reopening general mutation.

Required boundary behavior:
1. Override posture is evaluated before ticket/attestation acceptance.
2. Non-`normal` posture requires a machine-readable action-intent object.
3. Missing/invalid/mismatched action-intent packets fail closed.
4. Override violations surface explicit rejection codes (`override_stop_active`, `override_read_only_violation`, `override_drain_violation`).

## 4) Jeopardy rule (fail-closed)

`jeopardy_policy.fail_closed_on_lease_uncertainty` is mandatory and fixed `true` in v1.

When lease certainty is unavailable, runtime behavior is expected to:
1. block mutation at authority boundary,
2. emit explicit jeopardy/block reason,
3. require takeover protocol completion before normal mutation resumes.

## 5) Deterministic takeover sequence

v1 requires a deterministic, replay-friendly takeover sequence:
1. `freeze_mutation`
2. `revoke_stale_tickets`
3. `reconcile_by_replay`
4. `issue_new_leases`
5. `resume_with_new_tickets`

This sequence aligns A3 failover posture with A4 authority semantics.

## 6) Artifacts in this slice

- Contract schema: `docs/ops/schemas/lane_topology_authority_contract.schema.json`
- Contract template: `docs/ops/templates/lane_topology_authority_contract.template.json`
- Attestation schema: `docs/ops/schemas/mutation_attestation.schema.json`
- Attestation template: `docs/ops/templates/mutation_attestation.template.json`
- Action-intent schema: `docs/ops/schemas/lane_action_intent.schema.json`
- Action-intent template: `docs/ops/templates/lane_action_intent.template.json`
- Validation wiring: `ops/openclaw/architecture/validate_contracts.sh`
- Staged runtime gate helper (high-risk token path): `ops/openclaw/continuity/lane_authority_gate.py`

Current staged enforcement boundary:
- enforced for high-risk (`high|critical`) token-validated mutator ingress paths,
- requires lease-active certainty + attestation set (legacy names and/or structured `--attestation-object` artifacts) + valid ticket/fencing freshness,
- internal-bypass mutation paths run Stage B allowlist classification at ingress (allowlisted vs unknown callsites),
- Stage B closeout evidence is surfaced in `verify_gate_status.sh` (`internal_bypass_stage_b`) as a windowed unknown-callsite summary from `mutator_ingress_audit.jsonl`,
- Stage C trial behavior is armed only when `OPENCLAW_INTERNAL_BYPASS_ALLOWLIST_MODE=enforce` **and** `OPENCLAW_INTERNAL_BYPASS_ENFORCE_EVIDENCE_REF` is set: high-risk allowlisted internal callsites must pass lane authority gate checks before mutation is allowed.

## 7) Staged internal-bypass authority migration plan (bounded)

Internal bypass (`OPENCLAW_INTERNAL_MUTATION=1`) remains intentionally out-of-scope for hard enforcement in v1,
but migration is defined as a deterministic staged plan:

1. **Stage A — Audit-only baseline**
   - Keep current behavior (`internal_bypass` allowed with callsite required).
   - Require audit capture (`mutator_ingress_audit.jsonl`) with per-callsite volume tracking.
   - Exit condition: no unknown/uncategorized callsites in rolling audit window.

2. **Stage B — Soft gate / explicit allowlist**
   - Implemented in `mutator_ingress_guard.sh` via `OPENCLAW_INTERNAL_BYPASS_ALLOWLIST_MODE=soft` (default).
   - Bounded allowlist is seeded with known runtime callsites and may be extended by:
     - `OPENCLAW_INTERNAL_BYPASS_ALLOWLIST` (comma-separated),
     - `OPENCLAW_INTERNAL_BYPASS_ALLOWLIST_PATH` (newline list; `#` comments allowed).
   - Unknown callsites remain allowed temporarily but are surfaced as elevated warnings + explicit audit detail.
   - Closeout evidence is reviewed through `ops/openclaw/continuity/verify_gate_status.sh --json` under `internal_bypass_stage_b`:
     - windowed unknown-callsite totals,
     - top unknown callsites,
     - closeout readiness (`closeout_ready`) and failure reason.
   - Rollback toggle: `OPENCLAW_INTERNAL_BYPASS_ALLOWLIST_MODE=off`.
   - Exit condition: unknown callsites burn down to zero for two consecutive review windows with sufficient internal-bypass evidence.

3. **Stage C — Enforced authority for internal bypass (trial)**
   - Trial is explicitly armed with:
     - `OPENCLAW_INTERNAL_BYPASS_ALLOWLIST_MODE=enforce`, and
     - non-empty `OPENCLAW_INTERNAL_BYPASS_ENFORCE_EVIDENCE_REF`.
   - In this armed state, high-risk allowlisted internal callsites (`risk_tier=high|critical`) must also pass `lane_authority_gate.py` checks (ticket/attestation/fencing).
   - Unknown callsites continue to hard-block in enforce mode.
   - Break-glass rollback remains `OPENCLAW_INTERNAL_BYPASS_ALLOWLIST_MODE=off` (or reverting to `soft`).
   - Exit condition: replay fixtures + incident-free soak prove no false-block regressions.

Rollback condition:
- Any sustained false-block on critical continuity automation (>N incidents in review window)
  triggers immediate rollback to prior stage while preserving audit visibility and replay evidence.

## 8) Out of scope for v1

- Full mutation gateway runtime enforcement.
- End-to-end lease store/service implementation.
- Wide runtime migration of all mutation entrypoints.

These are follow-on Wave 4/5 enforcement slices after contract adoption evidence is stable.
