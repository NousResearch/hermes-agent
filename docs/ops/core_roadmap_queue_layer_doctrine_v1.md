# Core Roadmap Queue Layer Doctrine (v1)

Date: 2026-03-28  
Status: active (canonical queue-layer doctrine for core-roadmap autonomy)  
Scope: core-roadmap queue truth projection into execution-program and execution-frontier surfaces.

---

## 1) Purpose

Define one canonical queue layer for core-roadmap execution so autonomous executioner surfaces consume the same truth contract for:
- ready slices,
- dependency-blocked slices,
- next-candidate selection,
- and fail-closed behavior when queue truth is invalid.

This doctrine is additive and limited to core-roadmap queue surfaces.

---

## 2) Canonical queue truth hierarchy

Primary source (preferred):
- `state/continuity/latest/core_roadmap_execution_queue.json`

Mirror source (parity witness):
- `state/continuity/latest/core_roadmap_slice_queue_2026-03-28.json`

Derived canonical queue-layer artifact:
- `state/continuity/latest/core_roadmap_queue_layer.json`
- schema id: `clawd.core_roadmap_queue_layer.v1`
- schema file: `ops/openclaw/architecture/schemas/core_roadmap_queue_layer.schema.json`

Execution surfaces must consume the derived queue-layer artifact (via continuity-current projection), not ad-hoc direct reads of individual roadmap queue files.

---

## 3) Queue-layer contract

`clawd.core_roadmap_queue_layer.v1` must include:
- `contract_status`: `ok | degraded | missing`
- `authoritative`: boolean (`true` only when `contract_status=ok`)
- `issues[]`: explicit contract/parity faults
- source metadata (`source_paths`, `selected_source_path`, `source_details[]`)
- `status_taxonomy` contract (`done|ready|running|dependency_blocked|queued`)
- `contract_fingerprint` (deterministic SHA256 over contract-critical fields)
- `summary`:
  - `total_slices`, `done_count`, `ready_count`, `running_count`,
  - `dependency_blocked_count`, `queued_count`, `open_count`, `queue_empty`,
  - normalized `state_counts{done,ready,running,dependency_blocked,queued}`
  - source-preserving `source_state_counts{}`
- candidate sets:
  - `ready_candidates[]`
  - `running_candidates[]`
  - `dependency_blocked_candidates[]`
- `next_candidates[]` and `next_candidate`

`continuity_current.sh` must validate this artifact against
`ops/openclaw/architecture/schemas/core_roadmap_queue_layer.schema.json`.
If schema validation fails, contract status degrades and queue projection enters fail-closed mode.

Candidate IDs are normalized as:
- `core_roadmap:wave{slice_id}:{slug}`

This keeps downstream wave parsing compatible with existing execution-frontier logic.

---

## 4) State mapping

Core-roadmap slice state -> queue-layer category:
- `DONE|COMPLETE|COMPLETED|SKIPPED|CANCELLED` -> done
- `READY_NOW` -> ready
- `IN_PROGRESS|RUNNING|ACTIVE` -> running
- `DEPENDENCY_BLOCKED`, `*_BLOCKED`, `READY_PENDING_*` -> dependency_blocked
- unknown/non-done states -> queued (treated as blocked candidate unless explicitly ready/running)

Dependency resolution rule:
- A `READY_NOW` slice with unresolved dependencies is downgraded to dependency-blocked in the queue layer.

---

## 5) Executor integration

`ops/openclaw/continuity/continuity_current.sh` is the canonical projector.

It must:
1. Build `core_roadmap_queue_layer.json`.
2. Prefer queue-layer candidates for execution-program queue activity when `contract_status=ok`.
3. Write queue provenance into:
   - `state/continuity/latest/execution_program_status.json`
   - `state/continuity/latest/execution_frontier_ledger.json`
4. Expose `queue_source` and queue-layer contract metadata in those surfaces.
5. Emit `queue_truth_projection_parity` for both execution surfaces, comparing
   queue-layer truth vs projected frontier truth (with runtime deltas reported separately).

Queue source values:
- `core_roadmap_queue_layer`
- `core_roadmap_queue_layer_fail_closed`
- `continuity_os_queue_db` (legacy fallback when no core queue source exists)

---

## 6) Fail-closed behavior

If core-roadmap queue source exists but queue-layer contract is degraded/invalid:
- system must fail closed through queue-layer projection,
- emit dependency-blocked sentinel candidate (`queue_layer_contract_invalid`),
- set queue source to `core_roadmap_queue_layer_fail_closed`,
- prevent silent fallback to optimistic ready dispatch.

If queue-layer contract is authoritative but projection parity is mismatched:
- system must fail closed through queue-layer projection,
- emit dependency-blocked sentinel candidate (`queue_projection_parity_mismatch`),
- mark queue-layer contract projection issues explicitly,
- keep autonomous dispatch guarded until parity returns to `match`.

Fallback to legacy SQLite queue activity is allowed only when core-roadmap queue sources are absent.

---

## 7) Orchestrator and watchdog interaction

Autonomous controller surfaces consume queue-layer truth indirectly via:
1. `execution_program_status` frontier queue,
2. `execution_frontier_ledger` supervisor state and candidate selection,
3. execution-frontier controller tick + watchdog dispatch gates.

No controller should claim dispatch eligibility from ad-hoc queue interpretation when queue-layer contract is degraded.

---

## 8) Validation requirements

Minimum regression coverage for queue-layer slices:
- continuity-current prefers queue-layer candidates over legacy DB when authoritative,
- continuity-current writes queue-layer artifact and source refs,
- queue-layer artifact validates against the hard schema contract,
- queue status taxonomy is normalized to strict category counts,
- execution-program/execution-frontier parity reports `match` for valid projections,
- degraded queue-layer source or parity mismatch enters fail-closed dependency-blocked path.

---

## 9) Queue/executor meaningful-event reporting link

Queue-layer transitions that change dispatch posture are mandatory reporting events under:
- `docs/ops/execution_meaningful_event_reporting_doctrine_v1.md`
- `docs/ops/execution_meaningful_event_reporting_checklist_v1.md`
- `docs/ops/invalid_output_retry_relaunch_contract_v1.md` (invalid-output retry/relaunch canonical path)

At minimum, report when queue/frontier transitions into or out of:
- fail-closed parity mismatch,
- dependency-blocked only queue,
- dispatchable ready queue,
- relaunch after blocked/idle condition.

Runtime reinforcement surface (additive):
- `state/continuity/latest/execution_meaningful_event_reporting_latest.json`
- Produced by `ops/openclaw/continuity/continuity_current.sh` as machine-detectable transition packet + checklist projection.

The queue layer remains authoritative for these reports; narrative-only claims are non-authoritative.

## 9A) Queue recovery + bounded pressure fold-ins (additive)

Focused recovery-pattern fold-ins from the 2026-03-29 four-repo synthesis apply immediately to queue doctrine without changing the primary queue hierarchy:

1. **Stale-processing drain/recovery is mandatory.**
   - Queue/executor supervision must not leave work silently parked in stale `claimed`, `running`, or equivalent in-flight states once freshness/lease thresholds are exceeded.
   - Recovery action must deterministically resolve to `retry`, `blocked`, or explicit relaunch planning with evidence, never silent abandonment.
   - Recovery counters and last-recovery metadata must be projected into operator-visible surfaces.

2. **Bounded queue-pressure policy is mandatory for live/high-churn loops.**
   - Any live/browser/streaming-adjacent execution path must declare bounded queue depth or equivalent backpressure window.
   - Allowed pressure actions must be explicit: `drop_freshness_stage`, `skip_optional_stage`, or `fail_closed`.
   - `drop_freshness_stage` is prohibited for transactional, mutating, or evidence-bearing work; those paths must retry, block, or fail closed instead.

3. **Pressure/recovery transitions are meaningful events.**
   - If stale work is drained/recovered or queue pressure changes dispatch posture, the transition must emit meaningful-event evidence under the reporting doctrine/checklist.
   - Queue-layer/operator surfaces should expose at minimum queue-depth, bounded-drop count, recovery-attempt count, and recovered-stale count when those mechanisms exist.

These fold-ins borrow the recovery discipline, not the implementation stack, from external donors: no broker-heavy queue stack adoption is implied by this doctrine update.

## 10) Update protocol

Any queue-layer contract/schema change must co-update:
1. this doctrine,
2. `reports/openclaw_system_source_of_truth_map_2026-03-20.md` (lane mapping),
3. continuity/execution tests proving behavior,
4. meaningful-event reporting doctrine/checklist if queue transition semantics changed.

---

## 11) Phase-2 transactional claim/commit runtime (additive lane)

Phase-2 introduces a bounded transactional runtime lane that runs **in parallel** with
current execution-frontier dispatch logic and does not replace it.

Runtime entrypoint:
- `ops/openclaw/continuity/core_roadmap_queue_layer_txn.sh`

Runtime artifact:
- `state/continuity/latest/core_roadmap_queue_transaction_runtime.json`
- schema id: `clawd.core_roadmap_queue_transaction_runtime.v1`

Runtime lock:
- `state/continuity/locks/core_roadmap_queue_transaction_runtime.lock`

Deterministic transition contract:
1. `claim` creates a new fenced claim epoch and sets task state to `claimed`.
2. `commit --to-state running` is only legal from `claimed`.
3. terminal commit (`done|blocked|retry`) is only legal from `running`.
4. stale commit attempts (mismatched claim token/epoch or missing active claim)
   are fail-closed with `stale_claim_rejected`.
5. retry commits produce explicit cooldown state (`cooldown_until`) and prevent
   immediate re-claim until cooldown expiry.

Stale-claim expiry handling:
- expired active claims are auto-recovered into `retry` with explicit cooldown,
- no silent claim takeover is permitted.

This phase is additive: the canonical queue-layer truth remains authoritative for
candidate ordering and fail-closed projection, while the transactional runtime provides
strict claim fencing and commit semantics for bounded migration into queue-driven execution.

## 12) Kimi Repo-Pass Architecture Fold-ins

Following the 2026-03-26 clean Kimi synthesis of external repo patterns:

1. **Incremental Update Discipline (LightRAG pattern):**
   The queue layer and its transactional update path must enforce incremental delta-updates to the canonical truth artifacts. Blind overwrites are prohibited to prevent state-count corruption and assure high-fidelity provenance.

2. **Structured Judge Synthesis for Terminal Commits (TradingAgents pattern):**
   Before a Phase-2 transactional `commit --to-state done` or `commit --to-state dependency_blocked` can be executed, the execution frontier MUST provide a structured "judge synthesis" verdict. The queue layer only accepts terminal transitions if accompanied by synthesized evidence that conforms to the meaningful event reporting checklist.

## 13) Phase-3 runtime convergence + guarded dispatch handoff

Phase-3 integrates runtime convergence into canonical execution surfaces while preserving queue-layer parity fail-closed behavior.

Execution surface convergence:
- `ops/openclaw/continuity/continuity_current.sh` ingests
  `core_roadmap_queue_transaction_runtime.json` and emits normalized
  `transaction_runtime` projections in both:
  - `state/continuity/latest/execution_program_status.json`
  - `state/continuity/latest/execution_frontier_ledger.json`
- core-roadmap dispatch-ready candidates are lifecycle-masked by runtime state:
  - masked: `done`, `blocked`, `claimed`, `running`, `retry` with active cooldown
  - eligible: no runtime row or `retry` whose cooldown has elapsed.

Guarded handoff semantics:
- `ops/openclaw/continuity/execution_frontier_ledger.sh` may perform transactional
  handoff (`claim -> running -> terminal commit`) for core-roadmap candidate dispatch,
  but only when explicitly enabled:
  - `OPENCLAW_EXECUTION_FRONTIER_CORE_QUEUE_TXN_HANDOFF_ENABLED=1`
- Default posture remains disabled (safe bounded rollout).
- Any claim/commit mismatch or stale/inflight ownership is fail-closed and blocks dispatch.

Operator evidence contract:
- transition-attempt evidence must include whether txn handoff was enabled,
  plus phase/status/error when blocked (`claim`, `commit_running`, `commit_terminal`).
- stale/inconsistent claim-state must never silently advance wave-close dispatch.

## 14) Phase-4 transactional handoff soak/guard extension

Phase-4 adds bounded proof/soak accounting for transactional handoff attempts inside
`execution_frontier_ledger.sh` without changing queue-layer ordering semantics.

Soak artifact:
- `state/continuity/latest/core_roadmap_queue_transaction_handoff_soak.json`
- schema id: `clawd.core_roadmap_queue_txn_handoff_soak.v1`
- schema file: `ops/openclaw/architecture/schemas/core_roadmap_queue_txn_handoff_soak.schema.json`

Guard policy:
- enabled by default (`OPENCLAW_EXECUTION_FRONTIER_CORE_QUEUE_TXN_HANDOFF_GUARD_ENABLED=1`)
- threshold control:
  `OPENCLAW_EXECUTION_FRONTIER_CORE_QUEUE_TXN_HANDOFF_GUARD_CONSECUTIVE_BLOCK_THRESHOLD`
  (default `3`, minimum `1`)
- when threshold is reached, autonomous handoff enters fail-closed preflight block with
  `transaction_runtime_handoff_soak_guard_active` until operator intervention or successful reset.

Evidence requirements:
- every txn-handoff-enabled autonomous-dispatch attempt increments soak counters
  (`attempts_total`, `applied_total`, `blocked_total`, `skipped_total`, `consecutive_blocked`),
- transition-attempt evidence must include soak contract/guard/counter snapshot fields,
- apply/block command payloads must expose `transaction_runtime_handoff_soak` summary for
  operator visibility and post-soak verification.

## 15) Phase-5 continuity convergence + controlled reset lane

Phase-5 converges the transactional-handoff soak lane into continuity/operator surfaces and
adds an auditable reset path for guarded recovery.

Continuity convergence requirements:
- `ops/openclaw/continuity/continuity_current.sh` MUST ingest/normalize
  `state/continuity/latest/core_roadmap_queue_transaction_handoff_soak.json` as an explicit
  projection contract (`transaction_runtime_handoff_soak`) in:
  - `state/continuity/current.json`
  - `state/continuity/latest/execution_program_status.json`
  - `state/continuity/latest/execution_frontier_ledger.json`
- execution-program/frontier source refs MUST include
  `core_roadmap_queue_transaction_handoff_soak` path mapping.

Guard-posture projection requirements:
- when soak contract is invalid, canonical reasons MUST include
  `transaction_runtime_handoff_soak_contract_invalid`.
- when soak guard is active, canonical reasons MUST include
  `transaction_runtime_handoff_soak_guard_active`.
- frontier supervisor-state autonomous dispatch eligibility MUST be forced fail-closed when any
  soak guard reason is present.

Controlled reset requirements:
- `execution_frontier_ledger.sh supervisor-reset-txn-handoff-guard --reason <text>` is the
  explicit operator reset lane.
- reset is fail-closed unless:
  - soak contract is valid, and
  - soak guard is currently active, and
  - reason is present and non-trivial.
- successful reset MUST preserve cumulative counters while clearing
  `counters.consecutive_blocked` and deactivating guard state.
- reset MUST append auditable contract evidence (`reset.total`, `reset.last_reset*`,
  bounded `reset.history[]`) and emit transition-attempt evidence with
  `txn_handoff_status=guard_reset_applied`.
