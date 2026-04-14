# Execution Meaningful-Event Reporting Doctrine (v1)

Date: 2026-03-28  
Status: active (subordinate doctrine module)  
Parent doctrine: `docs/ops/unified_operating_doctrine_v1.md`  
Scope: A2/A6/C1 execution-supervisor communication and reporting behavior.

---

## 1) Purpose

Prevent the exact operating failures seen in recent runs:
- completion happened but no meaningful report was emitted,
- failure/junk output was recognized but corrective relaunch was delayed,
- queue/executor state transitions were discoverable in machine surfaces but not surfaced as actionable operator truth.

This doctrine makes meaningful execution transitions **mandatory reporting triggers**, not optional narration.

---

## 2) Failure-mode classification

1. **Narration-before-correction**
   - Talking about a failure before doing the deterministic next action (relaunch or explicit block declaration).

2. **State-transition silence**
   - Queue/executor/subagent changed state, but no bounded report packet was emitted.

3. **Completion without propagation**
   - Subagent/slice/phase finished, but downstream summary and next-step posture were not reported.

4. **Failure recognized without deterministic branch**
   - Invalid/junk output was acknowledged conversationally, but not transitioned to `RELAUNCHED` or `BLOCKED` with reason.

5. **Idle stall ambiguity**
   - Executor became idle/stalled and later relaunched, but no explicit “idle -> relaunch” event was reported.

---

## 3) Mandatory trigger matrix

For each transition below, emit a meaningful-event report packet in the same control turn (or immediately next turn if tooling fails).

| Trigger code | Trigger condition | Mandatory immediate decision | Minimum reporting outcome |
|---|---|---|---|
| `SUBAGENT_FINISHED` | Child run exits (success/fail/invalid) | classify outcome (`SUCCESS` \| `FAILED` \| `INVALID_OUTPUT`) | report worker id, outcome class, evidence refs, and next branch decision |
| `SLICE_LANDED` | A bounded slice is verified landed | update queue/frontier truth posture | report slice id/lane, verification refs, and next candidate posture |
| `PHASE_LANDED` | A declared phase/wave close condition is met | declare next phase state (`continue` \| `blocked`) | report phase id, close criteria evidence, next phase entry condition |
| `WORKER_FAILED_OR_JUNK` | Child output unusable / failure class set | **relaunch immediately** if retryable; otherwise explicit blocked state | report failure class, retryability, relaunch or blocked reason |
| `QUEUE_BLOCKED` | Queue transitions to no dispatchable ready work due blocker/parity/fail-close | set deterministic blocker reason | report blocking reason, blocker candidate(s), required unblock condition |
| `QUEUE_UNBLOCKED` | Queue regains dispatchable ready work | resume dispatch eligibility | report unblock reason and next dispatchable candidate |
| `QUEUE_RELAUNCHED` | Queue/work branch relaunched after block/failure | bind to deterministic relaunch reason | report relaunch target, why now, and expected next check |
| `EXECUTOR_IDLE_TO_RELAUNCHED` | Executor was idle/stalled then new launch occurs | mark idle interval closed + launch applied | report idle duration, relaunch command/session ref, and resulting active focus |

Hard rule: for `WORKER_FAILED_OR_JUNK`, `QUEUE_BLOCKED`, and `EXECUTOR_IDLE_TO_RELAUNCHED`, explanation cannot replace action; correction/transition decision must happen first.

Invalid-output canonicalization rule:
- `INVALID_OUTPUT` outcomes MUST follow `docs/ops/invalid_output_retry_relaunch_contract_v1.md`.
- At minimum this means `SUBAGENT_FINISHED(outcome_class=INVALID_OUTPUT)` + `WORKER_FAILED_OR_JUNK` in the same supervision cycle, followed by deterministic `relaunch` or explicit `blocked`.

Recovery/pressure reporting rule:
- Stale-task drain/recovery and bounded-queue pressure transitions are not silent internals.
- If stale work is reclassified, relaunched, or blocked, emit the corresponding queue/executor packet with recovery counters/reason.
- If bounded-pressure policy forces `drop_freshness_stage`, `skip_optional_stage`, or fail-closed gating, report why that action was allowed and confirm no transactional/evidence-bearing work was silently dropped.

---

## 4) Required event packet fields (minimum contract)

Every meaningful-event report must include:
- `event_code`
- `event_at` (transition timestamp)
- `detected_at`
- `lane` and `work_unit_ref` (subagent id, slice id, wave id, or queue candidate)
- `from_state` -> `to_state`
- `decision_taken` (`relaunch`, `blocked`, `continue`, `closed`, etc.)
- `decision_reason`
- `evidence_refs[]` (state/report/test artifacts used)
- `next_action` (or explicit `none` + why)

When queue/executor state is involved, include additionally:
- `queue_source`
- `queue_truth_projection_parity.status`
- `next_candidate` (or blocker sentinel)

### Runtime reinforcement surface (additive)

`continuity_current.sh` SHOULD materialize deterministic companion artifacts:
- `state/continuity/latest/execution_meaningful_event_reporting_latest.json`
- `state/continuity/latest/execution_meaningful_event_reporting_status_latest.json`

Minimum runtime expectations:
1. Emit `new_events[]` for transition detections in the current turn.
2. Carry forward recent events for a bounded window (`pending_required_event_codes`) so single-turn misses are harder.
3. Emit machine-evaluable checklist status (`checklist_status`) that mirrors the checklist sections.
4. Emit a low-noise status projection (`status`, `operator_attention_required`, pending/critical counts) for operator surfaces.
5. Explicitly declare detector coverage (`deterministic` / `heuristic` / `manual_only`) per trigger code.
6. Mission-control MUST surface a fail-closed contract signal when required packet expectation exists but status projection is missing/stale/mismatched (no silent fallback for required-event packets).
7. Mission-control SHOULD compact fail-close reason bursts into a deterministic digest token/status and emit a one-line expected-vs-delivered packet checksum signal for operator triage (`exp/del/miss/extra/stale/chk`).
8. The checksum `del` side MUST represent only actually delivered status-packet codes (from `execution_meaningful_event_reporting_status_latest.json`); projection fallback may be shown separately for visibility, but MUST NOT be treated as delivered parity.

This remains additive: it is **not** a replacement for operator-facing reporting, and it does not attempt full auto-messaging.

---

## 5) Delivery policy (low-noise compatible)

To stay consistent with `docs/ops/low_noise_interaction_policy_v1.md`:
- Routine heartbeats remain silent.
- Meaningful transitions in Section 3 are **not routine** and must be reported.

Allowed delivery forms:
1. immediate concise operator update (preferred for blocker/failure/relaunch transitions), and/or
2. same-turn durable closeout/event note that is explicitly referenced in operator status.

No-drop rule: batching is allowed only if all triggered events remain explicitly enumerated in the batch payload.

---

## 6) Missed-trigger recovery rule

If a required trigger was missed:
1. emit `MISSED_TRIGGER_RECOVERY` in the next turn,
2. include original transition time and why it was missed,
3. emit the missing event packet retroactively,
4. apply the deterministic action now (if still pending).

Treat repeated missed-trigger behavior as doctrine drift and A2/A6 incident input.

---

## 7) Checklist companion

Use: `docs/ops/execution_meaningful_event_reporting_checklist_v1.md`.

The checklist is required for:
- core-roadmap queue/executor slices,
- stalled-loop / relaunch-failure investigations,
- slices that claim phase/slice completion.

---

## 8) Update protocol

Changes to this doctrine must co-update:
1. `docs/ops/unified_operating_doctrine_v1.md` (module reference),
2. `reports/openclaw_system_source_of_truth_map_2026-03-20.md` (lane mapping),
3. any affected lane doctrine that consumes these triggers (`core_roadmap_queue_layer_doctrine_v1.md`, `subagent_slot_fill_protocol_v1.md`).
