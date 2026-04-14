# Execution Meaningful-Event Reporting Checklist (v1)

Date: 2026-03-28  
Status: active checklist (workflow support artifact)  
Parent doctrine: `docs/ops/execution_meaningful_event_reporting_doctrine_v1.md`

Use this checklist when supervising core-roadmap execution, especially after child completion, failure, queue-state change, or relaunch.

---

## A) Trigger detection

- [ ] I checked whether any mandatory trigger fired this turn:
  - [ ] `SUBAGENT_FINISHED`
  - [ ] `SLICE_LANDED`
  - [ ] `PHASE_LANDED`
  - [ ] `WORKER_FAILED_OR_JUNK`
  - [ ] `QUEUE_BLOCKED`
  - [ ] `QUEUE_UNBLOCKED`
  - [ ] `QUEUE_RELAUNCHED`
  - [ ] `EXECUTOR_IDLE_TO_RELAUNCHED`

- [ ] If none fired, I explicitly kept noise low (no fake status filler).

---

## B) Action-before-narration guard

- [ ] For failure/junk output, I made deterministic decision first (`relaunch` or explicit `blocked`).
- [ ] For `INVALID_OUTPUT`, I emitted both `SUBAGENT_FINISHED(outcome_class=INVALID_OUTPUT)` and `WORKER_FAILED_OR_JUNK` before/with relaunch decision.
- [ ] For queue blocked/unblocked/relaunch transitions, I set canonical queue state first.
- [ ] If stale-task recovery or bounded-queue pressure policy fired, I treated it as a queue/executor transition and recorded the recovery/pressure reason before narrating.
- [ ] I did not substitute explanation for corrective action.

---

## C) Event packet minimum fields

For each fired trigger:
- [ ] `event_code`
- [ ] `event_at`
- [ ] `detected_at`
- [ ] `lane`
- [ ] `work_unit_ref` (child id, slice id, wave id, or queue candidate)
- [ ] `from_state` and `to_state`
- [ ] `decision_taken`
- [ ] `decision_reason`
- [ ] `evidence_refs[]`
- [ ] `next_action` (or explicit `none` + reason)

Queue/executor transitions additionally:
- [ ] `queue_source`
- [ ] `queue_truth_projection_parity.status`
- [ ] `next_candidate` (or blocker sentinel)
- [ ] recovery/pressure metadata when applicable (for example recovery counts, queue-depth, bounded-drop counters, or explicit confirmation that only freshness-stage work was dropped)

---

## D) Delivery contract

- [ ] I emitted a concise operator-facing update and/or explicitly linked a durable event note.
- [ ] If batching, all fired triggers are still explicitly enumerated (no dropped events).
- [ ] For blocker/failure/relaunch transitions, delivery was immediate (same turn unless tooling failure).
- [ ] `state/continuity/latest/execution_meaningful_event_reporting_latest.json` exists and reflects this turn’s fired triggers (or explicitly records `none_fired`).
- [ ] `state/continuity/latest/execution_meaningful_event_reporting_status_latest.json` exists and correctly projects attention-required status for operator surfaces.
- [ ] `mission-control --json` truth strip includes `meaningful_event_reporting` and `meaningful_event_reporting_contract` rows.
- [ ] If required event packets are expected, `meaningful_event_reporting_contract.status` is not silently `ok` when status artifacts are missing/stale/mismatched.
- [ ] `meaningful_event_reporting_contract` exposes deterministic digest compaction (`digest_status`, `digest_token`) and a one-line expected-vs-delivered checksum signal (`exp/del/miss/extra/stale/chk`) for fast operator triage.
- [ ] Checksum `del` count is sourced from the real status packet payload (not projection fallback), with projection-vs-packet split explicit when fallback is used.

---

## E) Missed-trigger recovery

If any trigger was missed earlier:
- [ ] I emitted `MISSED_TRIGGER_RECOVERY`.
- [ ] I included original transition time and miss reason.
- [ ] I emitted the missing packet now.
- [ ] I applied still-pending corrective action now.
