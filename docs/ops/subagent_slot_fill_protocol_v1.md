# Subagent Slot-Fill Protocol (v1)

Date: 2026-03-13  
Status: active (subordinate doctrine module)  
Parent doctrine: `docs/ops/unified_operating_doctrine_v1.md`  
Companion doctrine: `docs/ops/execution_meaningful_event_reporting_doctrine_v1.md`  
Companion contract: `docs/ops/invalid_output_retry_relaunch_contract_v1.md`

## Why this exists
We hit an orchestration failure mode in live operation:
- child slot became free,
- assistant narrated that the slot was free,
- but did **not** spawn immediately in the same turn.

This protocol makes the expected behavior explicit and testable.

## Scope
Applies whenever subagent capacity is bounded and a child completion event arrives (or any event indicates active-child count dropped).

## Non-negotiable invariants
1. **main lane orchestration-only for non-trivial slices**
   - For non-trivial work, keep `worker_lane=subagent_default` and treat main lane as orchestrator/validator only.
   - `worker_lane=main_session_tiny_exception` is allowed only for clearly tiny slices (single-file, low-risk, bounded, no fan-out, no long-running supervision).

2. **Spawn-before-speak invariant**

If all are true:
1. active child count is below cap,
2. queued candidate slice exists,
3. no hard blocker (safety, missing permissions, duplicate active work),

then the assistant must:
1. call `sessions_spawn` first,
2. only then send narrative/status text.

Narration-only acknowledgment in this state is a protocol violation.

## Slot-fill loop (deterministic)
1. **Trigger detect**
   - completion event received, or
   - operator asks “are you doing more work after this one?”, or
   - any signal that a child slot likely freed.
2. **Capacity check**
   - compute `available_slots = cap - active_children`.
3. **Queue pick**
   - choose one highest-value next slice from ready queue.
   - avoid duplicate intent with active branches.
4. **Spawn attempt (mandatory first action)**
   - call `sessions_spawn` for chosen slice.
5. **Post-spawn report**
   - report what was launched and why it won.
6. **If spawn blocked**
   - report blocker with explicit reason (`at_cap`, `duplicate_active_work`, `missing_context`, `tool_error`),
   - include immediate next action (retry condition or what info is needed).

## Delegation trigger rules (explicit)
Default to delegation (`worker_lane=subagent_default`) when **any** of these are true:
- more than one file/surface is expected to change,
- verification requires one or more explicit commands,
- independent validation or evidence synthesis is required,
- expected duration can cross review/stale windows,
- rollback/risk impact is non-trivial.

Use `main_session_tiny_exception` only when **all** are true:
- bounded micro-change (typically single-file/minimal scope),
- low blast radius and reversible quickly,
- verification is immediate/local,
- no parallel fan-out or stale-worker supervision needed.

When using `main_session_tiny_exception`, report a concrete `delegation_basis` explaining why delegation was not used.

## Meaningful-event trigger obligations (slot context)
When slot-fill logic runs, treat the following as mandatory event triggers:
- child completion -> `SUBAGENT_FINISHED`
- successful completion that closes bounded objective -> `SLICE_LANDED`
- unusable/invalid child output -> `SUBAGENT_FINISHED(outcome_class=INVALID_OUTPUT)` + `WORKER_FAILED_OR_JUNK`
- relaunch after idle/stall or failure -> `QUEUE_RELAUNCHED` and/or `EXECUTOR_IDLE_TO_RELAUNCHED`

Action-first rule still applies: relaunch/block decision first, narration second.

## Tie-break policy for "next best slice"
Use this order:
1. no-nudge trust / safety / fail-close hardening,
2. orchestration reliability + recovery,
3. upgrade-enabling substrate,
4. cleanup/refactor.

## Stale-worker / closeout-bundle discipline
- At `stale_at` (or earlier if junk/failure is clear), force an explicit decision: `kill_and_relaunch`, `salvage_and_relaunch`, or `continue_with_reason`.
- Never leave stale workers in implied limbo; stale decision must be recorded before moving on.
- Any stale kill/salvage/relaunch action must produce a closeout bundle reference (`closeout_bundle_ref`) that captures:
  - what was attempted,
  - what evidence was salvaged,
  - why relaunch shape changed,
  - next bounded slice objective.
- Completion and stale-recovery updates must include `stale_worker_decision` and `closeout_bundle_ref` (or an explicit `none` + rationale when not applicable).

## Required reporting fields after each slot event
- `slot_event`: completion or capacity-drop trigger
- `event_code`: meaningful-event trigger (`SUBAGENT_FINISHED`, `WORKER_FAILED_OR_JUNK`, etc.)
- `transition`: `from_state -> to_state`
- `decision`: `spawned` or `blocked`
- `decision_taken`: deterministic control action (`relaunch`, `blocked`, `continue`)
- `chosen_slice` (or `blocker_reason`)
- `spawn_call`: tool call id or `none`
- `next_slice_after_this`
- `mode`: `BLOCKER_BURNDOWN` or `THROUGHPUT`
- `execution_mode`: `EXECUTE_NOW` or `PLAN_ONLY`
- `worker_lane`: `subagent_default` or `main_session_tiny_exception`
- `model_selection`: concrete worker/model (or `NO_LLM`) plus one-line reason
- `delegation_basis`: why delegated (or why tiny exception is justified)
- `fold_in_target`: `canonical_doctrine | roadmap_pair | queue_continuity | support_only`
- `escalation_evidence_refs`: required when `fold_in_target=support_only` and heavy Codex is selected
- `stale_worker_decision`: `not_applicable | continue_with_reason | kill_and_relaunch | salvage_and_relaunch`
- `closeout_bundle_ref`: bundle/report path for stale recovery or completion closeout (`none` only with rationale)
- `last_signal_age` for any still-active sibling lanes when relevant
- `risk_state` for the overall wave when blocker work is active
- `evidence_refs[]`: artifacts/surfaces supporting the event claim

## Violation recovery
If narration happened before spawn in a spawnable state:
1. acknowledge violation directly,
2. spawn immediately in same follow-up turn,
3. record the lapse and point to this protocol.

## Quick checklist (use on every slot-open event)
- [ ] I checked if a slot is actually available.
- [ ] I selected one concrete next slice.
- [ ] I attempted `sessions_spawn` **before** narrative text.
- [ ] If blocked, I reported blocker code + immediate next action.
- [ ] I recorded execution tuple fields (`execution_mode`, `worker_lane`, `model_selection`, `fold_in_target`).
- [ ] I recorded explicit delegation basis (`delegation_basis`) and justified any `main_session_tiny_exception`.
- [ ] If `fold_in_target=support_only` and heavy Codex was selected, I attached explicit escalation evidence refs.
- [ ] I made/recorded a stale decision when needed (`stale_worker_decision`) and attached a `closeout_bundle_ref`.
- [ ] I emitted a meaningful-event packet (`event_code`, transition, decision, evidence refs).
- [ ] I avoided claiming slot reuse without a real spawn call.
