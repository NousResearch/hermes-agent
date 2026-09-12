# Accepted Correction C10 — Minimal Kanban Claim-and-Defer Routing

Stage: PRODUCT_ARCHITECTURE_CORRECTION
Status: ACCEPTED
Execution-State: CANONICAL_CORRECTION_ONLY
Affected-Task: T005
Project-Version: 0.1.1
Architecture-Version: 1
Owner-Decision: «Минимальный T005: existing claim + defer в той же lane»
Supersedes: wider T005 reservation/state/schema/crash-matrix design

## Normative v1 behavior

1. `kanban_worker` remains a required v1 execution kind. Every actually proposed worker attempt routes before credentials, `task_run` creation, process or executor.
2. Reservation reuses existing `claim_lock` and `claim_expires`; task status remains exactly its original `ready` or `review` lane. Selection already excludes claimed rows. No `routing` status, source-status/planned-attempt task column, routing index/dashboard mapping, reservation table/store, queue, scheduler or dispatcher is added.
3. The bounded routing-reservation token lives in existing claim fields. Expired pre-run reservations are cleared and receive bounded not-started/recovery evidence without a phantom `task_run`.
4. Router request/decision lifecycle uses existing `task_events`. Route/open is one existing write transaction that revalidates claim token, expiry and unchanged lane, then inserts the canonical `task_run` plus accepted route metadata/events before status becomes running. Existing run primary identity and metadata/event payload correlate request and attempt. If implementation proves these cannot satisfy accepted identity/idempotency, STOP; do not add a schema column without a later exact owner decision.
5. `stop/router_error` creates no credentials, run, process or executor. It records bounded/redacted not-started evidence and leaves the task in its exact original lane under the existing claim until a short bounded defer expires; stale-claim release then makes it eligible. It does not increment worker failure accounting or circuit breaker. Repeated stop/error may defer again; no blocked status/policy is added.
6. A worker/model failure after run start is a real attempt and follows existing close, failure-budget, retry/requeue and circuit-breaker authority. A later ordinary eligible attempt reserves and routes again with fresh request/attempt IDs. No restart-receipt orchestrator or new retry engine is added. A minimal routed marker may prevent in-place fallback only if worker initialization requires it; reuse existing host receipt/fence and retry/requeue path.
7. No-router/pass-through preserves the exact current claim/open/spawn/retry path and error semantics. Router-specific behavior is guarded only after positive activation.
8. Crash behavior has two externally distinct classes: reserved/no run is released without a phantom run; opened run uses existing running/run recovery. No four-point ceremony is mandated unless implementation proves another distinct state.
9. Generic lifecycle, notices and redaction project only into existing `task_events` and run metadata. No Kanban-specific event subsystem, renderer or security hardening is added. Existing dashboard/status consumers remain unchanged because no new task status exists.
10. Prior solution-shaped planning required a `routing` state, new fields and a crash matrix before necessity was proven. Generic skills are not assigned blame without transcript evidence. This owner correction overrides the earlier wider T005 wording.

Unauthorized functional expansion: NONE.
Unnecessary security expansion: NONE.

## Authorization boundary

This correction changes product/architecture authority only. T005 and S001–S007 remain open and unimplemented. Source, tests, schema, runtime, profile, commit and LIVE changes require separate exact T005 implementation authorization.
