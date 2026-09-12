# Architecture Correction Set C8 — Native-Child Routed Re-entry Seam

Stage: ARCHITECTURE_CORRECTION
Status: SUPERSEDED / NOT_EXECUTABLE
Acceptance-State: SUPERSEDED_BY_C9
Execution-State: NOT_EXECUTABLE
Superseded-By: `architecture-correction-c9.md` — accepted owner simplification; C8 is historical only and MUST NOT be executed
Affected-Task: T004
Project-Version: 0.1.1
Architecture-Version: 1
Scope: `native_child` routed fallback re-entry only

## Review verdict

**TECHNICALLY FEASIBLE; STOP FOR HUMAN REVIEW.** The prior accepted-plan assumption about delegated-child re-entry and Python main-thread affinity is corrected. Existing detached unit runners already own batch execution in copied `ContextVars` and thread-local approval context and may synchronously prepare and construct a replacement through the existing process-wide thread-serialization seam; no marshal to the originating thread is required. The remaining blocker is only human approval of C8 clauses and canonical integration. This DRAFT does not authorize acceptance, integration or implementation.

## C8-01 — False accepted-plan assumption and exact source evidence

1. Accepted ARC-005 section 6 assumes that the native-child lifecycle already “receives child spec/factory and builds a new child only after a new decision.” That assumption is false in the current source.
2. `tools/delegate_tool.py::delegate_task` resolves delegation credentials before normalization at lines 450–474, builds every child at lines 486–495, and only then calls `_run_batch` at line 496.
3. `tools/delegate_tool.py::_build_child_agent` and `_build_children` use “main thread” at lines 180–181 and 365–366 to describe caller/batch construction before fan-out, not an enforced CPython main-thread affinity. No `threading.main_thread()`, signal or event-loop assertion enforces such affinity. `AIAgent` is already constructed in worker threads at `tui_gateway/server.py:1002–1063,2286–2304` and `gateway/platforms/api_server.py:3630–3660,2087–2150`.
4. `tools/delegate_tool_dispatch.py::_Batch` at lines 27–61 stores those built-child tuples, and `run_child` passes a built child to `_run_single_child`.
5. `tools/delegate_tool.py::_run_single_child` at lines 294–357 discovers only the normal child result. It runs output-schema correction at line 334 and returns an ordinary result entry; it does not propagate `RoutedAttemptRestartRequired` to the batch owner. Its `finally` invokes full cleanup at lines 356–357.
6. `tools/delegate_tool_dispatch.py::_run_children_parallel` at lines 103–154 receives completed futures, while `_execute_and_aggregate` at lines 156–190 has no per-child re-entry branch.
7. In detached mode, `_run_batch` at lines 428–432 returns after `_dispatch_background`; `_dispatch_unit` supplies `_execute_and_aggregate` as a later runner at lines 357–375. `tools/async_delegation.py:597–616` executes that runner in copied `ContextVars` and the existing thread-local approval context. The runner owns one async unit and its one final delivery; split `group`/independent units remain separate.
8. T002 already supplies the target-free immutable `agent/execution_router.py::RoutedAttemptRestartRequired` at lines 56–65, the existing SessionDB route lifecycle, credential-free resolver and host candidate validation, plus routed in-place-fallback suppression in `agent/chat_completion_helpers.py::try_activate_fallback` at lines 1828–1845. C8 does not duplicate or widen any of them.
9. `tools/delegate_tool_results.py:296–311` already owns the private process-wide `_CHILD_CONSTRUCTION_LOCK` and `_build_child_preserving_parent_tools`, which serialize delegated-child construction and the save/build/restore sequence for `model_tools._last_resolved_tool_names`. The narrow uncovered write is `_ChildRun.cleanup` at `tools/delegate_tool_child_run.py:812–817`, which restores that same global outside the lock.

## C8-02 — Private immutable child-attempt specification

1. T004 SHALL add one immutable private child-attempt specification owned by the existing delegation lifecycle. Its name is private and underscored; it is absent from `__all__`, facades, compatibility exports and plugin/public contracts.
2. The specification contains only:
   1. stable task index and concrete normalized task inputs: `goal`, optional `context`, normalized/coerced output schema, normalized depth/role decision and resolved workspace hint;
   2. concrete immutable toolset enable/disable selection, or the minimum immutable source projection from which the existing builder deterministically derives that selection without rereading a changing parent;
   3. inherited explicit model/provider/reasoning pins and concrete non-secret request-overrides, prefill and delegation/runtime-config projections necessary for deterministic rebuild;
   4. the current existing fallback cursor and bounded previous-attempt projection;
   5. parent/session/turn plus root/task/execution correlation identity and current request/attempt IDs;
   6. only narrowly necessary live transcript, attach/interrupt, progress and lifecycle ownership handles, including the parent lifecycle SessionDB authority needed by the existing pre-start route lifecycle.
3. The specification SHALL NOT reread mutable routing/runtime inputs from a changing parent after background handoff. It SHALL contain no API key, credential bundle, secret-bearing base URL, client, live agent, executor, factory, secret value or other dispatch capability. The parent lifecycle SessionDB authority and the narrowly listed existing live handles are the only permitted live references.
4. The specification is not a new store or retry record. It is process-local preparation input for one existing `_Batch` task slot and is discarded with that batch.
5. Credential lookup remains outside the specification and occurs only after each fresh route decision has been accepted and revalidated.

## C8-03 — Private attempt outcome and cleanup boundary

1. Existing `tools/delegate_tool.py::_run_single_child` remains the only single-child execution owner, but SHALL return one immutable private attempt outcome with exactly one variant:
   1. `result`: the existing normal result entry; or
   2. `restart`: the target-free `RoutedAttemptRestartRequired`, old request/attempt correlation and the child accounting required to close that attempt.
2. Immediately after `run.await_child()` returns a child result, `_run_single_child` SHALL inspect the child’s routed-restart receipt before `_validate_child_output_schema`. If the receipt exists, output-schema correction and its bounded model retry SHALL NOT run.
3. The restart outcome becomes observable to `_run_batch` only after the existing `_run_single_child` `finally` has completed `run.cleanup(...)`. That cleanup boundary must stop heartbeat, unregister the child, release any credential lease, restore process-global tool names, detach and close the child as currently defined by `tools/delegate_tool_child_run.py::_ChildRun.cleanup` at lines 796–836.
4. No generic callback fence is added. C8 identifies no concrete state-mutating callback that can run after the full non-timeout cleanup boundary. If implementation tracing later proves one concrete callback remains reachable, implementation SHALL STOP and return a named callback plus exact mutation mechanism for separate owner review; it SHALL NOT invent universal fencing.
5. A timed-out child whose close is deferred cannot produce a safe restart outcome. It remains the existing timeout/error result because full cleanup has not completed.

## C8-04 — Sole re-entry owner and per-surface sequence

1. Existing `tools/delegate_tool_dispatch.py::_run_batch` remains the sole batch dispatcher and architectural re-entry owner. `_run_single_child` does not resolve, build, retry or resubmit.
2. Active-router initial sequence per child SHALL be: `_normalize_task_list` and `_coerce_task_schemas`; form the private attempt specification and credential-free candidates/pins; persist pre-start route lifecycle in the parent SessionDB; invoke the existing resolver; validate the host-issued candidate; only then call `_resolve_delegation_credentials` and the existing `_build_children`/child-construction seam for allowed attempts. `stop/router_error` creates no credentials, client, child or executor.
3. Foreground serial sequence: `_execute_and_aggregate` receives the cleaned restart outcome from its only child; control is on the batch-owning caller thread. It closes the old parent route lifecycle, validates and carries the spent cursor, issues fresh IDs, invokes the existing resolver, resolves fresh credentials and uses the locked existing construction wrapper on that same batch-owner thread, then runs only the replacement attempt.
4. Foreground parallel sequence: `_run_children_parallel` receives completed outcomes in its existing `FIRST_COMPLETED` loop. For a cleaned restart outcome, the pool-owning unit-runner thread—not a child execution worker—closes the old lifecycle, prepares/resolves/builds the replacement through the locked existing construction wrapper, and submits only that task’s replacement future to the same existing executor. Pending and completed sibling futures are untouched. The task index remains stable; request/attempt and child/subagent/session identities are fresh.
5. Every delegated-child write or restore of `model_tools._last_resolved_tool_names` during construction and cleanup SHALL execute under the one existing process-wide `_CHILD_CONSTRUCTION_LOCK`. The lock remains privately owned by `tools/delegate_tool_results.py`; `tools/delegate_tool_child_run.py` imports that exact lock for `_ChildRun.cleanup`, consistent with its existing import direction and without a cycle. If source topology at implementation requires a move, only the same private lock may move to the lowest existing delegation helper module imported by both owners: identity remains one process-wide lock, and no public export, helper subsystem or generic tool-registry redesign is created. Initial and replacement children use exactly `_build_child_preserving_parent_tools`; no fallback child is prebuilt.
6. Detached-background sequence: the existing detached unit runner is the batch-owner construction thread. After receiving a fully cleaned restart outcome, it synchronously closes lifecycle, prepares, resolves, performs post-decision credential lookup, builds the replacement through `_build_child_preserving_parent_tools`, and resubmits or runs only the same unit slot. In parallel mode the pool-owning unit runner performs construction, never a child execution worker. No origin-thread marshal, completion replay, coordinator, queue, scheduler, dispatcher, retry engine, background manager or generic callback is added.
7. `_dispatch_unit` SHALL NOT capture only immutable initial `child_agents`. Each existing unit owns one private process-local per-unit current-child identity container plus its existing cancellation/interrupt state. Replacement checks cancellation after old-child cleanup and again immediately before construction/registration; if stopped, it never starts. The current-child identity is updated atomically under existing delegation registry synchronization (`tools.delegate_tool_registry.py::_active_subagents_lock`) so existing stop/steer ownership follows the concrete current child without a new registry. If the existing synchronization cannot express this private-container handoff during implementation, implementation SHALL STOP for owner decision rather than add a store, coordinator or registry.
8. Replacement invokes the existing `subagent_start` hook through the same existing hook call after construction of the new child through the locked wrapper. No hook dispatcher is added. Existing source proves no origin-thread affinity for `subagent_start`, and background finalization/hooks already run in unit workers; if a registered hook demonstrates an origin-thread-only contract, implementation SHALL STOP for a separate owner decision.

## C8-05 — Fallback cursor, identities and route ordering

1. The receipt’s `consumed_fallback_slot` proves that the existing child fallback authority consumed exactly one existing slot. C8 creates no budget, retry count, reset, refund or policy.
2. The replacement specification carries `next_fallback_cursor = consumed_fallback_slot + 1`. A missing, stale, contradictory, reused or exhausted cursor fails closed before fresh IDs, resolver, credentials or construction.
3. The old child route lifecycle SHALL close terminally before fresh resolver invocation. Only then may the host issue a fresh request ID and attempt ID and project bounded `previous_attempt`.
4. The active router is invoked exactly once for each fresh child attempt. Normalization precedes per-child route resolution; route resolution precedes credentials and construction.
5. Explicit model/provider/reasoning pins and host-issued candidate eligibility constraints remain unchanged. Router output cannot weaken a pin or introduce a non-host candidate.
6. No-router and explicit `pass_through` stay on the exact native `_resolve_delegation_credentials` → `_build_children` → `_run_batch` → in-agent fallback path. They create no private restart outcome/spec branch, synthetic route lifecycle or wrapper.

## C8-06 — Batch, transcript, progress and finalization invariants

1. Input ordering and final `results` ordering by stable `task_index` remain unchanged. A replacement occupies the same task slot and does not add a second result entry.
2. Mixed `route/pass_through/stop/router_error` siblings remain independent. A restart pauses or replaces only its own task; it neither cancels nor delays already-running siblings beyond the existing executor capacity semantics.
3. Existing `group` and independent-completion partitioning remains unchanged. One async unit still delivers once; C8 SHALL NOT emit the terminal old-attempt receipt as a completion and later deliver a replacement result again.
4. The same per-task live transcript writer/path remains attached to the replacement task slot. Attempt boundaries may be recorded through existing progress/lifecycle events, but no second transcript, replay or truncation is introduced.
5. Existing progress, steer/stop ownership and `_active_subagents` semantics apply per concrete child identity. The old child is removed by cleanup before the replacement is registered; the task/delegation identity remains correlated while the child/subagent/session identity is fresh.
6. Existing result finalization at `tools/delegate_tool_results.py::_finalize_child_results` lines 392–401 runs once per final task result. Memory notification, `subagent_stop`, summary budgeting and parent cost rollup SHALL NOT run on the intermediate restart receipt. Costs already incurred by the old attempt must be accumulated into the final task accounting exactly once through the existing cost fields; no delivery replay is used.
7. Parent SessionDB remains the pre-start route lifecycle authority. Each child’s dedicated SessionDB remains transcript authority under `tools/delegate_tool.py::_open_child_session_db` and child close ownership. No new database, journal or state store is added.
8. Group partition, independent-completion partition, sibling semantics, progress, cost, memory and one-final-delivery ownership remain unchanged for foreground and detached execution. A replacement is an internal attempt of the same unit slot, not a new unit or completion.

## C8-07 — Failure and crash semantics

1. Failure to normalize, project candidates/pins, persist the pre-start lifecycle, resolve or validate the initial route returns the existing bounded per-child not-started failure; valid siblings continue.
2. After a restart receipt, old lifecycle closure failure, invalid/exhausted cursor or inability to establish the full cleanup boundary is terminal for that child. No fresh resolver call or replacement construction occurs.
3. If fresh route resolution returns `stop/router_error`, the fresh attempt is recorded not-started under fresh IDs; the old child is not resurrected and the slot is not refunded.
4. If replacement credential resolution or construction fails after fresh route acceptance, close the fresh attempt through existing not-started/error projection, return one bounded failed task result, and preserve sibling execution. Do not retry construction and do not return to the old attempt.
5. If replacement execution fails, only the same existing fallback authority may consume the next existing slot and produce another receipt; all C8 ordering rules repeat. No new loop budget is introduced.
6. Process crash after the old receipt and before replacement build has uncertain child side effects. Existing durable transcript/lifecycle rows remain evidence, but C8 adds no automatic replay, reconstruction or cross-process continuation. Recovery exposes terminal/uncertain state for human action; it must not synthesize a new child attempt.
7. A crash after replacement construction/start remains governed by existing child SessionDB and delegation crash semantics. C8 does not claim durable background restart.
8. Regular non-delegation `AIAgent` builders can also write `model_tools._last_resolved_tool_names`; `agent/AGENTS.md:70–72` and `tools/AGENTS.md` already document this process-global/stale-value limitation. C8 does not claim to solve generic cross-builder races. Its proof boundary is: T004 adds no unguarded delegated-child write, and replacement uses exactly the same locked wrapper as initial delegated-child construction. If implementation tests demonstrate a new cross-builder corruption caused by replacement, implementation SHALL STOP for a separate owner decision rather than broaden T004.

## C8-08 — Exactly seven RED-capable T004 test nodes

These seven nodes are the complete proposed T004 ceiling. They extend existing delegation/router test modules; they do not create a broad new test family.

1. `test_native_child_active_route_normalizes_and_persists_before_credentials_and_build` — one resolver call per allowed initial child; host candidate/pins preserved; `stop/router_error` create no credentials/child/executor while valid siblings continue.
2. `test_native_child_no_router_and_pass_through_keep_exact_native_path` — unchanged kwargs, ordering, explicit pins, batch behavior, in-agent fallback and absence of synthetic route state.
3. `test_native_child_restart_skips_output_schema_and_returns_only_after_cleanup` — routed receipt is target-free; schema correction is not invoked; heartbeat/registry/lease/tool-name/attach/close cleanup completes before batch re-entry.
4. `test_native_child_serial_restart_consumes_cursor_closes_old_and_rebuilds_on_batch_owner` — exactly one spent slot, old close before fresh IDs/resolver, post-decision credentials/build through the locked wrapper and one final result.
5. `test_native_child_parallel_restart_resubmits_only_affected_slot` — replacement construction occurs on the pool-owning unit-runner thread through the locked wrapper, never a child execution worker; mixed siblings retain execution and input-result ordering.
6. `test_native_child_replacement_stop_or_build_failure_is_terminal_without_replay` — fresh not-started lifecycle, no refund/reuse, no second construction attempt and siblings unchanged.
7. `test_native_child_background_grouped_restart_uses_detached_batch_owner_and_delivers_once` — grouped/independent restart uses the detached unit runner as the locked batch-owner construction thread; a stop race after cleanup or before construction/registration prevents replacement; the same unit slot keeps one live transcript and finalizes progress/cost/memory and delivery exactly once without receipt delivery.

No eighth canonical T004 node may be added without explicit owner amendment. Existing directly affected tests may be adjusted only as necessary to preserve their accepted contracts; this does not authorize a new per-callback, per-provider, per-surface or crash-matrix family.

## C8-09 — Prohibited expansion, acceptance and rollback boundary

1. No public schema, API, capability, contract version or execution-kind change.
2. No credential owner, client/executor carrier in the attempt specification, prebuilt fallback child, generic factory callback, callback fence, coordinator service, queue, scheduler, dispatcher, retry engine, background manager or second store.
3. No T005/T006+, no Kanban changes, no new dispatcher, no documentation/package/release work, no commit, installation, profile/runtime mutation, external-system action or LIVE action.
4. C8 does not alter T004 product semantics: routed child failure still consumes the existing fallback slot, closes the old attempt, creates fresh IDs, invokes the resolver again and builds a replacement child; no-router/pass-through remains native.
5. Unauthorized functional expansion: **NONE**. Unnecessary security expansion: **NONE**. The one-lock rule is correctness preservation for the existing process-global delegated-child seam, not new hardening and not a generic registry redesign.
6. Acceptance requires human approval of every C8-01 through C8-09 clause and canonical integration. Until then—and solely for that reason—the DRAFT remains `NOT_ACCEPTED` and `NOT_EXECUTABLE`; no unresolved architecture gap remains in C8-04.6.
7. Abort criteria after acceptance: construction outside `_CHILD_CONSTRUCTION_LOCK`, construction by a child execution worker rather than its batch-owning caller/unit runner, any unguarded delegated-child write/restore of `_last_resolved_tool_names`, inability to express current-child replacement under existing registry synchronization, a proven origin-thread-only `subagent_start` hook, new replacement-caused cross-builder corruption, background delivery replay, prebuilt fallback, credential/agent before route, mutable parent routing/runtime reread, cursor reset/refund, old lifecycle left open, output-schema retry after receipt, stopped-unit replacement, sibling cancellation/drift, duplicate finalization/delivery, new public surface/store or any T005/T006+ change.
8. Rollback boundary is the private T004 native-child adapter/outcome/spec/re-entry delta and its seven tests only. Rollback restores the exact native delegation path and removes no T002 contract/lifecycle implementation. No data migration or public compatibility rollback is required.

## Integration steps only after acceptance

1. Stop for human review of the corrected C8; do not treat technical feasibility as clause acceptance.
2. On exact owner acceptance, integrate only the accepted C8 clauses into canonical `plan.md`, align only T004 wording/test topology in `tasks.md` if required, and re-enter exact T004 authority.
3. Implement the private immutable attempt specification/outcome, active-router ordering, one-lock delegated-child construction/cleanup writes and private per-unit current-child handoff without touching no-router/pass-through control flow.
4. Implement serial/parallel and detached-unit re-entry through the same batch-owner and locked-wrapper seam, then add exactly the seven test nodes.
5. Run only the accepted T004 proof after implementation. C8 itself authorizes none of these integration or execution steps.
