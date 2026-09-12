# Architecture Correction Set C6 — History-Safe Main-Turn Continuation

Stage: ARCHITECTURE_CORRECTION
Status: ACCEPTED
Project-Version: 0.1.1
Architecture-Version: 1
Reviewed-Plan-SHA256: `3a268443ba084699739a4017bf52a158d2a7733484fb8dd242ec1eb86d452a71`
Accepted-C5-Canonical-Record-SHA256: `3a2754b0206225a2541623337c4717f2e7dc56a0921cc5345c5843bd2406e603`
Affected-Task: T003
Supersedes-On-Acceptance: ARC-005 section 6 `main_turn` re-entry-owner bullet and only the implicit claim in ARC-005 sections 2/6 that Classic CLI and Gateway already have a history-safe same-semantic-turn re-entry path; ARC-003 section 6 is narrowed for routed `main_turn` continuation as stated below
Acceptance-State: ACCEPTED
Accepted-Source-SHA256: `b1d306fd1f6252d9034d0a84c5ba733f218cb101772ada0842d077bc4f7796f9`
Owner-Decision: `ACCEPT`
Owner-Acceptance-Reference/Text: `Owner accepted all C6-01 through C6-07 blocks and then explicitly accepted the unchanged whole DRAFT specs/002-execution-router-public-api/drafts/architecture-correction-c6.md by exact SHA-256 b1d306fd1f6252d9034d0a84c5ba733f218cb101772ada0842d077bc4f7796f9.`
Accepted-At: `2026-09-11T19:17:55+03:00`
Integrated-Into-Canonical-Plan: YES
Implementation-Authorization: T003_ONLY

***

## Purpose, evidence and preserved scope

The T003 implementation pass proved one material accepted-architecture gap and stopped correctly. Accepted ARC-003/ARC-005 and T003 S006 require a router-selected failed `main_turn` attempt to close and continue as a fresh routed attempt. Full source tracing showed that TUI/backend has an internal turn owner around `_invoke_agent`, while Classic CLI and Gateway have no existing history-safe API that can re-enter the same semantic user turn from an already-produced transcript. Their normal entry functions accept a new user message and would append/persist it again; Gateway queued-followup and crash-resume paths also represent a new turn or synthetic recovery input, not continuation of the same turn.

C6 closes only that gap. It preserves the accepted product contract and boundaries:

- one logical user turn;
- one policy-neutral host router and the existing surface retry/fallback budgets;
- a new request and attempt for each fallback;
- immutable route within each attempt;
- existing SessionDB and surface transcript owners;
- no product routing policy, credential owner, executor, dispatcher, queue, scheduler, retry engine, second store or plugin-visible transcript;
- no T004/T005 behavior, commit, publication, installation, runtime/profile change, pilot or LIVE effect.

The reviewed integrated plan and accepted C5 canonical record were verified immediately before this draft at the exact SHA-256 values in the header. A mismatch would have stopped this correction.

## C6-01 — Minimum host-owned same-semantic-turn continuation seam

### Normative rule

Hermes SHALL add one private, host-owned continuation seam for an already-started `main_turn`. Its architectural operation is:

`continue_main_turn_attempt(agent, continuation_record, *, existing_surface_callbacks) -> turn_result`

The name describes the exact internal role; it is not a public plugin capability or compatibility API. The seam is owned by the existing agent/turn host and is callable only by the four T003 surface owners after a routed attempt returns the accepted host-only restart receipt.

The seam SHALL continue from the exact transcript produced by the terminal old attempt. It SHALL NOT accept or synthesize a new user message. It SHALL initialize the new agent turn loop from the sealed transcript and the existing current-turn boundary, without executing the normal new-user-turn append path, without firing new-turn admission hooks for the same logical turn, and without converting the continuation into `/steer`, an interrupt, a queued prompt, an auto-continue prompt or a crash-recovery prompt.

The continuation transcript contains every assistant tool-call row and tool-result row already produced. The continuation begins after that tail. Completed tool calls and side effects are historical facts and SHALL NOT be dispatched again. A new model may inspect their existing results and continue; it may issue a genuinely new tool call, but the host never recreates a previously completed call from the continuation boundary.

Strict role alternation remains authoritative. No synthetic user replay is inserted to make continuation convenient. If the sealed transcript cannot form a provider-valid continuation under the existing role/tool-call repair rules without replaying, rewinding or deleting history, continuation fails closed and the surface returns the bounded terminal failure.

### Exact continuation record

The host SHALL seal one immutable internal `MainTurnContinuationRecord` containing only:

- owning profile/session identity and execution kind `main_turn`;
- old `request_id` and `attempt_id`;
- exact current `turn_id` and `current_turn_user_idx` exported by the old result;
- a recursively immutable snapshot of the exact returned `messages`, including assistant tool-call and tool-result metadata required by the existing provider projection;
- the existing persistence watermark/row-binding metadata already maintained by the session persistence path, so previously durable rows are recognized rather than appended again;
- old accepted route signature and router generation/contract generation;
- bounded old terminal reason and the accepted host-only fallback receipt;
- `consumed_fallback_slot` and `next_fallback_cursor = consumed_fallback_slot + 1`.

It SHALL NOT contain credentials, writable SessionDB/config objects, executor/dispatcher handles, a retry count invented by execution_router, a target route, a synthetic prompt or plugin payload. The record is not passed to the router. The router receives only the existing bounded `previous_attempt` projection from the fresh `ExecutionRouteRequestV1`.

The immutable snapshot is the cross-agent ownership boundary. A continuation agent receives a fresh mutable working copy for its loop; it cannot mutate the sealed record or the old attempt result. The existing surface remains owner of display state, callbacks and final transcript adoption.

## C6-02 — Mandatory close, budget and routing order

For router-selected `main_turn` fallback, every surface SHALL perform the following order exactly:

1. The active agent's existing fallback authority decides whether fallback is permitted and atomically consumes exactly one existing slot. It returns the host-only receipt with `consumed_fallback_slot`; it does not choose the next target.
2. The old attempt stops all model execution and returns its exact current transcript and current-turn boundary. No completed tool call is re-entered.
3. The surface seals the immutable continuation record under its existing transcript/persistence synchronization. It invokes the existing persistence path only for rows not already durable; the original user row is never appended again.
4. The old route lifecycle is terminally closed as `route_finished` with bounded terminal state/reason identifying routed restart. This closure occurs before any new `request_id` or `attempt_id` is issued.
5. The surface validates `next_fallback_cursor` against the same existing fallback budget/chain authority. Agent construction, primary-runtime restoration and route-signature rebuild cannot reset, decrement or replace this cursor. A stale, missing, contradictory or exhausted cursor fails closed without router invocation and without a new attempt.
6. Only after that validation does the host issue a fresh `request_id` and `attempt_id`, build the remaining credential-free candidate projection from the existing authority at `next_fallback_cursor`, include the bounded previous-attempt projection and invoke the active router exactly once.
7. `stop/router_error` records the fresh non-start lifecycle and returns a bounded terminal result. `route/pass_through` is revalidated under the accepted rules.
8. Only after the fresh decision is resolved and accepted may the surface bind credentials and construct or select an agent/executor. A changed route signature or router generation/contract generation forces the existing reconstruction path. Exact-signature reuse is allowed only where the existing surface already permits it and only if the immutable continuation record and budget cursor remain attached; reuse cannot resurrect the terminal old attempt.
9. The surface rebinds its existing callbacks to the selected continuation agent and calls the private continuation seam with the sealed transcript. The final result is adopted once by the existing surface owner.

`consumed_fallback_slot` is evidence that the existing authority spent a slot, not a new budget. The continuation record carries only the next cursor. The existing per-surface fallback chain, permission/depth and exhaustion rules remain the sole authority. C6 creates no extra retry, no refund and no reset on rebuild or `restore_primary_runtime`. A consumed slot and every earlier slot are excluded from the next candidate projection and cannot be selected again. Exhaustion is terminal and fail-closed.

No-router and explicit `pass_through` attempts retain their accepted native in-agent fallback behavior. They do not create this routed continuation record and do not enter the C6 seam.

## C6-03 — Transcript ownership and persistence ordering by surface

### Classic CLI

`CLIChatTurnMixin.chat()` remains the sole UI/turn owner. `_chat_stage_user_message()` runs exactly once for the logical turn. `_chat_run_agent()` owns an internal routed-continuation loop around agent execution; it never recursively calls `chat()` and never stages the original message again.

After the first routed attempt returns, Classic CLI seals `turn.result["messages"]` together with its exported current-turn boundary and the persistence watermark under the agent's existing session-persist lock. Missing transcript rows are flushed through the existing agent/SessionDB persistence path before the old route attempt is closed. The fresh route is then resolved before `_ensure_runtime_credentials()`, `_resolve_turn_agent_config(...)` and `_init_agent(...)` for the continuation agent. The existing stream, approval, secret, sudo, interim, TTS and status callbacks are rebound to that agent. `_chat_settle_turn()` adopts only the final continuation result into `self.conversation_history`; the original staged user row remains the sole user row.

### TUI/backend

The existing `_run_prompt_submit` admission, inflight marker and `message.start` describe one logical turn and run once. `_prepare_turn_input` snapshots `session["history"]` once. `_invoke_agent` owns the routed-continuation loop before `_absorb_turn_result`; it does not call `_run_prompt_submit`, `_dispatch_followup_turn`, `_enqueue_prompt` or queued-prompt drain for fallback.

On routed restart, TUI seals `st.result["messages"]` and its current-turn boundary, persists only missing rows through the agent's existing SessionDB path, closes the old route attempt, resolves the fresh route and rebuilds `session["agent"]` through the existing reconstruction path when the route signature changes. It rebinds the existing stream/interim/title/usage/approval/session-context callbacks and continues from the sealed transcript. `_commit_turn_history` runs once on the final result, preserving the existing `history_version` check. The inflight marker remains one marker for the logical turn and is cleared/retained only by the existing final outcome rules.

### Gateway

`TurnRunner.run_sync()` remains the sole owner of one admitted inbound turn. The routed-continuation loop lives inside `run_sync`, after the original history is selected and before stream finalization, `_sync_session_after_run` and outer delivery/persistence. It does not recursively call `run_sync`, re-enter inbound handling, enqueue `ctx.message`, invoke queued-followup machinery or use Gateway crash auto-resume as fallback.

On routed restart, Gateway seals the exact returned `messages`, current-turn boundary and persistence watermark from the active agent. Its normal `persist_user_message`, timestamp, display metadata and inbound platform ID apply only to the first user-row append; continuation omits every user-persistence argument and starts from the sealed transcript. Gateway closes the old route attempt, resolves the fresh route before `_resolve_session_agent_runtime` credential output and `_resolve_turn_agent`, applies the route signature, then rebinds the existing approval, stream, interim, tool/status, session-context and notification callbacks. Stream finalization, media extraction, session split synchronization and final delivery run only after the final continuation result.

### One-shot

`hermes_cli.oneshot._run_agent()` owns the entire logical turn and one SessionDB handle. It resolves the initial route and builds the first agent as accepted. If a routed restart receipt returns, `_run_agent()` keeps the existing SessionDB and sealed transcript, terminally closes the old attempt, validates the carried cursor, obtains fresh IDs/router decision before new runtime credentials and `AIAgent` construction, rebinds its existing noninteractive callbacks and calls the private continuation seam. It never calls `_run_agent()` recursively, reloads the original `prompt` as a user message or closes the SessionDB between attempts. The existing finalizer closes the final agent/store once. No background or process-level retry owner is added.

## C6-04 — Route signatures, prompt cache and callbacks

The accepted route signature remains the construction/reuse key and SHALL include the fresh accepted route identity plus router contract/generation identity. For continuation:

- a changed signature closes/discards the old agent for execution purposes and uses the existing surface reconstruction path;
- exact-signature reuse, where already supported, preserves the same sealed continuation and cursor and never clears the spent fallback state;
- credential and executor construction occur only after the fresh route decision;
- every surface rebinds the same callbacks it would bind for a normal attempt before the continuation call;
- no callback is copied into the public router request or continuation record.

The exact transcript prefix is not rewritten, reordered or truncated by C6. Existing sanctioned compression and provider-projection repair remain the only history transformations. Existing system-prompt restoration, route-identity validation, tool-schema freeze and prompt-cache policy remain authoritative; C6 does not introduce a second system prompt, a synthetic user cache break or history rewind. If the new route's existing prompt/runtime compatibility rules require an agent rebuild, that rebuild uses the existing persisted system-prompt authority and the exact sealed transcript rather than regenerating a logical turn.

## C6-05 — Notices, lifecycle and failure behavior

Events remain bounded and use the accepted route lifecycle:

- old routed attempt: `route_finished` with terminal state `routed_restart_required`, bounded reason and its original IDs;
- fresh attempt, only after budget/cursor validation: `route_requested`, then the accepted `route_accepted`/`route_started` or `route_not_started` sequence under fresh IDs;
- final started continuation attempt: `route_finished` with its actual terminal state.

A pre-start fallback notice is emitted through the existing surface renderer after the fresh decision and before executor start. It includes only execution kind, old attempt ID/terminal code, fresh request/attempt IDs, accepted route fields and bounded reason. Renderer failure remains non-authoritative and does not alter lifecycle or routing.

Exceptions or crashes are handled as follows:

1. Failure to seal a structurally valid transcript/current-turn boundary or a contradictory persistence watermark fails closed. No new attempt is created and no replay is attempted.
2. Budget exhaustion or invalid cursor returns the existing terminal fallback-exhausted surface result after closing the old attempt; it creates no fresh IDs and does not invoke the router.
3. Router `stop/error`, credential failure or construction failure closes the fresh attempt through existing `route_not_started` semantics and does not return to the old attempt.
4. Exception during continuation closes the fresh started attempt through existing `route_finished` failure handling and may consume another slot only if the same existing authority explicitly allows it.
5. Process crash preserves whatever the existing transcript owner made durable. C6 adds no second journal or automatic replay. On restart, absence of a complete valid continuation record/current-turn boundary is terminal for automatic routed continuation; existing crash recovery may expose the retained transcript to the user but SHALL NOT re-execute completed tools or synthesize/requeue the original user turn. One-shot has no cross-process automatic continuation.
6. A late result from a terminal old attempt is ignored by the existing attempt/request identity fencing and cannot mutate the fresh transcript or lifecycle.

## C6-06 — Required RED→GREEN tests

The following tests are required and no broader test family is added by C6.

### Shared private continuation seam

1. `test_main_turn_continuation_uses_exact_transcript_without_user_replay`: the first routed attempt appends one user row, emits an assistant tool call and receives a tool result; continuation starts from those exact rows, appends no user row, preserves strict alternation/current-turn identity and reaches a final assistant result.
2. `test_main_turn_continuation_does_not_repeat_completed_tool_side_effect`: a deterministic tool increments an external fixture counter once before fallback; the continuation sees the persisted tool result and the counter remains exactly one.
3. `test_main_turn_continuation_carries_spent_cursor_across_rebuild_and_primary_restore`: after slot N is consumed, a fresh/reused agent and `restore_primary_runtime` cannot select any slot `<= N`; the request projection starts at `N+1`, and exhaustion stops without fresh IDs/router call.
4. `test_main_turn_continuation_fresh_identity_and_lifecycle_order`: old `route_finished` precedes issuance/use of distinct fresh request/attempt IDs; the router is invoked once for each attempt; new credentials/agent/executor construction occurs only after the fresh accepted decision.

### Classic CLI

1. `test_cli_routed_fallback_continues_same_turn_once`: exercises one staged user row, one completed tool side effect, preserved assistant/tool transcript, fresh IDs and second router call, changed-signature rebuild with CLI callbacks rebound, and one final history adoption.
2. `test_cli_routed_fallback_budget_exhaustion_is_not_reset`: rebuild/primary restore cannot reuse the consumed slot and exhaustion is terminal without recursive `chat()` or queued input.
3. `test_cli_no_router_main_turn_and_native_fallback_are_unchanged`: existing user staging, kwargs, history persistence, credential/config/init order and native in-place fallback remain unchanged with no router events or continuation record.

### TUI/backend

1. `test_tui_routed_fallback_continues_inside_one_prompt_submit`: exactly one admission/inflight marker/message.start and one user row; tool side effect executes once; partial assistant/tool history survives; fresh IDs/router call and changed-signature reconstruction occur before private continuation; callbacks are rebound; history commits once.
2. `test_tui_routed_fallback_budget_exhaustion_is_not_a_queued_followup`: the consumed slot survives reconstruction, exhaustion fails closed and neither `_enqueue_prompt` nor `_dispatch_followup_turn` nor a second `_run_prompt_submit` is called.
3. `test_tui_no_router_main_turn_and_native_fallback_are_unchanged`: existing history-version, inflight, agent invocation and native fallback behavior remain unchanged without route events/continuation state.

### Gateway

1. `test_gateway_routed_fallback_continues_one_inbound_turn`: one inbound user row including its original platform metadata, one tool side effect, exact partial assistant/tool transcript, old terminal lifecycle before fresh IDs/router invocation, post-decision runtime/agent construction, callback rebinding and one final delivery/persistence path.
2. `test_gateway_routed_fallback_budget_exhaustion_never_queues_or_replays`: consumed budget survives agent replacement/primary restore; exhaustion invokes neither inbound dispatch, queued-followup, crash auto-resume nor the router again.
3. `test_gateway_no_router_main_turn_and_native_fallback_are_unchanged`: existing history selection, user persistence kwargs, runtime/construction ordering, streaming/delivery and native fallback remain unchanged with no synthetic route state.

### One-shot

1. `test_oneshot_routed_fallback_continues_before_single_close`: one SessionDB, one user row, one tool side effect, preserved partial transcript, fresh IDs/router invocation, post-decision credential/agent build and final close exactly once.
2. `test_oneshot_routed_fallback_budget_exhaustion_does_not_recurse`: consumed budget survives rebuild/primary restore; exhaustion does not call `_run_agent()` recursively, recreate the prompt as user input or invoke the router after exhaustion.
3. `test_oneshot_no_router_main_turn_and_native_fallback_are_unchanged`: existing prompt/resume history, construction kwargs, lifecycle closure and native fallback remain unchanged with no continuation record/events.

The positive per-surface tests SHALL assert directly that the persisted SessionDB transcript contains exactly one original user row and one copy of the completed assistant tool-call/tool-result pair. Mock call order alone is insufficient for those assertions.

## C6-07 — Traceability and exact supersession

C6 refines only accepted behavior already required by:

- FR-003 and FR-031: no-router compatibility;
- FR-011, FR-025 and FR-026: one-attempt identity/idempotency and immutable route;
- FR-027 and FR-029: host-owned fresh-attempt fallback and post-decision credential binding;
- FR-030: bounded pre-start fallback notice;
- ARC-001 sections 3, 4 and 7: one resolver, thin main-turn adapters and fresh resolution per fallback;
- ARC-003 sections 1, 4, 5, 6 and 8: lifecycle order, existing store, immutable route, existing retry authority and safe errors;
- ARC-004 sections 3, 6 and 7: requested/actual lifecycle, bounded notices and existing persistence authority;
- ARC-005 sections 2, 6 and 7: four main-turn surfaces, fallback and route-aware agent construction;
- ARC-006 sections 2 and 7: host runtime ownership and minimum surface integration set;
- T003 S002-S008: all four surface adapters, lifecycle, fallback re-entry and no-router characterization.

On owner acceptance, C6 supersedes exactly:

1. ARC-005 section 6 bullet `main_turn per-turn adapter creates new IDs, reruns router and constructs/reuses only matching fresh agent` with C6-01 through C6-05, because the old sentence omitted the required history-safe continuation mechanism and incorrectly assumed an existing one.
2. The phrase `existing surface re-entry` for routed `main_turn` in ARC-005 section 6 and T003 S006, only to the extent it claims Classic CLI/Gateway already provide that seam. Existing surfaces remain owners; the new private continuation seam is the minimum missing internal mechanism.
3. ARC-003 section 6 only for routed `main_turn` execution ordering: the existing authority still owns permission and budget, while C6 fixes immutable transcript/cursor carry and the fresh-attempt order.

All other ARC-001 through ARC-006, C1-C5 and T003 clauses remain unchanged. Acceptance of C6 would authorize integration into canonical architecture/task wording only under the appropriate owner-approved architecture-correction flow; it would not itself authorize source/test implementation, T004, T005, commit or any external/LIVE action.

## Explicitly rejected

C6 rejects:

- replaying the original user message or appending/persisting a second user row;
- rewinding, truncating, deleting or replacing the partial assistant/tool transcript;
- re-executing completed tool calls or side effects;
- using `/steer`, interrupt replay, queued follow-up, goal continuation, Gateway pending-message drain or crash auto-resume as routed fallback continuation;
- a second dispatcher, gateway, transcript/event store, continuation journal, retry engine, retry budget or fallback owner;
- exposing transcript/history, continuation records, callbacks, credentials or persistence handles to the plugin;
- a public continuation API, new model tool, new plugin capability or compatibility promise;
- queue-based continuation;
- T004 native-child or T005 Kanban expansion;
- product routing policy, commit, publication, installation, profile/runtime mutation, pilot or LIVE.

## Pre-release compatibility disposition

No public API change is proposed. C6 is private host orchestration for accepted `main_turn` semantics. The public `ExecutionRouteRequestV1`, decision/provider/event contracts, capability `execution.routing`, plugin consent tuple and contract version `1.0` do not change.

Compatibility was checked against the current project state before drafting: Git HEAD is `0b3c4e19ed457772dc7c45d413a259d0c11f1a8e`; the repository has no remotes and no tags; `agent/execution_router.py` is untracked at HEAD; canonical T007/T008 remain unchecked; no release, package publication, installation or consumer compatibility dependency exists. Therefore neither the execution-router contract version nor plugin capability/consent version needs to change if C6 remains entirely private and no public field/signature is added during implementation. Any later need for a public continuation surface would require a separate amendment and compatibility review.

## Acceptance placeholders

- Exact reviewed canonical plan SHA-256: `3a268443ba084699739a4017bf52a158d2a7733484fb8dd242ec1eb86d452a71`
- Exact accepted C5 canonical record SHA-256: `3a2754b0206225a2541623337c4717f2e7dc56a0921cc5345c5843bd2406e603`
- Exact C6 accepted-source SHA-256 presented for acceptance: `b1d306fd1f6252d9034d0a84c5ba733f218cb101772ada0842d077bc4f7796f9`
- Owner decision: `ACCEPT`
- Owner acceptance reference/text: `Owner accepted all C6-01 through C6-07 blocks and then explicitly accepted the unchanged whole DRAFT specs/002-execution-router-public-api/drafts/architecture-correction-c6.md by exact SHA-256 b1d306fd1f6252d9034d0a84c5ba733f218cb101772ada0842d077bc4f7796f9.`
- Accepted at: `2026-09-11T19:17:55+03:00`
- Integrated into canonical `plan.md`: `YES`
- T003 completion effect: `NONE`
