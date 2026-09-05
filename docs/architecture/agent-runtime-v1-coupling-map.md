# AgentRuntime v1 coupling map (Revision 4)

Historical Revision 4 implementation source before its documentation restamp:
`e6398e75c24be9b3e22f024d621a0221414cfe65`

The [canonical ADR](../adr/agent-runtime-v1.md) defines the rules. This map
records the final ownership split; it is not a second contract.

## Upstream module layout

The compatibility port targets upstream `006b1beb00d9d25230571d14277aca3d70e5e11f`.
It preserves AgentRuntime v1 and the ownership rules below. Selection and
host finalization now live in `agent/turn_runtime.py`, prompt projection in
`agent/turn_context.py`, and runtime state/usage in the
`hermes_state_runtime.py` mixin. Plugin teardown uses
`hermes_cli/plugins_ledger.py`; completion delivery uses
`gateway/run_notifications.py` and `gateway/run_turn.py`.
No Claude-specific runtime, SDK, authentication, or orchestration code is added
to core. This source port requires new CI and affected-path live evidence;
the earlier candidate's receipts are not silently rebound to it.

| Area | Hermes host owns | Runtime plugin owns | Boundary invariant |
| --- | --- | --- | --- |
| Registration and selection | API/version validation, capability manifest, built-in plus plugin registry, pure descriptor routing, provider-profile unload ledger | Frozen descriptor, selectors, required capabilities, zero-argument factory, provider-profile registration | Compatibility is rejected before factory, SDK, credentials, dependency installation, or model calls. |
| Prompt, messages, memory, and skills | Effective prompt composition, message context, memory/skill/context injection, prompt hash, turn correlation | Provider wire encoding only; no prompt rebuild or provider preset | SDK prompt equals the host `prompt_snapshot`; message order/content remain host-provided. |
| Tool inventory | Active tool resolution, stable schemas, `RuntimeToolInventory`, allowed-name checks, host/plugin declaration ownership | Mapping of active Hermes schemas to strict `mcp__hermes-tools__*` names | Inventory is the delivered-request snapshot; omitted or disabled tools are not silently rediscovered. |
| Provider SDK surface | Generic request/event types and host capability gate | SDK authentication/transport and stream parsing; native SDK `tools=[]`, `setting_sources=[]`, no `claude_code` preset or native `Agent` | Only active Hermes schemas cross the provider boundary; hidden provider tools and reasoning are not Hermes effects. |
| Approval and execution | Canonical executor, availability, middleware, approval, guardrails, progress, dispatch, result normalization, and persistence | Typed tool/approval requests and consumption of normalized results | Plugins cannot call a registry or alternate executor. Approval denial fails closed. |
| Tool transcript ordering | Paired assistant tool-call row, pre-effect database flush, canonical tool result row, request-ID validation, bounded per-turn fallback namespace, duplicate-id cache/conflict check | Request id, name, and arguments | No tool effect occurs before its paired assistant row is durable; same-id/same-payload retries are idempotent on the binding, and reused provider IDs remain local to their turn. |
| Visible content and lifecycle | `host_content_stream_v1` sanitizing funnel, final assistant-row finalization, terminal enforcement, status projection | Provider stream parsing and typed content/status events | Streaming can be visible before completion, but host finalization persists the response once and enforces one terminal outcome. |
| Delegation and background | `delegate_task`, handoff, fanout/synthesis, stale-child isolation, parent binding, completion queue, transcript re-entry, adapter retry | Optional bounded `RuntimeBackgroundResult` only for runtimes that declare the generic capability | The Revision 4 Claude plugin uses only Hermes `delegate_task`; it has no plugin child queue or background projector. Generic delivery is at-least-once and durable consumers dedupe after commit. |
| Runtime state and receipts | Session/runtime-keyed opaque state, validation, SQLite persistence, append-only usage receipts, correlated retry dedupe | Typed state and usage events plus provider classification | Payloads contain no credentials; host does not rewrite observed billing identity from a selection. |
| Compaction | Descriptor ownership and typed lifecycle projection; host compressor when ownership is `HOST` | Actual native compaction when ownership is `RUNTIME_NATIVE` | Generic code branches on declared ownership, never a provider name. |
| Failure, fallback, and cancellation | Host-observed effect/visibility evidence, replay gate, cancellation and exact-parent sealing | Explicit failure phase/replay classification and `close()` implementation | Any visible event or host side effect clears replay safety; fallback is never inferred from exception type. |
| Session lifecycle and removal | One cached binding and continuously running asyncio loop per exact parent session, per-turn context/approval rebinding, close-once ordering, unload and rollback to built-in path | One runtime instance per binding and provider-private async cleanup on the same loop | Long-lived SDK readers never cross async-runtime contexts; late calls after close or parent rebind fail closed; stale per-turn callbacks are cleared; plugin removal leaves generic state inert. |
| Provider policy and dependencies | Provider-neutral API only; no model, subscription, OAuth, dependency, Fable, or Claude policy | Provider SDK/dependency/version policy, auth, model selection, subscription handling, diagnostics, packaging | Provider policy cannot leak into generic Hermes core. |

## Generic background delivery sequence

This generic host facility is retained for other runtimes. The Revision 4
Claude subscription plugin does not require or use it; all of its child and
background work enters through Hermes `delegate_task`.

1. The plugin emits exactly one terminal event for its turn, then may call
   `emit_background_result()` on the same open host binding.
2. The host validates the bounded result, captures the exact parent/session route,
   and creates a host delivery identity; the plugin cannot supply either.
3. The existing completion consumer targets that parent, requeues while it is
   busy, and projects the result into the normal Hermes transcript/lifecycle.
4. The adapter may receive a replay. After the durable transcript/display append
   commits, the consumer suppresses a duplicate identity; a failed injection
   remains retryable.
5. Eviction or shutdown seals the binding before closing the runtime, so late
   background emission fails closed.

This is an at-least-once adapter boundary with idempotent durable consumption,
not an end-to-end exactly-once guarantee. v1 adds no new broker, daemon,
datastore, or auth boundary.
