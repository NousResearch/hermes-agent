# ADR: AgentRuntime v1 host boundary (Revision 4)

Status: accepted and frozen for Revision 4

Historical Revision 4 implementation source before its documentation restamp:
`e6398e75c24be9b3e22f024d621a0221414cfe65`

Contract authority: [Hermes-owned parity issue #19](https://github.com/100yenadmin/hermes-claude-agent-sdk/issues/19)

This ADR is the canonical architecture statement. The two v1 documents outside
`docs/adr/` are compatibility pointers; they must not introduce another set of
rules.

## Decision

AgentRuntime v1 is a provider-neutral whole-turn boundary. Hermes prepares an
immutable request, resolves one descriptor, and gives the selected runtime only
typed data plus a bounded `RuntimeHostServices` facade.

Hermes owns the prompt, messages, memory, skills, tool inventory, approval,
execution, delegation, background work, transcript, state, visible lifecycle,
and usage receipts. A runtime plugin owns only its provider transport and
provider continuity: authentication or SDK setup, model calls, SDK stream
parsing, and opaque provider state. The plugin cannot receive an `AIAgent`,
`SessionDB`, gateway route, credential object, or mutable host object.

The generic host contains no provider model, subscription, dependency, OAuth,
or provider-specific preset policy. In particular, no Claude or Fable policy is
part of this boundary; those choices, if needed, are private to a provider
plugin.

## Registration and capability handshake

The host exports `RUNTIME_API_VERSION = 1` and the concrete capabilities in
`agent/runtime_capabilities.json` and `runtime_api_manifest()`:

`background_delivery_v1`, `cancellation_v1`, `compaction_events_v1`,
`host_approval_v1`, `host_content_stream_v1`, `host_status_v1`,
`host_tool_execution_v1`, `host_tool_request_id_v1`,
`provider_profile_registration_v1`,
`runtime_model_provenance_v1`, `runtime_state_v1`,
`runtime_tool_inventory_v1`, and `usage_receipts_v1`.

A plugin registers a frozen `RuntimeDescriptor` and a zero-argument factory
through the existing plugin lifecycle. The host validates the API range,
required capabilities, selectors, state schema, and compaction ownership before
retaining the factory. Descriptor registration must not resolve credentials,
install dependencies, import a provider SDK, start a process, or call a model.
Selection is a pure match over the host's built-in and plugin registrations;
ambiguous matches are rejected.

## Whole-turn and prompt contract

`AgentRuntime.preflight(request)` is pure. `run_turn(request, host)` emits a
bounded union of typed content, status, tool-request, approval-request, state,
compaction, usage, completed, cancelled, or failed events. Dispatch rejects an
unknown event, any event after a terminal event, or a stream with no terminal
event. The runtime remains open across turns and is closed only when its exact
parent-session binding is evicted or hard-closed.

`RuntimeTurnRequest` is the host's already-resolved turn surface:

- `prompt_snapshot` is the effective Hermes prompt, and `messages` is the
  effective Hermes message sequence. Both are deep-frozen before crossing the
  boundary.
- The prompt equality invariant is strict: the provider adapter must pass the
  host `prompt_snapshot` to its SDK without adding a provider preset or hidden
  context. Message order and content must likewise remain the host-provided
  values; `prompt_hash` is computed from those effective values by the host.
- `tool_schemas` and `RuntimeToolInventory` describe the delivered request,
  including per-tool schema hashes and host/plugin declaration ownership. The
  inventory is a snapshot, not permission for a second discovery pass.
- State, attachments, and correlation identifiers are generic, bounded
  envelopes. Runtime state is opaque JSON and must not contain credentials.

### Provider-plugin options

The Revision 4 Claude Agent SDK adapter has a deliberately narrow SDK
configuration contract. Its native SDK `tools` option is exactly `[]`; native
Claude Bash, Read, Write, Edit, Web, and `Agent` tools are not enabled. Its
`setting_sources` option is exactly `[]`, and the `claude_code` system-prompt
preset is not used. The only SDK-visible functions are active Hermes schemas
mapped to strict `mcp__hermes-tools__*` names. This is a plugin adapter rule,
not a Claude policy in generic Hermes core.

The adapter must therefore pass the Hermes prompt snapshot and message context
through unchanged, emit typed runtime requests for Hermes tools, and return
provider content as host-projected events. Hidden provider reasoning is outside
the contract and is neither required nor claimed as parity evidence.

## Host-owned effects and security boundaries

### Tools, approvals, and transcript pairing

`RuntimeToolRequestEvent` crosses into `host.execute_tool()`. The host checks
the current session's allowed names and sends the call through Hermes' one
canonical executor funnel: scope and middleware, approval, guardrails,
progress, dispatch, result normalization, and persistence. A plugin cannot call
the registry directly or substitute an MCP server for that executor.

Before invoking the executor, the host appends the paired assistant tool-call
row to the live turn and flushes it to the session database. Only after that
pre-effect flush succeeds can the tool execute; the canonical tool result is
then appended and flushed. A repeated request id with the same name and
arguments returns the cached host result; a repeated id with different payload
fails closed. The public `execute_tool()` seam accepts that request id as an
optional keyword-only argument for provider-neutral plugins; callers that omit
it receive a bounded host-generated per-turn id. This preserves one durable
assistant/tool pair and prevents an unpaired effect from entering the
transcript.

Approval is host policy. A denied or malformed approval terminates the runtime
turn before completion and cannot be converted into a plugin-owned success.

### Visible content and terminal lifecycle

`host_content_stream_v1` projects runtime content through the same host
sanitizing stream funnel used by built-in responses. Host finalization is the
only durable assistant-row authority: streamed content is visible immediately,
and the final response is persisted once without a duplicate assistant row.
The typed event collector also enforces exactly one terminal outcome.

The host constrains a runtime's replay claim using observed host evidence. Any
visible event or host side effect clears `replay_safe`; fallback is allowed only
when host policy permits it and the runtime explicitly classified the failure
as replay-safe before output or effects.

### Delegation and background work

Hermes' `delegate_task` and provider-neutral host events own handoff, fanout and
synthesis, stale-child isolation, and background settlement. The generic v1
host can accept a frozen `RuntimeBackgroundResult` (normalized UTF-8 text,
bounded to 16 KiB, with a completed or failed outcome) through
`emit_background_result()` for runtimes that declare that capability. The
Revision 4 Claude subscription plugin neither requires nor uses this route;
its delegation and background work are ordinary Hermes `delegate_task` work.

The host binds that result to the captured parent session and host-generated
delivery identity, then reuses the existing completion consumer for exact-parent
preflight, busy-session requeue, transcript re-entry, and adapter retry. The
plugin supplies no route, latest-session lookup, or arbitrary metadata. The v1
queue/adapter boundary is at-least-once: transport delivery may replay and a
failed injection remains retryable. Durable consumers suppress a replay only
after their authoritative transcript/display append commits, using the stable
delivery identity. v1 does not claim end-to-end exactly-once delivery or add a
new broker, daemon, datastore, or authentication boundary.

### State, usage, and compaction

The host persists `RuntimeStateEnvelope` by (Hermes session, `runtime_id`) and
upserts only validated, bounded JSON. `RuntimeUsageReceipt` is an append-only
host audit stream. A non-secret correlation id is its sole deduplication key.
Hermes supplies an id that is stable for retries within one user turn and
distinct between turns; a session-scoped task id is not a valid receipt
correlation. Correlated retries are ignored, while receipts without a
correlation id remain independent events. The legacy `model` field stays the
runtime-observed billing/ledger identity; optional selected, effective,
canonical, and bounded resolution fields do not cause the host to invent
provider aliases or policy.

Compaction ownership is descriptor-declared (`HOST` or `RUNTIME_NATIVE`). A
runtime-native implementation emits typed lifecycle phases and the host merely
projects the bounded event; it does not invoke its compressor based on a
provider name.

## Lifecycle and rollback

One runtime and one host binding are cached for the exact parent Hermes session
and reused across its turns. Per-turn dispatch does not close the runtime.
Each binding owns one lifecycle-stable, continuously running asyncio loop. All
turns and `close()` execute on that loop so a provider SDK's long-lived reader,
task group, and subprocess lifecycle never cross async-runtime contexts and
remain serviceable between gateway turns. The caller's `ContextVars` and
thread-local approval/sudo callbacks are rebound for the duration of each turn
and cleared afterward; the stable loop is not permission to retain stale
per-turn security context.

Changing the selection, descriptor, plugin ownership, or parent session closes
the old binding before replacement. Closing seals the host first so late state,
tool, content, or background calls fail closed, then closes the runtime exactly
once.

Removing a plugin unregisters its runtime and provider-profile registrations;
generic persisted state remains inert and readable. Rolling the host back to
the exact upstream identity restores built-in operation. No provider-specific
session column, core dependency, or native-agent path is required by v1.

## Consequences and proof boundary

- A clean Hermes host with no matching plugin continues through its ordinary
  conversation loop.
- The plugin can be released and removed independently, while host-visible
  policy and effects remain auditable in normal Hermes state.
- The strongest delivery claim is idempotent durable consumption after commit
  over an at-least-once transport boundary; it is not exactly-once networking.
- This ADR proves the architecture and exact host-source contract only. It does
  not prove a plugin wheel, upstream merge, release/publication, installed
  runtime, fleet or customer readiness, future-main compatibility, identical
  provider prose, or hidden reasoning parity.
