# AgentRuntime Plugin API v1

This legacy path is retained for links and historical navigation. It is not a
second contract.

Read the [canonical AgentRuntime v1 ADR](adr/agent-runtime-v1.md) for the
frozen Revision 4 host boundary, including:

- Hermes ownership of prompt/messages, memory, skills, tools, approval,
  execution, delegation, background work, transcript, state, visible lifecycle,
  and usage receipts;
- the provider-plugin `tools=[]`, `setting_sources=[]`, no-preset/no-native-
  `Agent` contract and strict `mcp__hermes-tools__*` exposure;
- prompt equality, pre-effect paired tool transcript persistence, host content
  streaming, and typed terminal events;
- one continuously running per-session async lifecycle loop with fresh caller
  context and approval callbacks rebound for every turn; and
- a host-issued receipt correlation that is stable for retries within one
  Hermes turn and distinct between turns, never the session-scoped task id; and
- the honest at-least-once adapter boundary with idempotent durable consumers.

The host capability `host_tool_request_id_v1` exposes the provider-neutral
`RuntimeHostServices.execute_tool()` request-ID seam. Its `request_id` argument
is optional and keyword-only for compatibility with existing two-argument
callers; omitted IDs use a host-generated per-turn namespace.

Historical Revision 4 implementation source before its documentation-only restamp:
`e6398e75c24be9b3e22f024d621a0221414cfe65`.

Do not add requirements, provider policy, or implementation detail here; update
the ADR and its [canonical coupling map](architecture/agent-runtime-v1-coupling-map.md)
if the architecture changes.
