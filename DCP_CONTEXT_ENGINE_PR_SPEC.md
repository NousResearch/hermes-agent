# Temporary PR Spec: DCP Context Engine

Status: temporary agent-facing implementation spec.

Cleanup rule: before merge, either delete this file or fold the durable parts into:

- `website/docs/developer-guide/context-compression-and-caching.md`
- `website/docs/developer-guide/context-engine-plugin.md`
- `website/docs/developer-guide/prompt-assembly.md`
- `website/docs/developer-guide/agent-loop.md`
- `website/docs/developer-guide/architecture.md` if the file layout changes
- `website/docs/reference/cli-commands.md` if `/dcp` commands land

This file is for coordination during the PR. It is not intended to become permanent product documentation.

## Goal

Add a DCP-style context engine to Hermes Agent, based on the design pattern from `Opencode-DCP/opencode-dynamic-context-pruning`:

- model-callable `compress` tool
- stable message and compression block references
- outbound API-call-time context transform
- DCP-compatible config surface under `context.dcp`
- automatic strategies for duplicate tool output and old error cleanup
- nudges that teach the model when to call `compress`
- canonical transcript preservation

The target is not just “make the existing compressor more proactive.” The target is a separate `context.engine: dcp` mode whose primary compaction mechanism is model-guided compression over referenced message ranges or individual messages.

## Non-goals

- Do not copy DCP source or prompt text verbatim. DCP is AGPL-3.0-or-later. Reimplement behavior in Hermes-native Python with original prompts.
- Do not replace the existing `ContextCompressor` default.
- Do not mutate canonical conversation history during normal DCP operation.
- Do not run both the existing reactive compressor and DCP as primary compression systems in the same session.
- Do not implement the full `/dcp decompress` and `/dcp recompress` command surface in the first PR unless the core engine is already stable.

## Current Hermes architecture facts

Primary files:

- `agent/context_engine.py`: context engine ABC.
- `agent/context_compressor.py`: built-in default lossy compressor.
- `run_agent.py`: AIAgent loop, context engine loading, tool registration, API message assembly.
- `agent/prompt_caching.py`: Anthropic prompt caching.
- `hermes_cli/config.py`: `DEFAULT_CONFIG` and config migration.
- `hermes_cli/commands.py`, `cli.py`, `gateway/run.py`: slash command registry and dispatch.

Existing context engine behavior:

- One context engine is active per agent.
- `context.engine: "compressor"` selects the built-in default.
- Plugin/context engines can expose tools via `get_tool_schemas()`.
- Context engine tools are routed to `handle_tool_call()` by `run_agent.py`.
- Existing `compress()` returns a new valid OpenAI-format message list and can mutate the session history when invoked by the current compression loop.

DCP needs one new context engine capability:

- transform the API-call copy of messages before provider conversion/caching/request, without mutating canonical messages.

## Design decision

Implement DCP as a first-class context engine named `dcp`.

Recommended file layout for first PR:

```text
agent/dcp_config.py
agent/dcp_context_engine.py
agent/dcp_state.py
agent/dcp_ids.py
agent/dcp_prompts.py
agent/dcp_transform.py
agent/dcp_strategies.py
agent/dcp_compress.py
```

If this feels too many modules during implementation, start with fewer files but preserve the responsibility boundaries:

- config parsing
- state model
- ID assignment
- outbound transform
- strategies
- compress tool handling
- prompt/nudge generation

## Required context engine interface change

Add an optional method to `ContextEngine`:

```python
def transform_api_messages(
    self,
    api_messages: list[dict],
    *,
    canonical_messages: list[dict],
    system_prompt: str,
    tools: list[dict] | None,
    api_call_count: int,
    model: str,
    provider: str | None,
    session_id: str | None,
) -> list[dict]:
    return api_messages
```

Contract:

- Receives the API-call copy, not authoritative history.
- Must not mutate `canonical_messages`.
- Should avoid mutating `api_messages` in place unless documented and tested.
- Must preserve valid OpenAI message ordering.
- Must not separate an assistant `tool_calls` message from its required tool results.
- Must return messages that still pass Hermes provider sanitization.
- Should run before prompt-cache marker placement so caching logic sees the actual outgoing request.

Wire this into `run_agent.py` after API messages are assembled and before provider-specific cache-control/sanitization/request conversion.

## DCP engine behavior

### Canonical history invariant

DCP mode must preserve this invariant:

> The saved session transcript remains the full conversation. DCP only replaces or annotates content in the outbound API request copy.

The `compress` tool updates DCP state. It does not rewrite `messages` directly.

### Message references

The engine assigns stable refs that the model can use in `compress` calls.

Use:

- `m0001`, `m0002`, ... for canonical messages
- `b1`, `b2`, ... for compression blocks

Refs should be stable across turns within a session. If message IDs are not available in the canonical message dict, assign by append order and persist the mapping in DCP state.

The transform must expose refs to the model. Two implementation modes are acceptable:

1. Inline refs, closer to DCP
   - append or prefix a compact marker to user/assistant/tool content
   - examples: `<dcp-ref id="m0042" />`, `<dcp-ref id="m0043" tool="terminal" />`
   - risk: can disturb tool result adjacency or provider formatting if done carelessly

2. Index refs, safer first pass
   - inject a compact `<dcp-context-index>` block into the latest user-facing context
   - maps refs to role/tool/topic one-liners
   - less DCP-like, but lower provider risk

Recommendation for first PR:

- implement inline refs only for messages whose content can be safely represented as text
- keep a fallback index for messages that cannot be safely annotated
- add tests proving canonical messages are unchanged

### Compression blocks

A compression block is an active replacement over either:

- a contiguous range of message refs, range mode
- one or more individual message refs, message mode

Minimal state model:

```python
@dataclass
class CompressionBlock:
    block_id: int
    run_id: int
    mode: Literal["range", "message"]
    topic: str
    summary: str
    active: bool
    start_ref: str | None
    end_ref: str | None
    message_refs: list[str]
    included_block_ids: list[int]
    consumed_block_ids: list[int]
    created_at: float
    deactivated_at: float | None = None
    deactivated_by_block_id: int | None = None
```

Session state should also track:

```python
@dataclass
class DCPSessionState:
    session_id: str | None
    next_message_ref: int
    next_block_id: int
    next_run_id: int
    ref_by_message_key: dict[str, str]
    message_key_by_ref: dict[str, str]
    blocks_by_id: dict[int, CompressionBlock]
    active_block_ids: set[int]
    last_prompt_tokens: int
    last_user_turn_index: int
    turns_since_last_compress: int
    stats: dict[str, Any]
    manual_mode: bool | Literal["compress-pending"]
    pending_manual_focus: str | None
```

Persistence:

- First PR may keep state in memory for CLI and gateway agent lifetime.
- Prefer lightweight JSON persistence under Hermes home if session IDs are stable enough.
- Do not entangle DCP state with canonical `SessionDB` schema in first PR unless necessary.

### Compress tool

Tool name: `compress`.

Expose only when:

- `context.engine == "dcp"`
- `context.dcp.compress.permission != "deny"`
- `context.dcp.commands.enabled` does not disable tool exposure, if we choose to mirror DCP semantics there

Range mode schema:

```json
{
  "type": "object",
  "properties": {
    "topic": {
      "type": "string",
      "description": "Short 3-5 word label for the compression batch"
    },
    "content": {
      "type": "array",
      "items": {
        "type": "object",
        "properties": {
          "startId": {"type": "string"},
          "endId": {"type": "string"},
          "summary": {"type": "string"}
        },
        "required": ["startId", "endId", "summary"]
      }
    }
  },
  "required": ["topic", "content"]
}
```

Message mode schema:

```json
{
  "type": "object",
  "properties": {
    "topic": {"type": "string"},
    "content": {
      "type": "array",
      "items": {
        "type": "object",
        "properties": {
          "messageId": {"type": "string"},
          "topic": {"type": "string"},
          "summary": {"type": "string"}
        },
        "required": ["messageId", "topic", "summary"]
      }
    }
  },
  "required": ["topic", "content"]
}
```

Tool result should be JSON text:

```json
{
  "ok": true,
  "mode": "range",
  "created_blocks": [1, 2],
  "deactivated_blocks": [0],
  "estimated_tokens_removed": 42000,
  "estimated_summary_tokens": 1800,
  "message": "Compressed 2 ranges into b1 and b2."
}
```

Validation failures must be explicit and non-silent:

```json
{
  "ok": false,
  "error": "Unknown message ref: m0042",
  "hint": "Use refs visible in the current context."
}
```

### Applying blocks in outbound transform

When transforming API messages:

- detect active compression blocks
- replace covered content with a compact summary placeholder
- keep enough structural messages to satisfy provider format rules
- do not orphan tool results
- do not remove current active tail

Placeholder format should be concise and model-readable:

```text
<dcp-compressed-block id="b3" topic="Dependency investigation">
Summary: ...
Covers: m0018-m0036
</dcp-compressed-block>
```

For messages fully covered by a block but not chosen as the block anchor, replace with a very short placeholder or remove only if tool adjacency remains valid:

```text
[DCP: content moved into compressed block b3]
```

Do not delete assistant tool-call messages unless their paired tool results are also handled safely. It is safer in first PR to stub content than to physically remove messages.

### Nudges

DCP should teach the model to call `compress`.

Inject a DCP system extension or ephemeral message containing:

- `compress` tool exists
- refs are available as `mNNNN` and `bN`
- use range mode for closed contiguous spans
- use message mode for isolated large messages if configured
- do not compress the active task or very recent messages
- preserve concrete details in summaries

Nudge types:

- context limit nudge when prompt tokens exceed max limit
- turn nudge when prompt tokens exceed min limit and new user turns happen
- iteration nudge when many assistant/tool messages happen since last user turn
- manual nudge when `/dcp compress [focus]` is used

`nudgeForce`:

- `soft`: suggestion wording
- `strong`: direct instruction to compress before continuing if safe

### Automatic strategies

Deduplication:

- enabled by `context.dcp.strategies.deduplication.enabled`
- signature is `tool_name + "::" + stable_json(args)`
- keep latest matching output
- replace older repeated tool outputs with placeholder
- respect protected tools

Purge errors:

- enabled by `context.dcp.strategies.purgeErrors.enabled`
- after N turns, remove or summarize old failed tool inputs/large outputs
- preserve the actual error message
- respect protected tools

Turn protection:

- if `context.dcp.turnProtection.enabled`, never prune messages in the last N user turns

Protected tools:

Recommended Hermes defaults:

- `delegate_task`
- `todo`
- `memory`
- `skill_view`
- `skill_manage`
- `write_file`
- `patch`
- `clarify`
- `cronjob`
- `compress`

Also consider protecting MCP tools with mutating prefixes:

- create
- update
- delete
- remove
- write
- patch

Do not blindly protect all `terminal` calls. Terminal output is often the largest waste source. Instead, protect commands only if later evidence shows this is needed.

## Config surface

Add defaults under `context.dcp` in `hermes_cli/config.py`:

```yaml
context:
  engine: compressor
  dcp:
    enabled: true
    debug: false
    pruneNotification: detailed
    pruneNotificationType: chat
    commands:
      enabled: true
      protectedTools: []
    manualMode:
      enabled: false
      automaticStrategies: true
    turnProtection:
      enabled: false
      turns: 4
    experimental:
      allowSubAgents: false
      customPrompts: false
    protectedFilePatterns: []
    compress:
      mode: range
      permission: allow
      showCompression: false
      summaryBuffer: true
      maxContextLimit: 100000
      minContextLimit: 50000
      modelMaxLimits: {}
      modelMinLimits: {}
      nudgeFrequency: 5
      iterationNudgeThreshold: 15
      nudgeForce: soft
      protectedTools: []
      protectUserMessages: false
    strategies:
      deduplication:
        enabled: true
        protectedTools: []
      purgeErrors:
        enabled: true
        turns: 4
        protectedTools: []
```

Support numeric limits and percentage strings:

- `100000`
- `"50%"`

Model-specific limit maps should support Hermes provider/model naming. Use exact strings first and document the caveat:

```yaml
modelMaxLimits:
  openai/gpt-5.5: "80%"
  openrouter/anthropic/claude-sonnet-4.6: 120000
```

## Permission behavior

DCP supports `allow`, `ask`, and `deny`.

First PR recommendation:

- `allow`: expose and execute `compress`
- `deny`: do not expose `compress`
- `ask`: expose but route through Hermes approval if feasible; otherwise treat as `allow` with a warning in logs and status

Do not fake interactive approval if Hermes does not have the plumbing at the context-engine tool layer yet.

## Slash commands

First PR can defer slash commands, but if included, add only:

- `/dcp`
- `/dcp context`
- `/dcp stats`
- `/dcp manual [on|off]`
- `/dcp compress [focus]`

Later PR:

- `/dcp sweep [n]`
- `/dcp decompress <block>`
- `/dcp recompress <block>`

Commands should dispatch to the active context engine if `engine.name == "dcp"` and return a clear message otherwise.

## Interaction with existing compression

When `context.engine == "dcp"`:

- DCP should not routinely call the existing `ContextCompressor`.
- `DCPContextEngine.should_compress()` should return `False` by default.
- DCP uses nudges and the `compress` tool as its primary mechanism.
- Existing compressor may be used only as emergency fallback if the API request would exceed the hard provider limit or a context-length error is returned.

Gateway hygiene is still a safety net. It may need a DCP-aware branch eventually because hygiene compression mutates gateway session history. For first PR, document the interaction and avoid making gateway hygiene call DCP unless there is a clean state path.

## Appendix: Prompt caching internals and DCP interaction

### What Hermes prompt caching is

Prompt caching in Hermes is Anthropic-specific. It is not a generic cache across all providers. It exploits Anthropic's prefix caching API, where the provider caches the stable prefix of a prompt across turns and bills reused tokens at a reduced rate (~25% of full input price).

Non-Anthropic providers (OpenAI, Gemini, local models) do not participate. The cache-control logic is a no-op for them.

### Policy decision

`_anthropic_prompt_cache_policy()` in `run_agent.py` decides at init time whether caching is enabled and which layout to use. Returns `(should_cache, use_native_layout)`.

Enabled when:
- native Anthropic (provider=anthropic, api_mode=anthropic_messages)
- OpenRouter serving Claude models
- third-party Anthropic-compatible gateways serving Claude models
- MiniMax on its Anthropic-compatible endpoint
- Qwen/Alibaba on OpenCode/DashScope (OpenAI-wire, but they honor cache_control markers)

Two layouts:
- `use_native_layout=True` -- cache_control markers on inner content blocks. Required by native Anthropic API.
- `use_native_layout=False` -- markers on the message envelope. Expected by OpenRouter and OpenAI-wire proxies.

All other providers/models: caching off.

### The system_and_3 strategy

`apply_anthropic_cache_control()` in `agent/prompt_caching.py` is a pure function. Deep-copies the entire api_messages list, then places up to 4 `cache_control: {"type": "ephemeral"}` breakpoints:
- breakpoint 1: system prompt (index 0 if role=="system")
- breakpoints 2-4: last 3 non-system messages

Anthropic allows a max of 4 breakpoints. The system prompt is the most stable (rarely changes). The last 3 messages form a rolling window that shifts forward each turn.

TTL is configurable: "5m" (default) or "1h" (2x write cost, better for long pauses between turns).

### How Anthropic prefix caching works

Anthropic caches everything from the start of the prompt up to and including the message with a cache_control breakpoint. The cache key is the exact byte content of that prefix.

Example across 3 turns:

```
Turn 1: [system] [msg1] [msg2] [msg3]
                       ^bp2   ^bp3   ^bp4
         Cache miss. Full price. Cache stored for [system..msg3].

Turn 2: [system] [msg1] [msg2] [msg3] [msg4]
                                         ^bp4
         [system..msg3] is cached. Only [msg4] is full price.
         New cache stored for [system..msg4].

Turn 3: [system] [msg1] [msg2] [msg3] [msg4] [msg5]
                                                 ^bp4
         [system..msg4] is cached. Only [msg5] is full price.
```

The invariant: the prefix must be byte-for-byte identical. Any change to any cached message invalidates the entire cached prefix from that point forward.

### How Hermes protects cache stability

Several deliberate design choices in run_agent.py prevent cache churn:

1. Shallow copy per message, not deep copy. `api_msg = msg.copy()` at line 11097 creates a provider-bound copy. The canonical `messages` list is never mutated by anything downstream.

2. Plugin context injected into user message, not system prompt. Lines 11147-11150: "system prompt modifications break the prompt cache prefix." Plugin context from pre_llm_call hooks goes into the current turn's user message instead.

3. Ephemeral system prompt appended once. ephemeral_system_prompt changes the system message, but only at setup time, not per-turn.

4. Whitespace normalization. Lines 11215-11217 strip content whitespace on the API copy. Prevents accidental cache misses from trailing whitespace.

5. Deterministic JSON serialization for tool_calls. Lines 11218-11240 re-serialize tool_call arguments with `sort_keys=True` and compact separators `(",",":")`. Same logical tool call always produces same bytes.

6. Cache control deep-copies. `apply_anthropic_cache_control()` deep-copies the message list before placing markers, so the pre-cache messages are not aliased.

### The full API-call pipeline

This is the actual sequence in `run_agent.py` for every API call, showing where DCP was inserted:

```
messages (canonical, never mutated)
  |
  v
api_messages = shallow copy of messages              (line 11095-11138)
  - ephemeral context injected into user message
  - reasoning content handled
  - internal fields stripped (finish_reason, _thinking_prefill, etc.)
  |
  v
system prompt prepended                              (line 11151-11152)
  |
  v
prefill messages injected                            (line 11156-11159)
  |
  v
DCP transform_api_messages()                         (line 11166-11175)
  |
  v
apply_anthropic_cache_control()                      (line 11187-11191)
  - deep copies api_messages
  - places cache_control breakpoints
  |
  v
_sanitize_api_messages()                             (line 11197)
  - orphaned tool result safety net
  |
  v
_drop_thinking_only_and_merge_users()                (line 11207)
  - removes thinking-only assistant turns
  |
  v
whitespace normalization + deterministic JSON        (line 11215-11240)
  - strip content whitespace
  - sort_keys=True on tool_call arguments
  |
  v
surrogate sanitization                               (line 11246)
  |
  v
API call
```

### Why DCP runs before cache-control injection

The ordering DCP -> cache_control is not arbitrary. It matters because:

1. Cache-control deep-copies the already-transformed messages and places breakpoints on the DCP-transformed shape. The provider sees the final combined form.

2. If cache-control ran before DCP, Hermes would mark one prefix as cacheable, then DCP would rewrite it afterward. The cache-control decision would be stale or wrong.

3. By running DCP first, Hermes computes cache-control over the final provider-bound prompt shape.

### The cache risk with DCP

The system_and_3 strategy places breakpoints on the system prompt and the last 3 messages. DCP's transforms (refs, compression placeholders, nudges) primarily affect older messages in the middle of the transcript.

If DCP rewrites message content in the cached prefix region between turns, the cache breaks. Specifically:

- If DCP compresses messages 5-20 into a block placeholder while messages 21-25 are the last 3 (breakpoint targets), the cached prefix `[system, msg1..msg22]` from the previous turn is now invalid because messages 5-20 changed. Cache miss. But the new shape `[system, msg1-placeholder..msg22]` gets cached for the next turn.

- If DCP does nothing (no compression triggered), messages are identical to the previous turn and the cache works normally.

- The worst case is DCP making small changes every turn. That would defeat caching entirely.

### DCP's mitigation for cache churn

The implementation tries to reduce churn by:

1. Only compressing when the model explicitly calls the `compress` tool or when automatic strategies find actual duplicates/errors. Between compression events, the prefix stays stable.

2. Automatic strategies (dedup, purge-errors) are recalculated only when state changes (compress tool call, new turn), not randomly every request.

3. Compression block application is deterministic. Same active blocks always produce the same placeholder text.

4. Refs are stable across turns within a session. `m0001` always refers to the same canonical message.

5. Never appending moving counters or timestamps into the cached prefix.

### What is not yet solved

This does not prove every possible prompt-cache churn case is solved. Known open concerns:

1. When DCP compresses a range that overlaps the cached prefix, the entire prefix invalidates for one turn. The next turn re-caches with the new shape. This is a one-turn penalty per compression event, which is acceptable.

2. Nudges are appended to the live tail (not the cached prefix), so they should not cause churn. But this has not been instrumented or tested against real Anthropic cache behavior.

3. Non-Anthropic providers do not use this cache system. DCP transforms still run for them, but there is no prompt cache to worry about.

4. `apply_anthropic_cache_control` does `copy.deepcopy(api_messages)` internally. DCP transforms the first copy, then cache_control deep-copies again. Two full message copies exist briefly during each API call. For a 100k-token transcript this is noticeable but not catastrophic.

### Rules for future DCP changes

- Run DCP transform before cache-control marker placement (current ordering).
- Avoid changing old stable content every turn.
- Compression block application must be deterministic.
- Dedup and purge must avoid churn. Recalculate after compress tool calls or when state changes, not randomly every request.
- Never append moving counters into the cached prefix unless unavoidable.

## Tests

Use `scripts/run_tests.sh`, not direct pytest.

Add tests:

```text
tests/agent/test_dcp_config.py
tests/agent/test_dcp_context_engine.py
tests/agent/test_dcp_transform.py
tests/agent/test_dcp_compress_tool.py
tests/agent/test_dcp_strategies.py
```

If slash commands are implemented:

```text
tests/cli/test_dcp_command.py
```

Required coverage:

- config defaults and mode parsing
- percentage limit resolution
- `deny` permission hides tool
- range schema shape
- message schema shape
- canonical messages are not mutated by transform
- refs are stable across calls
- range compression creates active block
- overlapping range compression consumes/nests prior block or returns clear error
- invalid refs return clear error
- active blocks replace outbound content
- tool adjacency is preserved
- dedup keeps latest duplicate and prunes earlier duplicate
- protected tools are not deduped
- purgeErrors keeps error text but prunes old bulky input/output
- turn protection prevents pruning recent messages
- prompt-cache marker logic still receives valid messages

Run focused tests first:

```bash
scripts/run_tests.sh tests/agent/test_dcp_config.py tests/agent/test_dcp_context_engine.py tests/agent/test_dcp_transform.py tests/agent/test_dcp_compress_tool.py tests/agent/test_dcp_strategies.py
```

Then full suite before PR readiness:

```bash
scripts/run_tests.sh
```

## Documentation updates before merge

Permanent docs to update:

1. `website/docs/developer-guide/context-compression-and-caching.md`
   - describe `context.engine: dcp`
   - explain DCP vs built-in compressor
   - describe canonical transcript preservation
   - explain fallback and gateway hygiene interaction

2. `website/docs/developer-guide/context-engine-plugin.md`
   - add `transform_api_messages()` optional method
   - document API-call-time transform contract

3. `website/docs/developer-guide/prompt-assembly.md`
   - add context-engine API-call-time transforms
   - explain refs/nudges are ephemeral model-facing context

4. `website/docs/developer-guide/agent-loop.md`
   - add transform step to loop lifecycle
   - mention context-engine tools like DCP `compress`

5. `website/docs/developer-guide/architecture.md`
   - update file layout if new DCP modules are under `agent/`

6. `website/docs/reference/cli-commands.md`
   - only if `/dcp` commands are implemented

## Implementation order

1. Add DCP config parser and tests.
2. Add DCP state and ID assignment tests.
3. Add `transform_api_messages()` optional hook to `ContextEngine` and no-op default tests.
4. Wire the hook in `run_agent.py` using API-message copy only.
5. Add `DCPContextEngine` with status and tool schema only.
6. Add range-mode `compress` handling and block application.
7. Add message-mode handling.
8. Add nudges and system extension.
9. Add deduplication strategy.
10. Add purgeErrors strategy.
11. Add optional slash commands if scope allows.
12. Update permanent docs.
13. Delete or fold this temporary spec.

## Open questions

- Should DCP state be persisted in JSON under Hermes home in the first PR, or kept in memory until behavior stabilizes?
- Should inline refs or an index block be the default for provider compatibility?
- Should `ask` permission be implemented via existing approval callbacks in the first PR or deferred?
- How should gateway hygiene behave when `context.engine == "dcp"` and a session is already too large before the agent starts?
- Which MCP mutating tools should be protected by default?
- Should `terminal` be partially protected based on command classification, or left unprotected for token savings?

## Review checklist

Before marking the PR ready:

- No DCP AGPL source or prompt text copied.
- Canonical transcript preservation is tested.
- Provider-valid message ordering is tested.
- Existing compressor behavior still passes tests.
- DCP config is ignored unless `context.engine == "dcp"`.
- Prompt caching behavior is intentionally placed and documented.
- `/compress` existing Hermes command still works with the built-in compressor.
- `compress` tool only appears in DCP mode.
- Docs are updated or this temporary spec remains clearly marked as temporary if the PR is still draft.
