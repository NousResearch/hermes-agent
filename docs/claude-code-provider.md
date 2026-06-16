# Claude (Max subscription) provider — `claude-code`

This provider lets Hermes run inference on a **personal Claude Max/Pro
subscription** by delegating each model turn to the **official Claude Code
client** (the `claude` CLI, headless `claude -p`). It does **not** send
subscription tokens to `api.anthropic.com`, and it does **not** spoof a
`claude-cli` user-agent — the official client *is* the client, so there is
nothing to impersonate.

> Compliance scope: this is for using **your own** subscription via the
> official client. It is the most-compliant way to use a subscription, but
> only an Anthropic **API key** (the existing `anthropic` provider) is a hard
> guarantee. Do not use this to resell or pool subscription access.

## How it works

Hermes' agent loop is unchanged. For `provider = "claude-code"` the agent's
`api_mode` is `anthropic_messages`, but `agent._anthropic_client` is swapped
for a drop-in shim (`agent/claude_code_client.py`) instead of the real
`anthropic.Anthropic` SDK client. The shim exposes the exact surface Hermes
calls — `.messages.create(**kwargs)`, `.messages.stream(**kwargs)`, `.close()`
— so the existing dispatch, `normalize_response`, validation, and cache-stat
code all run untouched.

Under the hood the shim runs:

```
claude -p --output-format stream-json --verbose --include-partial-messages \
  --tools "" --strict-mcp-config --mcp-config <hermes-tools.json> \
  --allowedTools mcp__hermes__* \
  --system-prompt <hermes system prompt> \
  --max-turns 1 --permission-mode default --model <model>
```

and parses the streamed Anthropic events back into a `Message`-shaped object.

### Tool calls (the harness bridge)

Claude Code is itself an agent harness that owns tool execution; Hermes owns
*its* tool loop. To use Claude Code as a pure completion endpoint, Hermes'
tools are exposed to the model through a tiny stdio MCP server
(`agent/claude_code_bridge.py`, launched via `--mcp-config`). The model emits
a `tool_use` for `mcp__hermes__<toolname>`; `--max-turns 1` makes the official
client **stop at that tool boundary without executing the tool**
(`stop_reason=tool_use`). The shim captures the `tool_use`, strips the
`mcp__hermes__` prefix, and returns it to Hermes, which executes the tool
itself and feeds the result back on the next turn. The bridge's own tool
handlers are never the execution path (they return an inert sentinel if ever
reached).

## What maps cleanly vs. not

| Aspect | Status | Notes |
|---|---|---|
| Text completions | ✅ Clean | Parsed from `claude -p --output-format stream-json`. |
| Tool calls (single + parallel) | ✅ Clean | Captured from the assistant turn before execution; `--max-turns 1` makes the official client stop at the tool boundary (`stop_reason=tool_use`) without running the tool. Multiple `tool_use` blocks in one turn keep their order and `tool_use_id`s. |
| Token streaming | ⚠️ Buffered | Each turn is delivered as one buffered response (`agent._disable_streaming = True`) rather than token-by-token deltas. `claude -p` *does* emit Anthropic SSE events (`content_block_delta`), so live streaming is a clean future extension; it is intentionally left buffered here for stability (a partial-message stream that dies mid-turn is harder to reason about than a single parsed result). |
| System prompt | ✅ Clean | Hermes' system prompt replaces Claude Code's default via `--system-prompt-file` (no Claude-Code identity injection — we are not impersonating it). A large system prompt goes through a file, never argv. |
| Long prompts | ✅ Clean | The serialized conversation is piped to `claude` via **stdin**, not argv, so there is no `ARG_MAX` limit (verified at 500 KB). |
| Model selection | ✅ Clean | `sonnet` / `opus` / `haiku` aliases or full model ids passed through unchanged via `--model`. |
| Subscription auth | ✅ Clean | Handled entirely by the `claude` CLI (`claude /login` or `claude setup-token`). Hermes never sees or sends the token. |
| CLI failure / timeout / empty output | ✅ Clean | Non-zero exit, timeout, and silent/empty output all raise a clear `ClaudeCodeError` with the CLI's stderr surfaced — never a silently-empty "successful" turn. Subprocess, MCP bridge, and temp files are cleaned up on success, error, and cancellation (Ctrl-C). |
| **Conversation history** | ⚠️ Bounded approximation | `claude -p` is stateless per invocation, so Hermes' full message history (incl. tool_use/tool_result) is **serialized into the prompt text** each turn (native structured `tool_result` threading and cross-turn server-side caching are not used). Growth is linear per call (≈ +280 chars / turn measured on an 8-tool session) and **bounded by Hermes' own context compaction**, which is now driven by the real subscription window (see below) so a long session compacts before `claude -p` hits its limit rather than erroring. |
| **Token usage / cost** | ⚠️ Partial | `claude -p` reports `total_cost_usd` as an API-equivalent estimate, not subscription billing. Hermes does not treat it as spend. Because history is replayed each stateless turn, cumulative input tokens over a session are O(n²) in turns — bounded by compaction, but higher than a stateful API session would be. |
| `max_tokens`, `tool_choice`, `temperature`, `top_p`, thinking budget | ❌ Not mapped (ignored) | `claude -p` exposes **no** flags for these (only `--max-budget-usd`, which is API-key spend and irrelevant to subscriptions). They are silently ignored; output length and thinking follow the official client's own defaults. |
| Prompt caching, `service_tier`, betas | ❌ Not mapped | Owned by the official client; not controllable from Hermes. |

## History bounding policy (stress-test findings)

Because `claude -p` is stateless, the shim replays the whole conversation as
text on every call. A long, tool-heavy session was measured to confirm this is
safe:

| turn | messages | prompt chars | ~tokens |
|---|---|---|---|
| 1 | 1 | 281 | 70 |
| 5 | 9 | 1,469 | 367 |
| 9 | 17 | 2,559 | 639 |

Growth is **linear** per call (≈ +280 chars / +70 tokens per turn here), i.e.
O(n) per request and O(n²) cumulative over a session — exactly as expected for
stateless replay. There is no shim-side truncation (that would silently drop
context); instead the bound is **Hermes' existing context compaction**.

The one real risk found was that Hermes resolved the model's context window
from name heuristics (`sonnet` → 256K, `claude-sonnet-4-6` → 1M), while
`claude -p` on the subscription actually reports a **200K** window — so
compaction would have fired too late and long sessions could hit the CLI's real
limit. Fixed by pinning the `claude-code` context window to the real
subscription value (`agent/model_metadata.py`), so compaction triggers before
`claude -p`'s limit. Override with `HERMES_CLAUDE_CODE_CONTEXT_TOKENS` or
`model.context_length` in config if your subscription has a larger window.

## Auth

```
claude /login          # interactive, subscription
# or
claude setup-token     # long-lived token for headless/CI
```

No `ANTHROPIC_API_KEY` is required (and if set, it is irrelevant to this
provider — the `claude` CLI uses its own credentials).

## Selecting it

```
hermes model           # pick "Claude (Max subscription)"
# or
hermes setup
```

Config persisted to `~/.hermes/config.yaml`:

```yaml
model:
  default: sonnet
  provider: claude-code
  api_mode: anthropic_messages
```

## Overrides

- `HERMES_CLAUDE_CODE_COMMAND` / `CLAUDE_CODE_CLI_PATH` — path to the `claude`
  binary (default: `claude` on `PATH`).
- `HERMES_CLAUDE_CODE_ARGS` — extra args appended to every `claude -p` call.
- `HERMES_CLAUDE_CODE_TIMEOUT` — per-turn timeout in seconds (default 600).
- `HERMES_CLAUDE_CODE_CONTEXT_TOKENS` — override the context window that drives
  Hermes' compaction (default 200000, the standard subscription window).
