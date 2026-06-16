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
| Text completions | ✅ Clean | Streamed token-by-token via `--include-partial-messages`. |
| Tool calls (single + parallel) | ✅ Clean | Captured from the assistant turn before execution; `--max-turns 1` prevents the official client from running them. |
| Token streaming | ⚠️ Buffered | This version delivers each turn as one buffered response (`agent._disable_streaming = True`) rather than token-by-token deltas. `claude -p` *does* emit Anthropic SSE events (`content_block_delta`), so live streaming is a clean future extension, but it is not wired in the initial cut to keep the change minimal and robust. |
| System prompt | ✅ Clean | Hermes' system prompt replaces Claude Code's default via `--system-prompt` (no Claude-Code identity injection — we are not impersonating it). |
| Model selection | ✅ Clean | `sonnet` / `opus` / `haiku` aliases or full model ids passed via `--model`. |
| Subscription auth | ✅ Clean | Handled entirely by the `claude` CLI (`claude /login` or `claude setup-token`). Hermes never sees or sends the token. |
| **Conversation history** | ⚠️ Approximated | `claude -p` is stateless per invocation and has no first-class way to inject a prior assistant `tool_use` + externally-produced `tool_result`. Hermes' full message history (incl. tool results) is **serialized into the prompt text** each turn. The model sees the whole conversation, but native structured `tool_result` threading and server-side prompt caching across turns are not used. |
| **Token usage / cost** | ⚠️ Partial | `claude -p` reports `total_cost_usd` as an API-equivalent estimate, not subscription billing. Hermes does not treat it as spend. |
| `max_tokens`, `tool_choice`, exact thinking budgets | ⚠️ Best-effort | The CLI does not expose all Messages-API knobs; these are dropped or approximated rather than faked. |
| Prompt caching, `service_tier`, betas | ❌ Not mapped | Owned by the official client; not controllable from Hermes. |

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
