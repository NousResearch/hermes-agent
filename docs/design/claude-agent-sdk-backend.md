# Claude Agent SDK inference/agent backend

Hermes can drive inference — and, optionally, whole agentic turns — through
Anthropic's **[Claude Agent SDK](https://code.claude.com/docs/en/agent-sdk/overview)**
(`claude-agent-sdk`, import name `claude_agent_sdk`) instead of the native
`anthropic` Messages client.

The headline reason to pick this path is **authentication**: the Agent SDK
spawns the bundled/installed `claude` CLI, which can authenticate with a
Claude Code / Claude Pro-Max **OAuth subscription token** as well as a plain
API key. The native `anthropic` client path also supports OAuth, but the SDK
path additionally brings the CLI's own context management, prompt caching, and
(in delegate/hybrid) its full built-in tool loop.

> **Terms of Service note.** Using a Claude Pro/Max **subscription OAuth**
> token to power a third-party agent is outside Anthropic's intended use for
> those credentials (OAuth is "intended exclusively for … Claude Code and
> other native Anthropic applications"; developers "should use API key
> authentication"). See
> [Claude Code — Legal and Compliance](https://code.claude.com/docs/en/legal-and-compliance).
> For an unambiguously compliant setup, configure an **API key**
> (`ANTHROPIC_API_KEY`); the SDK path resolves and uses it automatically.

## Requirements

- `pip`/`uv` extra: `uv pip install 'hermes-agent[claude-agent-sdk]'`
  (pinned `claude-agent-sdk==0.2.110`; lazy-installed on first use).
- The **`claude` CLI** on `PATH` at runtime (Node). Install with
  `npm i -g @anthropic-ai/claude-code` or the native installer. This is a
  runtime dependency, not a pip one.
- Provider must be native **`anthropic`** (the OAuth/subscription path only
  makes sense there; third-party Anthropic-compatible endpoints that use
  bearer/custom-header auth are not supported in SDK mode).

## Enabling it

Add to `config.yaml` under `model`:

```yaml
model:
  provider: anthropic
  default: claude-opus-4-20250514
  claude_agent_sdk:
    mode: inference          # inference | delegate | hybrid
    # delegate/hybrid only:
    permission_mode: bypassPermissions   # default for delegate/hybrid
    max_turns: 24
    # optional:
    cwd: /path/to/workdir
    allowed_tools: []                     # extra allow-list (delegate)
    system_prompt_preset: claude_code     # use Claude Code's preset instead of Hermes' soul
    append_system_prompt: "…"             # appended to the preset
```

Shorthand — a bare mode string:

```yaml
model:
  claude_agent_sdk: inference
```

Set `mode: auto`/`off`/blank, or `enabled: false`, to disable.

## Modes

| Mode | Who drives the loop | Tools | Notes |
|------|--------------------|-------|-------|
| **`inference`** (default) | **Hermes** | Hermes' own tools, executed by Hermes | Single SDK model call per turn (`max_turns=1`, no SDK tools). The SDK is a pure transport-to-Anthropic + OAuth. **Tool-calling turns are not routed through this path** — see limitation below. |
| **`delegate`** | **SDK** | SDK built-ins (Read/Edit/Bash/Grep/…) | "Run Claude Code inside Hermes." Distinct trust surface; defaults to `permission_mode: bypassPermissions`. |
| **`hybrid`** (experimental) | **SDK** | **Hermes'** tools via an in-process MCP server | SDK loop + Hermes capabilities. Agent-level tools that need live agent state (todo/memory/clarify/delegate_task/read_terminal/session_search) are **not** exposed. Results are wrapped with the same `<untrusted_tool_result>` promptware defense as the native path. |

Cross-turn context is preserved through the SDK's own session (`resume`) — the
session id is stored on the agent and replayed on the next turn.

### Limitation (inference mode)

The Agent SDK is a **session-based agent**, not a stateless completion API: its
`query()` takes a single prompt and manages its own history, and its tools come
from built-ins / MCP rather than an arbitrary passed-in `tools` array. So in
`inference` mode the SDK performs a text model call and Hermes' registered
tools are **not** offered to the model on that path. Use `delegate` or `hybrid`
when you want tool-using agent behavior through the SDK. `inference` is best for
text turns and for unlocking subscription-billed inference with the least
disruption to Hermes' own loop, tools, soul, budget and safety wrapping.

## Implementation

No new `api_mode` string is introduced. When SDK mode is active the agent keeps
`api_mode == "anthropic_messages"` and sets `_claude_agent_sdk_mode`:

- **`agent/agent_init.py`** — in the `anthropic_messages` construction block,
  detects `model.claude_agent_sdk`, resolves the token, sets the auth fields,
  skips building the real `anthropic.Anthropic` client, and forces
  non-streaming (`_disable_streaming = True`).
- **`run_agent.py`** — `_anthropic_messages_create` routes to the adapter when
  `_claude_agent_sdk_mode` is set; the credential seams
  (`_rebuild_anthropic_client`, `_try_refresh_anthropic_client_credentials`,
  `_swap_credential`) no-op / recompute the env instead of touching a client.
- **`agent/claude_agent_sdk_adapter.py`** — the backend. Builds
  `ClaudeAgentOptions`, maps Hermes' resolved token onto the CLI's env
  (`CLAUDE_CODE_OAUTH_TOKEN` for OAuth, `ANTHROPIC_API_KEY` for keys,
  `ANTHROPIC_BASE_URL` for gateways), bridges async→sync, and returns an object
  shaped like a native Anthropic `Message` (`.content` blocks + `.stop_reason`
  + `.usage`).

Because the returned object is Anthropic-`Message`-shaped, the existing
`AnthropicTransport.normalize_response` consumes it unchanged — the hot agent
loop, transport registry, and `_VALID_API_MODES` need no edits.

## Shared tool-exposure layer (`hermes_tool_exposure.py`)

Two Hermes runtimes hand the agent loop to an external engine and must
re-expose Hermes' tools to it as MCP tools: the **codex_app_server** runtime
(`agent/transports/hermes_tools_mcp_server.py`, a separate stdio subprocess,
stateless via `handle_function_call`) and this **Claude Agent SDK** runtime
(in-process, stateful via `invoke_tool`). They legitimately differ on
execution, registration, and result shape, but were independently answering
the same three questions — *which tools, mapped how, wrapped/erred how*.

`agent/transports/hermes_tool_exposure.py` is the single source of truth for
exactly those, imported by both backends:

- `CURATED_STATELESS_TOOLS` — the stateless-safe tool list (codex's old
  `EXPOSED_TOOLS`, now an alias).
- `normalize_tool_spec` — Hermes tool spec → `(name, description, schema)`
  (OpenAI + Anthropic formats, `mcp__` strip).
- `resolve_curated_specs` — resolve curated names against `get_tool_definitions()`.
- `looks_like_tool_error` / `make_error_envelope` — one definition of the
  `{"error": ...}` envelope (fixes a prior producer/detector mismatch).
- `wrap_untrusted` — the `<untrusted_tool_result>` promptware defense (which
  the codex backend now also applies).

Each backend keeps its own dispatch, guardrails, registration and transport —
neither execution model is forced on the other, so there's no divergent
tool-logic to drift.

### Not yet shared: codex statefulness

The four **agent-level tools** (`todo` / `memory` / `delegate_task` /
`session_search`) need a live `AIAgent`. The Claude SDK backend runs handlers
in-process, so it reaches the live agent and these already work in `hybrid`.
The codex backend is a **separate process**; there is no existing
parent↔child channel that reaches the live agent (even `execute_code`'s RPC is
stateless), so codex cannot run those four until a dedicated parent↔child RPC
bridge is built. That bridge is a scoped follow-up, not part of this layer.

## Tests

`tests/agent/test_claude_agent_sdk_adapter.py` covers config resolution, auth
env mapping (OAuth vs API key vs base URL), all three modes, error handling,
and — critically — that the adapter's return value normalizes correctly through
the real `AnthropicTransport`. The tests use a fake SDK and never spawn the
`claude` CLI.

## SDK-surface coverage (claude-agent-sdk 0.2.110)

Every `ClaudeAgentOptions` field was reviewed against the Agent SDK docs
(code.claude.com/docs/en/agent-sdk). Coverage:

**Wired by the adapter** — `model`, `env` (auth), `setting_sources=[]`,
`system_prompt` (custom / preset+append), `tools`, `allowed_tools`,
`disallowed_tools`, `permission_mode`, `max_turns`, `max_budget_usd`,
`mcp_servers` + `strict_mcp_config` (hybrid), `hooks` (PreToolUse guardrail),
`resume` (gated to the same cwd — the CLI keys transcripts by directory),
`cwd` (resolved via `agent.runtime_cwd`, all modes), `include_partial_messages`
(stream bridge → live text/thinking/tool progress), `thinking` + `effort`
(derived from Hermes' reasoning settings; reasoning off → `{"type":
"disabled"}`).

**Config passthroughs** (`model.claude_agent_sdk.*`, forwarded verbatim) —
`cli_path`, `betas`, `fallback_model`, `sandbox`, `add_dirs`, `extra_args`.

**Interrupts** — one-shot `query()` has no native interrupt (that needs
`ClaudeSDKClient` + streaming input); the adapter polls Hermes' `/stop` flag
between SDK messages and raises `InterruptedError`, which closes the generator
and tears down the CLI transport.

**Deliberately not used** (Hermes owns the equivalent, or N/A):
`agents`/`skills`/`plugins` (Hermes has its own subagents/skills),
`output_format` (structured outputs — Hermes consumes plain text),
`enable_file_checkpointing` (Hermes has its own checkpoint manager),
`can_use_tool` (cannot execute-and-return a result — documented SDK
limitation; the PreToolUse hook covers deny/modify), `fork_session`,
`session_store`/`session_store_flush`, `task_budget`, `user`,
`include_hook_events`, `continue_conversation` (`resume` is used instead),
`session_id` (the SDK-generated id is captured from `ResultMessage`),
`max_thinking_tokens` (deprecated in favor of `thinking`),
`permission_prompt_tool_name`, `debug_stderr`/`stderr`, `max_buffer_size`,
`load_timeout_ms` (defaults).
