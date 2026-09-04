# MCP HTTP platform — chat with this Hermes from Claude Code

Serves Streamable HTTP MCP at `/mcp` from inside the gateway. A remote MCP client
(Claude Code, Codex, any MCP host) hands a task to the *running* Hermes agent — with
its memory, tools and skills — and long-polls for the reply.

Not the same as `hermes mcp serve`: that exposes Hermes' **tools** to a host; this
exposes the **agent** as a chat peer.

Full docs: `website/docs/user-guide/messaging/mcp-http.md`.

## Files

| File | Owns |
|---|---|
| `adapter.py` | `McpHttpAdapter` — MCP server, tools, turn flow, gateway callbacks |
| `security.py` | `Settings` (env > `extra` > default), tokens/auth, bind safety, DNS-rebinding hosts, filtering, redaction, audit, rate limit |
| `history.py` | Per-conversation JSONL transcripts under `cache/mcp_http/history/` |
| `__init__.py` | `register(ctx)` → `ctx.register_platform("mcp_http", …)`, setup flow |

## Contract with the gateway

The gateway delivers tool-progress bubbles through `send()`/`edit_message()` **without**
`metadata["notify"]` and the final reply **with** `notify=True`. Progress lines feed the
`wait_reply` "working … last activity" string; the notify-send resolves the waiter.
`on_processing_complete(SUCCESS)` with no notify-send resolves with a non-empty placeholder.

## Tests

`tests/gateway/test_mcp_http_platform.py` — bind safety, authentication, cross-peer
refusal, history persistence, and an end-to-end run over a real Streamable HTTP client
against an ephemeral port with a fake message handler.
