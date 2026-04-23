# Hermes ↔ Bindu A2A Adapter

Expose Hermes' `AIAgent` as an [A2A](https://a2aproject.github.io/A2A/) microservice so **other agents** can call it — with a cryptographic DID identity, OAuth2 auth, and optional x402 micropayments.

`hermes gateway` → talk to Hermes from **messaging platforms** (humans).
`hermes-bindu`   → talk to Hermes from **other agents** (A2A JSON-RPC).

Additive. Opt-in extra. No changes to the base install. ~200 lines of glue.

---

## Verified working

Tested end-to-end on this branch before the PR. Both paths round-trip:

| Mode | Result |
|---|---|
| **No auth** (dev) — direct `message/send` → `tasks/get` | ✅ `submitted` → `completed` in ~3s, correct artifact returned |
| **Auth on** (`AUTH__ENABLED=true AUTH__PROVIDER=hydra`) — Hydra JWT + DID Ed25519 signature on every call | ✅ Token issued, DID verified, artifact returned |

Abridged auth-on boot log:

```
INFO  Registering agent in Hydra OAuth2 server with DID-based authentication...
INFO  Extracted public key (base58) from DID extension
INFO  ✅ Agent registered in Hydra
INFO  Authentication middleware enabled
INFO  Hydra OAuth2 authentication enabled
INFO  Uvicorn running on http://localhost:3780
```

---

## Install

```bash
pip install -e '.[bindu]'
```

Nothing in the base Hermes install changes. The `[bindu]` extra only pulls [`bindu`](https://github.com/GetBindu/Bindu).

## Run

```bash
hermes-bindu                           # → http://localhost:3773
```

Uses `~/.hermes/.env` for env (same as the main CLI). Any `OPENROUTER_API_KEY` / `ANTHROPIC_API_KEY` / etc. already configured for Hermes works as-is.

## Call it

Standard A2A JSON-RPC. Submit a message, poll until the task completes:

```bash
curl -sS -X POST http://localhost:3773/ -H 'Content-Type: application/json' -d '{
  "jsonrpc":"2.0","method":"message/send","id":"<uuid>",
  "params":{"message":{
    "role":"user","kind":"message",
    "parts":[{"kind":"text","text":"summarize HN frontpage"}],
    "messageId":"<uuid>","contextId":"<uuid>","taskId":"<uuid>"
  }}
}'
```

With auth on, add `Authorization: Bearer <hydra-jwt>` plus the three `X-DID*` signature headers. The startup banner prints the exact curl to fetch a Hydra token.

---

## Safety tiers

Hermes has ~20 toolsets. Gate what's exposed via `HERMES_BINDU_TIER`:

| Tier | Toolsets | Use when |
|---|---|---|
| `read` *(default)* | `web` | Public / tunneled, untrusted callers |
| `sandbox` | `web` + `file` + `moa` | Trusted callers, ephemeral container |
| `full` | everything — terminal, browser, code-exec, MCP | **Localhost only** |

⚠️ **Never combine `full` with `HERMES_BINDU_EXPOSE=true`.** That's remote code execution over HTTP.

## Configuration

All env vars, nothing requires editing code:

| Variable | Default | Purpose |
|---|---|---|
| `HERMES_BINDU_MODEL` | `anthropic/claude-3.5-haiku` | LLM backing the A2A-exposed agent |
| `HERMES_BINDU_TIER` | `read` | Toolset tier (see above) |
| `HERMES_BINDU_MAX_ITERATIONS` | `30` | Max tool loops per A2A request |
| `HERMES_BINDU_URL` | `http://localhost:3773` | Public URL in the agent card |
| `HERMES_BINDU_NAME` | `hermes` | Agent display name |
| `HERMES_BINDU_AUTHOR` | `you@example.com` | Author field on the agent card |
| `HERMES_BINDU_EXPOSE` | `false` | If `true`, open a public FRP tunnel |
| `AUTH__ENABLED` | *(unset)* | Set `true` to enforce Hydra OAuth2 + DID sig |
| `AUTH__PROVIDER` | *(unset)* | Required with above — set `hydra` |

---

## How it fits

```
other agent ─► A2A JSON-RPC ─► Bindu server :3773
                                 │  OAuth2 (Hydra), DID verify, x402 (opt)
                                 ▼
                               bindu_adapter.handler(messages)
                                 ▼
                               AIAgent.chat(latest_user_text)
                                 ▼
                               Hermes tool loop → DID-signed artifact reply
```

One long-lived `AIAgent` per process so provider prompt caches stay valid across A2A calls. Bindu owns conversation history; Hermes owns the live model state.

## Files

| File | Purpose |
|---|---|
| `adapter.py` | `AIAgent` wrapped as a Bindu `handler(messages) -> str` |
| `entry.py` | CLI entry: loads env, builds config, calls `bindufy()` |
| `__main__.py` | Enables `python -m bindu_adapter` |

## Links

- Bindu: <https://github.com/GetBindu/Bindu>
- A2A protocol: <https://a2aproject.github.io/A2A/>
- x402: <https://www.x402.org/>
