# Claude Code -> Hermes Codex bridge

This bridge exposes a local Anthropic Messages-compatible HTTP API for Claude Code and routes requests through the Hermes `openai-codex` credential pool.

It is intended for local use only. It binds to `127.0.0.1`, reads its runtime config from `~/.hermes/cc-switch-codex-bridge/config.json`, and never requires raw Codex credentials in Claude Code.

## Features

- Anthropic `/v1/messages` compatibility for Claude Code.
- `/v1/models` exposes model aliases so Claude Code dynamic workflows and subagents can choose a fit-for-purpose profile.
- Model aliases map to backend model + Codex reasoning effort:
  - `gpt-5.5-xhigh`, `gpt-5.5-high`, `gpt-5.5-medium`, `gpt-5.5-low`, `gpt-5.5-fast`, `gpt-5.5-mini`
  - `gpt-5.4-xhigh`, `gpt-5.4-high`, `gpt-5.4-medium`, `gpt-5.4-low`
  - `claude-haiku-4-8`, `claude-haiku-4-8-latest`, `claude-sonnet-4-8`, `claude-sonnet-4-8-latest`, `claude-opus-4-8`, `claude-opus-4-8-latest` resolve to `gpt-5.5` while preserving the requested model id in Anthropic responses.
  - Older compatibility aliases remain available: `claude-sonnet-4-5`, `claude-sonnet-4-5-latest`, `claude-opus-4-1`, `claude-opus-4-1-latest`.
- Claude `thinking.budget_tokens` is translated into Codex reasoning effort when no explicit alias effort is present.
- Claude-looking aliases expose the real backend context window, not the cosmetic Claude model window. When an alias resolves to GPT-5.5, `/health` reports a 272k estimated-token window and `/v1/messages` fails closed with `context_length_exceeded` before calling the backend if the estimated input exceeds that window.
- Codex `response.reasoning_summary_text.delta` is streamed to Claude Code as Anthropic `thinking_delta` blocks, with a local placeholder `signature_delta` for Claude Code compatibility. The placeholder is marked as bridge-local and is not an Anthropic-origin signature.
- Tool-use history is flattened into non-protocol XML-like transcript markers to avoid Claude Code wrapper leakage.
- Optional protocol logging via `CODEX_ANTHROPIC_BRIDGE_PROTOCOL_LOG=1`; set `CODEX_ANTHROPIC_BRIDGE_PROTOCOL_LOG_FILE=/path/to/log.jsonl` to write compact no-secret JSONL logs.

## Local setup

```bash
mkdir -p ~/.hermes/cc-switch-codex-bridge
cp scripts/cc-switch-codex-bridge/config.example.json ~/.hermes/cc-switch-codex-bridge/config.json
python3 - <<'PY'
import json, secrets, pathlib
p = pathlib.Path.home() / '.hermes/cc-switch-codex-bridge/config.json'
c = json.loads(p.read_text())
c['auth_token'] = 'ccsb_' + secrets.token_urlsafe(32)
p.write_text(json.dumps(c, indent=2) + '\n')
print(p)
PY
```

Run foreground for smoke testing:

```bash
CODEX_ANTHROPIC_BRIDGE_CONFIG=~/.hermes/cc-switch-codex-bridge/config.json \
python scripts/cc-switch-codex-bridge/server.py
```

Point Claude Code at the bridge:

```bash
export ANTHROPIC_BASE_URL=http://127.0.0.1:15722
export ANTHROPIC_AUTH_TOKEN=BRIDGE_VALUE_FROM_CONFIG
export ANTHROPIC_API_KEY=BRIDGE_VALUE_FROM_CONFIG
claude -p --model gpt-5.5-low 'Reply exactly: OK'
```


## Smoke checks

```bash
curl -sS http://127.0.0.1:15722/health | python -m json.tool
curl -sS http://127.0.0.1:15722/v1/models | python -m json.tool
```

Streaming reasoning smoke should show `thinking_delta` frames. Use a prompt likely to trigger reasoning summary, for example arithmetic or debugging, because trivial prompts may not emit reasoning deltas.

Run the protocol smoke driver when a local bridge is running:

```bash
ANTHROPIC_BASE_URL=http://127.0.0.1:15722 \
ANTHROPIC_AUTH_TOKEN=BRIDGE_VALUE_FROM_CONFIG \
python scripts/cc-switch-codex-bridge/smoke_protocol.py
```

## Security notes

- Do not commit `~/.hermes/cc-switch-codex-bridge/config.json`; it contains `auth_token`.
- The server refuses to bind to non-local hosts.
- Do not log request headers or raw credentials.
