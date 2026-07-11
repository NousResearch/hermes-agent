# Operations — Agent Instructions

## Restart order (llama excluded)

1. Stop desktop Electron processes if rebuilding UI.
2. `scripts\windows\restart-hermes-stack.ps1` (no `-StartLlama` unless recovery).
3. Fix venv if desktop backend fails: `pip install -e ".[web]"` in `.venv`.
4. `hermes desktop --build-only --force-build` then launch packaged `Hermes.exe`.

## Port reference

| Port | Service |
|------|---------|
| 8787 | Messaging gateway |
| 9119 | `hermes serve` (headless backend) |
| 9120 | `hermes dashboard` |
| 8080 / 8081 | llama.cpp / proxy (optional; restart only with `-StartLlama`) |
| 8646 | LINE ngrok/webhook helper |
| 3001 | FreeLLMAPI local proxy |

## Tailscale

Run `tailscale up` if tailnet IP missing. Refresh Serve after stack restart when llama routes change.

## Do not commit

- `~/.hermes/` runtime state
- `apps/desktop/release/`, `dist/`, `node_modules/`
- Implementation logs in `_docs/`
- One-off probes under `tmp/probes/`; keep them out of the source tree

## Logs

- `~/.hermes/logs/agent.log`, `gateway.log`, `desktop.log`
- `hermes logs --follow`

## When gateway locks files

`uv sync` may fail on `PyNaCl` DLL locks — stop gateway first, or use `pip install -e ".[web]"` for quick dep repair.
