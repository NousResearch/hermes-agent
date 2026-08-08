# Implementation Plan — Run Your Own Self-Hosted Hermes on iPhone

End-to-end plan to stand up a private Hermes Agent on a server you control and
drive it from the **Hermes iOS app** in this folder. Values below are the real
defaults from the codebase (`gateway/config.py`, `gateway/platforms/api_server.py`).

The model: the **brain** (Hermes Core, 70+ tools, terminal, Python deps) runs on
your box; the **iPhone is a thin client** over HTTPS/SSE. Nothing agentic runs in
the iOS sandbox.

```
 iPhone (Hermes app)  ──HTTPS/SSE──▶  Tunnel (Tailscale/Cloudflare)  ──▶  Hermes Gateway :8642
   Keychain key                                                            API_SERVER_ENABLED=true
```

---

## Phase 0 — Prerequisites (done criteria: you can SSH to a box)
- A server that stays on: home machine, Raspberry Pi 5 (8GB+), or a small VPS.
  CPU-only is fine for hosted models (Anthropic/OpenAI/etc.); a GPU only matters
  if you run local weights.
- Python 3.11+ and `git`. A Mac with **Xcode 15+** to build the app (or TestFlight).
- An LLM provider API key (e.g. `ANTHROPIC_API_KEY`) — Hermes is the harness, not
  the model.

## Phase 1 — Install Hermes Core (done: `./hermes` starts a chat)
```bash
git clone https://github.com/<your-fork>/hermes-agent.git
cd hermes-agent
./setup-hermes.sh          # provisions the venv + dependencies
cp .env.example .env       # then edit .env (next phase)
./hermes                   # sanity-check the TUI works locally
```

## Phase 2 — Enable the OpenAI-compatible API Server (done: `/health` returns 200)
The iOS app talks to the API Server platform. Turn it on with env vars (in `.env`
or your service manager). Generate a strong key — this is the only thing standing
between the internet and your terminal-capable agent.

```bash
# in .env
API_SERVER_ENABLED=true
API_SERVER_KEY=$(openssl rand -hex 32)     # copy this; you'll paste it into the app
API_SERVER_HOST=127.0.0.1                  # keep loopback; the tunnel exposes it
API_SERVER_PORT=8642                       # default
# Optional: pin the default model the app sends
# API_SERVER_MODEL_NAME=claude-sonnet-4-6
```
Start the gateway, then verify locally:
```bash
./hermes                                   # starts gateway incl. API server
curl -s http://127.0.0.1:8642/health
curl -s http://127.0.0.1:8642/v1/models -H "Authorization: Bearer $API_SERVER_KEY"
```
> Keep `API_SERVER_HOST=127.0.0.1`. Do **not** bind `0.0.0.0` on a public box
> without a firewall — expose it through the tunnel in Phase 3 instead.

## Phase 3 — Expose it securely (done: the URL works from cellular, not just Wi-Fi)
Pick one. **Tailscale is the recommended default** — no open ports, encrypted, and
your phone reaches the box by its tailnet name.

- **Tailscale (recommended):** install on server + iPhone, same account. Use
  `http://<machine>.<tailnet>.ts.net:8642` as the server URL. Optionally
  `tailscale serve` to get HTTPS.
- **Cloudflare Tunnel:** `cloudflared tunnel --url http://127.0.0.1:8642` →
  gives a public `https://…trycloudflare.com` URL (or map a real domain). TLS for free.
- **Reverse proxy (Caddy/nginx):** terminate TLS on your domain and proxy to
  `127.0.0.1:8642`. Most work; full control.

## Phase 4 — Build & install the iOS app (done: app launches on your device)
```bash
brew install xcodegen
cd apps/ios && xcodegen generate && open HermesAgent.xcodeproj
```
Set your signing **Team** in `project.yml` or Xcode, pick your iPhone as the run
target, and Run. (For others/long-term: Archive → distribute via TestFlight.)

## Phase 5 — Connect (done: a streamed reply appears)
In the app's setup screen:
- **Server URL** → the Phase 3 URL (e.g. `https://hermes.example.com` or the
  Tailscale URL incl. `:8642`)
- **API Key** → the `API_SERVER_KEY` from Phase 2 (stored only in the Keychain)

Tap **Connect** → the app probes `/health`, loads `/v1/models`, and lands on the
session list. Start a session and send a message.

## Phase 6 — Run it as a service so it survives reboots (done: agent auto-starts)
- **Linux (systemd):** a unit that runs `./hermes` with `Restart=always` and an
  `EnvironmentFile=` pointing at your `.env`.
- **macOS:** a `launchd` plist, or just `tmux`/`screen` for a quick setup.
- Confirm with `systemctl status hermes` and a reboot test.

---

## Verification checklist (substitutes for an automated `/verify`)
SwiftUI can't be exercised headlessly on Linux, so verify by hand on a Mac/device:

| # | Check | Pass when |
|---|-------|-----------|
| 1 | Wrong key rejected | Connect with a bad key → "Authentication failed" |
| 2 | Health probe | Valid creds → reaches the session list |
| 3 | Streaming | Reply renders token-by-token (not one blob) |
| 4 | Tool progress | A web/terminal task shows chips that flip running→done |
| 5 | Tool failure | A failing tool shows a red failed chip (the `tool.failed` path) |
| 6 | Slash command | `/skills` returns gateway output (exercises the WebSocket) |
| 7 | Session sync | A CLI-started session appears after pull-to-refresh |
| 8 | Resume | Reopen a session → history loads from `/messages` |
| 9 | Model switch | Picker changes the model sent on the next turn |
| 10 | Keychain persistence | Force-quit + reopen → still connected, key not re-entered |
| 11 | Cellular | Works off Wi-Fi (confirms the tunnel, not just LAN) |

## Security notes
- The key grants an agent with **terminal + filesystem** access. Treat it like a
  root password: rotate it (`openssl rand -hex 32` → update `.env` + app), and
  scope what the agent can do server-side.
- Prefer HTTPS end-to-end in production and tighten the app's ATS exception
  (`Info.plist`) once you're not on plain-HTTP LAN.
- The "Dangerous Command" approval flow (sudo / file deletion) is on the Post-MVP
  roadmap — until then, review what you ask the agent to do remotely.

## Where this maps in the app
| Step | Code |
|------|------|
| Server URL + key, Keychain | `Networking/Connection.swift`, `KeychainStore.swift`, `Views/ConnectionSetupView.swift` |
| `/health`, `/v1/models`, `/api/sessions` | `Networking/HermesAPIClient.swift` |
| SSE streaming turn | `Networking/SSEClient.swift`, `Models/StreamEvent.swift` |
| Slash commands → gateway | `Networking/TUIGatewayClient.swift` |
