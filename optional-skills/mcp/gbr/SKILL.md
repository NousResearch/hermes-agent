---
name: gbr
description: Pair a phone spectator to this Hermes host session.
version: 1.0.0
author: David Rad (LinespottingPrivate), Hermes Agent
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [MCP, Mobile, Pairing]
    related_skills: [hermes-agent]
    homepage: https://grokbuildremote.com/
---

# GBR Skill

Pair a phone running Build Remote Agent as a spectator (and veto) on this Hermes host session. Protocol is `gbr/1` only: phone app, `gbr-agent pair` (QR and printed 8-char code), then `gbr-agent run`. This skill does not make the phone the orchestrator, does not treat Hermes chat channels as the pair path, and does not require Hermes to run on a GBR host — MCP add stays on the Hermes box.

Independent product by Linespotting AB. Not affiliated with xAI or SpaceX.

## When to Use

- The user wants a phone to spectate or veto this Hermes host session.
- The user asks to pair Build Remote Agent / `gbr-agent` with Hermes.

Do not use for:

- Making the phone the orchestrator
- Pairing through Hermes messaging (Telegram, WhatsApp, Discord, …)
- Any fourth pair protocol beyond phone app + `gbr-agent pair` + `gbr-agent run`
- Registering `http://127.0.0.1:8788` as a Hermes MCP `--url` (that port is Bot API REST, not MCP)

## Prerequisites

- `gbr-agent` ≥0.6.0 on PATH. Prefer a pinned GitHub Release binary from https://github.com/LinespottingOrg/GrokBuildRemote-Agents/releases/tag/v0.6.0 (assets `gbr-agent-<os>-<arch>`). The website `install.sh` is mutable; do not treat it as a pin.
- `node` on PATH. Hermes MCP attach is **stdio `gbr-mcp` only**.
- Clone `mcp/gbr-mcp` from that same tagged Agents repo (`npm install` in `mcp/gbr-mcp`). There is no npm package.
- Phone app: [Build Remote Agent](https://grokbuildremote.com/) → Connect.
- `gbr-agent run` serves Bot API REST at `http://127.0.0.1:8788` on this host. That URL is not an MCP endpoint.
- MCP setup is on the Hermes box. `gbr-mcp` then talks HTTP to local `:8788`.

Register MCP through `terminal` after `gbr-mcp` is present (`hermes mcp add` takes `--command` / `--args`, not a `--` remainder, and not `--url` against `:8788`):

```
terminal(command="hermes mcp add gbr --command node --args mcp/gbr-mcp/bin/gbr-mcp.js")
```

Equivalent `~/.hermes/config.yaml` (stdio):

```yaml
mcp_servers:
  gbr:
    command: node
    args: ["mcp/gbr-mcp/bin/gbr-mcp.js"]
```

Use an absolute path to `bin/gbr-mcp.js` if the working directory is not the Agents clone. Then `hermes mcp test gbr` (new chat to load tools).

## How to Run

Invoke host commands through the `terminal` tool. Do not substitute a fourth pair path.

```
terminal(command="gbr-agent version")
terminal(command="gbr-agent pair", pty=true, timeout=180)
terminal(command="gbr-agent run", background=true)
terminal(command="hermes mcp add gbr --command node --args mcp/gbr-mcp/bin/gbr-mcp.js")
```

`gbr-agent pair` prints an 8-char code and opens a browser QR. The phone scans the QR or types that code. Then keep `gbr-agent run` running so Bot API REST is on loopback `:8788`. Hermes talks to that API **through stdio `gbr-mcp`**, not as MCP HTTP.

## Quick Reference

| Command | Purpose |
|---------|---------|
| `gbr-agent version` | Confirm v0.6.0+ |
| `gbr-agent pair` | QR **and** printed 8-char code |
| `gbr-agent run` | Serve Bot API REST `http://127.0.0.1:8788` |
| `gbr-agent doctor` | Prove install and pair health |
| `gbr-agent status` | List local session |
| `hermes mcp add gbr --command node --args mcp/gbr-mcp/bin/gbr-mcp.js` | Stdio `gbr-mcp` on the Hermes box |
| `hermes mcp test gbr` | Handshake; expect 13 tools |

## Procedure

1. Confirm `gbr-agent version` reports v0.6.0 or newer via `terminal`. Done when stdout includes `v0.6` or higher.
2. On the phone, open Build Remote Agent → Connect.
3. On the host, invoke `gbr-agent pair` through `terminal` (`pty=true`). Done when a QR page is open **and** an 8-char code is printed.
4. Phone scans the QR **or** types the 8-char code. Done when the phone shows this host as paired.
5. Invoke `gbr-agent run` through `terminal` (`background=true`). Done when the process stays up and loopback `http://127.0.0.1:8788` answers Bot API REST (`/v1/status`).
6. On the Hermes box, register MCP through `terminal`: `hermes mcp add gbr --command node --args mcp/gbr-mcp/bin/gbr-mcp.js`. Done when `hermes mcp list` shows `gbr` and `hermes mcp test gbr` reports tools.
7. Phone spectates and may veto; it does not drive the Hermes tool loop.

## Pitfalls

- Unpair on the phone before pairing a new host. Force-close is not enough.
- Website `install.sh` / `install.ps1` can change without a tag; pin GitHub Release v0.6.0+.
- Do not `hermes mcp add gbr --url http://127.0.0.1:8788`. That port is Bot API REST. MCP handshake fails. Use stdio `gbr-mcp`.
- `http://127.0.0.1:8788` is loopback. If `gbr-agent run` is on another machine, stdio `gbr-mcp` must still run next to that agent (or use the agent's documented relay), not a remote Hermes `--url` to `:8788`.
- Hermes messaging channels are not `gbr/1`.
- Do not invent a fourth pair protocol.
- Phone is spectator + veto, not orchestrator.

## Verification

One command, through `terminal`:

```
terminal(command="gbr-agent doctor")
```

Done when `gbr-agent doctor` exits 0.
