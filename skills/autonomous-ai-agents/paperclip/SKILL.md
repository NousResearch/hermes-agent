---
name: paperclip
description: Deploy and manage Paperclip — open-source AI agent orchestration platform. Org charts, budgets, governance, heartbeats, ticket system for coordinating teams of AI agents (Claude Code, Codex, Cursor, OpenClaw, HTTP bots).
tags: [paperclip, ai-agents, orchestration, self-hosted, agent-management]
---

# Paperclip — AI Agent Orchestration

Paperclip is a Node.js server + React UI that orchestrates teams of AI agents as a company. It provides org charts, budgets, governance, goal alignment, ticket tracking, and scheduled heartbeats. Think of it as the operating system for running autonomous AI businesses.

**GitHub:** https://github.com/paperclipai/paperclip (MIT, 67k+ stars)
**Docs:** https://docs.paperclip.ing
**Discord:** https://discord.gg/m4HZY7xNG3

## When to Use

- User wants to coordinate multiple AI agents toward business goals
- User has many Claude Code / Codex / Cursor terminals open and loses track
- User wants 24/7 autonomous agent execution with monitoring
- User wants cost tracking and budget enforcement per agent
- User wants a dashboard to manage agents from phone/browser

## Requirements

- Node.js 20+
- pnpm 9.15+
- No external database needed (embedded PostgreSQL)

## Quick Start

```bash
# One-command setup (local trusted mode, loopback only)
npx paperclipai onboard --yes

# For external access (LAN or tunnel), bind to all interfaces:
npx paperclipai onboard --yes --bind lan
# Or for Tailscale:
npx paperclipai onboard --yes --bind tailnet
```

This creates config at `~/.paperclip/instances/default/` with embedded PostgreSQL.

## Running the Server

```bash
# Start server (loopback only)
npx paperclipai run

# Start with external access
npx paperclipai run --bind 0.0.0.0
```

- API: `http://localhost:3100/api`
- UI: `http://localhost:3100`
- Health: `http://localhost:3100/api/health`

### Background Running (Hermes)

Use `terminal(background=true)` for long-lived server processes:

```
terminal(background=true, command="npx paperclipai run --bind 0.0.0.0")
```

Wait ~10s for startup, then verify: `curl -s http://127.0.0.1:3100/api/health`

## External Access via Cloudflare Tunnel

Paperclip restricts which hostnames it responds to. When using a Cloudflare quick tunnel:

1. Start Paperclip: `npx paperclipai run --bind 0.0.0.0`
2. Start tunnel: `cloudflared tunnel --url http://127.0.0.1:3100`
3. Extract URL from tunnel logs (format: `https://<words>.trycloudflare.com`)
4. Allow the hostname: `npx paperclipai allowed-hostname <words>.trycloudflare.com`
5. **Restart Paperclip** (required for hostname change to take effect)
6. Verify: `curl -s https://<words>.trycloudflare.com/api/health`

See the `cloudflare-tunnel-testing` skill for general tunnel diagnostics (502 errors, multi-networking, health checks), and `references/paperclip-tunnel.md` under that skill for the full Paperclip-specific recipe.

## Key Directories

| Path | Purpose |
|------|---------|
| `~/.paperclip/instances/default/config.json` | Instance configuration |
| `~/.paperclip/instances/default/db/` | Embedded PostgreSQL data |
| `~/.paperclip/instances/default/logs/` | Server logs |
| `~/.paperclip/instances/default/data/backups/` | Auto-backups (every 60min, keep 30d) |
| `~/.paperclip/instances/default/secrets/master.key` | Local encrypted secrets key |
| `~/.paperclip/instances/default/.env` | Environment vars (JWT secret, etc.) |

## Key Commands

```bash
npx paperclipai onboard --yes          # Quickstart setup
npx paperclipai run                    # Start server
npx paperclipai configure              # Edit settings
npx paperclipai doctor                 # Diagnose issues
npx paperclipai allowed-hostname HOST  # Allow external hostname
```

## Supported Agent Adapters

Paperclip works with any agent that can receive a heartbeat:

- **Claude Code** — Anthropic's coding agent
- **Codex** — OpenAI's coding agent
- **Cursor** — AI code editor
- **OpenClaw** — Open-source agent employee
- **Bash** — Shell-based agents
- **HTTP/webhook** — Any HTTP endpoint

> "If it can receive a heartbeat, it's hired."

## Core Concepts

- **Company** — Top-level organization. One deployment can run multiple companies with data isolation.
- **Org Chart** — Agents have roles, titles, reporting lines, permissions, and budgets.
- **Goals** — Hierarchical objectives. Every task traces back to the company mission.
- **Tickets/Issues** — Structured tasks with owner, status, thread, full trace and immutable audit log.
- **Heartbeats** — Agents wake on schedule, check work, and act. Delegation flows up/down org chart.
- **Budgets** — Monthly budgets per agent. Hard stops when limit reached. No runaway costs.
- **Governance** — Board approval workflows. You approve hires, override strategy, pause/terminate agents.
- **Routines** — Recurring tasks with cron/webhook/API triggers.

## Paperclip vs Hermes

| Aspect | Hermes | Paperclip |
|--------|--------|-----------|
| Interface | Chat (Telegram, etc.) | Web dashboard |
| Scope | Single assistant | Team of agents |
| Management | Conversational | Org chart + tickets |
| Cost control | Manual | Automatic budgets |
| Best for | Direct tasks, personal assistant | Orchestrating multiple agents on projects |

They complement each other: Hermes for direct interaction, Paperclip for multi-agent coordination.

## Cost Considerations

Paperclip itself is free/open-source. **The real cost is the LLM agents it orchestrates.** Claude Code, Codex, and similar agents bill per-token. Running a team of agents 24/7 on commercial LLMs can rack up significant costs quickly, even with Paperclip's budget enforcement.

- **Bash agents** (shell-based, no LLM cost) are free to run
- **Local models** (Ollama, llama.cpp) eliminate per-token costs but sacrifice quality
- **Commercial LLM agents** (Claude Code, Codex) are where the spend happens

Always discuss cost expectations before setting up Paperclip for production use. The platform's budget feature helps cap spend, but it doesn't eliminate the underlying LLM costs.

## Pitfalls

- **Hostname restriction**: Paperclip rejects requests from unknown hostnames. Must run `allowed-hostname` and restart when using tunnels.
- **Embedded PostgreSQL port**: Uses port 54329 by default. Ensure it's not occupied.
- **Quick tunnel URLs are temporary**: Cloudflare quick tunnels generate random URLs that change on restart. For production, use a named tunnel with a custom domain.
- **`--bind 0.0.0.0` needed for tunnel access**: Default loopback-only binding won't work with external tunnels pointing to localhost.

## Coming Soon (Roadmap)

- Clipmart — Download pre-built company templates
- Cloud/Sandbox agents (Cursor/e2b)
- Memory/Knowledge
- Deep Planning, Work Queues, Self-Organization
- Desktop App
