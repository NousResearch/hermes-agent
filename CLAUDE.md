# CLAUDE.md — hermes-agent

## What This Repo Is

Hermes is a multi-component LLM orchestration framework. Python handles the core agent engine and CLI; JavaScript/TypeScript provides TUI and web UI. An extensible skill/plugin system lets the agent discover and invoke capabilities.

## Stack

| Component | Tech |
|-----------|------|
| Agent engine | Python 3.11+ |
| Package manager (Python) | **`uv`** (not pip, not poetry) |
| CLI entry | `cli.py` → `run_agent.py` |
| State management | `hermes_state.py` |
| MCP server | `mcp_serve.py` |
| TUI | `ui-tui/` (npm) |
| Web UI | `web/` (npm, Vercel deployed) |
| Desktop app | `apps/desktop/` |
| Infrastructure | Docker + `docker-compose.yml` |
| Nix | `flake.nix`, `nix/tui.nix`, `nix/web.nix` |

## Key Directories

| Path | Purpose |
|------|---------|
| `skills/` | Skill definitions |
| `plugins/` | Plugin extensions |
| `providers/` | LLM provider integrations |
| `tools/` | Tool definitions |
| `agent/` | Core agent logic |
| `apps/desktop/` | Electron/desktop app |
| `apps/bootstrap-installer/` | Installer bootstrap |
| `ui-tui/` | Terminal UI |
| `web/` | Vercel web UI |

## Development Commands

```bash
# Python agent
uv sync
uv run python cli.py

# TUI
cd ui-tui && npm install && npm run dev

# Web UI
cd web && npm install && npm run dev

# Full stack (Docker)
docker-compose up
```

## Nix Build Notes

This repo uses a Nix flake with an npm deps derivation. Known issue: 18 workspace packages (`@hermes/ink`, `ui-tui`, `web`, `apps/bootstrap-installer`, etc.) lack 'resolved' URLs in `package-lock.json`, which causes the `nix-lockfile-fix.yml` CI workflow to fail at the `npm-deps.drv` build step. This is a pre-existing limitation that requires modifying the nix derivation to handle npm workspace packages.

Do NOT manually edit hash values in `nix/tui.nix` or `nix/web.nix`.

## Rules

1. Never push to `main` directly — branch + PR only
2. Never commit secrets, API keys, or credentials
3. Python packages managed via `uv` only — never `pip install` manually
4. Check `skills/` before building new capabilities
5. Nix lockfile changes must go through `nix flake update` (not manual hash edits)
