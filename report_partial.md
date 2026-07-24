# Hermes Agent: Partial Architecture Report

## Project Overview

**Hermes** is a self-improving AI agent built by Nous Research (v0.18.2). It's a multi-platform agent that creates and refines skills from experience, maintains memory across sessions, and runs on diverse backends (local terminal, cloud VMs, serverless). Key differentiators: built-in learning loop (skill curation + FTS5 session search), support for any LLM provider (OpenRouter, OpenAI, Nous Portal, self-hosted), and messaging gateway spanning Telegram, Discord, Slack, WhatsApp, Signal, and ~20 other platforms.

## Tech Stack

- **Python 3.11–3.13** (exact-pinned deps via `uv` for supply-chain hardening)
- **Core framework:** OpenAI SDK (chat completions + function calling)
- **CLI:** `prompt_toolkit` (multiline editing, autocomplete), `rich` (formatting), curses (menus)
- **TUI:** React + Ink (Node.js), JSON-RPC over stdio to Python gateway
- **Web dashboard:** React + Tailwind, WebSocket PTY bridge (`xterm.js`)
- **Desktop (Electron):** React + nanostore, spawns headless `hermes serve` backend
- **Terminal backends:** local PTY, Docker, SSH, Singularity, Modal, Daytona
- **Data:** SQLite (sessions, memory, cron jobs), FTS5 (full-text search)

## Key Entry Points

| Entry Point | Purpose |
|---|---|
| `run_agent.py` (line 399) | `AIAgent` class—core conversation loop (~12k LOC) |
| `cli.py` (line 3703) | `HermesCLI` class—interactive CLI orchestrator (~11k LOC) |
| `hermes_cli/main.py` | CLI startup, profile switching, command dispatch |
| `gateway/run.py` | Messaging gateway runner (Telegram, Discord, Slack adapters) |
| `model_tools.py` | Tool orchestration, `discover_builtin_tools()`, `handle_function_call()` |
| `tui_gateway/server.py` | JSON-RPC backend for TUI (JSON-RPC over stdin/stdout) |
| `apps/desktop/electron/main.ts` | Electron app entry; spawns `hermes serve` backend |

## Core Modules

- **`agent/`** (120+ files): Adapters for LLM providers (Anthropic, Azure, Bedrock, Gemini), memory manager, auxiliary client (side-LLM for embedding/vision), caching, context compression, curator (skill lifecycle), memory providers (Honcho, Mem0, etc.)
- **`tools/`** (~100 files): Tool registry + implementations—terminal, file I/O, web, vision, delegation, delegation live-log, MCP client, skills hub, approval workflows, security tools (path validation, URL safety)
- **`hermes_cli/`** (50+ files): CLI commands, setup wizard, config loader, plugins manager, skin engine (theming), curses UI, tools config, cron CLI, kanban CLI
- **`gateway/`** (40+ files): Platform adapters (Telegram, Discord, Slack, Signal, WhatsApp, Matrix, IRC, Email, etc.), base adapter, message queue, lifecycle hooks
- **`plugins/`**: Memory providers, model-provider plugins, kanban dispatcher, observability, image gen, disk cleanup, etc.
- **`skills/`**: Built-in skills (organized by category); `optional-skills/` for heavier/niche ones
- **`cron/`**: Job store, tick loop, cron CLI
- **`hermes_state.py`**: Session database (SQLite + FTS5)
- **`toolsets.py`**: Tool groupings per platform (CLI, messaging, web, etc.)

## Directory Layout

```
hermes-agent/
├── run_agent.py, cli.py            # Core agent & CLI entry points
├── model_tools.py, toolsets.py     # Tool orchestration & groupings
├── hermes_cli/                     # CLI subcommands, setup, config
├── agent/                          # Provider adapters, memory, compression
├── tools/                          # Tool registry & implementations
│   └── environments/               # Terminal backends (local, docker, ssh, modal, daytona, singularity)
├── gateway/                        # Messaging platforms (telegram, discord, slack, signal, etc.)
├── plugins/                        # Memory providers, image-gen, model-providers, kanban, observability
├── skills/, optional-skills/       # Agent-creatable & bundled skills
├── cron/                           # Scheduler
├── ui-tui/                         # React + Ink terminal UI
├── tui_gateway/                    # JSON-RPC backend for TUI
├── apps/desktop/                   # Electron desktop app
├── web/                            # Dashboard (React + WebSocket PTY)
├── tests/                          # ~17k pytest tests
└── pyproject.toml, package.json    # Dependency manifests
```

## Notable Design Constraints

- **Prompt caching is sacred:** Avoid mid-conversation system-prompt mutations; cache invalidation multiplies costs
- **Core is a narrow waist:** New model tools are expensive (sent on every API call); prefer plugins, CLI commands + skills, or MCP servers
- **Profiles:** Multi-instance support via `HERMES_HOME` env var—all paths use `get_hermes_home()` (never hardcoded `~/.hermes`)
- **Exact-pinned deps:** Every Python dependency is pinned to `==X.Y.Z` (no ranges) to prevent supply-chain attacks

## Known TODOs / Blockers

1. **Python <3.14:** Currently capped at `<3.14` because some Rust-backed transitives lack CP314 wheels. Will lift once ecosystem matures.
2. **Windows native PTY:** No native Windows PTY backend yet (workaround: WSL2 or Daytona). SSH, Docker, Singularity, and Modal work on Windows via MinGit.
3. **Profile tests:** Some profile-aware tests need both `Path.home()` and `HERMES_HOME` mocking to work correctly.
4. **Third-party plugin friction:** Observability, vendor SaaS, and product integrations should be standalone repos in `~/.hermes/plugins/`, not in-tree (policy as of May–June 2026).

---

**Report generated:** 2026-07-20 | **Scope:** Partial architecture (entry points, modules, key design constraints)
