# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Hermes Agent is a self-improving AI agent by Nous Research. This is a **read-only reference clone** — not installed, not forked for production use.

**Status: ARCHIVED.** The security-hardening fork plan (`docs/plans/security-hardening-refactor.md`) was evaluated and dropped. The existing stack (Claude Code + NemoClaw/OpenClaw on openclaw-gs + claw-relay + osm MCP + shux) already covers every hermes capability. No redundant agent framework needed.

**Why this repo is kept:** Reference source for cherry-picking patterns into OpenClaw:
- `gateway/platforms/telegram.py` — polished Telegram adapter (chunking, files, voice, inline keyboards)
- `agent/context_compressor.py` — auto-summarization when context fills up
- `skills/` — Markdown-based skill format with YAML frontmatter and conditional activation
- `tools/registry.py` — self-registering tool registry pattern

## Commands

```bash
# Activate venv (required before any Python command)
source venv/bin/activate

# Run tests (parallel, skips integration tests by default)
pytest tests/ -q

# Run a single test file
pytest tests/agent/test_prompt_builder.py -v

# Run a specific test
pytest tests/agent/test_prompt_builder.py::TestClassName::test_method -v

# Run with integration tests included
pytest tests/ -m ""

# Interactive CLI
hermes

# Single query
hermes chat -q "Hello"

# Diagnostics
hermes doctor

# Install for development (all extras)
uv pip install -e ".[all,dev]"
uv pip install -e "./mini-swe-agent"
```

## Architecture

### Core Loop (run_agent.py → AIAgent)

```
User message → AIAgent._run_agent_loop()
  ├── Build system prompt (agent/prompt_builder.py)
  ├── Call LLM (OpenAI-compatible API)
  ├── If tool_calls → dispatch via tools/registry.py → loop back
  ├── If text → persist to SQLite (hermes_state.py) → return
  └── Auto-compress context if approaching token limit
```

### Tool System (Self-Registering Registry Pattern)

```
tools/registry.py  ← imported by all tool files (no deps)
       ↑
tools/*.py         ← each calls registry.register() at import time
       ↑
model_tools.py     ← triggers tool discovery by importing all tool modules (_modules list)
       ↑
run_agent.py, cli.py, batch_runner.py
```

**To add a tool:** Create `tools/my_tool.py` with schema + handler + `registry.register()` call, then add `"tools.my_tool"` to the `_modules` list in `model_tools.py`. If it's a new toolset, also update `toolsets.py`.

**Prefer skills over tools.** Tools need custom Python; skills are Markdown instructions + shell commands + existing tools. See CONTRIBUTING.md "Should it be a Skill or a Tool?"

### Toolsets (toolsets.py)

Named tool groups (`web`, `terminal`, `file`, `browser`, etc.) enabled/disabled per platform. Platform presets define which toolsets are active for CLI vs Telegram vs Discord.

### Session Persistence (hermes_state.py)

SQLite with WAL mode + FTS5 full-text search. Sessions chain via `parent_session_id` when context is compressed. System prompts and prefill are injected at API call time, never persisted.

### Skills System

Markdown files (`SKILL.md`) with YAML frontmatter in `skills/` (bundled) or `optional-skills/` (opt-in). Support platform filtering (`platforms: [macos]`), conditional activation (`fallback_for_toolsets`, `requires_toolsets`), and secure env var collection.

### Gateway (gateway/)

Async messaging platform adapters (Telegram, Discord, Slack, etc.) with per-user session isolation. Entry: `gateway/run.py`.

### Entry Points (pyproject.toml [project.scripts])

- `hermes` → `hermes_cli/main.py:main` (interactive CLI)
- `hermes-agent` → `run_agent.py:main` (programmatic)
- `hermes-acp` → `acp_adapter/entry.py:main` (IDE integration)

## Conventions

### Commit Messages

Conventional Commits: `<type>(<scope>): <description>`

Types: `fix`, `feat`, `docs`, `test`, `refactor`, `chore`
Scopes: `cli`, `gateway`, `tools`, `skills`, `agent`, `install`, `security`

### Code Style

- PEP 8 (no strict line length enforcement)
- Comments only for non-obvious intent, trade-offs, or API quirks
- Catch specific exceptions; log with `logger.warning()`/`logger.error()` with `exc_info=True`
- Cross-platform: always catch `ImportError` + `NotImplementedError` for Unix-only modules (`termios`, `fcntl`). Use `pathlib.Path` for paths. Handle Windows `cp1252` encoding.

### Security Rules

- `shlex.quote()` for all shell interpolation of user input
- `os.path.realpath()` to resolve symlinks before path-based ACL checks
- API keys stripped from `execute_code` sandbox environment
- Protected write-deny list for sensitive paths (`~/.ssh/authorized_keys`, `/etc/shadow`)
- Dangerous commands require user approval (`tools/approval.py`)

### Testing

- Framework: pytest + pytest-asyncio + pytest-xdist
- Config in `pyproject.toml`: `addopts = "-m 'not integration' -n auto"`
- Tests mirror source structure under `tests/`
- CI: GitHub Actions (`.github/workflows/tests.yml`) — runs on push to main and all PRs, 10-min timeout

## Key Files

| File | Lines | Purpose |
|------|-------|---------|
| `run_agent.py` | ~7.4K | Core agent loop — modify with extreme care |
| `cli.py` | ~7.3K | Interactive TUI (prompt_toolkit) |
| `hermes_state.py` | ~36K | SQLite session DB + FTS5 |
| `model_tools.py` | ~470 | Tool orchestration, discovery trigger |
| `toolsets.py` | ~575 | Tool grouping + platform presets |
| `tools/registry.py` | ~150 | Central registry (schema, handler, dispatch) |
| `tools/mcp_tool.py` | ~1050 | MCP client integration |
| `agent/prompt_builder.py` | — | System prompt assembly |
| `hermes_cli/config.py` | — | Config schema, DEFAULT_CONFIG, env vars |
| `hermes_cli/commands.py` | — | Slash command registry (CommandDef) |

## User Config

All user state lives in `~/.hermes/`:
- `config.yaml` — settings (model, terminal backend, toolsets, display)
- `.env` — API keys and secrets
- `state.db` — SQLite session database
- `skills/` — active skills (bundled + hub + agent-created)
- `memories/` — persistent memory (MEMORY.md, USER.md)

## Dependencies

Python 3.11+, managed with `uv`. ~40 core deps, unpinned.
Optional extras: `[messaging]`, `[mcp]`, `[cron]`, `[pty]`, `[voice]`, `[rl]`, `[dev]`, `[all]`.
