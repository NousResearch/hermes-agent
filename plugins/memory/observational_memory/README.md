# Observational Memory Provider

Shared local markdown memory across Hermes, Claude Code, and Codex.

## Requirements

- `pip install "observational-memory>=0.5.0"` (includes Hermes transcript parser plus the latest local search/status improvements)
- Optional but recommended: `om install` to configure Claude/Codex hooks and OM's background jobs
- For Hermes writeback: either an existing OM config, or set a direct Anthropic/OpenAI key during `hermes memory setup`

## Setup

```bash
hermes memory setup    # select "observational_memory"
```

If you want Claude Code and Codex to share the same memory store too:

```bash
om install
```

Or manually:

```bash
hermes config set memory.provider observational_memory
```

## Config

Config file: `$HERMES_HOME/observational_memory.json`

| Key | Default | Description |
|-----|---------|-------------|
| `llm_provider` | `inherit-existing` | Hermes-side writeback provider: `inherit-existing`, `anthropic`, or `openai` |
| `llm_model` | `""` | Optional observer/reflector model override |
| `memory_dir` | `~/.local/share/observational-memory` | Shared OM markdown memory directory |
| `env_file` | `~/.config/observational-memory/env` | OM env file path |
| `search_backend` | `bm25` | Search backend: `bm25`, `qmd`, `qmd-hybrid`, `none` |
| `writeback_mode` | `incremental` | `incremental`, `session_end`, or `off` |

Optional secret written to Hermes `.env`:

| Env var | Purpose |
|---------|---------|
| `OM_HERMES_API_KEY` | API key for the selected direct writeback provider |

## Tools

| Tool | Description |
|------|-------------|
| `om_context` | Load compact shared startup context plus task-relevant memory |
| `om_search` | Search Observational Memory for preferences, project history, and cross-agent context |
| `om_remember` | Store an explicit local observation immediately |
