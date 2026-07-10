# Local Workspace (Repo Root Scratch)

Files listed here often appear **directly under the repository root** on the
operator workstation. They are **gitignored** — kept on disk for local workflows,
not part of the canonical codebase.

**Do not delete** these files to "clean up" unless the operator asks. **Do not
commit** them. This document exists so agents understand what they are.

## Categories

### Media and social scratch

| Pattern | Examples | Purpose |
|---------|----------|---------|
| `*.mp4`, `*.wav`, `*.ogg`, `*.png` | `ohayo_tweet.mp4`, `latest_vrchat.png` | VRChat / Hakua content drafts |
| `hermes_v*.py`, `hermes_v*_with_audio.mp4` | versioned demo renders | Local video generation experiments |

### Disaster / monitoring JSON (ephemeral)

| Files | Purpose |
|-------|---------|
| `eq.json`, `quake.json`, `tsunami.json` | Earthquake API snapshots |
| `check_quake_tsunami.py`, `process_disaster.py`, `disaster_security*.py` | One-off processing scripts |

### Logs and state text

| Files | Purpose |
|-------|---------|
| `*.log`, `report.txt`, `state.txt`, `sync_result.txt` | Local run output |
| `current_time.txt`, `last_*.txt`, `.last_*_index` | Scratch indices |

### Diagnostics (gitignored)

| Pattern | Purpose |
|---------|----------|
| `_diag_*.py`, `temp_*.py`, `tmp_*.py` | Ad-hoc probes |
| `test_web_search.py` | Throwaway tool tests — use `tests/` instead |

### Secrets (never commit)

| Pattern | Purpose |
|---------|----------|
| `client_secret_*.json` | OAuth client secrets |
| `.env`, `cli-config.yaml` | Credentials and local paths |

## Where outputs should go

| Type | Preferred path |
|------|----------------|
| Cron media output | `output/` (gitignored) |
| Implementation logs | `_docs/` (gitignored) |
| Career / docs export | `career_docs_output/` (gitignored) |
| Temp merge work | `tmp/`, `_tmp/` (gitignored) |

## Official vs scratch

| Official (tracked) | Scratch (ignored) |
|--------------------|-------------------|
| `run_agent.py`, `cli.py` | `temp_script.py` |
| `scripts/daily_*.py` | root-level `test_*.py` probes |
| `plugins/`, `skills/` | root `*.mp4` renders |

See [`AGENTS.md`](AGENTS.md) for agent handling rules.
