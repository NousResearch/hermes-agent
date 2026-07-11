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
| `output/media/*` | `ohayo_tweet.mp4`, `latest_vrchat.png` | VRChat / Hakua content drafts |
| `tmp/snapshots/hermes_v*.py`, `output/media/hermes_v*_with_audio.mp4` | versioned demo renders | Local video generation experiments |

### Disaster / monitoring JSON (ephemeral)

| Files | Purpose |
|-------|---------|
| `output/reports/eq.json`, `output/reports/quake.json`, `output/reports/tsunami.json` | Earthquake API snapshots |
| `tmp/probes/check_quake_tsunami.py`, `tmp/probes/process_disaster.py`, `tmp/probes/disaster_security*.py` | One-off processing scripts |

### Logs and state text

| Files | Purpose |
|-------|---------|
| `output/logs/*.log`, `output/reports/report.txt`, `output/reports/state.txt`, `output/reports/sync_result.txt` | Local run output |
| `output/reports/current_time.txt`, `output/reports/last_*.txt`, `output/reports/.last_*_index` | Scratch indices |

### Diagnostics (gitignored)

| Pattern | Purpose |
|---------|----------|
| `tmp/probes/_diag_*.py`, `tmp/probes/temp_*.py`, `tmp/probes/tmp_*.py` | Ad-hoc probes |
| `test_web_search.py` | Throwaway tool tests — use `tests/` instead |

### Secrets (never commit)

| Pattern | Purpose |
|---------|----------|
| `client_secret_*.json` | OAuth client secrets |
| `.env`, `cli-config.yaml` | Credentials and local paths |

### Canonical local folders

Root scratch is being classified without deleting files. Use `output/media/`
for generated media, `output/reports/` for generated reports and snapshots,
`output/logs/` for generated logs, `tmp/probes/` for one-off diagnostics, and
`tmp/snapshots/` for generated source or configuration snapshots. These
directories remain ignored; only this guide and the fork harness are tracked.

Agents must not execute an untrusted generated script merely because it is in
the repository. Inspect it first, keep credentials out of its environment,
and run it only with an explicit temporary workspace and bounded permissions.

## Where outputs should go

| Type | Preferred path |
|------|----------------|
| Cron media output | `output/media/` (gitignored) |
| Implementation logs | `_docs/` (gitignored) |
| Career / docs export | `career_docs_output/` (gitignored) |
| Temp merge work | `tmp/`, `_tmp/` (gitignored) |

## Official vs scratch

| Official (tracked) | Scratch (ignored) |
|--------------------|-------------------|
| `run_agent.py`, `cli.py` | `temp_script.py` |
| `scripts/daily_*.py` | `tmp/probes/test_*.py` probes |
| `plugins/`, `skills/` | `output/media/*` renders |

See [`AGENTS.md`](AGENTS.md) for agent handling rules.
