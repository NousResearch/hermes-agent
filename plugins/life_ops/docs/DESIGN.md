# Hermes Agent — Architecture

## Runtime topology

- **Gateway process** (long-lived): hosts platform adapters, the agent loop,
  and the cron scheduler. Kept alive on macOS via a LaunchAgent managed by
  `hermes_cli/service_manager.py` (`launchd_start/stop/restart`).
- **Entry points** (`pyproject.toml`): `hermes` (CLI), `hermes-agent`
  (`run_agent.py`), `hermes-acp` (ACP adapter).
- **Repo layout**: `agent/` (core loop), `gateway/` (session/delivery),
  `plugins/platforms/discord/` (Discord adapter), `cron/` (scheduler + jobs),
  `tools/` (tool implementations), `skills/` + `optional-skills/` (packaged
  skills), `hermes_cli/` (CLI), `tests/` (mirrors source layout).

## Discord adapter

`plugins/platforms/discord/adapter.py` — a `discord.py` gateway bot:

- Slash commands are registered natively (`_register_slash_commands`,
  `_safe_sync_slash_commands`); commands are adapter-level code, not
  skill-declared.
- Interactive components (buttons/selects) run auth-checked callbacks
  (`_component_check_auth`, confirm-dialog patterns like
  `_on_expensive_confirm`). New interactions (bedtime Start/Skip prompt,
  approve-to-backlog, `/rpe`) follow these existing component patterns.

## Scheduling

`cron/scheduler.py` ticks every 60s inside the gateway (file-locked via
`~/.hermes/cron/.tick.lock`); jobs persist per-profile in
`~/.hermes/profiles/<p>/cron/jobs.json`. Jobs may attach skills and deliver
composed output to Discord (`deliver="discord"`). Unattended jobs follow a
bounded-tool-call discipline. An OS-level launchd chain (deploy/) may invoke
morning steps that must run even if a job was missed (macOS runs launchd
calendar jobs on wake).

## MVP1 integration seams

- **Contract reader**: leaf projects write versioned JSON files
  (`journal_brief.latest.json`, `perfcoach_brief.latest.json`,
  `commander_report.latest.json`) with `schema_version` / `for_date` /
  `generated_at` (+07:00) and atomic writes. Hermes validates freshness
  (`for_date == today`, Asia/Bangkok) per file and renders stale/missing
  sections as "unavailable."
- **Commander dispatch**: via the `commander-api` skill's client script
  (bearer token; confirm-gated writes; per-project run lock respected by
  checking `GET /api/sprints/running-all` before `POST /api/sprints/run`).
- **perf-coach feedback**: `/rpe` writes through perf-coach's worker API
  feel-entry endpoint; Hermes never interprets training data.

## Testing

- `tests/` mirrors the source tree; run via `scripts/run_tests.sh`
  (hermetic env, xdist parallel; CI shards 8 ways).
- Naming: `test_<feature>.py`, frequently suffixed with issue numbers
  (`test_<slug>__<N>.py` style also appears). Tests are authored separately
  from implementation in the Commander sprint flow (tester agent writes
  acceptance tests; coder must not edit grading tests).
- Git model: `main` is the default branch; `develop` is the integration
  branch for automated sprint merges; feature branches merge via PR.
