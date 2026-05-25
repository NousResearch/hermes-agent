# Trend Discovery Center

Reliable trend and business discovery for Hermes Agent.

This plugin was added because the previous n8n/Open Crawl workflow was brittle:
if either service failed, the whole discovery process could go silent. This
plugin makes Hermes the control layer instead:

- persistent phase and issue tracking in SQLite
- local run logs and delivery receipts
- watchdog alerts for overdue phases and repeated source failures
- independent source adapters with fallback behavior
- optional Open Crawl and n8n adapters, never required for the core pipeline
- review-queue writeback for Hermes knowledge capture
- macOS `launchd` jobs for real scheduled execution on this machine

## Code

Core files:

- `store.py` - SQLite schema, phase registry, compliance percentages
- `scanner.py` - RSS/webpage/Open Crawl/n8n/Searxng-style source scanning
- `notifications.py` - local, macOS, and webhook notification dispatch
- `knowledge.py` - digest, reliability report, review-queue writeback
- `health.py` - local runtime health checks
- `cli.py` - `hermes trend-discovery ...` CLI
- `cli_runner.py` - direct `python -m` runner
- `plan.py` - canonical phase and issue definitions

Tests:

- `tests/plugins/test_trend_discovery_plugin.py`

## Runtime State

Persistent DB:

```bash
~/.hermes/trend-discovery/trend_discovery.db
```

Runtime logs:

```bash
~/.hermes/trend-discovery/logs/
```

Review queue output:

```bash
~/.hermes/review-queue/trend-discovery/
```

macOS scheduled jobs:

```bash
~/Library/LaunchAgents/com.hermes.trend-discovery.scan.plist
~/Library/LaunchAgents/com.hermes.trend-discovery.watchdog.plist
```

## Operator Commands

```bash
venv/bin/python -m hermes_cli.main trend-discovery ops
venv/bin/python -m hermes_cli.main trend-discovery status
venv/bin/python -m hermes_cli.main trend-discovery comply
venv/bin/python -m hermes_cli.main trend-discovery health
venv/bin/python -m hermes_cli.main trend-discovery launchd-status
venv/bin/python -m hermes_cli.main trend-discovery scan --write-review-queue
venv/bin/python -m hermes_cli.main trend-discovery watchdog
venv/bin/python -m hermes_cli.main trend-discovery sources list
venv/bin/python -m hermes_cli.main trend-discovery logs
```

## Operating Modes

This system has seven practical modes:

1. `scheduled-scan` - macOS launchd runs the scanner daily and writes review-queue output.
2. `scheduled-watchdog` - macOS launchd runs the watchdog hourly and alerts only when needed.
3. `manual-scan` - operator-triggered scan through `trend-discovery scan`.
4. `manual-watchdog` - operator-triggered watchdog through `trend-discovery watchdog`.
5. `compliance` - numeric phase/issue completion through `trend-discovery comply`.
6. `source-admin` - manage adapters through `trend-discovery sources ...`.
7. `knowledge-writeback` - digest/review-queue output through `digest` or scan with `--write-review-queue`.

Control source adapters with:

```bash
venv/bin/python -m hermes_cli.main trend-discovery sources list
venv/bin/python -m hermes_cli.main trend-discovery sources add --name my-rss --adapter rss --url https://example.com/feed.xml --priority 30
venv/bin/python -m hermes_cli.main trend-discovery sources disable --name my-rss
venv/bin/python -m hermes_cli.main trend-discovery sources enable --name my-rss
venv/bin/python -m hermes_cli.main trend-discovery sources delete --name my-rss
```

Control scheduled execution with:

```bash
venv/bin/python -m hermes_cli.main trend-discovery install-launchd
venv/bin/python -m hermes_cli.main trend-discovery launchd-status
venv/bin/python -m hermes_cli.main trend-discovery uninstall-launchd
```

## Current Installed Schedule

Installed on 2026-05-26:

- `com.hermes.trend-discovery.scan` - every 86400 seconds
- `com.hermes.trend-discovery.watchdog` - every 3600 seconds

Both were verified with `launchctl print`, `runs = 2`, and `last exit code = 0`
at installation time.

## Verification Evidence

Latest verified state when this README was created:

```text
tests/plugins/test_trend_discovery_plugin.py: 11 passed
compileall plugins/trend_discovery: pass
trend-discovery health: database/hostname/phases/issues/sources all 100 0
launchd scan: last exit code 0
launchd watchdog: last exit code 0
notification target: macos with local fallback
findings recorded: 41
PROJECT_TOTAL 100 0
```

## Important Continuation Notes

- Do not make Open Crawl or n8n required dependencies. They are optional import
  adapters only.
- If optional Open Crawl or n8n URLs are empty, source status should be
  `skipped_optional` / `skipped`, not success.
- If this is moved to a VPS, install the equivalent systemd timers or cron jobs
  and update this README plus Obsidian handoff with the new runtime target.
- If adding external notifications, keep `local` as fallback so delivery
  failures still leave receipts.
