# Hermes Agent Repo Context

This repo has a local runtime feature called Trend Discovery Center.

Related local AI cards:

- `.hermes/context.md`
- `.hermes/active.md`
- `.hermes/decisions.md`
- `.hermes/progress.md`

## Trend Discovery Center

Implemented as a bundled Hermes plugin at:

- `plugins/trend_discovery/`
- `tests/plugins/test_trend_discovery_plugin.py`

Purpose:

- replace brittle n8n/Open Crawl-only trend discovery
- keep a persistent phase/issue registry so AI does not rely on memory
- run real scheduled scan/watchdog jobs on this Mac through `launchd`
- store evidence, logs, delivery receipts, findings, and compliance numbers

Runtime DB:

```bash
~/.hermes/trend-discovery/trend_discovery.db
```

Runtime logs:

```bash
~/.hermes/trend-discovery/logs/
```

Review queue:

```bash
~/.hermes/review-queue/trend-discovery/
```

Installed launchd jobs:

```bash
~/Library/LaunchAgents/com.hermes.trend-discovery.scan.plist
~/Library/LaunchAgents/com.hermes.trend-discovery.watchdog.plist
```

Use these commands for continuation:

```bash
venv/bin/python -m hermes_cli.main trend-discovery ops
venv/bin/python -m hermes_cli.main trend-discovery status
venv/bin/python -m hermes_cli.main trend-discovery comply
venv/bin/python -m hermes_cli.main trend-discovery health
venv/bin/python -m hermes_cli.main trend-discovery launchd-status
venv/bin/python -m hermes_cli.main trend-discovery sources list
venv/bin/python -m hermes_cli.main trend-discovery logs
```

Do not make Open Crawl or n8n mandatory. They are optional adapters only.
