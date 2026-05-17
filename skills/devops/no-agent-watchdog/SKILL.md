---
name: no-agent-watchdog
description: "Use when creating deterministic cron watchdogs that should send alerts without an LLM call: health checks, stale backups, disk/memory thresholds, endpoint probes, and other quiet-on-OK monitoring jobs."
version: 1.0.0
author: Hermes Agent
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [cron, watchdog, monitoring, alerts, no-agent, automation]
    related_skills: [webhook-subscriptions]
---

# No-Agent Watchdog Cron Jobs

## Overview

Use Hermes cron in `no_agent=true` mode for deterministic monitoring jobs where a script already knows exactly what to report. This pattern avoids spending model tokens, keeps alerts working during provider/API outages, and prevents the agent from over-reasoning about simple health checks.

The core contract is simple:

- **OK state:** print nothing and exit `0` so the scheduler stays silent.
- **Warning state:** print the exact alert message to stdout and exit non-zero or zero depending on whether you want the scheduler to record failure.
- **Broken watchdog:** exit non-zero with useful stderr/stdout so the user is alerted that the monitor itself is failing.

Good watchdogs are boring: they are read-only, bounded by timeouts, explicit about thresholds, and quiet unless something needs attention.

## When to Use

Use this skill for:

- Endpoint health checks (`curl /health`, HTTP status probes, local TCP listeners)
- Backup freshness checks (latest file age, snapshot count, last successful export marker)
- Resource thresholds (disk, memory, GPU, process count)
- External API pollers that can format their own alert text
- Heartbeats where missed or stale data is the only decision criterion
- Any recurring check where an LLM would add cost and failure modes but no value

Do **not** use this pattern for:

- Reports that need summarization, judgment, ranking, or natural-language synthesis
- Workflows that need human clarification
- Jobs that write production systems without explicit approval
- Checks that might print secrets, cookies, raw messages, tokens, or private row data
- Long-running daemons; cron scripts should start, check, print-or-stay-quiet, then exit

## Script Contract

A no-agent cron job delivers the script stdout verbatim.

Design scripts with these rules:

1. **Quiet on OK**
   - Normal output should be empty.
   - Do not print progress logs, debug banners, or timestamps during normal operation.

2. **Actionable on warning**
   - Include what failed, where, when, threshold, observed value, and safe next step.
   - Keep it short enough for chat notifications.

3. **Bounded checks**
   - Use short timeouts for network calls.
   - Avoid unbounded `find`, broad recursive scans, or commands that can hang.

4. **Read-only by default**
   - Health monitors should inspect state, not fix it.
   - If recovery is safe, make it a separate explicitly-approved job.

5. **Secret hygiene**
   - Never print env files, cookies, credentials, request headers, or raw private content.
   - Redact command outputs before printing if they can include sensitive values.

6. **Portable paths**
   - Put the script under `~/.hermes/scripts/`.
   - Register the cron job with the script filename only, not an absolute path.

## Minimal Python Watchdog

Save as `~/.hermes/scripts/service_watchdog.py`:

```python
#!/usr/bin/env python3
from __future__ import annotations

import datetime as dt
import subprocess
import sys


def http_code(url: str, timeout: int = 3) -> str:
    proc = subprocess.run(
        ["curl", "-sS", "-m", str(timeout), "-o", "/dev/null", "-w", "%{http_code}", url],
        text=True,
        capture_output=True,
        timeout=timeout + 2,
    )
    if proc.returncode != 0:
        return "000"
    return proc.stdout.strip() or "000"


def main() -> int:
    checks = [
        ("api", "http://127.0.0.1:8000/health", "200"),
        ("dashboard", "http://127.0.0.1:3000/", "200"),
    ]
    warnings = []
    for name, url, expected in checks:
        got = http_code(url)
        if got != expected:
            warnings.append(f"{name}: expected HTTP {expected}, got {got}")

    if not warnings:
        return 0  # quiet on OK

    now = dt.datetime.now().strftime("%Y-%m-%d %H:%M")
    print("[Watchdog] service health warning")
    print(f"- time: {now}")
    for warning in warnings:
        print(f"- {warning}")
    print("- next step: check process status and recent service logs")
    return 1


if __name__ == "__main__":
    sys.exit(main())
```

Make it executable and test both modes:

```bash
chmod +x ~/.hermes/scripts/service_watchdog.py
~/.hermes/scripts/service_watchdog.py
printf 'exit=%s\n' "$?"
```

## Backup Freshness Pattern

Use modification time rather than reading backup contents. This avoids exposing private data and keeps the check cheap.

```python
#!/usr/bin/env python3
from __future__ import annotations

import datetime as dt
from pathlib import Path

MAX_AGE_HOURS = 36
BACKUP_DIRS = [Path.home() / "backups", Path.home() / ".app" / "snapshots"]


def newest_file(paths: list[Path]) -> Path | None:
    newest: Path | None = None
    for base in paths:
        if not base.is_dir():
            continue
        for path in base.rglob("*"):
            if path.is_file() and (newest is None or path.stat().st_mtime > newest.stat().st_mtime):
                newest = path
    return newest


def main() -> int:
    latest = newest_file(BACKUP_DIRS)
    if latest is None:
        print("[Watchdog] backup warning")
        print("- latest backup: missing")
        print("- next step: verify backup job path and scheduler status")
        return 1

    age_hours = int((dt.datetime.now().timestamp() - latest.stat().st_mtime) // 3600)
    if age_hours <= MAX_AGE_HOURS:
        return 0

    print("[Watchdog] backup freshness warning")
    print(f"- latest age: {age_hours}h")
    print(f"- threshold: {MAX_AGE_HOURS}h")
    print(f"- path: {latest}")
    print("- next step: inspect the backup job logs and run a manual dry-run")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
```

For very large trees, prefer a bounded shell command (`find` with `-maxdepth`, or a metadata file written by the backup job) rather than `Path.rglob("*")`.

## Register the Cron Job

Create the job with `no_agent=true`:

```python
cronjob(
    action="create",
    name="service-health-watchdog",
    schedule="*/15 * * * *",
    deliver="origin",
    script="service_watchdog.py",
    no_agent=True,
    enabled_toolsets=["terminal", "file"],
)
```

Important details:

- `script` should be a filename under `~/.hermes/scripts/`.
- Omit `prompt` and `skills`; they are ignored when `no_agent=True`.
- Empty stdout means silent success.
- Non-empty stdout is delivered exactly as written.
- Non-zero exit or timeout alerts the user, so failures in the monitor are visible.

## Verification Flow

After writing the script:

1. **Manual OK test**
   ```bash
   ~/.hermes/scripts/service_watchdog.py
   echo $?
   ```
   Expected: no stdout, exit `0`.

2. **Manual warning test**
   Temporarily point one check at a closed port or set an impossible threshold.
   Expected: compact alert text with observed value and next step.

3. **Cron registration check**
   ```python
   cronjob(action="list")
   ```
   Confirm `script`, `no_agent: true`, schedule, delivery target, and enabled state.

4. **Scheduler trigger**
   ```python
   cronjob(action="run", job_id="...")
   cronjob(action="list")
   ```
   Confirm `last_status` becomes `ok` after the scheduler tick.

5. **Noise check**
   Let it run through at least one normal interval. The user should receive no message when everything is OK.

## Recovery Boundaries

Keep watchdogs separate from recovery automation unless the user explicitly asks for auto-remediation.

Safe watchdog actions:

- Probe local endpoints
- Check process lists or service status
- Inspect file modification times and sizes
- Read bounded, non-secret log excerpts for a specific service
- Print a concise warning with rollback or next-step guidance

Actions that need explicit approval or a separate runbook:

- Restarting production services
- Editing databases or running migrations
- Activating/deactivating workflows
- Deleting executions, logs, backups, or queues
- Changing credentials, tokens, or encryption keys
- Uploading private files or alert payloads to third-party services

## Alert Message Template

```text
[Watchdog] <system> warning
- time: <timestamp and timezone>
- status: warning
- failed check: <endpoint/process/backup/resource>
- observed: <actual value>
- expected: <threshold or status>
- impact: <short user-facing impact>
- next step: <safe inspection or rollback step>
```

For chat platforms, prefer short bullet lists over tables.

## Common Pitfalls

1. **Printing success logs.** In `no_agent` mode, stdout is the notification body. If the script prints "OK" every run, the user gets spammed.

2. **Using an LLM for deterministic checks.** If the script can decide pass/fail, use `no_agent=true`. Use an agent only when interpretation is required.

3. **Absolute script paths in cron config.** Store scripts in `~/.hermes/scripts/`, then reference only `service_watchdog.py`.

4. **No timeouts.** A stuck `curl`, `ssh`, or API call can make the watchdog look broken. Always set short timeouts.

5. **Alerting on expected maintenance.** Add maintenance windows or pause the cron job during planned downtime to avoid false positives.

6. **Bundling recovery into monitoring.** Restart loops can hide the root cause or make outages worse. Alert first; automate recovery only after a reviewed runbook exists.

7. **Leaking sensitive output.** Commands like `env`, verbose HTTP clients, database dumps, or raw logs can print secrets. Whitelist what you print.

## Verification Checklist

- [ ] Script lives under `~/.hermes/scripts/`
- [ ] OK path prints nothing and exits `0`
- [ ] Warning path prints a concise, actionable alert
- [ ] Every external command has a timeout
- [ ] Checks are read-only unless a separate approved recovery runbook exists
- [ ] No secrets, raw messages, cookies, tokens, or private rows are printed
- [ ] Cron job uses `no_agent=true`
- [ ] Cron job references the script by filename, not absolute path
- [ ] Manual run and cron-triggered run both verified
- [ ] Rollback is known: pause/remove the cron job or restore the previous script
