# UGW Health Check

A standalone health check script for the Hermes Unified Gateway (UGW).

## After Install

Hermes automatically syncs this skill to `~/.hermes/skills/devops/ugw-health-check/` on next startup. To activate it for the UGW Health Monitor cron job, copy or symlink the script:

```bash
# Copy (simple)
cp ~/.hermes/skills/devops/ugw-health-check/scripts/ugw-health-check.py ~/.hermes/scripts/ugw-health-check.py

# Or symlink (stays up to date with skill updates)
ln -sf ~/.hermes/skills/devops/ugw-health-check/scripts/ugw-health-check.py ~/.hermes/scripts/ugw-health-check.py
```

Then verify it works:
```bash
python3 ~/.hermes/scripts/ugw-health-check.py
```

Expected output when gateway is running:
```
🟢 Unified Gateway OK
Status: RUNNING
...
```

## Problem

The original `~/.hermes/scripts/ugw-health-check.py` fails with `report parse failed`
because it expects a `report` key in `gateway_state.json` that does not exist.
The actual schema written by `gateway/status.py` uses `gateway_state` (not `report`).

## What it checks

Reads `~/.hermes/gateway_state.json` (or `$HERMES_HOME/gateway_state.json`) and
maps `gateway_state` to a health outcome:

| `gateway_state` value                          | Result     | Exit code |
|------------------------------------------------|------------|-----------|
| `running`                                      | OK         | 0         |
| `degraded`                                     | WARNING    | 2         |
| `starting`, `draining`, `stopped`, `startup_failed`, or file missing | CRITICAL | 1 |

## Sample output

**Healthy:**
```
Unified Gateway OK
Status: RUNNING (report_type: gateway_state_check)
Active agents: 3
Platforms: slack:connected, telegram:connected
PID: 12345 | Last updated: 4m 12s ago
```

**Critical:**
```
Unified Gateway CRITICAL
Status: CRITICAL (gateway_state: stopped)
Exit reason: signal SIGTERM
PID: 12345 | Last updated: 1h 2m 5s ago
Platforms: slack:disconnected
```

## Install

```bash
cp scripts/ugw-health-check.py ~/.hermes/scripts/ugw-health-check.py
chmod +x ~/.hermes/scripts/ugw-health-check.py
```

## Dependencies

None. Pure Python 3.9+ stdlib only (`json`, `pathlib`, `datetime`, `os`, `sys`).
No imports from the hermes agent codebase.

## Environment

- `HERMES_HOME` — override the hermes home directory (default: `~/.hermes`)
