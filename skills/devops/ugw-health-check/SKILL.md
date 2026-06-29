---
name: ugw-health-check
description: Checks Hermes gateway health from its state file.
version: 1.0.0
author: dizhaky
platforms: [linux, macos]
metadata:
  hermes:
    tags: [devops, gateway, health, monitoring, ops]
    related_skills: []
---

# UGW Health Check Skill

Standalone script that reports the health status of the Hermes Unified Gateway process. It reads `~/.hermes/gateway_state.json`, verifies the recorded PID is alive and belongs to the gateway (not a reused PID), and prints a structured status summary.

## When to Use

- After a suspected gateway crash or OOM kill to confirm whether the process is still running.
- In monitoring/alerting pipelines (Nagios-compatible exit codes).
- To diagnose stale state files left by SIGKILL.
- To detect PID reuse — where an unrelated process inherited the gateway's PID after a crash.

## Prerequisites

- Python 3.9+ (no external dependencies — stdlib only)
- Hermes gateway must write `gateway_state.json` (requires gateway v1.0+)
- On Linux, `/proc` filesystem must be available for PID identity verification

## How to Run

```bash
# Run directly from the skill directory
python3 ugw-health-check.py

# Install to scripts directory for repeated use
cp ugw-health-check.py ~/.hermes/scripts/ugw-health-check.py
chmod +x ~/.hermes/scripts/ugw-health-check.py
~/.hermes/scripts/ugw-health-check.py
```

The script reads `$HERMES_HOME/gateway_state.json` (defaults to `~/.hermes/gateway_state.json`).

## Quick Reference

| Exit code | Meaning |
|-----------|---------|
| 0 | Gateway is running and healthy |
| 1 | CRITICAL — not running, stale state, PID reuse, or state file missing |
| 2 | DEGRADED — running with warnings |

## Procedure

1. Run the script (see **How to Run** above).
2. Read the first output line:
   - `Unified Gateway OK` — process is healthy.
   - `Unified Gateway DEGRADED` — running but check the exit reason.
   - `Unified Gateway CRITICAL` — not running; see status line for details.
3. If CRITICAL due to `stale state file` or `PID reuse after crash`, restart the gateway:
   ```bash
   hermes gateway start
   ```
4. If the state file is missing entirely, verify `HERMES_HOME` is set correctly and that the gateway has been started at least once.

### Example output

```
Unified Gateway OK
Status: RUNNING (report_type: gateway_state_check)
Active agents: 2
Platforms: slack:connected, telegram:connected
PID: 12345 | Last updated: 5s ago
```

```
Unified Gateway CRITICAL
Status: CRITICAL (PID 12345 is not the gateway process — possible PID reuse after crash)
Last updated: 2m 30s ago
Platforms: slack:connected
```
