---
name: hermes-diag
description: One-shot Hermes diagnostics — check profiles, gateways, configs, disk, memory, and recent errors.
version: 1.0.0
author: Vijay Selvaraj (vijays365), Hermes Agent
license: MIT
platforms: [linux]
metadata:
  hermes:
    tags: [devops, diagnostics, health, troubleshooting, monitoring]
    category: devops
    requires_toolsets: [terminal]
    related_skills: [config-validator, system-health]
---

# Hermes Diagnostics

Single command to assess overall Hermes system health. Checks gateways, configs, disk, memory, uptime, recent errors, and skills count.

## When to Use

- User reports Hermes isn't working correctly
- Before and after config changes to verify system health
- Routine system check for self-hosted Hermes instances
- Troubleshooting gateway connection issues

## What it checks

| Check | What it inspects | Source |
|-------|------------------|--------|
| Gateway states | Each profile's gateway status | `gateway_state.json` per profile |
| Config validity | YAML syntax and required fields | Delegates to `config-validator` |
| Disk usage | Root partition fill percentage | `df -h` |
| Memory | RAM and swap usage | `free -m` |
| Uptime | System uptime and load averages | `uptime`, `/proc/loadavg` |
| Errors | Recent gateway log errors (24h) | Profile log files |
| Skills | Count of installed skills | `$HERMES_HOME/skills/` tree |

## Usage

```bash
bash $HERMES_HOME/skills/devops/hermes-diag/scripts/hermes-diag.sh
```

The script uses `$HERMES_HOME` (defaults to `~/.hermes`).

## Output format

Color-coded with emoji indicators:

```
  Hermes Diagnostics — myserver
  2026-06-12 21:06:44

── Gateway States ──
  ✅ default: running
  ✅ codi: running

── Config ──
  ✅ All configs valid

── Disk Usage ──
  ✅ /: 42% used

── Memory ──
  ✅ Memory: 62% used
  ✅ Swap: 0% used

── Uptime ──
  ✅ up 5 days, 14 hours
  ✅ Load: 0.45 0.52 0.48

── Recent Errors ──
  ✅ No recent errors

── Skills ──
  ✅ 52 skills installed
```

## Exit codes

| Code | Meaning |
|------|---------|
| 0 | All clear |
| 1 | Warnings (non-critical) |
| 2 | Errors detected |

## Common Pitfalls

1. **Log paths depend on profile directories** under `$HERMES_HOME/profiles/`.
2. **Gateway status in `gateway_state.json` may lag a few seconds** behind the actual process state.
3. **The disk check inspects root (`/`)** — adjust if Hermes data is on a separate volume.
4. **The config validator sub-check requires `validate-config.sh`** at the expected path. Install `config-validator` for full coverage.

## Verification Checklist

- [ ] Run `bash $HERMES_HOME/skills/devops/hermes-diag/scripts/hermes-diag.sh`
- [ ] Confirm all checks return ✅ or acceptable ⚠️
- [ ] Verify the exit code matches the expected system state (0 for clean)
