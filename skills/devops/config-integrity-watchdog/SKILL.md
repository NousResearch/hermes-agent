---
name: config-integrity-watchdog
description: Detects and restores tampered Hermes config via git log.
version: 1.0.0
author: dizhaky
platforms: [linux, macos]
metadata:
  hermes:
    tags: [devops, security, config, integrity, watchdog]
    related_skills: [ugw-health-check]
---

## When to Use

Use this skill when you need to detect or recover from unauthorized changes to `~/.hermes/config.yaml`. The watchdog stores a tamper-evident fingerprint in the dotfiles git repository — unlike a mutable `.sha256` sidecar, a git commit cannot be silently overwritten.

## Prerequisites

- `~/.hermes/config.yaml` must exist (symlink or real file)
- The dotfiles directory must be a git repository (configurable via `HERMES_DOTFILES_DIR`, defaults to `~/Dev/dotfiles`)
- Python 3.9+, no third-party dependencies

## How to Run

```bash
# Seal the current config as canonical
python3 ~/.hermes/skills/devops/config-integrity-watchdog/scripts/seal.py

# Verify config integrity
python3 ~/.hermes/skills/devops/config-integrity-watchdog/scripts/verify.py

# Restore canonical config if tampered
python3 ~/.hermes/skills/devops/config-integrity-watchdog/scripts/restore.py
```

## Quick Reference

| Exit code | Meaning |
|---|---|
| 0 | Config matches canonical baseline |
| 1 | Config has been tampered |
| 2 | Integrity log itself has been modified (log tampering) |
| 3 | No baseline found (run seal.py first) |

## Procedure

1. After any intentional config change, run `seal.py` to commit the new baseline.
2. Schedule `verify.py` as a cron job to detect tampering.
3. If tampering is detected, run `restore.py` to revert and re-seal.
