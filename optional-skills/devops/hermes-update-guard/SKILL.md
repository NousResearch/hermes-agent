---
name: hermes-update-guard
description: Guard safe Hermes Agent updates: preflight gate, WAL-safe backup, provider-drift tripwire, verify.
version: 0.1.1
author: MCPVOT
license: MIT
platforms: [macos, linux, windows]
metadata:
  hermes:
    tags: [hermes, update, backup, drift, guardian, devops]
    homepage: https://github.com/MCPVOT/hermes-update-guard
    related_skills: [hermes-agent]
    requires_toolsets: [terminal]
---

# hermes-update-guard

Guardian orchestrator for safe `hermes update` runs. It wraps the official update
CLI with a preflight gate, WAL-safe SQLite backups, and a provider-surface
tripwire — the whole scripted update ritual in one static binary.

## When to Use

- Before ANY `hermes update`: run the guarded flow below instead of updating raw.
- After an update: prove nothing drifted (version, DB integrity, provider surface).
- When a previous update failed and left a stale lock, a stale `running` receipt,
  or a silently stashed core file.

## Quick Reference

| Task | Command |
|---|---|
| Peek before updating | `hermes-update-guard preflight` |
| Pin the known-good surface | `hermes-update-guard export` |
| Prove no drift | `hermes-update-guard diff --baseline last-known` |
| WAL-safe backup | `hermes-update-guard backup` |
| Guarded update (all-in-one) | `hermes-update-guard update --close-desktop` |
| Verify after | `hermes-update-guard verify` |
| Identity banner | `hermes-update-guard vibe` |

## Procedure

1. **Confirm the binary is available** (see *Binary availability* below).
2. **Run the ritual in order** — adjust `GUARD` to the absolute path if the
   binary is not on `PATH`:

   ```bash
   GUARD=hermes-update-guard

   $GUARD preflight                        # read-only: any surprises?
   $GUARD export                           # pin known-good provider surface
   $GUARD diff --baseline last-known       # prove clean (exit 0)
   $GUARD backup                           # WAL-safe snapshot of every profile
   $GUARD update --close-desktop           # preflight -> export -> backup -> update -> verify
   $GUARD verify                           # post-update integrity + drift
   ```

3. **Apply the decision rules** (below) at each gate before continuing.

## Binary availability

The expected binary is `hermes-update-guard` on `PATH` (or `target/release/` of a
local build). If it is not installed:

- Build it: `cargo install --path <repo>` from the hermes-update-guard repo.
- Or grab the per-OS artifact from the repo's GitHub Releases.

## Decision rules

- `preflight` reports `dirty: N file(s)` → those files will be stashed by the
  updater. If a tracked core file is intentionally patched (for example
  `hermes_cli/models_catalog_static.py`), verify the stash list after the update
  and restore it if the update dropped it.
- `diff` exits 1 (drift found) → **do not update**; investigate `model.*`,
  `fallback_*`, `delegation`, and `display.language` first.
- `update` refuses to run while a venv holder or the Desktop app is running; pass
  `--close-desktop` only when the operator has already closed their session.
- Backup FAIL → never update from an unverified backup.

## Pitfalls

- **Updating with the desktop app open** → the update refuses; close the app (or
  pass `--close-desktop`) and retry.
- **Ignoring a `diff` drift** → you lose the guarantee that the update keeps your
  model/provider config; resolve drift first.
- **Trusting a partial backup** → the `backup` gate already refuses incomplete
  backups; do not override it.
- **Stale lock after a previous failed update** → `preflight` surfaces it; clear
  it before retrying.

## Manual restore (emergency)

```bash
rsync -a "$HOME/.hermes/backups/pre-update-<stamp>/<profile>/state.db" \
      "<HERMES_HOME>/<profile>/state.db"
```

The same applies to `.env`, `auth.json`, and `config.yaml`. Restore only after
stopping the gateway (`hermes gateway stop`) and closing Hermes.

## Verification

- `hermes-update-guard verify` exits `0` and reports DB integrity `ok` with no
  drift versus the pre-update export.
- Exit codes: `0` ok · `1` drift found, backup incomplete, verify failure, or any
  runtime error · `2` CLI usage error (bad flags/arguments).
