# Config Integrity Watchdog — Implementation Plan

**Slack thread:** https://mfc-nyc.slack.com/archives/C0BD8QBUSJF/p1782742870774319

## Problem

The Config Integrity Watchdog has triggered 25+ times in 19 days. Root cause: the `.sha256` sidecar file is mutable — any process that writes `config.yaml` can also overwrite the fingerprint, masking tampering.

## Solution

Replace mutable sidecar with git-backed append-only integrity log stored in the dotfiles repository.

## Issues (Linear not available — tracked here)

| # | Title | Status |
|---|---|---|
| 1 | Create config-integrity-watchdog skill scaffold | Done |
| 2 | Implement seal.py | Done |
| 3 | Implement verify.py | Done |
| 4 | Implement restore.py | Done |
| 5 | Write tests | Done |
| 6 | Open PR | Done |
| 7 | Add hermes config seal/verify/restore CLI commands | Done |
| 8 | Write CLI integration tests | Done |

## Assumptions

- Dotfiles git repo is at `~/Dev/dotfiles` (configurable via `HERMES_DOTFILES_DIR`)
- `config.yaml` may be a symlink; scripts follow symlinks for hashing
- Canonical model decision (deepseek-v4-pro vs Nemotron-free) deferred — restore.py uses whatever is in the sealed baseline, not a hardcoded model

## Out of scope

- Changing the canonical model ID (requires user decision)
- Config integrity for non-config files
- 1Password integration (future enhancement)

## Outcome

**Shipped:** 2026-07-02 — PR #67 (merged `38d8fdb`)

All 8 issues above are complete. The mutable `.sha256` sidecar is replaced end-to-end:

- `skills/devops/config-integrity-watchdog/` ships `seal.py`, `verify.py`, `restore.py`, and shared `config_integrity.py` core
- `hermes config seal/verify/restore` CLI commands wired in `hermes_cli/config_integrity_cli.py`
- 50 tests (29 skill-level + 21 CLI integration) all pass
- Config Integrity Watchdog cron job (MacBook Pro, `fbe11786e4d1`) confirmed calling `hermes config verify` — verified 2026-07-02
- `restore_deepseek_config.py` confirmed absent from all machines — fully retired
