# Scanner / Watchdog Conflict Resolution — Session Archive

**Date:** 2026-07-02
**Trigger:** Slack question: "Why did the Config Integrity Watchdog 'fix' the config — and how did it know the fix was correct?"

**Goal:** Diagnose and fix a two-cron-job conflict where `free-model-scanner-am` legitimately updated `model.default` and `Config Integrity Watchdog` treated the change as tampering and reverted it. Harden config integrity end-to-end so authorized changes are never flagged.

**Architecture:** Two-layer fix — (1) make authorized writes self-sealing so the watchdog never sees a mismatch after a legitimate change; (2) replace the mutable `.sha256` sidecar with a git-backed integrity log so the fingerprint itself cannot be silently overwritten.

---

## Root cause

`free-model-scanner-am` wrote `config.yaml` via `save_config()`, which did not seal the SHA256 sidecar atomically. The watchdog ran in the TOCTOU window between write and seal, saw a digest mismatch, and reverted the config. Neither job was buggy in isolation — the conflict was architectural.

## What shipped

| PR | Title | What it fixed |
|---|---|---|
| #52 | fix(backup): reseal config after quick snapshot restore | Snapshot restores left seal stale; added reseal call and included `.sha256` in `_QUICK_STATE_FILES` |
| #57 | feat(config): interprocess lock + config integrity API | `save_config()` now holds an exclusive OS file lock and atomically seals the sidecar; 17 new tests |
| #65 | feat(skills): auto-deploy skill scripts + UGW health check fix | `deploy_skill_scripts()` in `skills_sync.py` copies `skill/scripts/*.py` → `~/.hermes/scripts/` on every startup; fixed UGW script that was reading the wrong JSON key (`"report"` → `"gateway_state"`) |
| #66 | docs(agents): document config integrity seal API and skill scripts auto-deploy | AGENTS.md — config seal API section + skill script auto-deploy note |
| #68 | docs(agents): update config integrity section for git-backed seal system | AGENTS.md updated to document `hermes config seal/verify/restore`, the new skill, env vars, and guidance to use CLI commands (not low-level helpers) in new automation |

PR #67 (git-backed watchdog, shipped independently by user same day) is tracked in `.plans/config-integrity-watchdog.md`.

## Key decisions

- **Self-sealing writes**: `save_config()` acquires `config_write_lock()` (fcntl.flock / msvcrt.locking), writes atomically, seals — all inside one exclusive lock. External readers use `config_read_lock()` (shared lock) before checking integrity.
- **No re-entrancy deadlock**: `restore_config()` holds the write lock and calls `_write_config_to_disk()` directly — never re-enters `save_config()`.
- **Sidecar vs git-backed**: SHA256 sidecar remains the internal mechanism for `save_config()`; git-backed log (`config_integrity.jsonl`) is the external verification tool for cron jobs and watchdogs.
- **Skill script auto-deploy**: `deploy_skill_scripts()` runs on every `sync_skills()` call — no manual copy step for `scripts/*.py` files.

## Verified closed

- Config Integrity Watchdog cron job (`fbe11786e4d1`) on MacBook Pro confirmed using `hermes config verify` — verified 2026-07-02
- `restore_deepseek_config.py` confirmed absent from all machines
- MacBook Air: no watchdog cron job configured (intentional — single-machine deployment)
- All session branches deleted, `main` clean at `17766db`
