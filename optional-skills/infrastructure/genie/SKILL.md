---
name: genie
description: VPS disk cleanup, root filesystem audit, and backup retention. Deletes old snaps, logs, and cron files; investigates disk spikes; enforces one historical backup on the VPS. Not for database maintenance beyond analysis, log rotation configuration, or real-time monitoring.
version: 1.7.0
author: Indigo Karasu (indigokarasu)
license: MIT
platforms: [linux]
source: https://github.com/indigokarasu/genie
includes:
  - references/**
  - scripts/**
metadata:
  hermes:
    tags: [cleanup, disk-space, filesystem, maintenance]
    category: infrastructure
    config:
      - key: genie.snapshot_max_age_days
        description: "Delete snapshots older than N days"
        default: "7"
      - key: genie.log_compress_age_days
        description: "Compress logs older than N days"
        default: "7"
      - key: genie.log_delete_age_days
        description: "Delete compressed logs older than N days"
        default: "30"
      - key: genie.cron_output_compress_age_days
        description: "Compress cron output files older than N days"
        default: "7"
      - key: genie.session_compress_age_days
        description: "Compress session JSONs older than N days"
        default: "14"
      - key: genie.tmp_stale_hours
        description: "Delete /tmp files older than N hours (0 to skip)"
        default: "24"
      - key: genie.git_clone_max_age_days
        description: "Delete git clones in /root/projects/ untouched for N days (must have remote)"
        default: "5"
      - key: genie.dry_run
        description: "If true, only report — don't delete/compress"
        default: "false"
---

# Genie — VPS Disk Cleanup

Safely reclaims disk space on Linux VPS and workstations by deleting or compressing stale state snapshots, logs, cron output, temp files, and package caches. Also owns `/root` disk-spike investigation, duplicate repo detection, and backup-retention audits formerly covered by `util-vps-cleanup`. Does not touch live databases, auth files, or active sessions.

## When to Use

- VPS disk usage is high (above 50%)
- User asks to "clean up disk space" or "check disk usage"
- User asks why disk usage grew or where space went
- `/root` needs a safe audit/classification pass
- Backup directories, snapshots, or `.bak-*` files may violate retention
- Weekly maintenance cron fires
- Before/after large operations (backups, migrations)

## When NOT to Use

- Database maintenance tasks (genie does not manage databases)
- Real-time monitoring or alerting
- Log rotation configuration (genie handles specific files, not system-wide logrotate)

## Prerequisites

No external dependencies — uses only Python 3.11+ stdlib.

Configuration lives in `config.yaml` under `skills.config.genie.*` (see Configuration below). Behavioral settings are never read from environment variables.

## How to Run

```bash
# Assess disk usage and identify cleanup targets
python3 /root/.hermes/profiles/indigo/skills/ocas-genie/scripts/genie.py --assess

# Execute cleanup (Tier 1 + Tier 2)
python3 /root/.hermes/profiles/indigo/skills/ocas-genie/scripts/genie.py --clean

# Dry run — preview without deleting
python3 /root/.hermes/profiles/indigo/skills/ocas-genie/scripts/genie.py --clean --dry-run

# Map filesystem and generate FILESYSTEM.md manifest
python3 /root/.hermes/profiles/indigo/skills/ocas-genie/scripts/genie.py --discover
```

For large file counts (>2,000), run in background: `terminal(background=True, notify_on_complete=True)`.

## Quick Reference

| Flag | Action |
|------|--------|
| `--assess` | Report disk usage and cleanup targets |
| `--clean` | Execute Tier 1 + Tier 2 cleanup |
| `--clean --tier 1` | Only Tier 1 (zero risk) |
| `--dry-run` | Preview without modifying anything |
| `--discover` | Map filesystem, create/update FILESYSTEM.md |
| `--analyze` | Tier 3 analysis only (read-only) |
| `--json` | Output as JSON |

## Backup Retention Rule

The VPS should keep **only one historical backup at a time**, plus current live data. Genie owns detecting and reporting violations.

Historical backup candidates include:
- `/root/backup/*`
- `/root/backups/*`
- `/root/.hermes/state-snapshots/*`
- `/root/.hermes/migrations/*/backups/*`
- profile DB `.bak-*` files under `/root/.hermes/profiles/*/commons/db/`

Default behavior:
1. local full `state.db` backup copies are invalid unless explicitly requested
2. keep the newest valid historical backup
3. mark older or invalid historical backups as reclaimable
4. do not create an additional local historical backup unless replacing the retained one
5. after a replacement backup is verified, remove the superseded backup
6. report path, size, timestamp, and reclaimed/expected space

For the full audit workflow, see `references/root-audit-and-backup-retention.md`.

## Procedure

1. **Locate the script** — check these paths in order:
   - `/root/.hermes/profiles/indigo/skills/ocas-genie/scripts/genie.py` (profile — note `ocas-` prefix)
   - `/root/.hermes/profiles/indigo/scripts/genie.py` (profile scripts dir — alternate location)
   - `/root/.hermes/skills/ocas-genie/scripts/genie.py` (skill-bundled)
   - Use absolute paths only — never `~` in cron context
   - **Gotcha**: the skill folder is `ocas-genie/`, not `genie/`. A literal read of the old path will fail with ENOENT.

2. **Assess** — run `--assess` to identify targets

3. **Execute** — run `--clean` (or `--clean --dry-run` to preview)

4. **Report** — verify with `df -h /` after cleanup

## Manual / Investigative Targets

Some large disk consumers require an audit pass before cleanup because they may be live data, backups, or duplicate worktrees. Use `references/root-audit-and-backup-retention.md` for the full procedure.

1. **Manual backups** (`/root/backup/`, `/root/backups/`) — enforce one historical backup total on the VPS. Keep newest valid backup; older backups are reclaimable.
2. **Pre-update snapshots** (`/root/.hermes/state-snapshots/`) — count as historical backups. Keep only the newest valid one once the post-update gateway is confirmed healthy.
3. **Migration backups** (`/root/.hermes/migrations/*/backups/`) — count as historical backups. Keep only if they are the newest/only valid historical backup.
4. **Pre-migration `.bak-*` files** (`profiles/*/commons/db/*/.bak-*`) — count as historical backups. Reclaim older ones once the live DB has newer writes and integrity checks pass.
5. **Duplicate git repos** — compare remote + HEAD before removing. Same remote and same HEAD = duplicate candidate.
6. **Browser caches** (`~/.cache/camoufox/`) — safe to delete when no creating process is running.
7. **Stale /tmp extracts** (`/tmp/camoufox-*/`, `/tmp/uc_*/`, `/tmp/body_*`) — safe to delete when no creating process is running.
8. **state.db VACUUM** — Genie does not run `VACUUM` automatically. Report bloat/freelist only.

When disk is critically high, flag these in the report even if `--clean` cannot auto-clean them.

## Safety Rules

- NEVER modify `state.db` directly — Tier 3 analysis only. Note: `/root/.hermes/state.db` is typically a symlink to a profile's DB (e.g., `→ profiles/indigo/state.db`). Resolve symlinks before analyzing or reporting.
- Enforce the one-historical-backup rule: keep current live data plus the newest valid historical backup; older backup copies/snapshots are reclaimable.
- NEVER delete files without compressing first (Tier 2+) unless they are superseded historical backups being removed under the one-backup retention rule.
- ALWAYS report what was done
- If disk usage is below 50%, report "no action needed"
- If any operation fails, report the error and continue
- If a snapshot deletion fails, leave the snapshot in place and report at the end
- The most recent snapshot is always preserved (never auto-deleted)
- **Do not preserve multiple backups**: The VPS policy is one historical backup plus current live data. Count all historical backup classes together (`/root/backup`, `/root/backups`, snapshots, migration backups, `.bak-*`) and report/remove older valid copies.

## Error Handling

- **gzip failures:** If `gzip -t` reports a `.gz` is corrupt but the original exists, keep the original and regenerate. Never delete the original while the gzip is corrupt.
- **Concurrent gzip race:** Only one gzip per file. A timed-out foreground command can leave a background shell that spawns a second gzip. Verify with `gzip -t`, then remove the original.
- **Disk-at-100%:** Focus on immediate space recovery (Tier 1 cleanup) before any backup workflow.
- **Snapshot deletion failure:** Leave the snapshot in place, report the error, continue to next target.

## What Genie Cleans

### Tier 1 — Zero Risk (auto-executed)
1. **Stale state snapshots** — directories older than 7 days (most recent always preserved)
2. **Old log files** — compress after 7 days, delete compressed after 30 days
3. **Old cron output** — compress after 7 days
4. **Stale `/tmp` files** — delete after 24 hours
5. **Package caches** — pip, uv, npm (all rebuildable)
6. **Browser profile caches** — `~/.cache/camoufox/` (rebuildable, often 1+ GB)
7. **Inactive git clones** — `/root/projects/` dirs untouched >5 days with confirmed remote (safe to re-clone)

### Tier 2 — Low Risk (requires confirmation)
1. **Session JSON duplicates** — compress after 14 days (data also in `state.db`)

### Tier 3 — Analysis Only (never auto-executes)
1. **state.db bloat analysis** — reports DB size, freelist waste
2. **Large directories** — reports git checkpoints, commons/data, commons/db for manual review

## Configuration

Genie reads all behavioral settings from `config.yaml` under
`skills.config.genie.*`. Set them with `hermes config set` or by editing
`config.yaml` directly. CLI flags (e.g. `--dry-run`) override config at runtime.
Any setting not present falls back to the built-in default shown below.

| Config key (`skills.config.genie.*`) | Default | Description |
|---|---|---|
| `snapshot_max_age_days` | 7 | Delete snapshots older than N days |
| `log_compress_age_days` | 7 | Compress logs older than N days |
| `log_delete_age_days` | 30 | Delete compressed logs older than N days |
| `cron_output_compress_age_days` | 7 | Compress cron output files older than N days |
| `session_compress_age_days` | 14 | Compress session JSONs older than N days |
| `tmp_stale_hours` | 24 | Delete /tmp files older than N hours (0 to skip) |
| `git_clone_max_age_days` | 5 | Delete git clones in /root/projects/ untouched for N days (must have remote) |
| `dry_run` | false | If true, only report — don't delete/compress |
| `filesystem_md` | (empty) | Optional override path for FILESYSTEM.md |
| `allow_local_state_db_backup` | false | If true, local full state.db copies count as valid retained backups |

## Verification

- `df -h /` — check disk usage dropped
- `du -sh /root/.hermes/` — check .hermes size
- Session search still works (confirms state.db intact)

## Support File Map

| File | When to read |
|---|---|
| `references/root-audit-and-backup-retention.md` | Root disk-spike investigation, duplicate repo checks, and one-historical-backup retention rule |
| `references/genie-gotchas.md` | Before first production run or when debugging |
| `references/operational-notes.md` | Real-world examples and case studies |
| `references/os-walk-pitfall.md` | Debugging nested directory traversal issues |
| `references/session-2026-05-29-disk-recovery.md` | Disk emergency case study |
| `references/snapshot-backup-redaction.md` | Backing up snapshots to git/LFS |
| `references/snapshot-structures.md` | Snapshot format breakdown |
| `references/state-db-compaction.md` | Tackling state.db bloat |
| `references/state-db-size-breakdown.md` | State DB composition analysis |
| `references/disk-growth-patterns.md` | Recurring disk hogs: pre-update snapshots, /root/backup/, browser caches |
| `references/state-db-retention.md` | State DB retention policy: audit all instances, keep current + one backup, delete oldest first |
| `references/self-update-genie.md` | Self-update hash comparison procedure |
| `references/repo-path-conventions.md` | Repo path convention — all remote clones under `projects/github*` |
| `scripts/genie.py` | Main cleanup script |
| `scripts/genie_rebuild_fts.py` | FTS rebuild after restoring no-FTS backup |
