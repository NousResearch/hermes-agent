# Disk Growth Patterns — VPS

Common disk hogs on Indigo's VPS that cause rapid growth after updates or normal operation.

## Pre-Update Snapshots (state-snapshots/)

After `hermes update`, a full pre-update snapshot is created in `/root/.hermes/state-snapshots/YYYYMMDD-HHMMSS-pre-update/`. The largest file is always `state.db` — a full copy of the live state database.

**Indigo's state.db is 11 GB** (session history, FTS index). A single snapshot = 11 GB on disk.

**Rule of thumb**: if state.db is N GB, each snapshot costs N GB. These are NOT tracked by genie's FILESYSTEM.md cleanup — they live outside the manifest.

**Cleanup**: safe to delete once the post-update gateway is confirmed healthy (usually 24h after update). Keep only the most recent.

```bash
ls -la /root/.hermes/state-snapshots/
rm -rf /root/.hermes/state-snapshots/20260620-XXXXXX-pre-update/  # old one
```

## /root/backup/

Manually created backups in `/root/backup/YYYYMMDD_HHMMSS/` follow the same pattern: state.db dominates. On June 24 the backup was 9.4 GB (9.0 GB state.db + 253 MB chroma.sqlite3 + smaller DBs).

**These are separate from state-snapshots/** and from genie's retention. Ask the user before deleting — they may want to keep one historical restore point.

```bash
du -sh /root/backup/*/ | sort -rh
```

## Migration Backups (migrations/)

`/root/.hermes/migrations/` holds backup copies of databases created during migration scripts (e.g., peopledb takeout enrichment). The largest is typically a chronicle snapshot matching the live DB size.

**June 2026 example**: `migrations/peopledb/backups/chronicle.2026-06-28.db` was 6.1 GB — a full copy of chronicle.db from the takeout import. The migration completed successfully, making this backup stale.

**Cleanup**: safe to delete once the migration that created them is confirmed complete. Check for a `CHECKPOINT.md` or similar status marker in the migration directory.

```bash
du -sh /root/.hermes/migrations/*/backups/ 2>/dev/null
cat /root/.hermes/migrations/*/CHECKPOINT.md
```

## Pre-Migration Database Backups (`.bak-*` files)

Migration scripts sometimes create `.bak-*` copies of live databases before modifying them. These live alongside the production DB and are NOT tracked by genie's cleanup.

**June 2026 example**: `profiles/indigo/commons/db/chronicle/chronicle.db.bak-pretakeout-2026-06-28` was 6.1 GB — a full copy of chronicle.db before the takeout import. The import completed, making this backup stale.

**Cleanup**: safe to delete once the migration is complete. Verify the live DB has a recent modification time (confirming writes happened after the backup timestamp).

```bash
# Find all .bak files in profiles
find /root/.hermes/profiles/ -name "*.bak*" -ls
# Compare timestamps
ls -lh /root/.hermes/profiles/indigo/commons/db/chronicle/
```

## /tmp Stale Extracts

Beyond genie's 24-hour /tmp cleanup, some tools extract large directories that persist:

- `/tmp/camoufox-*/` — full browser engine extraction (680 MB observed). Not tracked by npm/pip cache cleanup.
- `/tmp/uc_*/` — undetected-chromedriver extracts
- `/tmp/body_*` and `/tmp/b_*` — HTTP response bodies from debugging (nubank.com.br observed)

These are safe to delete when the tool that created them is not running.

```bash
# Check if camoufox is in use before deleting
ps aux | grep camoufox
rm -rf /tmp/camoufox-*
```

## Large Caches Not Tracked by Package Managers

- `~/.cache/camoufox/` — browser profile cache for stealth browsing. 1.4 GB in this session. Safe to delete entirely; rebuilds on next browser use.
- `~/.cache/uv/` — Python package cache (genie handles this)
- `~/.cache/pip/` — pip cache (genie handles this)
- `~/.npm/` — npm cache (genie handles this)

## Symlink Indirection

`/root/.hermes/state.db` is a symlink to the active profile's DB (currently `→ /root/.hermes/profiles/indigo/state.db`). When auditing "how many state.db files exist", resolve symlinks first — `find / -name state.db` may report the same file twice (once as symlink, once as target). Use `readlink -f` and `ls -li` (inode check) to deduplicate.

To identify what grew since last check:

```bash
# Top-level summary
du -sh /root/*/ /root/.*/ 2>/dev/null | sort -rh | head -20

# Deep dive into .hermes
du -sh /root/.hermes/*/ /root/.hermes/profiles/*/ 2>/dev/null | sort -rh

# Inside a specific profile
du -sh /root/.hermes/profiles/indigo/*/ 2>/dev/null | sort -rh
```

**Indigo's typical breakdown** (June 2026):
- .hermes/ : 33 GB (profiles 22 GB + state-snapshots 11 GB)
- backup/ : 9.4 GB
- projects/ : 3.5 GB
- hermes-agent/ : 2.8 GB
