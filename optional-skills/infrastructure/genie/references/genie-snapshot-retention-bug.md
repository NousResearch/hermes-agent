# genie snapshot retention bug

## What broke

On 2026-07-12 a `--clean` run deleted the 13 GB pre-update rollback snapshot
from `state-snapshots/`. That snapshot was the only way back to the pre-update
system state. After the run, `clean_snapshots` printed `snapshots: freed 0.0 B`,
but that line is meaningless here because the directory was already empty,
so there was nothing left for it to preserve.

## Root cause

`clean_backup_retention` ranks every historical backup candidate together
(`keep = valid[:1]`): snapshots, `/root/backup` copies, migration backups,
`.bak-*`, the lot. The most-recent pre-update `state-snapshot` is itself one of
those candidates. When some *other* backup file had a newer mtime (a
`transactions.db` copy, in this case), the snapshot fell below the cutoff,
landed in `reclaim`, and was `shutil.rmtree`'d. That directly violates the
safety rule that the most recent snapshot is always preserved.

## Fix (applied 2026-07-16)

`backup_retention_plan()` now drops `state-snapshot` candidates before the
cross-class ranking. Snapshots are still preserved independently by
`clean_snapshots()`, which always keeps the newest one. The combined "keep one
historical backup" rule can no longer reach into the snapshot directory.

## Reproduction recipe

```bash
# In a throwaway HERMES_HOME with a populated state-snapshots/ dir:
python3 scripts/genie.py --clean --dry-run
# Confirm the snapshot path is reported as preserved, not reclaimed.
ls -la "$HERMES_HOME/profiles/<profile>/state-snapshots"
```

If the snapshot directory is non-empty after a dry run, the fix holds. A real
run (`--clean`, no `--dry-run`) should still leave it intact.
