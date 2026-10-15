# State DB Retention Policy

User's stated policy (June 2026): keep only the current `state.db` and **one** backup copy. Delete the oldest extra copies before running genie.

## Audit: find all state.db instances

Always start by enumerating every `state.db` on the full filesystem (not just `/root`):

```bash
find / -name "state.db" -type f 2>/dev/null
```

**Symlink dedup**: `/root/.hermes/state.db` is often a symlink to a profile DB. Resolve with `readlink -f` and check inodes (`ls -li`) to determine true duplicates. Multiple found paths may point to the same underlying file.

## Identify candidates for deletion

For each found file, check:

1. **Current live DB** — the one inside the active profile (usually `/root/.hermes/state.db` or `/root/.hermes/profiles/<name>/state.db`). KEEP.
2. **Other profiles** — per-profile copies. Keep only if the profile is still active. Ask before deleting.
3. **Backups** — dated directories in `/root/backup/`. Keep the newest, delete older ones.
4. **Snapshots** — `/root/.hermes/state-snapshots/*/state.db`. Keep the most recent (genie never auto-deletes these).

## Deletion sequence

1. Run `find / -name state.db -type f 2>/dev/null` — report all instances to user
2. Note sizes via `du -sh` and dates via `stat --format='%y'`
3. Ask user which to delete (or follow "delete oldest" instruction)
4. Delete the chosen candidates
5. Run genie to reclaim ancillary space (caches, logs, tmp)

## User preference

- "Only need the current state.db and only one backup"
- "Delete the oldest one" — unambiguous instruction
- Then immediately: "run genie to free up space"

This two-step pattern (prune DBs, then genie) is the user's standard maintenance flow. Genie alone won't touch state.db or backups — those require human judgment and should always be a separate step before genie runs.
