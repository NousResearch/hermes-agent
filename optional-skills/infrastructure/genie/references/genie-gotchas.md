# Genie — Gotchas

Operational lessons learned from running Genie in production. Read this after your first production run or when debugging unexpected behavior.

## Snapshot Gotchas

- **Snapshot directory mtime**: Directory mtime updates when files are created inside it. Genie checks the oldest file inside the snapshot, not the directory itself.
- **Snapshot structure varies by era**: Snapshots from different periods have different internal structures. Older snapshots may contain full environment backups and are much larger. Newer snapshots may only contain `state.db` and a few config files. See `references/snapshot-structures.md` for the breakdown.
- **Snapshot retention is user-managed**: ALWAYS ask the user before deleting a snapshot, even if it exceeds the age threshold. Snapshots are the only rollback mechanism. **EXCEPTION — Emergency disk-100%**: When `/` is at 100% and blocking critical operations (MCP auth, cron execution), delete snapshots WITHOUT user confirmation to restore functionality. State-snapshots are pre-update rollback points — they are the safest and largest deletion target. Document what was deleted for audit.
- **Most recent snapshot is NEVER auto-deleted**: Genie always skips the newest snapshot (by mtime) regardless of age. The most recent snapshot is the most valuable rollback point — it's the one most likely to be needed. Only older snapshots beyond `max_age_days` are candidates for deletion.
- **Snapshot backup requires credential redaction**: When backing up snapshots to git/GitHub LFS, `auth.json`, `config.yaml`, and `.env` files contain API keys, OAuth tokens, and secrets. NEVER commit these as-is. Replace with redacted stubs before committing. Only `state.db` and `state.db-journal` are safe to push directly. See `references/snapshot-backup-redaction.md`.
- **Snapshot compression as space recovery**: When disk is critically full and snapshot deletion is unacceptable (user-managed rollback), compressing `state.db` inside snapshots is the correct middle ground — it preserves the rollback capability while typically saving 50-75% of the space. After gzip, **always remove the original uncompressed file** to actually reclaim space. Verify the `.gz` passes `gzip -t` before removing the original.

## Compression Gotchas

- **Gzip empty file artifact**: After compressing a log, a 0-byte original may remain. The .gz has all the data.
- **Concurrent gzip race**: When compressing large files, ensure only ONE gzip process targets a file at a time. A timed-out foreground command can leave a background shell that later spawns a second gzip, resulting in both `file` and `file.gz` coexisting — zero space reclaimed. Use `gzip -t` after compression completes to verify, and if both file and .gz exist, remove the uncompressed original (after verifying the .gz is valid).
- **gzip -t timeouts scale with file size**: For multi-GB files, `gzip -t` can take minutes. A 5G file may need 60-120s, a 10G file 3-5min. Use `timeout 300 gzip -t <file>` or larger. Do NOT assume a gzip is corrupt just because the check takes a long time — it's just reading every byte.
- **Corrupt .gz with existing original**: If `gzip -t` reports a `.gz` file is corrupt but the uncompressed original still exists, NEVER delete the original. The corrupt gzip means the compressed copy lost data. Keep the original and regenerate the gzip (`gzip <original>`). Only delete the original after verifying the new `.gz` passes `gzip -t`. Deleting the original while the gzip is corrupt causes irreversible data loss.

## Filesystem Gotchas

- **Off-hermes .cache consumers**: When `/root/.cache` is large, the usual culprits are: `ms-playwright/` (browser binaries, 1G+ rebuildable), `puccinialin/` (cargo/rustup cache, rebuildable), `chroma/` (embedding DB — check before deleting). These are all Tier 1 safe rebuildable caches.
- **`/tmp` not in assess output**: The `genie.py --assess` command does NOT report `/tmp/` contents, even when they contain gigabytes of reclaimable stale data. When assess shows zero or minimal Tier 1 targets but disk is still high, always manually check `/tmp/` with `du -sh /tmp/` and `find /tmp -type f -size +50M -mtime +1`. All `/tmp/` files older than 24h are Tier 1 zero-risk deletions.
- **`backups/` is outside genie's scope**: The genie script only scans `state-snapshots/`, `logs/`, `cron/output/`, and `sessions/`. It does NOT scan `backups/` — which is often the largest space consumer after snapshots. When genie's `--assess` shows minimal targets but disk is still high, always run `du -sh /root/backups/*/` and `ls -la /root/backups/` to inspect.
- **Nested directory traversal**: Ensure `os.walk()` is used for recursive directory traversal. Always use `dirpath` from `os.walk()` — never rejoin filenames against the root path. See `references/os-walk-pitfall.md`.
- **FILESYSTEM.md manifest can report stale/phantom targets**: The discovered targets section may show large file counts that don't match reality. Always verify manifest targets with direct filesystem checks (`find ... | wc -l`, `du -sh`) before reporting to the user. Run `--discover` to refresh the manifest if numbers seem off.
- **Built-in snapshot path default is wrong for profile-scoped setups**: The default `GENIE_SNAPSHOTS_PATH` is `/root/.hermes/state-snapshots`, but in profile-scoped setups snapshots live at `/root/.hermes/profiles/indigo/state-snapshots`. The FILESYSTEM.md manifest corrects this, but if the manifest is missing or stale, genie won't find snapshots.

## State DB Gotchas

- **FTS footprint is significant**: FTS trigram indexes add substantial overhead proportional to message count. For large DBs (100K+ messages), expect FTS overhead in the multi-GB range. FTS can be dropped and rebuilt on demand — it contains no unique data, only search indexes.
- **VACUUM INTO rebuilds FTS indexes**: `VACUUM INTO` on a DB with FTS tables rebuilds all FTS indexes during the copy. The result is as large as the input — it does NOT compact FTS overhead. To truly compact: copy the DB, drop FTS tables on the copy, then `.backup` to a new file. See `references/state-db-compaction.md`.
- **SQLite timeouts on large state.db**: Large state.db files cause Python sqlite3 queries (especially `dbstat` and `COUNT(*)`) to timeout at 30-60s. Use the `sqlite3` CLI binary for lightweight queries. Avoid `dbstat` aggregation on large DBs.

## Operational Gotchas

- **Large file counts**: With 5,000+ files to gzip, use `terminal(background=True, notify_on_complete=True)`.
- **Disk-at-100% blocks operations**: When disk is at 100%, Genie cannot write journal files, create temp files, or stage data. Check `df -h /` first — if at 100%, focus on immediate space recovery (Tier 1 cleanup, snapshot deletion) before attempting any backup workflow.
- **Script/version desync during self-update**: Always compare script hashes even when versions match. See `references/self-update-genie.md`.
- **`~` path resolution in cron context**: When running as a cron job, `HOME` is set to the profile-scoped path, not `/root/`. **Always use absolute paths** (`/root/.hermes/...`) in cron context.
- **GitHub default branch is `main`**: The genie repo's default branch is `main`, not `master`. Using `master` in raw GitHub URLs returns 404 or stale content.
- **Script path deduplication**: The three script paths in Step 0 may resolve to the same file (hardlink or symlink). `cp` between them will fail with "same file" — this is expected. Use `cmp` or `ls -i` (inode check) to verify before assuming they're independent copies.
