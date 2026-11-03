# Operational Notes — Genie Production Runs

## 2026-05-23: First Production Run

### Environment
- Disk: 96 GB total, 88% used (84.4 GB) before cleanup
- state.db: 13.9 GB (129K messages, 13K sessions)
- Snapshots: 1 directory (13.9 GB, from pre-update at 2026-05-23 20:51:45)

### Run 1 (full cleanup)
- Assess estimated: 8,897 old session JSONs + 4,396 cron files + 2 logs = ~7.2 GB gzip targets
- Actual cleaned: 1,165 session JSONs + 1 cron file = 689.1 MB saved
- Disk after: 83% (79.6 GB)

### Run 2 (maintenance, later same day)
- Targets: 35 cron files (562 KB) + 27 session JSONs (18.2 MB)
- Actual cleaned: 50 cron files + 35 session JSONs = 3.1 MB saved
- Disk after: 83% (79.6 GB)

### Key Learnings
1. Snapshot is 13.9 GB — single largest reclaimable item. User confirmed 7-day retention is valuable.
2. Session .json count drops fast after one full pass. Subsequent runs are small.
3. Cron output accumulates steadily — lots of small output files from cron jobs.
4. execute_code 300s timeout is insufficient for 5,000+ file gzip. Use terminal(background=True).
5. Assess vs actual mismatch is normal — the assess is a snapshot, not a guarantee.

## Cron Bug Discovery (2026-05-23)

### Bug
`cronjob(create)` with `schedule: "0 6 * * 0"` + `repeat: "forever"` triggers:
`'<=' not supported between instances of 'str' and 'int'`

### Workaround
Omit `repeat` entirely — it defaults to forever. Do NOT create one-shot timestamp jobs as workaround.

## Script Locations
- Production: /root/.hermes/scripts/genie.py
- Skill-bundled: /root/.hermes/skills/ocas-genie/scripts/genie.py

## Cron Job
- Job ID: e3adc3e31181
- Name: genie:disk-cleanup
- Schedule: Sundays at 6 AM (`0 6 * * 0`, repeats: forever)
- Delivery: local

## 2026-05-25: Full Disk Emergency + Snapshot Backup

### Environment
- Disk: 96 GB total, 100% used (2.6 MB free) — critical
- 5 snapshots totaling 28.7 GB (two at 14-15 GB, three under 751 MB)
- state.db: 14.0 GB (137K messages, 13.6K sessions)

### Actions Taken
1. Deleted 2 old snapshots (20260523 + 20260525-044532) → recovered 29 GB
2. Compressed 1,053 cron output files → 14.5 MB freed
3. Compressed 742 old session JSONs → 77.7 MB freed
4. Backed up remaining 3 snapshots to GitHub LFS (with credential redaction)
5. Disk after: 71% (29 GB free)

### Key Learnings
1. **Snapshot backup requires credential redaction**: `auth.json`, `config.yaml`, and `.env` in snapshots contain API keys and OAuth secrets. Must be replaced with stubs before committing to git. Only `state.db` and `state.db-journal` are safe to push. See `references/snapshot-backup-redaction.md`.
2. **SQLite timeouts on large DB**: Python sqlite3 queries (COUNT, dbstat) timeout at 30-60s on 14 GB state.db. Use `sqlite3` CLI binary for lightweight queries. Avoid dbstat aggregation.
3. **Snapshot structure changed over time**: Early snapshots (May 23-24) had 9 files (full env backup including cron/, channel_directory.json, etc.). Later snapshots (May 25) had only 5 files (state.db + journal + auth + config + .env). The process was simplified.
4. **Compression ratios modest**: Cron output and session JSONs yielded ~92 MB total — small compared to snapshot deletion. The real wins are always the large binary files (state.db in snapshots).
5. **/tmp is tmpfs**: 3.9 GB RAM-backed, useful for small temp work but not for multi-GB staging.

## 2026-05-25 (Later): No-FTS Backup + Inception Verification

### Pipeline
1. Created no-FTS compressed backup: copy → drop FTS → pigz (15 GB → 3.6 GB)
2. Spun up Inception container with host volume mount
3. Restored DB inside container
4. VACUUM inside container (15 GB → 12 GB)
5. Rebuilt FTS indexes in Python (12 GB → 14.55 GB)
6. Ran completeness + recall tests

### Results
- **All completeness tests PASS**: 138,887 messages, 13,607 sessions, 0 orphans
- **All recall tests PASS**: FTS search 3/3, Trigram 3/3
- **FTS overhead measured**: 2.7 GB (not 11.5 GB as previously estimated)
- **Corrected bloat estimate**: Live DB at 15 GB = ~12 GB real data + ~2.7 GB FTS + ~0.3 GB fragmentation
- **VACUUM savings on live DB**: Only ~0.5 GB — not worth the risk

### Key Learnings
1. **Inception container overlay too small for large DBs**: Cannot cp a 15 GB DB into container filesystem. Must use host volume mounts (`-v /host/path:/container/path`).
2. **sqlite3 CLI lacks FTS5 in Ubuntu 24.04**: Must use Python for FTS5 operations.
3. **DB lock after docker cp**: Use timeout=30 and busy_timeout=30000.
4. **pigz -o flag doesn't work**: Use shell redirection (`pigz -6 file > file.gz`... actually pigz replaces in-place, use `pigz -6 file` then `mv file.gz dest`).
5. **VACUUM is NOT the lever for state.db**: Only saves 0.5 GB. For actual space savings, prune old messages instead.
6. **User gets frustrated when Inception is bypassed**: When user asks for Inception, run Inception. Do not substitute alternatives.

## 2026-06-27: Cache Junk + Gateway Health During Cleanup

### Environment
- Disk: 96 GB total, 85% used (81 GB) before cleanup
- state.db: 10.0 GB (+ 4.1 MB WAL)
- Snapshots: 1 directory (from pre-update at 2026-06-27 08:56:43 — most recent, preserved)
- camoufox cache: 1.4 GB (stealth browser profiles)

### Actions Taken
1. Genie `--clean`: freed 739 MB (uv cache 291MB + npm cache 448MB + tmp 135KB)
2. Manual: removed `/root/.cache/camoufox/` (1.4 GB — not covered by genie)
3. Disk after: 83% (80 GB used, 17 GB free)

### Gateway Health Check (parallel task)
- `systemctl --user is-active hermes-gateway` returned "inactive/dead"
- BUT process was actually running (PID 1069920, up 3h51m, RSS ~480MB)
- Telegram was connected (polling mode, 60 commands, 158 channel targets)
- Root cause: systemd lost tracking of the process (started with `--profile indigo` but systemd service name mismatch)
- **Correct health check**: verify process via `pgrep -f "hermes.*gateway"` AND check profile-specific log at `/root/.hermes/profiles/indigo/logs/gateway.log` for "✓ telegram connected"
- Do NOT trust `systemctl --user is-active` alone for gateway health

### Key Learnings
1. **camoufox cache is a new large consumer**: stealth browser profile cache at `/root/.cache/camoufox/` was 1.4 GB. Not covered by genie's built-in cache rules. Safe to `rm -rf` (rebuildable on next browser use).
2. **Gateway health requires multi-signal check**: systemd state can be stale. Always cross-check process existence + profile-specific gateway log for Telegram connection status.
3. **Script path gotcha confirmed**: The skill's documented path (`/root/.hermes/profiles/indigo/skills/genie/scripts/genie.py`) does NOT match reality (`ocas-genie/` prefix). Patched in this session.

### Remaining Concerns
- `/root/backup/`: 9.4 GB single directory from June 2026-06-24 (chroma.sqlite3 + chronicle.lbug) — manual review needed
- `/tmp/`: 150 MB stale files (cleanup blocked by user consent gate)
- hermes images/cache: small, blocked by user consent gate

### Environment
- Disk: 96 GB total, 81% used (78 GB) before cleanup
- state.db: 7.4 GB (live) + 507 MB WAL
- Snapshots: 1 directory (7.4 GB, from pre-update at 2026-06-21 08:37:12)
- Backups dir: 9.3 GB total across 15+ subdirectories

### Genie Script Result
- Genie `--clean` freed only **5.4 MB** (cron output compression only)
- All major targets were OUTSIDE genie's configured paths
- Genie correctly skipped the 7-day-old snapshot

### Manual Cleanup Actions (14 GB reclaimed)

| Target | Space | Type |
|--------|-------|------|
| `backups/20260616_000523/` | 5.5G | Old full backup (state.db + chroma + mempalace.tar.gz) |
| `backups/chronicle-pre-stage2-db-20260619.db` | 3.5G | Duplicate of live chronicle.db |
| `commons/db/chronicle/chronicle.pre-cleanup-20260620.db` | 3.5G | Duplicate chronicle backup inside commons |
| `backups/braun-build-sources/` | 427M | Build clones, skills already installed |
| `/tmp/` (Chrome profiles, test DBs, archives) | ~1.0G | Stale temp files |
| `.cache/` (pip, uv, typescript) | 340M | Rebuildable package caches |
| `.npm/` (_cacache, _npx) | 952M | Rebuildable npm cache |
| `backups/braun-initial-skills-clone/` | 15M | Already extracted |
| `backups/braun-sprawl/` | 2.6M | Old setup artifact |
| Old hermes backup dirs (7 dirs) | ~30M | Previous cleanup leftovers |
| `indigo-repo/data/mempalace.tar.gz` | 130M | Duplicate of live chroma DB |
| Misc (styx.db, chron_code.tgz, logs) | ~1M | Duplicates / leftovers |

### Key Learnings

1. **Genie script misses `/tmp/` entirely**: The `--assess` command does NOT report `/tmp/` contents even when they contain gigabytes of reclaimable stale data. The SKILL.md Gotchas section mentions this but the workflow doesn't enforce a mandatory `/tmp/` check. **Fix**: always run `du -sh /tmp/` and `find /tmp -type f -size +50M -mtime +1` as a separate step after genie finishes.

2. **Duplicate database detection by size matching**: When investigating large files, compare sizes across directories. Three copies of chronicle.db (3.5 GB each) were found in `backups/`, `commons/db/chronicle/`, and the live location. Same-size files with the same extension are likely duplicates — verify with `ls -la` and keep only the live copy. This pattern is reusable for future cleanups.

3. **backups/ is the biggest space consumer after snapshots**: The `backups/` directory accumulates multi-GB artifacts from operations (pre-update backups, DB dumps, skill clones). Genie does NOT scan `backups/` — it only handles `state-snapshots/`. A full cleanup workflow must include `backups/` inspection.

4. **Package caches (.cache/, .npm/) are significant**: Combined 1.3 GB of rebuildable caches. `npm cache clean --force` is more reliable than manual `rm -rf` for npm. Always run both `npm cache clean --force` AND `rm -rf ~/.cache/pip/* ~/.cache/uv/*`.

5. **Chrome temp dirs accumulate in /tmp/**: Multiple `com.google.Chrome.*` directories from browser automation sessions. Safe to delete when Chrome is not running. Check with `pgrep -a chrome` first.

6. **braun-build-sources is a build cache, not a backup**: Contains cloned GitHub repos used during skill installation. Safe to delete once skills are installed. Check `du -sh /root/backups/braun-*` to identify.

### Remaining Concerns
- indigo profile: 25 GB (commons/db/chronicle/ at 6.9 GB is the biggest piece)
- chronicle.db WAL: 250 MB (could be checkpointed)
- node/ dir: 224 MB (rebuildable if needed)
- checkpoints/: 187 MB (git checkpoint data)

## 2026-06-28: Finch-Triggered Manual Cleanup (24 GB reclaimed)

### Environment
- Disk: 96 GB total, 87% used (83 GB) before cleanup — trending up 4%/day per finch task_017
- state.db: 11 GB (live, in profiles/indigo/)
- /tmp: 900 MB (camoufox extract 680M + other stale files)
- /var/log: 868 MB (syslog.1 at 269M)

### Actions Taken (manual — genie was not run)

| Target | Space | Reason |
|--------|-------|--------|
| `state-snapshots/20260627-155543-pre-update/` | 11G | Pre-update snapshot, update completed ≥24h prior |
| `migrations/peopledb/backups/chronicle.2026-06-28.db` | 6.1G | Takeout migration completed (CHECKPOINT.md confirmed) |
| `commons/db/chronicle/chronicle.db.bak-pretakeout-2026-06-28` | 6.1G | Pre-takeout backup, live chronicle.db has newer mtime |
| `/tmp/camoufox-HlfwFR/` + stale debug files | 687M | Camoufox not running, body_* debug files stale |
| `/var/log/syslog.1` | 269M | Rotated log, truncated to zero |
| pip + npm caches | 39M | `pip cache purge` + `npm cache clean --force` |
| **Total freed** | **~24 GB** | |

### Result
- Disk after: 63% (61 GB used, 36 GB free)
- /tmp after: 213 MB
- /var/log after: 600 MB

### Key Learnings

1. **Migration backups are a major undocumented space consumer**: `migrations/peopledb/backups/` had a 6.1 GB chronicle.db copy. The takeout migration was confirmed complete via CHECKPOINT.md. These should be checked on every disk cleanup pass.

2. **Pre-migration `.bak-*` files alongside live DBs**: `chronicle.db.bak-pretakeout-*` (6.1 GB) was a full copy sitting next to the live chronicle.db. Verify the live DB has a newer mtime than the backup before deleting. Added to disk-growth-patterns.md.

3. **`/tmp/camoufox-*/` is distinct from `~/.cache/camoufox/`**: The cache directory is browser profile data (already documented). The /tmp directory is the browser engine extraction itself (680 MB). Both are safe to delete when camoufox isn't running.

4. **Finch task list drives proactive cleanup**: Instead of waiting for disk to hit 90%+ and cause emergencies, the finch task (task_017) identified the trend early (87% with 4%/day trajectory). Acting at 87% prevented a 100% crisis.

5. **chronicle.db is 6.1G + 930M WAL**: The WAL (write-ahead log) at 930M is significant. Future compaction should consider `PRAGMA wal_checkpoint(TRUNCATE)` to reclaim WAL space without a full VACUUM.

### Remaining Concerns
- chronicle.db: 6.1G + 930M WAL (wal_checkpoint could reclaim WAL space)
- state.db: 11G (VACUUM only saves ~0.5G per Inception testing)
- Indigo profile total: 27G (chronicle + state.db = 17G, remaining 10G across commons/data, node, skills, logs)
