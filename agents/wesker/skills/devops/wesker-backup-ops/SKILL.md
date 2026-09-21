---
name: wesker-backup-ops
description: Operate daily Hermes backups — retention policy, gap investigation, health monitoring, and script maintenance.
version: 1.0.0
adoption_status: permanent
---

# Wesker Backup Operations

## Backup architecture

- **Script:** `/home/kensei/scripts/daily-hermes-backup.sh`
- **Cron:** `0 3 * * *` via crontab (system cron, not Hermes scheduler — exists before Hermes was installed)
- **Destination:** `/home/kensei/backups/daily/`
- **Weekly archive:** `/home/kensei/backups/weekly/` (created first Sunday after script update)
- **Log:** `/home/kensei/backups/daily/backup.log`
- **Retention:** 7 daily + 4 weekly (Sunday promotion)
- **Watchdog:** `backup-health-check.sh` (cron `01c927f8a1a7`, 06:00 daily, no_agent, delivers to `discord:#ops`, silent when healthy)

## Script improvements (2026-06-12)

The original script had `2>/dev/null || true` which silently swallowed all tar errors. If tar OOM'd or hit a transient issue, the backup silently failed — the script printed nothing and the `|| true` absorbed the non-zero exit.

**Fix:** Capture stderr to a temp file, check exit code explicitly, print error and exit non-zero on failure. This means cron will log the actual failure message.

**Retention:** Was KEEP_COUNT=5 (very tight). Now KEEP_DAILY=7 with weekly promotion on Sunday (KEEP_WEEKLY=4). Safe margin for a 2TB disk.

## Gap investigation process

When a gap is found (missing backup dates):

1. **Check backup.log** — does it show the gap as missing entries or as failed attempts?
2. **Check crontab** — `crontab -l` — is the entry still there?
3. **Check cron service** — `systemctl is-active cron && systemctl status cron --no-pager -l`
4. **Check script permissions** — `ls -la /home/kensei/scripts/daily-hermes-backup.sh`
5. **Check system load** — look for OOM kills (dmesg), heavy disk, or concurrent updates in `/home/kensei/backups/` (dated backup dirs indicate heavy activity)
6. **Check log for the gap period** — missing log entries with no error = cron didn't fire (possible OOM kill of cron, transient system issue)
7. **Check journal** — `journalctl -u cron --since "YYYY-MM-DD 03:00" --until "YYYY-MM-DD 04:00"` (may need sudo)

Most common cause on this VPS: OOM during multi-profile rollout. The cron daemon itself peaked at 12GB RSS. The tar+gzip of a ~2.8GB directory under that load would OOM the backup subshell.

## Repairing a gap

- The script is self-healing — once cron fires again, it resumes daily. No backfill needed unless the gap is critical.
- For manual backfill: `bash /home/kensei/scripts/daily-hermes-backup.sh` and the script handles retention.
- Cron only runs on system crontab, not via Hermes scheduler — the script path never changes unless you explicitly edit the crontab.

## Testing

- Syntax check: `bash -n /home/kensei/scripts/daily-hermes-backup.sh`
- Dry run: `bash /home/kensei/scripts/daily-hermes-backup.sh` (writes a real backup — runs at ~2.8GB, takes ~30s)
- Watchdog test: `bash /home/kensei/.hermes/profiles/wesker/scripts/backup-health-check.sh`
- Tar exclude validation: `tar --exclude='...' -czf /dev/null --warning=no-file-changed -C /home/kensei .hermes 2>&1`

## Pitfalls

- **`2>/dev/null || true` is dangerous** — it absorbs tar failures. Always capture stderr and check exit codes.
- **cron uses system crontab, not Hermes scheduler.** The entry is in `crontab -l`, not in `hermes cron`. If you want to change the schedule, edit crontab.
- **Tar exclude patterns** must match the actual paths inside the tarball (relative to `-C /home/kensei`). The `.hermes/` prefix is correct.
- **OOM risk** on ops-heavy days. The script runs at 03:00, which is typically idle. But if a multi-profile rollout or large migration ran the previous day, remaining processes can push memory over the limit.
- **Weekly archive directory** is created by the script on first Sunday run. Do not pre-create it — the script does `mkdir -p`.
- **No_agent watchdog** lives under the wesker profile scripts dir and is registered in the Hermes root scheduler. Scripts must use bare filenames (relative to profile scripts dir).