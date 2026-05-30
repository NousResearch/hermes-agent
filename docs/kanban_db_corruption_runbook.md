# Kanban DB Corruption Runbook

## Scope
This runbook is for recurring corruption on a Hermes Kanban board database (for example `~/.hermes/kanban/boards/cryptotrader/kanban.db`) where dashboard or gateway reports `database disk image is malformed` or `disk I/O error`.

## Symptoms
- Dashboard board endpoint fails with `500`.
- Gateway logs show repeated `kanban notifier tick failed: disk I/O error`.
- Later logs escalate to `database disk image is malformed`.
- Multiple `kanban.db.corrupt.*.bak` files appear quickly.

## Root-Cause Signals
The canonical risk pattern is:
1. Repeated SQLite `disk I/O error` on board reads/writes.
2. Followed by malformed image errors.
3. Followed by dispatcher/notifier disable messages.

This pattern indicates storage-layer instability or abrupt write-path interruption, not just a dashboard bug.

## Immediate Triage
1. Stop writers:
```bash
systemctl --user stop hermes-gateway.service hermes-dashboard.service
```
2. Validate the live DB:
```bash
sqlite3 ~/.hermes/kanban/boards/cryptotrader/kanban.db 'PRAGMA integrity_check;'
```
3. If bad, restore from last healthy recovered snapshot and re-check.
4. Start services and validate endpoint path.

## Host Diagnostics
Run these checks before concluding app-level root cause:
```bash
journalctl -k --since '24 hours ago' --no-pager | rg -i 'i/o|ext4|xfs|btrfs|blk_update|buffer i/o|nvme|vda'
findmnt -no SOURCE,FSTYPE,OPTIONS /
df -h /
```

If kernel/storage logs show I/O faults in the same window as SQLite errors, prioritize host remediation.

## Runtime Guardrail
Enable a no-agent watchdog script that:
- Verifies quick integrity on the live board DB.
- Quarantines the broken DB and sidecars on failure.
- Restores from the latest healthy snapshot.
- Restarts gateway/dashboard.
- Emits JSON status for cron delivery.

Use the runtime script:
- `~/.hermes/scripts/kanban_db_guard.py`

## Maintenance Policy
- Keep aggressive maintenance (`full` mode with vacuum path) disabled during an active incident.
- Re-enable maintenance only after:
  - Board stays healthy for multiple cycles.
  - No new `disk I/O error` bursts.
  - Host-level checks are clean.

## Verification Checklist
- `hermes kanban --board cryptotrader stats` succeeds.
- Plugin handler returns board payload without exception.
- `sqlite3 ... 'PRAGMA quick_check;'` returns `ok`.
- No fresh gateway `disk I/O error` / `malformed` messages after restart.

## Notes
- Preserve broken DBs as evidence (`*.broken.*.bak` / `*.corrupt.*.bak`).
- Do not delete forensic snapshots during active investigation.
