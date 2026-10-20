# Weekly Pipeline Recovery

When prospect_discovery_cf.py hangs:

1. Kill the hung script
2. Generate weekly brief directly from SQLite:
   - `SELECT * FROM prospects WHERE status='new' ORDER BY score DESC LIMIT 10;`
   - `SELECT * FROM competitor_snapshots WHERE created_at > datetime('now', '-7 days');`
   - `SELECT * FROM monitor_runs ORDER BY created_at DESC LIMIT 20;`
3. Assemble report to `~/built-lms-agent-system/trackers/weekly_brief_YYYYMMDD.txt`

## Null-Safety Bug
`generate_weekly_report()` crashes on `summary[:120]` when `signal_summary` is NULL.
Fix: Use `(summary or "")[:120]` and `(angle or "")[:120]`.

## Recovery Queries
```sql
-- Top new prospects
SELECT company_name, url, source, employment_count, score, signal_summary
FROM prospects
WHERE status = 'new'
ORDER BY score DESC
LIMIT 10;

-- Recent competitor changes
SELECT company_name, url, field, old_value, new_value, created_at
FROM competitor_snapshots
WHERE created_at > datetime('now', '-7 days')
ORDER BY created_at DESC;

-- Lane health
SELECT lane, status, started_at, finished_at, error_message
FROM monitor_runs
ORDER BY created_at DESC
LIMIT 20;
```