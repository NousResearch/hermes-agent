# AUDIT_LOG_STANDARD

## Purpose
Standardize event records for all data access, configuration changes, agent actions, and scheduled jobs.

## Log sources
- Hermes gateway logs
- Hermes cron logs
- Approval request/answer logs
- Data access logs from ingestion tooling
- Windows event security log

## Canonical schema
- `timestamp_utc`
- `source`
- `agent`
- `action`
- `resource`
- `tier`
- `result`
- `session_id`
- `request_id`
- `evidence_path`
- `opaque_signature`

## Storage
- Append-only local files in `C:\Users\13464\AppData\Local\hermes\data\audit\`
- Optional FTS indexing for investigations.
- Tamper-evident via hash chaining.

## Principles
- Never disable auditing without Arthur or Naomi authorization.
- Logs are not stored in the same writable path as application code.
