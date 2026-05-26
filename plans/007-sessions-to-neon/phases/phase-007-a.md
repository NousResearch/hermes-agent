# Phase 007-A: Migration — sessions/messages/raw_events tables + RLS

## Goal
Confirm Neon has the schema Plan 007 will write to (`sessions`, `messages`, `raw_events`) with Row-Level Security policies enforcing tenant isolation, so subsequent phases can implement backend code against a stable contract.

## Context
Plan 001-D shipped `migrations/001_neon_schema.sql` which created `tenants`, `sessions`, `messages` tables — but those tables may not have all the columns `NeonBackend.append_message` writes, and `raw_events` (compliance-grade audit log) does not exist. Without this phase, Phase 007-B has no target.

## Dependencies
None — first phase. Requires Neon write access via the `neondb_owner` DSN (DDL) and that `hermes_app` role exists (from Plan 001-D Phase D).

## Scope

### Files to Create
- `migrations/006_raw_events.sql` — DDL for `raw_events(id, tenant_id, session_id, event_kind, platform_message_id, raw_payload jsonb, recorded_at)` + RLS policy `tenant_id = current_setting('app.tenant_id')::uuid` + GRANT SELECT/INSERT to `hermes_app`

### Files to Modify
- None (Phase A is migration-only)

### Explicitly Out of Scope
- Backfilling historical data from filesystem JSONL into Neon
- Indexes beyond the primary key + `(session_id, recorded_at)` — wait for query patterns
- Encryption beyond Neon's default at-rest
- Cross-region replication

## Implementation Notes

1. **Column shape verification first, additions second.** Before writing the new migration, run `\d sessions` and `\d messages` in psql against the live Neon DB to confirm Plan 001-D's tables match what `NeonBackend.append_message` (at `hermes_storage/neon_backend.py:382`) writes. If columns are missing, add them as part of `006_raw_events.sql` rather than touching `001_neon_schema.sql`.
2. **RLS pattern mirrors existing policies.** Use the same `current_setting('app.tenant_id')::uuid` pattern already in `tenants` / `messages` / `skill_feedback` policies. Don't invent a new isolation model.
3. **Idempotent migration.** Use `CREATE TABLE IF NOT EXISTS` and `DROP POLICY IF EXISTS … CREATE POLICY` so re-runs are safe (Blake's habit is to re-apply migrations during iteration).
4. **Grants for `hermes_app`.** Both DML grants (SELECT/INSERT) AND sequence usage. The pattern from Plan 004-A's `002_skill_feedback.sql` is the reference.
5. **DO NOT auto-apply.** The migration file should be created locally but the actual `psql` apply against Neon must be a deliberate step in the verification section — Blake's habit is human-in-the-loop for any Neon DDL (per the migration-002 guard comment Blake authored).

## Acceptance Criteria
- [x] `migrations/006_raw_events.sql` exists and is syntactically valid (psql parses without error)
- [x] Migration applied to live Neon DB; `\d raw_events` shows the expected 7 columns
- [x] RLS is enabled on `raw_events` (`SELECT relrowsecurity FROM pg_class WHERE relname='raw_events'` returns `t`)
- [x] RLS policy exists on `raw_events` enforcing `tenant_id = current_setting('app.tenant_id')::uuid`
- [x] `hermes_app` role has INSERT + SELECT on `raw_events` (verified via `\dp raw_events` and successful test insert as hermes_app under tenant `ac85d33a`; 2-tenant isolation verified: tenant B sees 0 of tenant A's rows)
- [x] Existing tables match what `NeonBackend.append_message` writes — adaptation noted: table is `conversations` (not `sessions`), and `messages` extras land in `metadata` jsonb (no ALTER needed). See STATUS.md Adaptations.
- [x] Re-running the migration is a no-op (idempotent — re-applied with only NOTICEs about IF NOT EXISTS skips)

## Verification Steps

```bash
# 1. Inspect existing tables (run as owner)
/opt/homebrew/opt/libpq/bin/psql "<owner-dsn>" -c "\d sessions"
/opt/homebrew/opt/libpq/bin/psql "<owner-dsn>" -c "\d messages"

# 2. Apply the new migration
/opt/homebrew/opt/libpq/bin/psql "<owner-dsn>" -f migrations/006_raw_events.sql

# 3. Confirm raw_events exists with RLS
/opt/homebrew/opt/libpq/bin/psql "<owner-dsn>" -c "\d raw_events"
/opt/homebrew/opt/libpq/bin/psql "<owner-dsn>" -c "SELECT relrowsecurity FROM pg_class WHERE relname='raw_events';"
/opt/homebrew/opt/libpq/bin/psql "<owner-dsn>" -c "\dp raw_events"

# 4. Confirm hermes_app can write under RLS
/opt/homebrew/opt/libpq/bin/psql "<hermes_app-dsn>" -c "SELECT set_config('app.tenant_id', '<known-tenant-uuid>', false); INSERT INTO raw_events (tenant_id, session_id, event_kind, raw_payload) VALUES (current_setting('app.tenant_id')::uuid, gen_random_uuid(), 'slack_inbound', '{\"test\": true}'::jsonb) RETURNING id;"

# 5. Confirm RLS isolation (insert under tenant A, query under tenant B → 0 rows)
```

## Status
Complete — 2026-05-25
