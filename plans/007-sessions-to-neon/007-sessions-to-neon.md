# Plan 007 — Sessions to Neon (Compliance-Grade Audit + Cloud-Survivable State)

**Status:** DRAFT 2026-05-25
**Run after:** Plan 001-D complete (NeonBackend infrastructure exists, `append_message` already implemented)
**Blocks:** Plan 005 (Fargate cutover) — cloud Hermes loses conversation memory on every task restart without this
**Estimated effort:** 1.5–2 working days

## Context — Why this plan exists

Today every Slack interaction goes through `SessionStore.append_to_transcript()` at `gateway/session.py:1262`, which writes the message to **two** stores: SQLite via `SessionDB()` AND a per-session JSONL file under `~/.hermes/sessions/`. Both are filesystem-local. There is no Neon write path for conversation history.

This is currently fine because:
- Hermes runs on Blake's laptop with persistent disk
- `NeonBackend` (Plan 001-D) is wired for tenants, skill_output_map, and skill_feedback — but NOT for session messages, despite `NeonBackend.append_message` existing at `hermes_storage/neon_backend.py:382`

It becomes broken the moment Hermes moves to Fargate (Plan 005):
- Fargate tasks have no persistent disk; the SQLite file dies with the container
- Every task restart = total amnesia for every active Slack conversation
- This is a regression vs. today's local-only behavior — therefore **Plan 005 cannot ship cleanly until Plan 007 lands**

Beyond the cloud-survivability case, Blake explicitly asked (2026-05-25) for compliance-grade message persistence: "every Slack message + every Hermes reply + every tool invocation gets a raw audit row in Neon." That's a slightly bigger ask than "session continuity," and this plan covers both in one pass.

## Discovery — what already exists

| Component | State | File |
|---|---|---|
| `NeonBackend.append_message(session_id, role, content, tool_name, tool_calls, tool_call_id, ...)` | EXISTS (Plan 001-D) | `hermes_storage/neon_backend.py:382` |
| `SQLiteBackend.append_message(...)` (same signature) | EXISTS | `hermes_storage/sqlite_backend.py:138` |
| `StorageBackend` abstract base with `append_message` | EXISTS | `hermes_storage/backend.py:55` |
| `SessionStore.append_to_transcript()` | EXISTS — calls `self._db.append_message(...)` where `self._db = SessionDB()` (SQLite) | `gateway/session.py:1262` |
| `SessionStore` initialization | Hard-wired to SQLite; no `HERMES_MODE` check | `gateway/session.py:694–698` |
| `SessionEntry` index (`sessions.json`) | Filesystem JSON; no Neon equivalent | `gateway/session.py:711` |
| JSONL per-session transcript | Filesystem-only; primarily legacy | `gateway/session.py:1260` |
| Schema for `sessions` + `messages` Neon tables | Migration `001_neon_schema.sql` already creates these — verify column shapes match `append_message` signature | `migrations/001_neon_schema.sql` |

This is mostly a wiring job, not a from-scratch build. Most of the code already exists; what's missing is a backend-selection layer + a Neon implementation of `SessionEntry` metadata + RLS-aware reads.

## Key design decisions (locked)

1. **Single backend selection point.** `SessionStore.__init__` becomes the only place that decides SQLite vs. Neon, gated on `HERMES_MODE=saas`. Every other call site stays SQLite-shaped — the backend object handles the routing.
2. **`StorageBackend` interface owns both message AND session-entry CRUD.** Today `append_message` exists on both backends, but session-entry CRUD (`get_session`, `list_sessions`, `update_session_metadata`) is split between `SessionStore`'s JSON code and `SessionDB`'s SQLite code. Plan 007 lifts session-entry CRUD into the `StorageBackend` interface.
3. **RLS-enforced tenancy** for every Neon write. Reuses `_RLSTransaction` from Plan 001-D. Tenant resolved via `_tenant_id_by_team` cache (Plan 004-A, already wired).
4. **JSONL legacy writes stay on for SQLite mode**, get dropped for Neon mode. JSONL is filesystem-only and serves no purpose in cloud. SaaS-mode reads come from Neon exclusively.
5. **Idempotent message ingest.** Each message gets a deterministic `(session_id, platform_message_id, role)` unique constraint so Slack redeliveries don't double-write.
6. **Raw-audit table separate from messages.** Compliance-grade audit (every inbound + every outbound + every tool call) writes to a dedicated `raw_events` table with full payload + provenance — separate from the cleaned-up `messages` table the agent reads back. This lets us evolve the agent's view independently from the audit retention.

## Phase Index

| Phase | Title | Effort | Risk | Priority | Status |
|---|---|---|---|---|---|
| 007-A | Migration + schema: ensure `sessions`, `messages`, `raw_events` tables + RLS policies | 0.5 day | Low | P0 | Not started |
| 007-B | `NeonBackend`: implement session-entry CRUD + raw_events write | 0.5 day | Med | P0 | Not started |
| 007-C | `SessionStore`: backend selection + JSONL drop in saas mode | 0.5 day | Med | P0 | Not started |
| 007-D | Raw audit: inbound + outbound + tool-call hooks → `raw_events` | 0.5 day | Med | P0 | Not started |
| 007-E | End-to-end UAT: Slack mention → both `messages` + `raw_events` rows visible in Neon under RLS | 0.25 day | Low | P0 | Not started |

## Execution Sequence

A → B → C → D → E. Each phase has its own acceptance criteria; do not advance until prior gates pass.

## Phase Detail

### Phase 007-A — Migration: confirm sessions/messages/raw_events tables + RLS

**What:** Plan 001-D shipped `migrations/001_neon_schema.sql` which already creates `tenants`, `sessions`, `messages` tables. Verify column shapes match what `NeonBackend.append_message` writes; add `raw_events` table if missing; add RLS policies for any table that doesn't have one yet.

**Files to read:**
- `migrations/001_neon_schema.sql`
- `hermes_storage/neon_backend.py:382` (NeonBackend.append_message column list)
- `gateway/session.py:1274–1287` (call site — confirms which fields are passed)

**Files to create:**
- `migrations/006_raw_events.sql` — `raw_events(id, tenant_id, session_id, event_kind, platform_message_id, raw_payload jsonb, recorded_at)` + RLS policy `tenant_id = current_setting('app.tenant_id')::uuid`

**Acceptance criteria:**
- [ ] `\d sessions` in psql shows columns matching `SessionEntry` shape (id, tenant_id, platform, external_chat_id, thread_id, created_at, updated_at, metadata jsonb)
- [ ] `\d messages` shows columns covering all `append_message` parameters
- [ ] `raw_events` table exists with RLS enabled
- [ ] Migration is idempotent (re-runs cleanly)
- [ ] Migration applied to the live Neon DB (verify by querying for new table)

### Phase 007-B — NeonBackend: session-entry CRUD + raw_events write

**What:** Add `get_session`, `create_session`, `update_session_metadata`, `list_sessions_for_tenant`, `append_raw_event` methods to `NeonBackend`. Mirror to `SQLiteBackend` for local-mode parity (or stub with `NotImplementedError` — Plan accepts either).

**Files to modify:**
- `hermes_storage/neon_backend.py` — add 5 new methods, each wrapped in `_RLSTransaction`
- `hermes_storage/backend.py` — add abstract method signatures
- `hermes_storage/sqlite_backend.py` — add minimal implementations (or NotImplementedError + skip-on-saas guard)

**Acceptance criteria:**
- [ ] `pytest tests/test_neon_backend.py -v` — at least 5 new tests covering each method
- [ ] Each method uses `_RLSTransaction(conn, tenant_id)` for the writes/reads
- [ ] `append_raw_event` accepts `event_kind` enum: `slack_inbound`, `slack_outbound`, `tool_call_request`, `tool_call_response`
- [ ] `list_sessions_for_tenant` correctly filters by RLS GUC — verified by 2-tenant fixture test (tenant A query never sees tenant B sessions)

### Phase 007-C — SessionStore: backend selection + JSONL drop in saas mode

**What:** Add a thin `_select_backend()` helper at `SessionStore.__init__` that returns `NeonBackend._pool`-backed adapter when `HERMES_MODE=saas`, else `SessionDB()`. Adapt all `self._db.append_message` etc. call sites to work through a uniform interface. Disable JSONL writes in saas mode (filesystem absent on Fargate; pointless writes would crash).

**Files to modify:**
- `gateway/session.py:694–698` — replace direct `SessionDB()` instantiation with backend selection
- `gateway/session.py:1262–1300` — make JSONL writes conditional on `not self._saas_mode`
- `gateway/session.py:1321–1371` — make `load_transcript` go through backend (no JSONL fallback in saas mode)
- `gateway/session.py:700–727` — `_ensure_loaded_locked` reads `sessions.json` from disk; in saas mode this becomes `backend.list_sessions_for_tenant(tenant_id)`

**Acceptance criteria:**
- [ ] In `HERMES_MODE=saas`, no writes to `~/.hermes/sessions/` (verified by snapshotting dir before + after a Slack interaction)
- [ ] In `HERMES_MODE=saas`, `SessionStore.append_to_transcript` calls `NeonBackend.append_message`
- [ ] In default mode (no env var), behavior is byte-identical to today
- [ ] `pytest tests/test_session_store.py -v` — all existing tests still pass + at least 3 new tests for saas mode

### Phase 007-D — Raw audit: hook inbound + outbound + tool-call → raw_events

**What:** Three call sites need to fire `append_raw_event`:
1. **Slack inbound**: `gateway/platforms/slack.py:_handle_slack_message` — capture every event as `slack_inbound` with full event dict
2. **Slack outbound**: `gateway/platforms/slack.py:send` — already hooks `register_output` for Plan 004-A; add parallel `append_raw_event(event_kind="slack_outbound")` with full payload
3. **Tool calls**: `agent/conversation_loop.py` (or wherever tool invocations happen) — emit `tool_call_request` + `tool_call_response` events with arguments + return value

Each write must be best-effort (try/except, log on failure) so audit-store outages never block the user-facing path.

**Files to modify:**
- `gateway/platforms/slack.py:_handle_slack_message` — add 5-line `append_raw_event` hook
- `gateway/platforms/slack.py:send` (around line 840 where `register_output` already fires) — add parallel raw_outbound hook
- `agent/conversation_loop.py` or `agent/tool_executor.py` — add hooks around tool call begin + end

**Acceptance criteria:**
- [ ] After a 1-message Slack interaction with 1 tool call, `raw_events` table has ≥4 rows: 1 inbound, 1 outbound, 1 tool_call_request, 1 tool_call_response
- [ ] Each row has correct `tenant_id` matching the team's tenant
- [ ] Each row has non-empty `raw_payload` jsonb
- [ ] Failed Neon writes log a debug line but do not raise (verified by stopping Neon mid-test → conversation still completes)

### Phase 007-E — End-to-end UAT

**What:** Manual + automated verification that the whole Slack → Neon write path works under realistic load.

**Procedure:**
1. Start gateway with `HERMES_MODE=saas`
2. Send `@Hermes hello world` from Slack
3. Wait for reply
4. React 👍 (covers Plan 004-A integration)
5. Query Neon as `hermes_app` (with RLS):
   ```sql
   SELECT set_config('app.tenant_id', '<team_uuid>', false);
   SELECT 'sessions' AS tbl, COUNT(*) FROM sessions
   UNION ALL SELECT 'messages', COUNT(*) FROM messages
   UNION ALL SELECT 'raw_events', COUNT(*) FROM raw_events
   UNION ALL SELECT 'skill_feedback', COUNT(*) FROM skill_feedback;
   ```

**Acceptance criteria:**
- [ ] `sessions` has at least 1 row for the test interaction
- [ ] `messages` has ≥2 rows (user + assistant)
- [ ] `raw_events` has ≥3 rows (inbound + outbound + 0-N tool calls)
- [ ] `skill_feedback` has 1 row after the 👍 (Plan 004-A unchanged)
- [ ] Restart the gateway — re-query `messages` for the same `session_id` returns the same rows (proves persistence survives restart)
- [ ] `~/.hermes/sessions/` is empty (or contains only files from a prior non-saas run)

## Out of scope (deliberate)

- Migrating historical filesystem sessions into Neon (one-shot script can come later; Phase 007 is forward-looking only)
- S3 archival of raw_events (compliance retention policy is its own decision)
- Encryption-at-rest beyond what Neon provides by default
- Read-side query optimizations / indexes — wait until query patterns emerge
- Removing the `SessionDB` SQLite class — keep for local-mode tests and rollback
- UI for browsing raw_events / sessions in Neon — not in scope

## Critical files (summary)

### Read for context
- `hermes_storage/neon_backend.py` (existing `append_message`)
- `hermes_storage/backend.py` (existing interface)
- `gateway/session.py:675–745, 1262–1371` (SessionStore + transcript)
- `migrations/001_neon_schema.sql` (existing tenants/sessions/messages tables)

### Will be modified
- `hermes_storage/neon_backend.py` (Phase 007-B)
- `hermes_storage/backend.py` (Phase 007-B)
- `hermes_storage/sqlite_backend.py` (Phase 007-B)
- `gateway/session.py` (Phase 007-C)
- `gateway/platforms/slack.py` (Phase 007-D)
- `agent/conversation_loop.py` or `agent/tool_executor.py` (Phase 007-D)

### Will be created
- `migrations/006_raw_events.sql` (Phase 007-A)
- `tests/test_neon_backend_sessions.py` (Phase 007-B)
- `tests/test_session_store_saas_mode.py` (Phase 007-C)
- `tests/test_raw_events_audit.py` (Phase 007-D)

## Verification (end-state)

| Phase | Gate |
|---|---|
| A | Migrations applied to Neon, `raw_events` table exists with RLS |
| B | 5 new NeonBackend methods + tests pass |
| C | `HERMES_MODE=saas` routes session writes to Neon; default mode unchanged |
| D | Inbound + outbound + tool-call audit rows appear in `raw_events` |
| E | End-to-end Slack interaction produces durable rows across 4 tables; survives gateway restart |
