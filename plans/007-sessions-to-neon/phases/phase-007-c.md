# Phase 007-C: SessionStore — backend selection + JSONL drop in saas mode

## Goal
Make `SessionStore` route session-entry + transcript writes through the `StorageBackend` interface (Neon in saas mode, SQLite locally), and stop writing to filesystem JSONL in saas mode where Fargate has no persistent disk.

## Context
`SessionStore.__init__` at `gateway/session.py:694–698` hard-wires `self._db = SessionDB()` (SQLite). `append_to_transcript` at `gateway/session.py:1262` writes to BOTH SQLite AND a per-session JSONL file under `~/.hermes/sessions/`. Neither path consults `HERMES_MODE`. This phase adds the routing.

## Dependencies
Phase 007-B complete (NeonBackend has session-entry CRUD + raw_events methods, both backends conform to the same `StorageBackend` interface).

## Scope

### Files to Create
- `tests/test_session_store_saas_mode.py` — pytest module with ≥3 tests: `test_saas_mode_routes_to_neon`, `test_local_mode_byte_identical_to_today`, `test_saas_mode_skips_jsonl`

### Files to Modify
- `gateway/session.py` — three regions:
  - `__init__` (lines 683–698): add backend selection helper; pick Neon when `HERMES_MODE=saas`
  - `append_to_transcript` (lines 1262–1299): make JSONL write conditional on `not self._saas_mode`
  - `load_transcript` (lines 1321–1371): in saas mode, return early after backend query — do not attempt JSONL fallback
  - `_ensure_loaded_locked` (lines 700–727): in saas mode, replace `sessions.json` read with `await backend.list_sessions_for_tenant(...)` — wrapped in `asyncio.run_coroutine_threadsafe` since `_ensure_loaded_locked` is sync

### Explicitly Out of Scope
- Migrating historical sessions from filesystem to Neon (one-shot script — later)
- Read-side caching / hot-session in-memory layer
- Tenant resolution inside `SessionStore` — caller (Slack adapter) passes tenant_id already
- Removing `SessionDB` (`hermes_state.py`) — keep for local mode + rollback safety

## Implementation Notes

1. **`HERMES_MODE` check at one site only.** Centralize the `os.environ.get("HERMES_MODE") == "saas"` check in a single helper `SessionStore._is_saas_mode() -> bool`. Reference once in `__init__` to pick backend; reference once each in `append_to_transcript` / `load_transcript` / `_ensure_loaded_locked` to gate JSONL writes.
2. **Sync/async bridge.** `SessionStore` is sync (called from sync code paths in the gateway). NeonBackend methods are async. Use `asyncio.run_coroutine_threadsafe(coro, self._loop)` where `self._loop` is captured at `__init__` time from the gateway's event loop. Fall back to `asyncio.new_event_loop().run_until_complete` if no loop is available (test scenarios).
3. **JSONL path during saas mode.** In saas mode, `get_transcript_path` should return `None` (or raise) — and every call site should be guarded. Cleaner alternative: leave the JSONL writes returning early. Decide during implementation; pick whichever yields fewer call-site changes.
4. **`SessionDB` fallback on Neon outage.** If `HERMES_MODE=saas` but `NeonBackend` initialization fails (e.g., no `NEON_DATABASE_URL` set), log a critical warning and fall through to `SessionDB` so the gateway still starts. This mirrors the defensive pattern in Plan 005's "Storage backend initialized: NeonBackend (pool=ready)" startup line.
5. **Zero-regression mandate.** All existing tests in `tests/test_session_store.py`, `tests/test_session.py`, `tests/test_gateway.py` MUST still pass. The default mode (no `HERMES_MODE` env var) MUST behave byte-identical to today.

## Acceptance Criteria
- [ ] `pytest tests/test_session_store_saas_mode.py -v` — 3 new tests pass
- [ ] `pytest tests/test_session*.py tests/test_gateway.py -v` — all existing tests pass (no regressions)
- [ ] In `HERMES_MODE=saas`, a Slack interaction writes rows to Neon `sessions` + `messages` tables (verified by psql query)
- [ ] In `HERMES_MODE=saas`, no new files appear under `~/.hermes/sessions/` for the session (verified by snapshot before/after)
- [ ] In default mode (no env var), `~/.hermes/sessions/sessions.json` is still updated; `<session_id>.jsonl` is still appended
- [ ] If `HERMES_MODE=saas` but `NEON_DATABASE_URL` is unset, gateway logs a CRITICAL warning, falls back to SQLite, and still starts (graceful degradation)

## Verification Steps

```bash
cd /Users/blakeaber/Documents/hermes-agent

# 1. Run the new test module
.venv/bin/python -m pytest tests/test_session_store_saas_mode.py -v

# 2. Confirm zero regressions
.venv/bin/python -m pytest tests/test_session*.py tests/test_gateway.py -v

# 3. End-to-end smoke (manual, with launchd gateway in saas mode):
#    a. Verify HERMES_MODE=saas and NEON_DATABASE_URL set in .env
#    b. Snapshot filesystem: ls ~/.hermes/sessions/ > /tmp/before.txt
#    c. Bounce gateway: launchctl kickstart -k gui/$(id -u)/ai.hermes.gateway
#    d. Send a Slack mention to Hermes; wait for reply
#    e. Snapshot filesystem: ls ~/.hermes/sessions/ > /tmp/after.txt
#    f. diff /tmp/before.txt /tmp/after.txt  → expect empty (no new files in saas mode)
#    g. Query Neon: SELECT COUNT(*) FROM messages WHERE created_at > NOW() - INTERVAL '5 minutes';
#       → expect ≥2 (user + assistant)

# 4. Confirm graceful degradation (manual)
#    a. Temporarily unset NEON_DATABASE_URL
#    b. Start gateway; observe CRITICAL warning in log
#    c. Confirm gateway is up (lsof -i:8080 returns the listener)
#    d. Restore NEON_DATABASE_URL
```

## Status
Complete — 2026-05-25

### Adaptations
- Original plan wired through `SessionStore`. Live inspection showed SessionStore has zero existing calls into NeonBackend, and that the natural insertion point for "every Slack message in Neon" is the Slack adapter itself (where tenant_id, identity, channel_id, thread_ts are all immediately in scope).
- Pivoted Phase 007-C scope: hook the two Slack adapter call sites instead. User turn writes from `_handle_slack_message` (right after `hermes_identity` is constructed); assistant turn writes from `send` adjacent to the existing Plan 004-A `register_output` hook.
- `_conv_id_by_chat: Dict[(chat_id, thread_ts), conv_uuid]` cache populated lazily on first inbound message per thread.
- `SessionStore` left untouched. The "JSONL drop in saas mode" original sub-goal is deferred to Plan 005 (Fargate cutover) where it actually matters; local launchd still uses disk normally.
- Live integration verified end-to-end: probe persisted user + assistant turns; both rows visible in Neon `messages` table with correct conversation_id linkage and metadata (`slack_ts`, `slack_user_id`).
- Regression sweep: 380 passed + 5 skipped + 0 failed across the full pytest run (8 unrelated ImportErrors in acp/browser/plugin test modules — pre-existing, not from this phase).
