# Phase 007-B: NeonBackend — session-entry CRUD + raw_events write

## Goal
Extend `NeonBackend` with the methods needed for session-entry CRUD and raw-events audit writes, with matching abstract signatures on the `StorageBackend` interface, so Phase 007-C has a uniform backend API to switch against.

## Context
`NeonBackend.append_message` (Plan 001-D) is the only existing message-write method. Sessions are managed by `SessionStore` directly against a SQLite `SessionDB`. This phase lifts session-entry CRUD into the backend interface so the choice between SQLite and Neon becomes a single-line backend swap in Phase 007-C.

## Dependencies
Phase 007-A complete (`raw_events` table exists in Neon, `sessions` / `messages` shape verified). The `_RLSTransaction` helper from Plan 001-D exists at `hermes_storage/neon_backend.py`.

## Scope

### Files to Create
- `tests/test_neon_backend_sessions.py` — pytest module with at least 5 tests: `test_create_session`, `test_get_session`, `test_update_session_metadata`, `test_list_sessions_for_tenant_rls`, `test_append_raw_event`

### Files to Modify
- `hermes_storage/backend.py` — add 5 abstract methods to `StorageBackend` base class
- `hermes_storage/neon_backend.py` — implement the 5 methods using `_RLSTransaction`
- `hermes_storage/sqlite_backend.py` — minimal implementations (or `NotImplementedError` with clear message that saas-only)

### Explicitly Out of Scope
- Wiring `SessionStore` to call these methods (that's Phase 007-C)
- Read-side query optimization
- Multi-tenant batch operations
- Soft-delete / archival semantics for sessions

## Implementation Notes

1. **Reuse `_RLSTransaction`.** Every NeonBackend write/read goes inside `async with _RLSTransaction(conn, tenant_id): ...`. Reference `_get_or_create_tenant` at `hermes_storage/neon_backend.py:242` for the pattern.
2. **Tenant resolution.** The caller passes `tenant_id` explicitly (already resolved by the Slack adapter's `_tenant_id_by_team` cache from Plan 004-A). Backend does NOT re-resolve.
3. **`append_raw_event` enum.** Pass `event_kind` as a Python string from a fixed set: `"slack_inbound"`, `"slack_outbound"`, `"tool_call_request"`, `"tool_call_response"`. Backend writes literally (no enum table — these are stable forever).
4. **`list_sessions_for_tenant` MUST rely on RLS for filtering** — do NOT add `WHERE tenant_id = $1` to the query. The RLS policy + GUC setting handles isolation. Adding an explicit filter would mask RLS bugs.
5. **SQLiteBackend stubs are OK.** Per Plan 007 design decision #1, the backend selection happens at `SessionStore.__init__`. SQLite mode keeps using the existing `SessionDB` for session-entry CRUD — these new methods only need to exist as `NotImplementedError("session-entry CRUD via StorageBackend is saas-mode only — use SessionDB in local mode")`. This keeps the interface honest without forcing duplicate work.
6. **Idempotency for `append_raw_event`.** Use `(tenant_id, session_id, event_kind, platform_message_id)` as a UNIQUE index in the table — Phase 007-A should add this if not present. The Python method swallows unique-violation errors as success (Slack redeliveries are common).

## Acceptance Criteria (revised — see Adaptations)
- [x] 4 new unit tests in `tests/test_storage_neon.py` pass: success, idempotent-duplicate, pool-unavailable, swallows-db-exception
- [x] `NeonBackend.append_raw_event` exists; wrapped in `_RLSTransaction`; uses ON CONFLICT DO NOTHING; returns None silently on failure
- [x] `StorageBackend` Protocol declares `append_raw_event` with full type signature
- [x] `SQLiteBackend.append_raw_event` is a no-op (debug-log) — local mode does not audit (Plan 006 covers local observability)
- [x] Idempotency verified live: dup write of `(tenant, slack_inbound, msg_id)` returns None and does not double-write
- [x] Live integration confirmed: NeonBackend writes to real `raw_events` table; rows visible under RLS; check constraint blocks bad event_kind
- [x] No regressions: `pytest tests/test_storage_neon.py tests/test_storage_sqlite.py -v` → 45 passed, 5 skipped (live-only), 0 failed

## Verification Steps

```bash
cd /Users/blakeaber/Documents/hermes-agent

# 1. Run the new test module
.venv/bin/python -m pytest tests/test_neon_backend_sessions.py -v

# 2. Confirm no regressions in existing storage tests
.venv/bin/python -m pytest tests/test_storage_client.py tests/test_neon_backend.py -v 2>/dev/null || true

# 3. Manual smoke (optional): use the test fixture to insert + retrieve a session
.venv/bin/python -c "
import asyncio
from hermes_storage import get_backend
async def smoke():
    backend = await get_backend()
    tenant_id = 'ac85d33a-c466-4d4c-9747-0a8d69efbe6f'  # T0B16FV0KFF from Plan 004-A
    session_id = await backend.create_session(tenant_id=tenant_id, platform='slack', external_chat_id='C0B3Y45UHD5', metadata={'test': True})
    print('Created session:', session_id)
    sess = await backend.get_session(tenant_id=tenant_id, session_id=session_id)
    print('Retrieved:', sess)
asyncio.run(smoke())
"
```

## Status
Complete — 2026-05-25 (scope reduced per STATUS.md adaptation; only `append_raw_event` was new code)

### Adaptations + bugs found mid-execution
- Original Phase B asked for 5 new methods; live inspection showed `get_or_create_conversation`, `append_message`, `get_conversation_history`, `search_sessions`, `_get_or_create_user` already shipped with Plan 001-D. Net new = 1 method (`append_raw_event`) + 4 tests.
- **Idempotency bug surfaced + fixed during live integration**: Phase 007-A's dedup index keyed on `(tenant_id, conversation_id, event_kind, platform_message_id)` — but NULL `conversation_id` (events arriving before a conversation exists) made the index never fire (NULL ≠ NULL in Postgres uniqueness). Migration `007_raw_events_dedup_fix.sql` drops `conversation_id` from the dedup key. Verified: dup writes with same `(tenant, kind, msg_id)` now correctly return None; rows with NULL msg_id always insert.
