# Phase D: Cloud Storage Backend

**Status**: TODO
**Depends on**: Phase 0
**Blocks**: Phase E

## Goal

Replace SQLite session store with Neon PostgreSQL for SaaS mode. Maintain WAL for durability. Keep local SQLite for dev. The switch is gated by `HERMES_MODE=saas` — zero local regression.

## Context

`hermes_state.py` today wraps SQLite with FTS5 for session search. In SaaS mode this needs to be Neon PostgreSQL for: cloud-native persistence, WAL replication, multi-worker access, and RLS-based tenant isolation.

The approach is a **thin protocol abstraction**: define `StorageBackend` as a Python `Protocol`, implement `SQLiteBackend` (wraps existing code) and `NeonBackend` (new, async/asyncpg), and select via factory at startup.

## Specifications

### S1: StorageBackend protocol

`hermes_storage/backend.py` — defines the interface: `get_or_create_conversation`, `append_message`, `get_conversation_history`, `search_sessions`. Both implementations satisfy this protocol.

### S2: SQLiteBackend

`hermes_storage/sqlite_backend.py` — wraps existing `hermes_state.py` `SessionDB`. Provides the same interface. Used when `HERMES_MODE != saas`.

### S3: NeonBackend

`hermes_storage/neon_backend.py` — uses `asyncpg` connection pool. Sets `app.tenant_id` on each connection for RLS. Implements all protocol methods against the Phase 0 schema.

### S4: Backend factory

`hermes_storage/__init__.py` — `get_backend()` reads `HERMES_MODE` and `NEON_DATABASE_URL`, returns singleton.

### S5: Gateway wires backend

`gateway/session.py` (or wherever messages are persisted) calls `get_backend()` and uses the protocol interface. Existing local behavior untouched.

## Steps

| # | Action | File | Expected Result |
|---|--------|------|-----------------|
| 1 | Create `hermes_storage/` package with `backend.py` protocol | `hermes_storage/backend.py` | Protocol defined, importable |
| 2 | Create `hermes_storage/sqlite_backend.py` wrapping `SessionDB` | `hermes_storage/sqlite_backend.py` | Existing SQLite behavior behind new interface |
| 3 | Write tests against SQLiteBackend (local behavior) | `tests/test_storage_sqlite.py` | All existing session operations pass |
| 4 | Create `hermes_storage/neon_backend.py` with asyncpg pool | `hermes_storage/neon_backend.py` | Connects to Neon, RLS context set per connection |
| 5 | Implement all protocol methods in NeonBackend | same | get_or_create_conversation, append_message, etc. |
| 6 | Write tests against NeonBackend (test DSN) | `tests/test_storage_neon.py` | Messages persisted + RLS isolates tenants |
| 7 | Create backend factory in `hermes_storage/__init__.py` | `hermes_storage/__init__.py` | `HERMES_MODE=saas` → Neon; else → SQLite |
| 8 | Wire `get_backend()` into gateway session runner | `gateway/session.py` | Messages stored via backend on every turn |
| 9 | Integration test: simulate container restart, verify history survives | `tests/test_storage_integration.py` | History retrieved after reconnection |
| 10 | Commit + push | git | `feat: phase-D cloud storage backend` |

## Acceptance Criteria

- [ ] `StorageBackend` is a `Protocol` with `get_or_create_conversation`, `append_message`, `get_conversation_history`, `search_sessions`
- [ ] `SQLiteBackend` satisfies the protocol and all existing session tests pass
- [ ] `NeonBackend` connects to Neon via `NEON_DATABASE_URL`, sets `app.tenant_id` per connection
- [ ] Cross-tenant reads blocked: User A's messages not returned for User B's tenant_id via RLS
- [ ] History survives simulated container restart (messages retrieved from Neon after reconnect)
- [ ] `HERMES_MODE=local` → SQLiteBackend selected (no Neon connection attempted)
- [ ] `HERMES_MODE=saas` + `NEON_DATABASE_URL` set → NeonBackend selected
- [ ] `pytest tests/test_storage_sqlite.py tests/test_storage_neon.py -v` — all pass

## Key Code

```python
# hermes_storage/backend.py
from typing import Protocol, runtime_checkable

@runtime_checkable
class StorageBackend(Protocol):
    async def get_or_create_conversation(
        self, identity: "HermesIdentity", channel_id: str, thread_id: str | None
    ) -> str: ...  # Returns conversation UUID

    async def append_message(
        self, conversation_id: str, role: str, content: str,
        tool_calls: dict | None = None, metadata: dict | None = None
    ) -> str: ...  # Returns message UUID

    async def get_conversation_history(
        self, conversation_id: str, limit: int = 50
    ) -> list[dict]: ...

    async def search_sessions(
        self, query: str, identity: "HermesIdentity", limit: int = 5
    ) -> list[dict]: ...
```

```python
# hermes_storage/__init__.py
import os
from hermes_storage.backend import StorageBackend

_backend: StorageBackend | None = None

async def get_backend() -> StorageBackend:
    global _backend
    if _backend is None:
        if os.environ.get("HERMES_MODE") == "saas":
            from hermes_storage.neon_backend import NeonBackend
            _backend = NeonBackend(dsn=os.environ["NEON_DATABASE_URL"])
            await _backend.initialize()
        else:
            from hermes_storage.sqlite_backend import SQLiteBackend
            _backend = SQLiteBackend()
    return _backend
```

## Neon Setup Commands

```bash
# One-time provisioning
neon project create --name hermes-saas
neon database create --name hermes --project-id <id>
psql $NEON_DATABASE_URL -f migrations/001_tenants_and_users.sql

# Connection string format:
# postgresql://user:pass@ep-xxx.us-east-2.aws.neon.tech/hermes?sslmode=require
```
