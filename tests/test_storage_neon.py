"""
tests/test_storage_neon.py — Tests for hermes_storage.NeonBackend.

Two test tiers:

  1. UNIT TESTS (always run) — mock asyncpg pool + connections.
     Verify protocol compliance, RLS transaction wrapping, DSN resolution,
     tenant/user bootstrapping, and error paths — all without network access.

  2. LIVE INTEGRATION TESTS (skipped unless NEON_DATABASE_URL is set) —
     connect to the real Neon instance and verify:
       - Connection pool initialises.
       - get_or_create_conversation creates + retrieves rows.
       - append_message persists messages.
       - get_conversation_history returns them in order.
       - Cross-tenant isolation: User A's messages not visible to User B's tenant.
       - History survives reconnect (simulates container restart).

Run live tests locally:
    NEON_DATABASE_URL="postgres://..." pytest tests/test_storage_neon.py -v -m live

Run unit tests only (CI):
    pytest tests/test_storage_neon.py -v -m "not live"
"""

from __future__ import annotations

import asyncio
import os
import uuid
from typing import Optional
from unittest.mock import AsyncMock, MagicMock, patch, call

import pytest

from hermes_identity import HermesIdentity
from hermes_storage import StorageBackend, get_backend, reset_backend
from hermes_storage.neon_backend import NeonBackend, _RLSTransaction, _resolve_dsn


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_identity(
    platform: str = "slack",
    team_id: str = "TTEAM01",
    user_id: str = "UUSER01",
    channel_id: str = "CCHAN01",
    thread_id: Optional[str] = None,
) -> HermesIdentity:
    return HermesIdentity(
        platform=platform,
        team_id=team_id,
        user_id=user_id,
        channel_id=channel_id,
        thread_id=thread_id,
    )


LIVE = pytest.mark.skipif(
    not os.environ.get("NEON_DATABASE_URL"),
    reason="NEON_DATABASE_URL not set — skipping live Neon tests",
)

pytest.ini_options = {}  # keep pytest from complaining about unknown marks


# ---------------------------------------------------------------------------
# DSN resolution tests
# ---------------------------------------------------------------------------

def test_resolve_dsn_explicit_kwarg():
    """Explicit dsn kwarg wins over env vars."""
    dsn = _resolve_dsn("postgres://explicit/db")
    assert dsn == "postgres://explicit/db"


def test_resolve_dsn_env_var(monkeypatch):
    """NEON_DATABASE_URL env var is used when no kwarg."""
    monkeypatch.setenv("NEON_DATABASE_URL", "postgres://from_env/db")
    monkeypatch.delenv("NEON_DSN", raising=False)
    dsn = _resolve_dsn(None)
    assert dsn == "postgres://from_env/db"


def test_resolve_dsn_neon_dsn_fallback(monkeypatch):
    """NEON_DSN env var is the secondary fallback."""
    monkeypatch.delenv("NEON_DATABASE_URL", raising=False)
    monkeypatch.setenv("NEON_DSN", "postgres://from_neon_dsn/db")
    dsn = _resolve_dsn(None)
    assert dsn == "postgres://from_neon_dsn/db"


def test_resolve_dsn_secrets_manager(monkeypatch):
    """Falls back to Secrets Manager when no env vars set."""
    monkeypatch.delenv("NEON_DATABASE_URL", raising=False)
    monkeypatch.delenv("NEON_DSN", raising=False)

    mock_boto = MagicMock()
    mock_client = MagicMock()
    mock_client.get_secret_value.return_value = {
        "SecretString": "postgres://from_secrets_manager/db"
    }
    mock_boto.client.return_value = mock_client

    with patch.dict("sys.modules", {"boto3": mock_boto}):
        dsn = _resolve_dsn(None)

    assert dsn == "postgres://from_secrets_manager/db"
    mock_client.get_secret_value.assert_called_once_with(
        SecretId="agentic-stack/neon/hermes-saas"
    )


def test_resolve_dsn_secrets_manager_json_format(monkeypatch):
    """Secrets Manager returns a JSON-encoded DSN."""
    monkeypatch.delenv("NEON_DATABASE_URL", raising=False)
    monkeypatch.delenv("NEON_DSN", raising=False)
    import json

    mock_boto = MagicMock()
    mock_client = MagicMock()
    mock_client.get_secret_value.return_value = {
        "SecretString": json.dumps({"dsn": "postgres://from_json/db"})
    }
    mock_boto.client.return_value = mock_client

    with patch.dict("sys.modules", {"boto3": mock_boto}):
        dsn = _resolve_dsn(None)

    assert dsn == "postgres://from_json/db"


def test_resolve_dsn_no_source_raises(monkeypatch):
    """RuntimeError when no DSN source is available."""
    monkeypatch.delenv("NEON_DATABASE_URL", raising=False)
    monkeypatch.delenv("NEON_DSN", raising=False)

    mock_boto = MagicMock()
    mock_boto.client.side_effect = Exception("no AWS creds")

    with patch.dict("sys.modules", {"boto3": mock_boto}):
        with pytest.raises(RuntimeError, match="cannot resolve Neon DSN"):
            _resolve_dsn(None)


# ---------------------------------------------------------------------------
# Protocol compliance
# ---------------------------------------------------------------------------

def test_neon_backend_satisfies_protocol():
    """NeonBackend satisfies the StorageBackend Protocol (isinstance check)."""
    backend = NeonBackend(dsn="postgres://fake/db")
    assert isinstance(backend, StorageBackend)


# ---------------------------------------------------------------------------
# _RLSTransaction unit tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_rls_transaction_sets_local_guc():
    """_RLSTransaction issues SET LOCAL app.tenant_id = '...' inside a txn."""
    # conn.transaction() is a SYNC method returning a Transaction object.
    # Use MagicMock for transaction(); its methods (start/commit/rollback) are async.
    txn = MagicMock()
    txn.start = AsyncMock()
    txn.commit = AsyncMock()
    txn.rollback = AsyncMock()

    conn = AsyncMock()
    conn.transaction = MagicMock(return_value=txn)  # sync, returns txn directly

    tenant_id = str(uuid.uuid4())
    async with _RLSTransaction(conn, tenant_id):
        pass

    txn.start.assert_awaited_once()
    conn.execute.assert_any_await(f"SET LOCAL app.tenant_id = '{tenant_id}'")
    txn.commit.assert_awaited_once()
    txn.rollback.assert_not_awaited()


@pytest.mark.asyncio
async def test_rls_transaction_rollback_on_exception():
    """_RLSTransaction rolls back on exception."""
    txn = MagicMock()
    txn.start = AsyncMock()
    txn.commit = AsyncMock()
    txn.rollback = AsyncMock()

    conn = AsyncMock()
    conn.transaction = MagicMock(return_value=txn)

    tenant_id = str(uuid.uuid4())
    with pytest.raises(ValueError, match="test error"):
        async with _RLSTransaction(conn, tenant_id):
            raise ValueError("test error")

    txn.rollback.assert_awaited_once()
    txn.commit.assert_not_awaited()


# ---------------------------------------------------------------------------
# NeonBackend unit tests (mocked pool)
# ---------------------------------------------------------------------------

class _FakeRecord(dict):
    """Dict subclass that supports asyncpg Record-style attribute access."""
    def __getitem__(self, key):
        return super().__getitem__(key)


def _make_mock_pool_and_conn():
    """
    Return (pool, conn, txn) mocks suitable for NeonBackend tests.

    asyncpg quirks to replicate in mocks:
    - conn.transaction() is a SYNC call returning a transaction object
      (not a coroutine). Use MagicMock for it.
    - txn.start(), txn.commit(), txn.rollback() ARE awaitable (coroutines).
    - conn.execute(), conn.fetchrow(), conn.fetch() ARE awaitable.
    - pool.acquire() returns an async context manager.
    - asyncpg.create_pool IS awaitable.
    """
    txn = MagicMock()  # sync object, but methods on it are async
    txn.start = AsyncMock()
    txn.commit = AsyncMock()
    txn.rollback = AsyncMock()

    conn = AsyncMock()
    # transaction() is a SYNC method on asyncpg Connection that returns a
    # Transaction object. Use MagicMock (not AsyncMock) so it returns txn
    # without being a coroutine.
    conn.transaction = MagicMock(return_value=txn)

    # acquire() is a context manager (async with pool.acquire() as conn:)
    acquire_ctx = MagicMock()
    acquire_ctx.__aenter__ = AsyncMock(return_value=conn)
    acquire_ctx.__aexit__ = AsyncMock(return_value=False)

    pool = MagicMock()
    pool.acquire = MagicMock(return_value=acquire_ctx)
    pool.close = AsyncMock()

    return pool, conn, txn


@pytest.mark.asyncio
async def test_initialize_creates_pool():
    """initialize() calls asyncpg.create_pool with the DSN."""
    backend = NeonBackend(dsn="postgres://test/db", min_size=1, max_size=5)

    mock_pool = MagicMock()
    mock_pool.close = AsyncMock()
    # asyncpg.create_pool is a coroutine — use AsyncMock so await works.
    with patch("asyncpg.create_pool", new_callable=AsyncMock, return_value=mock_pool) as mock_create:
        await backend.initialize()

    mock_create.assert_awaited_once()
    call_kwargs = mock_create.call_args
    assert call_kwargs[0][0] == "postgres://test/db"
    assert call_kwargs[1]["min_size"] == 1
    assert call_kwargs[1]["max_size"] == 5


@pytest.mark.asyncio
async def test_initialize_idempotent():
    """initialize() called twice doesn't create a second pool."""
    backend = NeonBackend(dsn="postgres://test/db")
    mock_pool = MagicMock()
    mock_pool.close = AsyncMock()

    with patch("asyncpg.create_pool", new_callable=AsyncMock, return_value=mock_pool) as mock_create:
        await backend.initialize()
        await backend.initialize()

    assert mock_create.await_count == 1


@pytest.mark.asyncio
async def test_close_closes_pool():
    """close() closes the pool and resets to None."""
    backend = NeonBackend(dsn="postgres://test/db")
    mock_pool = AsyncMock()
    backend._pool = mock_pool

    await backend.close()

    mock_pool.close.assert_awaited_once()
    assert backend._pool is None


@pytest.mark.asyncio
async def test_close_safe_when_not_initialised():
    """close() on uninitialised backend doesn't raise."""
    backend = NeonBackend(dsn="postgres://test/db")
    await backend.close()  # No exception.


@pytest.mark.asyncio
async def test_require_pool_raises_if_not_initialised():
    """Calling a storage method before initialize() raises RuntimeError."""
    backend = NeonBackend(dsn="postgres://test/db")
    identity = make_identity()
    with pytest.raises(RuntimeError, match="not initialised"):
        await backend.get_or_create_conversation(identity, "#ch", None)


@pytest.mark.asyncio
async def test_get_or_create_conversation_creates_new():
    """Creates tenant + user + conversation on first call."""
    backend = NeonBackend(dsn="postgres://test/db")
    pool, conn, txn = _make_mock_pool_and_conn()
    backend._pool = pool

    identity = make_identity()
    tenant_id = str(uuid.uuid4())
    user_id = str(uuid.uuid4())

    # Simulate: tenant not found → insert → re-fetch → found
    # user not found → insert → re-fetch → found
    # conversation not found → insert → return new_id
    conn.fetchrow.side_effect = [
        None,                                    # _get_or_create_tenant: SELECT → not found
        _FakeRecord({"id": tenant_id}),          # _get_or_create_tenant: re-fetch after INSERT
        None,                                    # _get_or_create_user: SELECT → not found
        _FakeRecord({"id": user_id}),            # _get_or_create_user: re-fetch after INSERT
        None,                                    # get_or_create_conversation: SELECT → not found
    ]

    result = await backend.get_or_create_conversation(identity, "#general", None)

    assert isinstance(result, str)
    assert len(result) == 36  # UUID format


@pytest.mark.asyncio
async def test_get_or_create_conversation_returns_existing():
    """Returns existing conversation_id when one already exists."""
    backend = NeonBackend(dsn="postgres://test/db")
    pool, conn, txn = _make_mock_pool_and_conn()
    backend._pool = pool

    identity = make_identity()
    tenant_id = str(uuid.uuid4())
    user_id = str(uuid.uuid4())
    existing_conv_id = str(uuid.uuid4())

    conn.fetchrow.side_effect = [
        _FakeRecord({"id": tenant_id}),        # _get_or_create_tenant: found
        _FakeRecord({"id": user_id}),          # _get_or_create_user: found
        _FakeRecord({"id": existing_conv_id}), # conversation: found
    ]

    result = await backend.get_or_create_conversation(identity, "#general", None)
    assert result == existing_conv_id


@pytest.mark.asyncio
async def test_append_message_persists():
    """append_message inserts a row and returns a UUID string."""
    backend = NeonBackend(dsn="postgres://test/db")
    pool, conn, txn = _make_mock_pool_and_conn()
    backend._pool = pool

    tenant_id = str(uuid.uuid4())
    conv_id = str(uuid.uuid4())

    # First fetchrow: resolve tenant_id from conversations
    conn.fetchrow.return_value = _FakeRecord({"tenant_id": tenant_id})

    msg_id = await backend.append_message(conv_id, "user", "Hello, Neon!")

    assert isinstance(msg_id, str)
    assert len(msg_id) == 36  # UUID

    # Verify an INSERT was called.
    insert_calls = [
        c for c in conn.execute.await_args_list
        if "INSERT INTO messages" in str(c)
    ]
    assert len(insert_calls) == 1


@pytest.mark.asyncio
async def test_append_message_raises_if_conversation_missing():
    """append_message raises ValueError when conversation_id not found."""
    backend = NeonBackend(dsn="postgres://test/db")
    pool, conn, txn = _make_mock_pool_and_conn()
    backend._pool = pool

    conn.fetchrow.return_value = None  # conversations not found

    with pytest.raises(ValueError, match="not found"):
        await backend.append_message("nonexistent-conv-id", "user", "Hello")


@pytest.mark.asyncio
async def test_get_conversation_history_returns_messages():
    """get_conversation_history returns messages in chronological order."""
    backend = NeonBackend(dsn="postgres://test/db")
    pool, conn, txn = _make_mock_pool_and_conn()
    backend._pool = pool

    tenant_id = str(uuid.uuid4())
    conv_id = str(uuid.uuid4())

    conn.fetchrow.return_value = _FakeRecord({"tenant_id": tenant_id})
    # fetch returns DESC order (latest first); NeonBackend reverses to oldest-first.
    conn.fetch.return_value = [
        _FakeRecord({"role": "assistant", "content": "Hi!", "tool_calls": None, "metadata": "{}"}),
        _FakeRecord({"role": "user", "content": "Hello", "tool_calls": None, "metadata": "{}"}),
    ]

    history = await backend.get_conversation_history(conv_id, limit=50)

    assert len(history) == 2
    # reversed — oldest first
    assert history[0]["content"] == "Hello"
    assert history[1]["content"] == "Hi!"


@pytest.mark.asyncio
async def test_get_conversation_history_missing_conv_returns_empty():
    """Returns empty list for unknown conversation_id."""
    backend = NeonBackend(dsn="postgres://test/db")
    pool, conn, txn = _make_mock_pool_and_conn()
    backend._pool = pool

    conn.fetchrow.return_value = None  # not found

    history = await backend.get_conversation_history("unknown-id")
    assert history == []


@pytest.mark.asyncio
async def test_search_sessions_returns_results():
    """search_sessions returns matching conversation snippets."""
    backend = NeonBackend(dsn="postgres://test/db")
    pool, conn, txn = _make_mock_pool_and_conn()
    backend._pool = pool

    tenant_id = str(uuid.uuid4())
    conv_id = str(uuid.uuid4())

    conn.fetchrow.return_value = _FakeRecord({"id": tenant_id})
    conn.fetch.return_value = [
        _FakeRecord({"conversation_id": conv_id, "snippet": "The quick brown fox"}),
    ]

    identity = make_identity()
    results = await backend.search_sessions("quick brown fox", identity)

    assert len(results) == 1
    assert results[0]["conversation_id"] == conv_id
    assert "quick brown fox" in results[0]["snippet"]


@pytest.mark.asyncio
async def test_search_sessions_no_tenant_returns_empty():
    """Returns empty list if tenant doesn't exist yet."""
    backend = NeonBackend(dsn="postgres://test/db")
    pool, conn, txn = _make_mock_pool_and_conn()
    backend._pool = pool

    conn.fetchrow.return_value = None  # no tenant

    identity = make_identity()
    results = await backend.search_sessions("query", identity)
    assert results == []


# ---------------------------------------------------------------------------
# Factory: HERMES_MODE=saas → NeonBackend
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_get_backend_saas_mode_returns_neon(monkeypatch, tmp_path):
    """get_backend() returns NeonBackend when HERMES_MODE=saas."""
    await reset_backend()
    monkeypatch.setenv("HERMES_MODE", "saas")
    monkeypatch.setenv("NEON_DATABASE_URL", "postgres://fake/db")

    mock_pool = MagicMock()
    mock_pool.close = AsyncMock()
    with patch("asyncpg.create_pool", new_callable=AsyncMock, return_value=mock_pool):
        backend = await get_backend()

    assert isinstance(backend, NeonBackend)
    await reset_backend()


# ===========================================================================
# LIVE INTEGRATION TESTS — require NEON_DATABASE_URL
# ===========================================================================

@LIVE
@pytest.mark.asyncio
@pytest.mark.live
async def test_live_pool_initialises():
    """Live: NeonBackend connects to Neon and pool is ready."""
    dsn = os.environ["NEON_DATABASE_URL"]
    backend = NeonBackend(dsn=dsn, min_size=1, max_size=3)
    await backend.initialize()
    assert backend._pool is not None
    # Simple ping via pool.
    async with backend._pool.acquire() as conn:
        result = await conn.fetchval("SELECT 1")
    assert result == 1
    await backend.close()


@LIVE
@pytest.mark.asyncio
@pytest.mark.live
async def test_live_get_or_create_conversation():
    """Live: creates conversation in Neon, idempotent on second call."""
    dsn = os.environ["NEON_DATABASE_URL"]
    backend = NeonBackend(dsn=dsn, min_size=1, max_size=3)
    await backend.initialize()

    # Use unique team/user IDs to avoid test collisions.
    run_id = uuid.uuid4().hex[:8]
    identity = make_identity(
        team_id=f"TTEST_{run_id}",
        user_id=f"UTEST_{run_id}",
        channel_id=f"CTEST_{run_id}",
    )

    conv_id_1 = await backend.get_or_create_conversation(identity, identity.channel_id, None)
    conv_id_2 = await backend.get_or_create_conversation(identity, identity.channel_id, None)

    assert conv_id_1 == conv_id_2, "Idempotency: same conv_id returned"
    assert len(conv_id_1) == 36  # UUID

    await backend.close()


@LIVE
@pytest.mark.asyncio
@pytest.mark.live
async def test_live_append_and_retrieve_messages():
    """Live: messages are persisted and retrievable from Neon."""
    dsn = os.environ["NEON_DATABASE_URL"]
    backend = NeonBackend(dsn=dsn, min_size=1, max_size=3)
    await backend.initialize()

    run_id = uuid.uuid4().hex[:8]
    identity = make_identity(
        team_id=f"TTEST_{run_id}",
        user_id=f"UTEST_{run_id}",
        channel_id=f"CTEST_{run_id}",
    )

    conv_id = await backend.get_or_create_conversation(identity, identity.channel_id, None)
    await backend.append_message(conv_id, "user", f"Hello from test {run_id}")
    await backend.append_message(conv_id, "assistant", "Got it!")

    history = await backend.get_conversation_history(conv_id)

    assert len(history) == 2
    assert history[0]["role"] == "user"
    assert run_id in history[0]["content"]
    assert history[1]["role"] == "assistant"
    assert history[1]["content"] == "Got it!"

    await backend.close()


@LIVE
@pytest.mark.asyncio
@pytest.mark.live
async def test_live_history_survives_reconnect():
    """Live: messages persist after pool close + re-initialise (simulates container restart).

    Realistic gateway pattern: on reconnect, the gateway calls get_or_create_conversation
    first (using the platform event context it always has), which populates the
    in-process tenant cache.  Then get_conversation_history works.
    This test mirrors that realistic flow.
    """
    dsn = os.environ["NEON_DATABASE_URL"]
    run_id = uuid.uuid4().hex[:8]
    identity = make_identity(
        team_id=f"TTEST_{run_id}",
        user_id=f"UTEST_{run_id}",
        channel_id=f"CTEST_{run_id}",
    )

    # First "container": write messages.
    backend_1 = NeonBackend(dsn=dsn, min_size=1, max_size=3)
    await backend_1.initialize()
    conv_id = await backend_1.get_or_create_conversation(identity, identity.channel_id, None)
    await backend_1.append_message(conv_id, "user", f"Persistent message {run_id}")
    await backend_1.close()

    # Second "container": realistic reconnect pattern.
    # The gateway always calls get_or_create_conversation on each incoming event
    # (idempotent: returns the same conv_id), which populates the tenant cache.
    # Then get_conversation_history is called to load context.
    backend_2 = NeonBackend(dsn=dsn, min_size=1, max_size=3)
    await backend_2.initialize()
    # Simulate gateway reconnect: resolve conversation first (re-populates tenant cache).
    reconnected_conv_id = await backend_2.get_or_create_conversation(
        identity, identity.channel_id, None
    )
    assert reconnected_conv_id == conv_id, "Reconnected conv_id must match original"
    history = await backend_2.get_conversation_history(conv_id)
    await backend_2.close()

    assert any(run_id in m.get("content", "") for m in history), (
        "Messages must survive pool close + re-initialise (container restart simulation)"
    )


@LIVE
@pytest.mark.asyncio
@pytest.mark.live
async def test_live_cross_tenant_rls_isolation():
    """Live: User A's messages are NOT visible when queried as User B's tenant."""
    dsn = os.environ["NEON_DATABASE_URL"]
    backend = NeonBackend(dsn=dsn, min_size=1, max_size=3)
    await backend.initialize()

    run_id = uuid.uuid4().hex[:8]

    # Tenant A.
    identity_a = make_identity(
        team_id=f"TTEAM_A_{run_id}",
        user_id=f"UUSER_A_{run_id}",
        channel_id=f"CCHAN_{run_id}",
    )
    conv_a = await backend.get_or_create_conversation(identity_a, identity_a.channel_id, None)
    await backend.append_message(conv_a, "user", f"tenant_a_secret_{run_id}")

    # Tenant B.
    identity_b = make_identity(
        team_id=f"TTEAM_B_{run_id}",
        user_id=f"UUSER_B_{run_id}",
        channel_id=f"CCHAN_{run_id}",
    )
    conv_b = await backend.get_or_create_conversation(identity_b, identity_b.channel_id, None)

    # Search as Tenant B — must NOT see Tenant A's data.
    results_b = await backend.search_sessions(f"tenant_a_secret_{run_id}", identity_b)
    conv_ids_b = [r["conversation_id"] for r in results_b]

    assert conv_a not in conv_ids_b, (
        "RLS isolation FAILED: Tenant B can see Tenant A's conversation"
    )

    await backend.close()


# ---------------------------------------------------------------------------
# Plan 007-B: append_raw_event tests (compliance audit log)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_append_raw_event_success_returns_id():
    """append_raw_event INSERTs into raw_events under RLS and returns the row id."""
    pool, conn, txn = _make_mock_pool_and_conn()
    expected_id = str(uuid.uuid4())
    conn.fetchrow = AsyncMock(return_value=_FakeRecord(id=expected_id))

    backend = NeonBackend(dsn="postgres://fake/db")
    backend._pool = pool

    tenant_id = str(uuid.uuid4())
    returned = await backend.append_raw_event(
        tenant_id=tenant_id,
        conversation_id=str(uuid.uuid4()),
        event_kind="slack_inbound",
        platform_message_id="1234567890.123456",
        raw_payload={"event": "test", "ts": "1234567890.123456"},
    )

    assert returned == expected_id
    # RLS GUC was set inside the transaction
    conn.execute.assert_any_await(f"SET LOCAL app.tenant_id = '{tenant_id}'")
    # The INSERT itself was issued
    assert conn.fetchrow.await_count == 1
    # Transaction committed cleanly
    txn.commit.assert_awaited_once()
    txn.rollback.assert_not_awaited()


@pytest.mark.asyncio
async def test_append_raw_event_idempotent_duplicate_returns_none():
    """ON CONFLICT DO NOTHING — fetchrow returns None on duplicate; method returns None."""
    pool, conn, txn = _make_mock_pool_and_conn()
    conn.fetchrow = AsyncMock(return_value=None)  # duplicate hit ON CONFLICT

    backend = NeonBackend(dsn="postgres://fake/db")
    backend._pool = pool

    returned = await backend.append_raw_event(
        tenant_id=str(uuid.uuid4()),
        conversation_id=str(uuid.uuid4()),
        event_kind="slack_outbound",
        platform_message_id="1234567890.654321",
        raw_payload={"ts": "1234567890.654321"},
    )

    assert returned is None  # idempotent — no error, no double-write


@pytest.mark.asyncio
async def test_append_raw_event_pool_unavailable_returns_none():
    """No pool → return None silently (audit path NEVER raises)."""
    backend = NeonBackend(dsn="postgres://fake/db")
    backend._pool = None  # not initialised

    returned = await backend.append_raw_event(
        tenant_id=str(uuid.uuid4()),
        conversation_id=None,
        event_kind="tool_call_request",
        platform_message_id=None,
        raw_payload={"tool": "x", "args": {}},
    )

    assert returned is None


@pytest.mark.asyncio
async def test_append_raw_event_swallows_db_exception():
    """If the INSERT raises, return None and do not propagate (best-effort)."""
    pool, conn, txn = _make_mock_pool_and_conn()
    conn.fetchrow = AsyncMock(side_effect=RuntimeError("simulated DB error"))

    backend = NeonBackend(dsn="postgres://fake/db")
    backend._pool = pool

    # Must NOT raise; must return None.
    returned = await backend.append_raw_event(
        tenant_id=str(uuid.uuid4()),
        conversation_id=str(uuid.uuid4()),
        event_kind="tool_call_response",
        platform_message_id=None,
        raw_payload={"result": "ok"},
    )

    assert returned is None
