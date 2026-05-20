"""
tests/test_storage_sqlite.py — Unit tests for hermes_storage.SQLiteBackend.

All tests use a tmpdir-scoped SQLite DB so they never touch the real state.db.
No network or credentials required.

Tests verify:
  - SQLiteBackend satisfies StorageBackend protocol (isinstance check).
  - get_or_create_conversation: creates new session, returns same ID on repeat call.
  - append_message: persists message, returns message ID string.
  - get_conversation_history: returns messages in chronological order.
  - search_sessions: finds sessions matching a query substring.
  - HERMES_MODE != 'saas' → get_backend() returns SQLiteBackend.
  - SQLiteBackend.close() is safe to call multiple times.
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import Optional
from unittest.mock import patch

import pytest

from hermes_identity import HermesIdentity
from hermes_storage import StorageBackend, get_backend, reset_backend
from hermes_storage.sqlite_backend import SQLiteBackend


# ---------------------------------------------------------------------------
# Fixtures
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


@pytest.fixture
def db_path(tmp_path: Path) -> Path:
    return tmp_path / "test_state.db"


@pytest.fixture
def backend(db_path: Path) -> SQLiteBackend:
    return SQLiteBackend(db_path=db_path)


# ---------------------------------------------------------------------------
# Protocol compliance
# ---------------------------------------------------------------------------

def test_sqlite_backend_satisfies_protocol(backend: SQLiteBackend) -> None:
    """isinstance() check confirms runtime_checkable Protocol is satisfied."""
    assert isinstance(backend, StorageBackend), (
        "SQLiteBackend must satisfy the StorageBackend Protocol"
    )


# ---------------------------------------------------------------------------
# get_or_create_conversation
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_get_or_create_conversation_creates_new(backend: SQLiteBackend) -> None:
    identity = make_identity()
    conv_id = await backend.get_or_create_conversation(identity, "#general", None)
    assert conv_id, "Should return a non-empty conversation ID"
    assert isinstance(conv_id, str)


@pytest.mark.asyncio
async def test_get_or_create_conversation_idempotent(backend: SQLiteBackend) -> None:
    """Same (identity, channel, thread) returns the same conversation_id."""
    identity = make_identity()
    conv_id_1 = await backend.get_or_create_conversation(identity, "#general", None)
    conv_id_2 = await backend.get_or_create_conversation(identity, "#general", None)
    assert conv_id_1 == conv_id_2, "Repeated call must return same conversation ID"


@pytest.mark.asyncio
async def test_get_or_create_conversation_different_channels(backend: SQLiteBackend) -> None:
    """Different channel → different conversation."""
    identity = make_identity()
    conv_id_1 = await backend.get_or_create_conversation(identity, "#general", None)
    conv_id_2 = await backend.get_or_create_conversation(identity, "#random", None)
    assert conv_id_1 != conv_id_2


@pytest.mark.asyncio
async def test_get_or_create_conversation_thread_vs_toplevel(backend: SQLiteBackend) -> None:
    """Thread ID produces a different conversation from top-level."""
    identity = make_identity()
    top_conv = await backend.get_or_create_conversation(identity, "#general", None)
    thread_conv = await backend.get_or_create_conversation(identity, "#general", "thread_ts_123")
    assert top_conv != thread_conv


# ---------------------------------------------------------------------------
# append_message
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_append_message_returns_id(backend: SQLiteBackend) -> None:
    identity = make_identity()
    conv_id = await backend.get_or_create_conversation(identity, "#general", None)
    msg_id = await backend.append_message(conv_id, "user", "Hello, Hermes!")
    assert isinstance(msg_id, str)
    assert msg_id  # non-empty


@pytest.mark.asyncio
async def test_append_message_multiple(backend: SQLiteBackend) -> None:
    """Each append returns a distinct ID."""
    identity = make_identity()
    conv_id = await backend.get_or_create_conversation(identity, "#general", None)
    id_1 = await backend.append_message(conv_id, "user", "First message")
    id_2 = await backend.append_message(conv_id, "assistant", "Response")
    assert id_1 != id_2


# ---------------------------------------------------------------------------
# get_conversation_history
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_get_conversation_history_returns_messages(backend: SQLiteBackend) -> None:
    identity = make_identity()
    conv_id = await backend.get_or_create_conversation(identity, "#general", None)
    await backend.append_message(conv_id, "user", "Hello, Hermes!")
    await backend.append_message(conv_id, "assistant", "Hello! How can I help?")

    history = await backend.get_conversation_history(conv_id)
    assert len(history) == 2
    roles = [m["role"] for m in history]
    assert "user" in roles
    assert "assistant" in roles


@pytest.mark.asyncio
async def test_get_conversation_history_chronological_order(backend: SQLiteBackend) -> None:
    """Messages are returned oldest-first."""
    identity = make_identity()
    conv_id = await backend.get_or_create_conversation(identity, "#ch", None)
    await backend.append_message(conv_id, "user", "First")
    await backend.append_message(conv_id, "assistant", "Second")
    await backend.append_message(conv_id, "user", "Third")

    history = await backend.get_conversation_history(conv_id)
    assert history[0]["content"] == "First"
    assert history[1]["content"] == "Second"
    assert history[2]["content"] == "Third"


@pytest.mark.asyncio
async def test_get_conversation_history_limit(backend: SQLiteBackend) -> None:
    """The limit parameter caps returned messages."""
    identity = make_identity()
    conv_id = await backend.get_or_create_conversation(identity, "#ch", None)
    for i in range(10):
        await backend.append_message(conv_id, "user", f"Message {i}")

    history = await backend.get_conversation_history(conv_id, limit=3)
    assert len(history) == 3


@pytest.mark.asyncio
async def test_get_conversation_history_empty(backend: SQLiteBackend) -> None:
    """New conversation with no messages returns empty list."""
    identity = make_identity()
    conv_id = await backend.get_or_create_conversation(identity, "#ch", None)
    history = await backend.get_conversation_history(conv_id)
    assert history == []


# ---------------------------------------------------------------------------
# search_sessions
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_search_sessions_finds_matching(backend: SQLiteBackend) -> None:
    identity = make_identity()
    conv_id = await backend.get_or_create_conversation(identity, "#general", None)
    await backend.append_message(conv_id, "user", "The quick brown fox jumps")

    results = await backend.search_sessions("quick brown fox", identity)
    assert len(results) >= 1
    assert any(r["conversation_id"] == conv_id for r in results)


@pytest.mark.asyncio
async def test_search_sessions_no_cross_user_results(
    db_path: Path,
) -> None:
    """User A's messages are not returned when searching as User B."""
    backend = SQLiteBackend(db_path=db_path)
    identity_a = make_identity(user_id="UUSER_A")
    identity_b = make_identity(user_id="UUSER_B")

    conv_a = await backend.get_or_create_conversation(identity_a, "#ch", None)
    await backend.append_message(conv_a, "user", "secret_message_user_a")

    # User B searches — should not see User A's messages.
    results_b = await backend.search_sessions("secret_message_user_a", identity_b)
    conv_ids_b = [r["conversation_id"] for r in results_b]
    assert conv_a not in conv_ids_b, "User B must not see User A's messages"

    await backend.close()


@pytest.mark.asyncio
async def test_search_sessions_no_match(backend: SQLiteBackend) -> None:
    identity = make_identity()
    results = await backend.search_sessions("zzznomatch_xyzzy", identity)
    assert results == []


# ---------------------------------------------------------------------------
# close
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_close_safe_multiple_times(backend: SQLiteBackend) -> None:
    """close() is idempotent — calling twice should not raise."""
    await backend.close()
    await backend.close()  # Second call must not raise.


# ---------------------------------------------------------------------------
# Factory: HERMES_MODE != 'saas' → SQLiteBackend
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_get_backend_local_mode_returns_sqlite(tmp_path: Path) -> None:
    await reset_backend()
    with patch.dict(os.environ, {"HERMES_MODE": "local", "HERMES_HOME": str(tmp_path)}):
        backend = await get_backend()
        assert isinstance(backend, SQLiteBackend)
    await reset_backend()


@pytest.mark.asyncio
async def test_get_backend_no_mode_env_returns_sqlite(tmp_path: Path) -> None:
    await reset_backend()
    env = {k: v for k, v in os.environ.items() if k != "HERMES_MODE"}
    env["HERMES_HOME"] = str(tmp_path)
    with patch.dict(os.environ, env, clear=True):
        backend = await get_backend()
        assert isinstance(backend, SQLiteBackend)
    await reset_backend()


@pytest.mark.asyncio
async def test_get_backend_singleton(tmp_path: Path) -> None:
    """get_backend() returns the same object on repeated calls."""
    await reset_backend()
    with patch.dict(os.environ, {"HERMES_MODE": "local", "HERMES_HOME": str(tmp_path)}):
        b1 = await get_backend()
        b2 = await get_backend()
        assert b1 is b2
    await reset_backend()
