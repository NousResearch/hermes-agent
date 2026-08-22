"""Regression tests for #81513: CLI→gateway handoff must not hijack a live
gateway session.

Bug: ``_process_handoff`` blindly called ``switch_session`` on the
destination session key. When that key was already bound to a gateway
session the gateway was actively serving (e.g. an ongoing QQ conversation
on the home channel), ``switch_session`` ENDED the live session and
rebound the key to the CLI session — silently replacing the active
conversation and making its history unreachable from the platform.

The fix adds a source/ownership check before the switch: the handoff is
rejected when the destination key is already bound to a live gateway
session (different session id, not ended, with messages). It remains a
no-op rejection when the destination key is already bound to the CLI
session itself (the CLI resumed the gateway session). Rebind is allowed
when the bound session is ended or an empty fresh reset, and when the
destination key is unbound (the normal first-handoff path).
"""

import asyncio
from datetime import datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from hermes_state import AsyncSessionDB, SessionDB
from gateway.config import GatewayConfig, HomeChannel, Platform, PlatformConfig
from gateway.session import SessionEntry, SessionSource, build_session_key


def _make_qq_runner(session_db, *, prebound_entries=None, active_processes=False):
    """Build a GatewayRunner stand-in wired for qqbot handoff.

    ``prebound_entries`` maps session_key → session_id and is seeded into
    the fake session store's routing index so the ownership check sees an
    existing binding (as the real gateway would after a QQ conversation).
    ``active_processes`` makes the store report in-flight agent work on
    every key (mirrors process_registry.has_active_for_session).
    """
    from gateway.run import GatewayRunner

    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(
        platforms={Platform.QQBOT: PlatformConfig(enabled=True, token="***")}
    )
    runner.config.platforms[Platform.QQBOT].home_channel = HomeChannel(
        platform=Platform.QQBOT,
        chat_id="OPENID123",
        name="QQ DM",
    )
    adapter = MagicMock()
    adapter.send = AsyncMock(return_value=SimpleNamespace(success=True))
    adapter.create_handoff_thread = AsyncMock(return_value=None)
    adapter._bot = None
    runner.adapters = {Platform.QQBOT: adapter}
    runner.hooks = SimpleNamespace(
        emit=AsyncMock(),
        emit_collect=AsyncMock(return_value=[]),
        loaded_hooks=False,
    )

    class _FakeSessionStore:
        def __init__(self):
            self._entries = {}
            for key, session_id in (prebound_entries or {}).items():
                self._entries[key] = SessionEntry(
                    session_key=key,
                    session_id=session_id,
                    created_at=datetime.now(),
                    updated_at=datetime.now(),
                    platform=Platform.QQBOT,
                    chat_type="dm",
                )
            self.switch_calls = []
            self.create_calls = []

        def _ensure_loaded(self):
            return None

        def _has_active_processes_safe(self, session_key, *, context):
            return active_processes

        def _generate_session_key(self, source):
            return build_session_key(
                source,
                group_sessions_per_user=True,
                thread_sessions_per_user=False,
            )

        def get_or_create_session(self, source, force_new=False):
            key = self._generate_session_key(source)
            self.create_calls.append(key)
            if key not in self._entries:
                self._entries[key] = SessionEntry(
                    session_key=key,
                    session_id="fresh-session",
                    created_at=datetime.now(),
                    updated_at=datetime.now(),
                    platform=source.platform,
                    chat_type=source.chat_type,
                    origin=source,
                )
            return self._entries[key]

        def switch_session(self, session_key, target_session_id):
            self.switch_calls.append((session_key, target_session_id))
            return SessionEntry(
                session_key=session_key,
                session_id=target_session_id,
                created_at=datetime.now(),
                updated_at=datetime.now(),
                platform=Platform.QQBOT,
                chat_type="dm",
            )

    runner.session_store = _FakeSessionStore()
    runner._session_db = AsyncSessionDB(session_db)
    runner._evict_cached_agent = MagicMock()
    runner._release_running_agent_state = MagicMock()
    runner._handle_message = AsyncMock(return_value="handoff ok")
    return runner


def _dest_key() -> str:
    """The session key a qqbot handoff produces for home chat OPENID123."""
    source = SessionSource(
        platform=Platform.QQBOT,
        chat_id="OPENID123",
        chat_name="QQ DM",
        chat_type="dm",
        user_id="system:handoff",
        user_name="Handoff",
        thread_id=None,
    )
    return build_session_key(
        source,
        group_sessions_per_user=True,
        thread_sessions_per_user=False,
    )


def _seed_live_session(db, session_id, *, message_count=5, ended=False):
    """Create a real gateway-session row in state.db for the bound entry."""
    db.create_session(session_id, source="qqbot", chat_id="OPENID123", chat_type="dm")
    if message_count:
        db._conn.execute(
            "UPDATE sessions SET message_count = ? WHERE id = ?",
            (message_count, session_id),
        )
        db._conn.commit()
    if ended:
        db.end_session(session_id, "user")


@pytest.mark.asyncio
async def test_handoff_rejects_when_dest_key_bound_to_live_gateway_session(tmp_path):
    """A handoff must NOT end a live gateway conversation on the destination key."""
    db = SessionDB(db_path=tmp_path / "state.db")
    key = _dest_key()
    _seed_live_session(db, "gw-live-session", message_count=7)
    runner = _make_qq_runner(db, prebound_entries={key: "gw-live-session"})

    with pytest.raises(RuntimeError, match="already bound to live gateway session"):
        await runner._process_handoff({
            "id": "cli-session",
            "title": "CLI work",
            "handoff_platform": "qqbot",
        })

    # The live session must not be switched away, and no synthetic turn runs.
    assert runner.session_store.switch_calls == []
    runner._handle_message.assert_not_called()


@pytest.mark.asyncio
async def test_handoff_rejects_when_session_already_gateway_owned(tmp_path):
    """Handing off the gateway session itself is a no-op, not a re-switch."""
    db = SessionDB(db_path=tmp_path / "state.db")
    key = _dest_key()
    _seed_live_session(db, "cli-session", message_count=3)
    runner = _make_qq_runner(db, prebound_entries={key: "cli-session"})

    with pytest.raises(RuntimeError, match="already the gateway session"):
        await runner._process_handoff({
            "id": "cli-session",
            "title": "CLI work",
            "handoff_platform": "qqbot",
        })

    assert runner.session_store.switch_calls == []
    runner._handle_message.assert_not_called()


@pytest.mark.asyncio
async def test_handoff_allows_when_bound_session_ended(tmp_path):
    """An ended destination session is safe to rebind (normal re-handoff)."""
    db = SessionDB(db_path=tmp_path / "state.db")
    key = _dest_key()
    _seed_live_session(db, "old-qq-session", message_count=9, ended=True)
    runner = _make_qq_runner(db, prebound_entries={key: "old-qq-session"})

    await runner._process_handoff({
        "id": "cli-session",
        "title": "CLI work",
        "handoff_platform": "qqbot",
    })

    assert runner.session_store.switch_calls == [(key, "cli-session")]
    runner._handle_message.assert_awaited_once()


@pytest.mark.asyncio
async def test_handoff_allows_when_bound_session_is_empty_fresh_reset(tmp_path):
    """A fresh /new session (zero messages) is safe to replace."""
    db = SessionDB(db_path=tmp_path / "state.db")
    key = _dest_key()
    _seed_live_session(db, "fresh-empty", message_count=0)
    runner = _make_qq_runner(db, prebound_entries={key: "fresh-empty"})

    await runner._process_handoff({
        "id": "cli-session",
        "title": "CLI work",
        "handoff_platform": "qqbot",
    })

    assert runner.session_store.switch_calls == [(key, "cli-session")]
    runner._handle_message.assert_awaited_once()


@pytest.mark.asyncio
async def test_handoff_allows_unbound_destination(tmp_path):
    """First handoff to a never-used home channel still works."""
    db = SessionDB(db_path=tmp_path / "state.db")
    key = _dest_key()
    runner = _make_qq_runner(db, prebound_entries={})

    await runner._process_handoff({
        "id": "cli-session",
        "title": "CLI work",
        "handoff_platform": "qqbot",
    })

    assert runner.session_store.switch_calls == [(key, "cli-session")]
    runner._handle_message.assert_awaited_once()


@pytest.mark.asyncio
async def test_handoff_rejects_when_dest_has_active_agent_work(tmp_path):
    """In-flight agent work on the destination key counts as live, even with
    zero persisted messages (first turn still running)."""
    db = SessionDB(db_path=tmp_path / "state.db")
    key = _dest_key()
    _seed_live_session(db, "gw-busy", message_count=0)
    runner = _make_qq_runner(
        db,
        prebound_entries={key: "gw-busy"},
        active_processes=True,
    )

    with pytest.raises(RuntimeError, match="has active agent work"):
        await runner._process_handoff({
            "id": "cli-session",
            "title": "CLI work",
            "handoff_platform": "qqbot",
        })

    assert runner.session_store.switch_calls == []
    runner._handle_message.assert_not_called()


@pytest.mark.asyncio
async def test_handoff_fails_closed_when_bound_session_state_unverifiable(tmp_path):
    """If the bound session's state cannot be read, refuse the handoff rather
    than risk hijacking a live conversation (fail-closed)."""
    db = SessionDB(db_path=tmp_path / "state.db")
    key = _dest_key()
    _seed_live_session(db, "gw-unknown", message_count=1)
    runner = _make_qq_runner(db, prebound_entries={key: "gw-unknown"})

    def _boom(*_args, **_kwargs):
        raise RuntimeError("db exploded")

    runner._session_db.get_session = _boom

    with pytest.raises(RuntimeError, match="cannot verify session state"):
        await runner._process_handoff({
            "id": "cli-session",
            "title": "CLI work",
            "handoff_platform": "qqbot",
        })

    assert runner.session_store.switch_calls == []
    runner._handle_message.assert_not_called()


@pytest.mark.asyncio
async def test_handoff_proceeds_when_no_session_db(tmp_path):
    """Without a SessionDB the guard falls back to routing-entry + active-work
    signals only, and does NOT fail closed on the absent DB (JSONL fallback /
    memory-only store must still be able to hand off)."""
    key = _dest_key()
    runner = _make_qq_runner(None, prebound_entries={key: "in-memory-session"})
    runner._session_db = None

    await runner._process_handoff({
        "id": "cli-session",
        "title": "CLI work",
        "handoff_platform": "qqbot",
    })

    assert runner.session_store.switch_calls == [(key, "cli-session")]
    runner._handle_message.assert_awaited_once()
