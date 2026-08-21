"""Tests for auto-unarchive on inbound activity (#89325).

Archiving is a soft hide: the session row keeps every message and inbound
delivery still routes to it, but the archived flag hides it from the default
session list on every surface. A conversation that receives a real user
message from any platform channel must come back — otherwise a chat you are
actively using on WhatsApp, BlueBubbles, Photon, Telegram, ... stays hidden
on the desktop UI forever while messages pile up invisibly.

Internal/system events (cron deliveries, background-process completions,
startup-restore replays, plugin injections) are NOT user activity and must
keep the archive flag — otherwise background traffic drags chats the user
deliberately hid back into the main list.
"""

import sys
import types
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock

import pytest

import gateway.run as gateway_run
from gateway.config import GatewayConfig, Platform
from gateway.platforms.base import MessageEvent
from gateway.session import SessionEntry, SessionSource
from hermes_state import AsyncSessionDB, SessionDB


# ---------------------------------------------------------------------------
# Helper under test — real DB
# ---------------------------------------------------------------------------

def _make_db(tmp_path):
    """Real SessionDB + async door on an isolated temp file."""
    db = SessionDB(db_path=tmp_path / "state.db")
    return AsyncSessionDB(db)


def _create_session(db: SessionDB, session_id: str = "sess-1") -> str:
    """Create a session row directly (bypassing the async door)."""
    db.create_session(session_id=session_id, source="test")
    return session_id


@pytest.mark.asyncio
async def test_unarchives_archived_session(tmp_path):
    db = _make_db(tmp_path)
    session_id = _create_session(db._db)
    db._db.set_session_archived(session_id, True)
    assert db._db.get_session(session_id)["archived"] == 1

    result = await gateway_run._unarchive_session_on_activity(db, session_id)

    assert result is True
    assert db._db.get_session(session_id)["archived"] == 0


@pytest.mark.asyncio
async def test_leaves_unarchived_session_alone(tmp_path):
    db = _make_db(tmp_path)
    session_id = _create_session(db._db)
    assert db._db.get_session(session_id)["archived"] == 0

    result = await gateway_run._unarchive_session_on_activity(db, session_id)

    assert result is False
    assert db._db.get_session(session_id)["archived"] == 0


@pytest.mark.asyncio
async def test_missing_session_is_noop(tmp_path):
    db = _make_db(tmp_path)
    result = await gateway_run._unarchive_session_on_activity(db, "sess-nope")
    assert result is False


@pytest.mark.asyncio
async def test_none_db_is_noop(tmp_path):
    result = await gateway_run._unarchive_session_on_activity(None, "sess-1")
    assert result is False


@pytest.mark.asyncio
async def test_empty_session_id_is_noop(tmp_path):
    db = _make_db(tmp_path)
    result = await gateway_run._unarchive_session_on_activity(db, "")
    assert result is False


# ---------------------------------------------------------------------------
# Wiring in _handle_message_with_agent
# ---------------------------------------------------------------------------

def _bootstrap(monkeypatch, tmp_path):
    """Minimal GatewayRunner setup shared by the wiring tests."""
    fake_dotenv = types.ModuleType("dotenv")
    fake_dotenv.load_dotenv = lambda *args, **kwargs: None
    monkeypatch.setitem(sys.modules, "dotenv", fake_dotenv)

    config = GatewayConfig()
    runner = gateway_run.GatewayRunner(config)
    runner.adapters = {}
    runner._running_agents = {}
    runner._running_agents_ts = {}
    runner._pending_messages = {}
    runner._pending_approvals = {}
    runner._is_user_authorized = lambda _source: True
    runner._set_session_env = lambda _context: None
    runner._handle_active_session_busy_message = AsyncMock(return_value=False)
    runner._session_db = MagicMock()
    runner._recover_telegram_topic_thread_id = lambda _source: None
    runner._cache_session_source = lambda _key, _source: None
    runner._is_session_run_current = lambda _key, _gen: True
    runner._begin_session_run_generation = lambda _key: 1
    runner._reply_anchor_for_event = lambda _event: None
    runner._get_guild_id = lambda _event: None
    runner._should_send_voice_reply = lambda *_a, **_kw: False
    runner.hooks = MagicMock()
    runner.hooks.emit = AsyncMock()
    # Telegram topic lane check — must be False for the plain DM path.
    runner._is_telegram_topic_lane = lambda _source: False

    runner.session_store = MagicMock()
    runner.session_store.get_or_create_session.return_value = SessionEntry(
        session_key="agent:main:telegram:group:-1001:12345",
        session_id="sess-wired",
        created_at=datetime.now(),
        updated_at=datetime.now(),
        platform=Platform.TELEGRAM,
        chat_type="group",
    )
    runner.session_store.load_transcript.return_value = []
    runner.session_store.append_to_transcript = MagicMock()
    runner.session_store.has_platform_message_id.return_value = False
    runner.session_store.update_session = MagicMock()

    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
    monkeypatch.setattr(
        gateway_run, "_resolve_runtime_agent_kwargs", lambda: {"api_key": "fake"}
    )
    monkeypatch.setattr(
        "agent.model_metadata.get_model_context_length",
        lambda *_args, **_kwargs: 100_000,
    )
    return runner


def _event(*, internal: bool = False):
    return MessageEvent(
        text="hello world",
        source=SessionSource(
            platform=Platform.TELEGRAM,
            chat_id="-1001",
            chat_type="group",
            user_id="12345",
        ),
        message_id="msg-42",
        internal=internal,
    )


def _source():
    return SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="-1001",
        chat_type="group",
        user_id="12345",
    )


@pytest.mark.asyncio
async def test_user_message_unarchives_session(monkeypatch, tmp_path):
    runner = _bootstrap(monkeypatch, tmp_path)
    unarchive = AsyncMock(return_value=True)
    monkeypatch.setattr(gateway_run, "_unarchive_session_on_activity", unarchive)
    runner._run_agent = AsyncMock(
        return_value={
            "final_response": "Hello!",
            "messages": [],
            "tools": [],
            "history_offset": 0,
            "last_prompt_tokens": 0,
        }
    )

    await runner._handle_message_with_agent(
        _event(), _source(), "agent:main:telegram:group:-1001:12345", 1
    )

    unarchive.assert_awaited_once()
    args = unarchive.await_args.args
    assert args[0] is runner._session_db
    assert args[1] == "sess-wired"


@pytest.mark.asyncio
async def test_internal_event_keeps_session_archived(monkeypatch, tmp_path):
    runner = _bootstrap(monkeypatch, tmp_path)
    unarchive = AsyncMock(return_value=False)
    monkeypatch.setattr(gateway_run, "_unarchive_session_on_activity", unarchive)
    runner._run_agent = AsyncMock(
        return_value={
            "final_response": "Hello!",
            "messages": [],
            "tools": [],
            "history_offset": 0,
            "last_prompt_tokens": 0,
        }
    )

    await runner._handle_message_with_agent(
        _event(internal=True),
        _source(),
        "agent:main:telegram:group:-1001:12345",
        1,
    )

    unarchive.assert_not_awaited()
