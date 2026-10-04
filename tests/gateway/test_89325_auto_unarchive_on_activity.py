"""A chat the idle sweep archived must resurface when real user activity arrives (#89325).

Archiving is a soft hide: the session row keeps every message and inbound delivery still routes
to it, but the flag hides it from the default session list on every surface (desktop sidebar,
TUI, CLI). The sweep's archive (``sessions.auto_archive``) is bookkeeping, not a decision, so a
conversation still receiving messages from any channel is live again and must come back —
otherwise a chat you keep using from WhatsApp, BlueBubbles, Photon, Telegram, ... stays hidden on
the desktop UI forever while messages pile up invisibly.

The DELIBERATE archive (user, CLI, API) is the opposite decision and must survive exactly this
traffic: ``auto_archived`` is the provenance that separates the two (#127019), and this feature
clears the sweep's stamp only. Internal/system events (cron deliveries, background-process
completions, startup-restore replays) are not user activity and keep the flag either way.
"""

import sys
import time
import types
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock

import pytest

import gateway.run as gateway_run
import gateway.run_turn as gateway_run_turn
from gateway.config import GatewayConfig, Platform
from gateway.platforms.base import MessageEvent
from gateway.session import SessionEntry, SessionSource
from hermes_state import AsyncSessionDB, SessionDB


# ---------------------------------------------------------------------------
# SessionDB door: the sweep's archive clears, the user's does not
# ---------------------------------------------------------------------------

@pytest.fixture
def db(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
    return SessionDB(tmp_path / "state.db")


def _sidebar_ids(db) -> set:
    """Same flags as the desktop sidebar slice (tips projected from roots)."""
    return {
        row["id"]
        for row in db.list_sessions_rich(
            limit=50, offset=0, min_message_count=1, include_archived=False,
            archived_only=False, order_by_last_active=True, compact_rows=True, include_pinned=True,
        )
    }


def _live_chat(db, session_id="chat", source="telegram"):
    db.create_session(session_id, source)
    db.append_message(session_id, "user", "hello")
    return session_id


def _lineage(db):
    """root -> mid (compression) -> tip, all with messages."""
    db.create_session("root", "telegram")
    db.append_message("root", "user", "hello")
    _compress(db, "root", "mid")
    _compress(db, "mid", "tip")
    db.append_message("tip", "user", "latest")


def _compress(db, parent, child):
    holder = f"holder-{child}"
    assert db.try_acquire_compression_lock(parent, holder, ttl_seconds=60)
    db.publish_compression_child(
        parent_session_id=parent, child_session_id=child, source="telegram", system_prompt="p",
        messages=[{"role": "user", "content": f"summary for {child}"}], compression_lock_holder=holder)


def _sweep(db):
    """The idle sweep: archives every stale lineage and stamps ``auto_archived``."""
    time.sleep(0.02)
    return db.archive_stale_sessions(0)


def _flags(db, *ids):
    return {sid: (db.get_session(sid)["archived"], db.get_session(sid)["auto_archived"]) for sid in ids}


def test_swept_chat_resurfaces_on_activity(db):
    sid = _live_chat(db)
    assert _sweep(db) == 1
    assert _flags(db, sid) == {sid: (1, 1)}
    assert _sidebar_ids(db) == set()

    assert db.unarchive_on_activity(sid) is True

    assert _flags(db, sid) == {sid: (0, 0)}
    assert _sidebar_ids(db) == {sid}


def test_activity_unhides_the_whole_lineage(db):
    """The sidebar admits a lineage by its ROOT, so the sweep's stamp must clear across it."""
    _lineage(db)
    assert _sweep(db) == 1
    assert _flags(db, "root", "mid", "tip") == {s: (1, 1) for s in ("root", "mid", "tip")}
    assert _sidebar_ids(db) == set()

    assert db.unarchive_on_activity("tip") is True

    assert _flags(db, "root", "mid", "tip") == {s: (0, 0) for s in ("root", "mid", "tip")}
    assert _sidebar_ids(db) == {"tip"}


def test_deliberate_archive_survives_activity(db):
    sid = _live_chat(db)
    db.set_session_archived(sid, True)  # the user's own archive
    assert db.get_session(sid)["auto_archived"] == 0

    assert db.unarchive_on_activity(sid) is False

    assert db.get_session(sid)["archived"] == 1
    assert _sidebar_ids(db) == set()


def test_deliberate_archive_wins_over_an_earlier_sweep(db):
    """Re-archiving a swept chat makes it deliberate, so later traffic leaves it hidden."""
    sid = _live_chat(db)
    assert _sweep(db) == 1
    db.set_session_archived(sid, True)
    assert db.get_session(sid)["auto_archived"] == 0

    assert db.unarchive_on_activity(sid) is False
    assert _sidebar_ids(db) == set()


def test_unarchived_chat_is_a_noop(db):
    sid = _live_chat(db)

    assert db.unarchive_on_activity(sid) is False

    assert _flags(db, sid) == {sid: (0, 0)}
    assert _sidebar_ids(db) == {sid}


def test_unknown_and_empty_ids_are_noops(db):
    assert db.unarchive_on_activity("") is False
    assert db.unarchive_on_activity("sess-nope") is False


@pytest.mark.asyncio
async def test_async_door_offloads_the_same_call(db):
    sid = _live_chat(db)
    assert _sweep(db) == 1

    assert await AsyncSessionDB(db).unarchive_on_activity(sid) is True

    assert _sidebar_ids(db) == {sid}


# ---------------------------------------------------------------------------
# Gateway helper + wiring
# ---------------------------------------------------------------------------

def _make_db(tmp_path):
    """Real SessionDB + async door on an isolated temp file."""
    monkeypatch_home = tmp_path / "home"
    monkeypatch_home.mkdir(exist_ok=True)
    db = SessionDB(db_path=tmp_path / "state.db")
    return AsyncSessionDB(db)


@pytest.mark.asyncio
async def test_helper_none_db_is_noop(tmp_path):
    assert await gateway_run_turn._unarchive_session_on_activity(None, "sess-1") is False


@pytest.mark.asyncio
async def test_helper_empty_session_id_is_noop(tmp_path):
    db = _make_db(tmp_path)
    assert await gateway_run_turn._unarchive_session_on_activity(db, "") is False


@pytest.mark.asyncio
async def test_helper_swallows_a_broken_store(tmp_path):
    """A store error must never break the inbound turn it rides on."""
    db = MagicMock()
    db.unarchive_on_activity = AsyncMock(side_effect=RuntimeError("db gone"))
    assert await gateway_run_turn._unarchive_session_on_activity(db, "sess-1") is False


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


async def _run_turn(runner, event):
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
        event, _source(), "agent:main:telegram:group:-1001:12345", 1
    )


@pytest.mark.asyncio
async def test_user_message_unarchives_session(monkeypatch, tmp_path):
    runner = _bootstrap(monkeypatch, tmp_path)
    unarchive = AsyncMock(return_value=True)
    monkeypatch.setattr(gateway_run_turn, "_unarchive_session_on_activity", unarchive)

    await _run_turn(runner, _event())

    unarchive.assert_awaited_once()
    args = unarchive.await_args.args
    assert args[0] is runner._session_db
    assert args[1] == "sess-wired"


@pytest.mark.asyncio
async def test_internal_event_keeps_session_archived(monkeypatch, tmp_path):
    runner = _bootstrap(monkeypatch, tmp_path)
    unarchive = AsyncMock(return_value=False)
    monkeypatch.setattr(gateway_run_turn, "_unarchive_session_on_activity", unarchive)

    await _run_turn(runner, _event(internal=True))

    unarchive.assert_not_awaited()
