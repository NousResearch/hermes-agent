"""Tests for Telegram private-chat topic-mode routing.

Topic mode makes the root Telegram DM a system lobby while user-created
Telegram topics act as independent Hermes session lanes.
"""

import asyncio
import threading
from collections import OrderedDict
from datetime import datetime
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock, call, patch

import pytest

from hermes_state import SessionDB
from gateway.config import GatewayConfig, HomeChannel, Platform, PlatformConfig
from gateway.platforms.base import MessageEvent
from gateway.session import SessionEntry, SessionSource, build_session_key


def _make_source(*, thread_id: str | None = None) -> SessionSource:
    return SessionSource(
        platform=Platform.TELEGRAM,
        user_id="208214988",
        chat_id="208214988",
        user_name="tester",
        chat_type="dm",
        thread_id=thread_id,
    )


def _make_event(text: str, *, thread_id: str | None = None) -> MessageEvent:
    return MessageEvent(
        text=text,
        source=_make_source(thread_id=thread_id),
        message_id="m1",
    )


def _make_group_source(*, thread_id: str | None = None) -> SessionSource:
    return SessionSource(
        platform=Platform.TELEGRAM,
        user_id="208214988",
        chat_id="-100123",
        user_name="tester",
        chat_type="group",
        thread_id=thread_id,
    )


def _make_group_event(text: str, *, thread_id: str | None = None) -> MessageEvent:
    return MessageEvent(
        text=text,
        source=_make_group_source(thread_id=thread_id),
        message_id="gm1",
    )


def _make_runner(session_db=None):
    from gateway.run import GatewayRunner

    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(
        platforms={Platform.TELEGRAM: PlatformConfig(enabled=True, token="***")}
    )
    adapter = MagicMock()
    adapter.send = AsyncMock()
    adapter.send_image_file = AsyncMock()
    adapter._bot = None
    adapter._create_dm_topic = AsyncMock(return_value=None)
    adapter.rename_dm_topic = AsyncMock()
    adapter.get_forum_topic_icon_options = AsyncMock(return_value=[])
    adapter.dm_topic_custom_icon_state = MagicMock(return_value=None)
    runner.adapters = {Platform.TELEGRAM: adapter}
    runner._voice_mode = {}
    runner.hooks = SimpleNamespace(
        emit=AsyncMock(),
        emit_collect=AsyncMock(return_value=[]),
        loaded_hooks=False,
    )

    runner.session_store = MagicMock()
    runner.session_store._generate_session_key.side_effect = lambda source: build_session_key(
        source,
        group_sessions_per_user=getattr(runner.config, "group_sessions_per_user", True),
        thread_sessions_per_user=getattr(runner.config, "thread_sessions_per_user", False),
    )
    runner.session_store.get_or_create_session.side_effect = lambda source, force_new=False: SessionEntry(
        session_key=build_session_key(
            source,
            group_sessions_per_user=getattr(runner.config, "group_sessions_per_user", True),
            thread_sessions_per_user=getattr(runner.config, "thread_sessions_per_user", False),
        ),
        session_id="sess-topic" if source.thread_id else "sess-root",
        created_at=datetime.now(),
        updated_at=datetime.now(),
        platform=Platform.TELEGRAM,
        chat_type="dm",
        origin=source,
    )
    runner.session_store.load_transcript.return_value = []
    runner.session_store.has_any_sessions.return_value = True
    runner.session_store.append_to_transcript = MagicMock()
    runner.session_store.rewrite_transcript = MagicMock()
    runner.session_store.update_session = MagicMock()
    runner.session_store.reset_session = MagicMock(return_value=None)

    # Default switch_session impl: returns a SessionEntry carrying the target
    # session_id. Mirrors SessionStore.switch_session semantics for tests that
    # exercise Telegram topic binding rebinds without a real store.
    def _switch_session(session_key, target_session_id):
        return SessionEntry(
            session_key=session_key,
            session_id=target_session_id,
            created_at=datetime.now(),
            updated_at=datetime.now(),
            platform=Platform.TELEGRAM,
            chat_type="dm",
            origin=None,
        )
    runner.session_store.switch_session = MagicMock(side_effect=_switch_session)
    runner._running_agents = {}
    runner._running_agents_ts = {}
    runner._pending_messages = {}
    runner._pending_approvals = {}
    runner._queued_events = {}
    runner._busy_ack_ts = {}
    runner._session_model_overrides = {}
    runner._pending_model_notes = {}
    # Gateway holds the async facade; the slash handlers await it.
    if session_db is not None:
        from hermes_state import AsyncSessionDB
        session_db = AsyncSessionDB(session_db)
    runner._session_db = session_db
    runner._reasoning_config = None
    runner._provider_routing = {}
    runner._fallback_model = None
    runner._show_reasoning = False
    runner._draining = False
    runner._busy_input_mode = "interrupt"
    runner._is_user_authorized = lambda _source: True
    runner._session_key_for_source = lambda source: build_session_key(
        source,
        group_sessions_per_user=getattr(runner.config, "group_sessions_per_user", True),
        thread_sessions_per_user=getattr(runner.config, "thread_sessions_per_user", False),
    )
    runner._set_session_env = lambda _context: None
    runner._should_send_voice_reply = lambda *_args, **_kwargs: False
    runner._send_voice_reply = AsyncMock()
    runner._capture_gateway_honcho_if_configured = lambda *args, **kwargs: None
    runner._emit_gateway_run_progress = AsyncMock()
    runner._invalidate_session_run_generation = MagicMock()
    runner._begin_session_run_generation = MagicMock(return_value=1)
    runner._is_session_run_current = MagicMock(return_value=True)
    # Bypass the destructive-slash confirm gate — these tests focus on
    # /new topic-mode mechanics, not the confirm prompt itself.
    runner._read_user_config = lambda: {
        "approvals": {"destructive_slash_confirm": False}
    }
    runner._release_running_agent_state = MagicMock()
    runner._evict_cached_agent = MagicMock()
    runner._clear_session_boundary_security_state = MagicMock()
    runner._set_session_reasoning_override = MagicMock()
    runner._format_session_info = MagicMock(return_value="")
    return runner


@pytest.mark.asyncio
@pytest.mark.parametrize("thread_id", [None, "1"])
async def test_internal_root_telegram_dm_event_bypasses_topic_lobby(
    monkeypatch, thread_id
):
    import gateway.run as gateway_run

    runner = _make_runner()
    runner._telegram_topic_mode_enabled = lambda source: True
    runner._handle_message_with_agent = AsyncMock(return_value="agent response")

    monkeypatch.setattr(
        gateway_run, "_resolve_runtime_agent_kwargs", lambda: {"api_key": "***"}
    )

    event = MessageEvent(
        text="[SYSTEM: kanban task completed]",
        source=_make_source(thread_id=thread_id),
        message_id="wake-1",
        internal=True,
    )
    result = await runner._handle_message(event)

    assert result == "agent response"
    assert runner._handle_message_with_agent.await_count == 1
    assert runner._handle_message_with_agent.await_args.args[0] is event


@pytest.mark.asyncio
async def test_root_telegram_dm_new_shows_create_topic_instruction(monkeypatch):
    import gateway.run as gateway_run

    runner = _make_runner()
    runner._telegram_topic_mode_enabled = lambda source: True
    runner._run_agent = AsyncMock(
        side_effect=AssertionError("/new in root Telegram DM must not start an agent")
    )

    monkeypatch.setattr(
        gateway_run, "_resolve_runtime_agent_kwargs", lambda: {"api_key": "***"}
    )

    result = await runner._handle_message(_make_event("/new"))

    assert "create a new topic" in result
    assert "All Messages" in result
    assert "Use /new inside" in result
    runner._run_agent.assert_not_called()
    runner.session_store.reset_session.assert_not_called()
    runner.session_store.get_or_create_session.assert_not_called()


@pytest.mark.asyncio
async def test_managed_topic_binding_reuses_restored_session_over_static_lane_session(
    tmp_path, monkeypatch
):
    import gateway.run as gateway_run

    session_db = SessionDB(db_path=tmp_path / "state.db")
    session_db.enable_telegram_topic_mode(chat_id="208214988", user_id="208214988")
    session_db.create_session(
        session_id="restored-session",
        source="telegram",
        user_id="208214988",
    )
    session_db.bind_telegram_topic(
        chat_id="208214988",
        thread_id="17585",
        user_id="208214988",
        session_key=build_session_key(_make_source(thread_id="17585")),
        session_id="restored-session",
        managed_mode="restored",
    )
    runner = _make_runner(session_db=session_db)
    captured = {}

    async def fake_run_agent(*args, **kwargs):
        captured["session_id"] = kwargs.get("session_id")
        return {
            "success": True,
            "final_response": "restored response",
            "session_id": kwargs.get("session_id"),
            "messages": [],
        }

    runner._run_agent = AsyncMock(side_effect=fake_run_agent)

    monkeypatch.setattr(
        gateway_run, "_resolve_runtime_agent_kwargs", lambda: {"api_key": "***"}
    )

    result = await runner._handle_message(_make_event("continue restored", thread_id="17585"))

    assert result == "restored response"
    assert captured["session_id"] == "restored-session"


@pytest.mark.asyncio
async def test_telegram_group_prompt_is_not_topic_lobby_even_when_dm_topic_mode_enabled(
    tmp_path, monkeypatch
):
    import gateway.run as gateway_run

    session_db = SessionDB(db_path=tmp_path / "state.db")
    session_db.enable_telegram_topic_mode(chat_id="208214988", user_id="208214988")
    runner = _make_runner(session_db=session_db)
    runner._handle_message_with_agent = AsyncMock(return_value="group agent response")

    monkeypatch.setattr(
        gateway_run, "_resolve_runtime_agent_kwargs", lambda: {"api_key": "***"}
    )

    result = await runner._handle_message(_make_group_event("hello group", thread_id="555"))

    assert result == "group agent response"
    runner._handle_message_with_agent.assert_awaited_once()
    assert session_db.get_telegram_topic_binding(chat_id="-100123", thread_id="555") is None


@pytest.mark.asyncio
async def test_group_new_keeps_existing_reset_semantics_when_dm_topic_mode_enabled(
    tmp_path, monkeypatch
):
    import gateway.run as gateway_run

    session_db = SessionDB(db_path=tmp_path / "state.db")
    session_db.enable_telegram_topic_mode(chat_id="208214988", user_id="208214988")
    runner = _make_runner(session_db=session_db)
    group_source = _make_group_source(thread_id="555")
    group_key = build_session_key(group_source)
    new_entry = SessionEntry(
        session_key=group_key,
        session_id="new-group-session",
        created_at=datetime.now(),
        updated_at=datetime.now(),
        platform=Platform.TELEGRAM,
        chat_type="group",
        origin=group_source,
    )
    runner.session_store.reset_session.return_value = new_entry

    monkeypatch.setattr(
        gateway_run, "_resolve_runtime_agent_kwargs", lambda: {"api_key": "***"}
    )
    # /new appends a random tip from hermes_cli.tips; one tip's text contains
    # the phrase "parallel work", which collides with the negative assertion
    # below (observed as a 1-in-N CI flake). Pin the tip.
    monkeypatch.setattr(
        "hermes_cli.tips.get_random_tip", lambda: "pinned tip for test"
    )

    result = await runner._handle_message(_make_group_event("/new", thread_id="555"))

    assert "Started a new Hermes session in this topic" not in result
    assert "parallel work" not in result
    runner.session_store.reset_session.assert_called_once_with(group_key)


@pytest.mark.asyncio
async def test_new_inside_telegram_topic_rewrites_binding_to_new_session(tmp_path, monkeypatch):
    """Regression: /new inside a topic must rewrite the binding table.

    Previously /new reset the SessionStore entry but the
    telegram_dm_topic_bindings row still pointed at the old session_id;
    the next inbound message would look up the stale binding and switch
    back to the old session, making /new a no-op.
    """
    import gateway.run as gateway_run

    session_db = SessionDB(db_path=tmp_path / "state.db")
    session_db.enable_telegram_topic_mode(chat_id="208214988", user_id="208214988")
    session_db.create_session(
        session_id="old-topic-session",
        source="telegram",
        user_id="208214988",
    )
    topic_source = _make_source(thread_id="17585")
    topic_key = build_session_key(topic_source)
    session_db.bind_telegram_topic(
        chat_id="208214988",
        thread_id="17585",
        user_id="208214988",
        session_key=topic_key,
        session_id="old-topic-session",
    )

    runner = _make_runner(session_db=session_db)
    new_entry = SessionEntry(
        session_key=topic_key,
        session_id="new-topic-session",
        created_at=datetime.now(),
        updated_at=datetime.now(),
        platform=Platform.TELEGRAM,
        chat_type="dm",
        origin=topic_source,
    )
    # Mirror SessionStore.reset_session: in production it calls
    # SessionDB.create_session() for the new id before returning, so the
    # bindings FK can reference it.
    session_db.create_session(
        session_id="new-topic-session",
        source="telegram",
        user_id="208214988",
    )
    runner.session_store.reset_session.return_value = new_entry
    runner._agent_cache_lock = None

    monkeypatch.setattr(
        gateway_run, "_resolve_runtime_agent_kwargs", lambda: {"api_key": "***"}
    )

    await runner._handle_message(_make_event("/new", thread_id="17585"))

    binding = session_db.get_telegram_topic_binding(
        chat_id="208214988", thread_id="17585",
    )
    assert binding is not None
    assert binding["session_id"] == "new-topic-session"


@pytest.mark.asyncio
async def test_topic_binding_follows_compression_tip_on_read(tmp_path, monkeypatch):
    """Stale topic bindings auto-heal to the compression child on next inbound.

    Regression for #20470 / #29712 / #33414. After compression rotates the
    session_id, the binding row still pointed at the parent. On the next
    inbound message in that topic, the gateway used to reload the oversized
    parent transcript and re-run preflight compression — sometimes in a loop.
    The read path now walks ``SessionDB.get_compression_tip()`` and rewrites
    the binding to the descendant.
    """
    import gateway.run as gateway_run

    session_db = SessionDB(db_path=tmp_path / "state.db")
    session_db.enable_telegram_topic_mode(chat_id="208214988", user_id="208214988")
    # Build a parent -> compression child chain. end_session sets ended_at;
    # create_session sets started_at to "now", so the child's started_at is
    # always >= parent's ended_at on a real clock.
    session_db.create_session(
        session_id="parent-session", source="telegram", user_id="208214988",
    )
    session_db.end_session("parent-session", end_reason="compression")
    session_db.create_session(
        session_id="child-session",
        source="telegram",
        user_id="208214988",
        parent_session_id="parent-session",
    )
    topic_source = _make_source(thread_id="17585")
    topic_key = build_session_key(topic_source)
    # Pre-bug binding: topic still pointed at the pre-compression parent.
    session_db.bind_telegram_topic(
        chat_id="208214988",
        thread_id="17585",
        user_id="208214988",
        session_key=topic_key,
        session_id="parent-session",
    )

    runner = _make_runner(session_db=session_db)
    # switch_session() returns a SessionEntry pointing at whatever id was
    # requested; capture the requested id for assertion.
    switched_to: dict = {}

    def fake_switch(_key, new_session_id):
        switched_to["id"] = new_session_id
        return SessionEntry(
            session_key=topic_key,
            session_id=new_session_id,
            created_at=datetime.now(),
            updated_at=datetime.now(),
            platform=Platform.TELEGRAM,
            chat_type="dm",
            origin=topic_source,
        )

    runner.session_store.switch_session = MagicMock(side_effect=fake_switch)
    runner._run_agent = AsyncMock(
        return_value={
            "success": True,
            "final_response": "ok",
            "session_id": "child-session",
            "messages": [],
        }
    )

    monkeypatch.setattr(
        gateway_run, "_resolve_runtime_agent_kwargs", lambda: {"api_key": "***"}
    )

    await runner._handle_message(_make_event("follow up after compression", thread_id="17585"))

    # The route was advanced to the compression tip, not the stale parent.
    assert switched_to.get("id") == "child-session"
    # The binding row was rewritten to point at the descendant so future
    # inbound messages skip the tip walk and resolve directly.
    refreshed = session_db.get_telegram_topic_binding(
        chat_id="208214988", thread_id="17585",
    )
    assert refreshed is not None
    assert refreshed["session_id"] == "child-session"


@pytest.mark.asyncio
async def test_topic_root_command_lists_unlinked_sessions_for_restore(tmp_path, monkeypatch):
    import gateway.run as gateway_run

    session_db = SessionDB(db_path=tmp_path / "state.db")
    session_db.enable_telegram_topic_mode(chat_id="208214988", user_id="208214988")
    session_db.create_session(
        session_id="old-unlinked",
        source="telegram",
        user_id="208214988",
    )
    session_db.set_session_title("old-unlinked", "Old research")
    session_db.append_message("old-unlinked", "user", "first prompt")
    session_db.append_message("old-unlinked", "assistant", "old answer")
    session_db.create_session(
        session_id="already-linked",
        source="telegram",
        user_id="208214988",
    )
    session_db.set_session_title("already-linked", "Already linked")
    session_db.bind_telegram_topic(
        chat_id="208214988",
        thread_id="11111",
        user_id="208214988",
        session_key="agent:main:telegram:dm:208214988:11111",
        session_id="already-linked",
    )
    session_db.create_session(
        session_id="other-user",
        source="telegram",
        user_id="someone-else",
    )
    runner = _make_runner(session_db=session_db)
    runner._run_agent = AsyncMock(
        side_effect=AssertionError("root /topic status must not enter the agent loop")
    )

    monkeypatch.setattr(
        gateway_run, "_resolve_runtime_agent_kwargs", lambda: {"api_key": "***"}
    )

    result = await runner._handle_message(_make_event("/topic"))

    assert "Telegram multi-session topics are enabled" in result
    assert "Previous unlinked sessions" in result
    assert "Old research" in result
    assert "old-unlinked" in result
    assert "Send /topic old-unlinked inside a topic" in result
    assert "Already linked" not in result
    assert "other-user" not in result
    runner._run_agent.assert_not_called()


@pytest.mark.asyncio
async def test_first_message_inside_topic_records_topic_binding(tmp_path, monkeypatch):
    import gateway.run as gateway_run

    session_db = SessionDB(db_path=tmp_path / "state.db")
    session_db.enable_telegram_topic_mode(chat_id="208214988", user_id="208214988")
    session_db.create_session(
        session_id="sess-topic",
        source="telegram",
        user_id="208214988",
    )
    runner = _make_runner(session_db=session_db)
    runner._handle_message_with_agent = AsyncMock(return_value="agent response")

    monkeypatch.setattr(
        gateway_run, "_resolve_runtime_agent_kwargs", lambda: {"api_key": "***"}
    )

    source = _make_source(thread_id="17585")
    entry = runner.session_store.get_or_create_session(source)
    runner._record_telegram_topic_binding(source, entry)

    binding = session_db.get_telegram_topic_binding(
        chat_id="208214988",
        thread_id="17585",
    )
    assert binding is not None
    assert binding["user_id"] == "208214988"
    assert binding["session_id"] == "sess-topic"
    assert binding["session_key"] == build_session_key(_make_source(thread_id="17585"))


@pytest.mark.asyncio
async def test_handoff_to_telegram_dm_topic_uses_dm_lane_not_generic_thread(tmp_path):
    """Handoff-created Telegram DM topics must use the real DM-topic lane.

    A positive Telegram chat_id is a private chat. If handoff treats the new
    topic as generic chat_type="thread" with user_id="system:handoff", the
    synthetic turn lands under agent:...:thread:chat:topic while real user
    replies arrive as chat_type="dm" with user_id=chat_id. Recovery then sees
    the topic as unbound and can rewrite it to another recent topic.
    """
    session_db = SessionDB(db_path=tmp_path / "state.db")
    session_db.enable_telegram_topic_mode(chat_id="208214988", user_id="208214988")
    runner = _make_runner(session_db=session_db)
    runner.config.platforms[Platform.TELEGRAM].home_channel = HomeChannel(
        platform=Platform.TELEGRAM,
        chat_id="208214988",
        name="Tester DM",
    )
    adapter = cast(Any, runner.adapters[Platform.TELEGRAM])
    adapter.create_handoff_thread = AsyncMock(return_value="17585")
    adapter.send.return_value = SimpleNamespace(success=True)
    captured = {}

    async def fake_handle_message(event):
        captured["source"] = event.source
        return "handoff ok"

    runner._handle_message = AsyncMock(side_effect=fake_handle_message)

    await runner._process_handoff({
        "id": "cli-session",
        "title": "CLI work",
        "handoff_platform": "telegram",
    })

    expected_source = _make_source(thread_id="17585")
    expected_key = build_session_key(expected_source)
    runner.session_store.switch_session.assert_called_once_with(expected_key, "cli-session")
    assert captured["source"].chat_type == "dm"
    assert captured["source"].user_id == "208214988"
    assert captured["source"].thread_id == "17585"


@pytest.mark.asyncio
async def test_auto_generated_title_renames_bound_telegram_topic(tmp_path):
    db = SessionDB(db_path=tmp_path / "state.db")
    db.apply_telegram_topic_migration()
    db.create_session("sess-topic", source="telegram", user_id="208214988")
    db.bind_telegram_topic(
        chat_id="208214988",
        thread_id="42",
        user_id="208214988",
        session_key="agent:main:telegram:dm:208214988:42",
        session_id="sess-topic",
    )
    runner = _make_runner(session_db=db)
    runner._telegram_topic_mode_enabled = lambda source: True

    await runner._rename_telegram_topic_for_session_title(
        _make_source(thread_id="42"),
        "sess-topic",
        "  Build   Telegram Topic UX  ",
    )

    runner.adapters[Platform.TELEGRAM].rename_dm_topic.assert_awaited_once_with(
        chat_id="208214988",
        thread_id="42",
        name="Build Telegram Topic UX",
    )


@pytest.mark.asyncio
async def test_topic_refuses_unauthorized_user(tmp_path, monkeypatch):
    """Unauthorized DMs cannot flip multi-session mode on."""
    import gateway.run as gateway_run

    db = SessionDB(db_path=tmp_path / "state.db")
    runner = _make_runner(session_db=db)
    runner._is_user_authorized = lambda _source: False  # Deny

    monkeypatch.setattr(
        gateway_run, "_resolve_runtime_agent_kwargs", lambda: {"api_key": "***"}
    )

    result = await runner._handle_topic_command(_make_event("/topic"))

    assert "not authorized" in result.lower()
    # Tables must not be created for an unauthorized caller.
    tables = {
        row[0]
        for row in db._conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name LIKE 'telegram_dm%'"
        ).fetchall()
    }
    assert tables == set()


# ──────────────────────────────────────────────────────────────────────
# Cross-topic Reply leak / stripped-reply recovery
# ──────────────────────────────────────────────────────────────────────


def _seed_two_topic_bindings(session_db):
    """Create two topics for the same user in topic mode, oldest first."""
    session_db.enable_telegram_topic_mode(chat_id="208214988", user_id="208214988")
    # Seed two distinct sessions so the bind FK resolves.
    session_db.create_session(
        session_id="sess-A",
        source="telegram",
        user_id="208214988",
    )
    session_db.create_session(
        session_id="sess-B",
        source="telegram",
        user_id="208214988",
    )
    # Old topic A first, then current topic B (so B is "most recent").
    src_a = _make_source(thread_id="111")
    session_db.bind_telegram_topic(
        chat_id=src_a.chat_id,
        thread_id=src_a.thread_id,
        user_id=src_a.user_id,
        session_key=build_session_key(src_a),
        session_id="sess-A",
    )
    src_b = _make_source(thread_id="222")
    session_db.bind_telegram_topic(
        chat_id=src_b.chat_id,
        thread_id=src_b.thread_id,
        user_id=src_b.user_id,
        session_key=build_session_key(src_b),
        session_id="sess-B",
    )


def test_recover_preserves_unknown_thread_id_for_new_topic(tmp_path):
    # A newly-created Telegram DM topic arrives with a real, previously-unbound
    # message_thread_id. It must become its own session lane rather than being
    # rewritten to whichever older topic was most recently active.
    db = SessionDB(db_path=tmp_path / "state.db")
    _seed_two_topic_bindings(db)
    runner = _make_runner(session_db=db)

    assert runner._recover_telegram_topic_thread_id(_make_source(thread_id="9999")) is None


def test_recover_returns_none_for_brand_new_topic(tmp_path):
    # Regression for #31086: bindings exist for a prior topic but the user
    # opened a fresh one (thread_id "99999"). Recovery must return None so the
    # new topic gets its own session rather than being silently merged into
    # the previous topic's session. The hijack was self-reinforcing — because
    # the rewrite ran before _record_telegram_topic_binding, the new topic's
    # binding row never got written, so every subsequent message in that topic
    # looked "unknown" and was hijacked again.
    db = SessionDB(db_path=tmp_path / "state.db")
    db.enable_telegram_topic_mode(chat_id="208214988", user_id="208214988")
    db.create_session(session_id="sess-old", source="telegram", user_id="208214988")
    src_old = _make_source(thread_id="12345")
    db.bind_telegram_topic(
        chat_id=src_old.chat_id,
        thread_id=src_old.thread_id,
        user_id=src_old.user_id,
        session_key=build_session_key(src_old),
        session_id="sess-old",
    )
    runner = _make_runner(session_db=db)

    # "99999" is non-lobby and not in the binding table — brand-new topic.
    assert runner._recover_telegram_topic_thread_id(_make_source(thread_id="99999")) is None


def test_list_telegram_topic_bindings_for_chat_no_table(tmp_path):
    # Missing topic-mode tables → [] without auto-migrating.
    db = SessionDB(db_path=tmp_path / "state.db")
    assert db.list_telegram_topic_bindings_for_chat(chat_id="208214988") == []
    tables = {
        row[0]
        for row in db._conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name LIKE 'telegram_dm%'"
        ).fetchall()
    }
    assert tables == set()


# ---------------------------------------------------------------------------
# Tests for get_telegram_topic_binding_by_session (issue #27166)
# ---------------------------------------------------------------------------

def test_get_telegram_topic_binding_by_session_returns_binding(tmp_path):
    """Reverse lookup by session_id returns the binding row."""
    db = SessionDB(db_path=tmp_path / "state.db")
    db.enable_telegram_topic_mode(chat_id="208214988", user_id="208214988")
    db.create_session(session_id="sess-27166", source="telegram", user_id="208214988")
    db.bind_telegram_topic(
        chat_id="208214988",
        thread_id="17585",
        user_id="208214988",
        session_key="agent:main:telegram:dm:208214988:17585",
        session_id="sess-27166",
    )

    binding = db.get_telegram_topic_binding_by_session(session_id="sess-27166")

    assert binding is not None
    assert binding["chat_id"] == "208214988"
    assert binding["thread_id"] == "17585"
    assert binding["session_id"] == "sess-27166"


# ---------------------------------------------------------------------------
# Test for session-split thread_id recovery (issue #27166)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_first_auto_title_assigns_icon_when_creation_state_is_unknown(tmp_path):
    db = SessionDB(db_path=tmp_path / "state.db")
    db.apply_telegram_topic_migration()
    db.create_session("sess-topic", source="telegram", user_id="208214988")
    db.bind_telegram_topic(
        chat_id="208214988",
        thread_id="42",
        user_id="208214988",
        session_key="agent:main:telegram:dm:208214988:42",
        session_id="sess-topic",
    )
    runner = _make_runner(session_db=db)
    runner._telegram_topic_mode_enabled = lambda source: True
    adapter = cast(Any, runner.adapters[Platform.TELEGRAM])
    runner.config.platforms[Platform.TELEGRAM].extra["auto_topic_icons"] = True
    runner.config.platforms[Platform.TELEGRAM].extra["preserve_manual_topic_icons"] = True
    # Private DM topics do not reliably emit forum_topic_created updates. The
    # one-shot first-title path must remain eligible when state is unknown, or
    # every such topic keeps Telegram's default letter bubble.
    adapter.dm_topic_custom_icon_state.return_value = None
    adapter.get_forum_topic_icon_options.return_value = [
        {"emoji": "📊", "custom_emoji_id": "chart-id"},
        {"emoji": "🚀", "custom_emoji_id": "rocket-id"},
    ]

    with patch("agent.title_generator.choose_topic_icon", return_value="🚀") as choose:
        await runner._rename_telegram_topic_for_session_title(
            _make_source(thread_id="42"),
            "sess-topic",
            "ProjectAtlas",
            user_message="Improve the ProjectAtlas planning flow",
        )

    choose.assert_called_once_with(
        "ProjectAtlas",
        "Improve the ProjectAtlas planning flow",
        ["📊", "🚀"],
        recent_emojis=[],
    )
    adapter.rename_dm_topic.assert_awaited_once_with(
        chat_id="208214988",
        thread_id="42",
        name="ProjectAtlas",
        icon_custom_emoji_id="rocket-id",
    )


@pytest.mark.asyncio
async def test_auto_topic_icon_uses_secondary_transport_profile_config():
    runner = _make_runner()
    default_adapter = cast(Any, runner.adapters[Platform.TELEGRAM])
    default_adapter.config = runner.config.platforms[Platform.TELEGRAM]
    runner.config.platforms[Platform.TELEGRAM].extra["auto_topic_icons"] = False

    secondary_config = PlatformConfig(
        enabled=True,
        token="***",
        extra={
            "auto_topic_icons": True,
            "preserve_manual_topic_icons": False,
            "disable_topic_auto_rename": True,
        },
    )
    secondary_adapter = MagicMock()
    secondary_adapter.config = secondary_config
    secondary_adapter.get_forum_topic_icon_options = AsyncMock(
        return_value=[{"emoji": "🚀", "custom_emoji_id": "rocket-id"}]
    )
    runner._profile_adapters = {
        "work": {Platform.TELEGRAM: secondary_adapter},
    }
    source = _make_source(thread_id="42")
    source.profile = "work"

    with patch("agent.title_generator.choose_topic_icon", return_value="🚀"):
        selected = await runner._select_telegram_topic_icon_id(
            secondary_adapter,
            source,
            "ProjectAtlas",
            "Work on ProjectAtlas",
        )

    assert selected == "rocket-id"
    assert runner._telegram_topic_auto_rename_disabled(source) is True


@pytest.mark.asyncio
async def test_auto_topic_icon_selection_remembers_recent_choices_per_chat():
    runner = _make_runner()
    adapter = cast(Any, runner.adapters[Platform.TELEGRAM])
    extra = runner.config.platforms[Platform.TELEGRAM].extra
    extra["auto_topic_icons"] = True
    adapter.dm_topic_custom_icon_state.return_value = False
    adapter.get_forum_topic_icon_options.return_value = [
        {"emoji": "💻", "custom_emoji_id": "computer-id"},
        {"emoji": "🎨", "custom_emoji_id": "art-id"},
        {"emoji": "🧪", "custom_emoji_id": "lab-id"},
    ]
    source = _make_source(thread_id="42")

    with patch(
        "agent.title_generator.choose_topic_icon",
        side_effect=["💻", "🎨"],
    ) as choose:
        first = await runner._select_telegram_topic_icon_id(
            adapter,
            source,
            "Developer Tools",
            "debug the agent",
        )
        second = await runner._select_telegram_topic_icon_id(
            adapter,
            source,
            "Creative Assets",
            "design a campaign",
        )

    assert first == "computer-id"
    assert second == "art-id"
    assert choose.call_args_list == [
        call(
            "Developer Tools",
            "debug the agent",
            ["💻", "🎨", "🧪"],
            recent_emojis=[],
        ),
        call(
            "Creative Assets",
            "design a campaign",
            ["💻", "🎨", "🧪"],
            recent_emojis=["💻"],
        ),
    ]


@pytest.mark.asyncio
async def test_auto_topic_icon_selection_serializes_same_chat_history():
    runner = _make_runner()
    adapter = cast(Any, runner.adapters[Platform.TELEGRAM])
    runner.config.platforms[Platform.TELEGRAM].extra["auto_topic_icons"] = True
    adapter.dm_topic_custom_icon_state.return_value = False
    adapter.get_forum_topic_icon_options.return_value = [
        {"emoji": "💻", "custom_emoji_id": "computer-id"},
        {"emoji": "🎨", "custom_emoji_id": "art-id"},
    ]
    source = _make_source(thread_id="42")
    first_started = threading.Event()
    release_first = threading.Event()
    observed_recent: list[list[str]] = []

    def choose(*args, recent_emojis, **kwargs):
        observed_recent.append(list(recent_emojis))
        if len(observed_recent) == 1:
            first_started.set()
            assert release_first.wait(timeout=2)
            return "💻"
        return "🎨"

    with patch("agent.title_generator.choose_topic_icon", side_effect=choose):
        first_task = asyncio.create_task(
            runner._select_telegram_topic_icon_id(
                adapter,
                source,
                "Developer Tools",
                "debug the agent",
            )
        )
        assert await asyncio.to_thread(first_started.wait, 2)
        second_task = asyncio.create_task(
            runner._select_telegram_topic_icon_id(
                adapter,
                source,
                "Creative Assets",
                "design a campaign",
            )
        )
        await asyncio.sleep(0.05)
        release_first.set()
        assert await asyncio.gather(first_task, second_task) == [
            "computer-id",
            "art-id",
        ]

    assert observed_recent == [[], ["💻"]]
    entry = runner._telegram_topic_icon_locks["208214988"]
    assert entry["users"] == 0


@pytest.mark.asyncio
async def test_auto_topic_icon_history_is_isolated_per_chat():
    runner = _make_runner()
    adapter = cast(Any, runner.adapters[Platform.TELEGRAM])
    runner.config.platforms[Platform.TELEGRAM].extra["auto_topic_icons"] = True
    adapter.dm_topic_custom_icon_state.return_value = False
    adapter.get_forum_topic_icon_options.return_value = [
        {"emoji": "💻", "custom_emoji_id": "computer-id"},
        {"emoji": "🎨", "custom_emoji_id": "art-id"},
    ]
    first_source = _make_source(thread_id="42")
    second_source = SessionSource(
        platform=Platform.TELEGRAM,
        user_id="999",
        chat_id="999",
        user_name="other",
        chat_type="dm",
        thread_id="43",
    )

    with patch(
        "agent.title_generator.choose_topic_icon",
        side_effect=["💻", "🎨"],
    ) as choose:
        await runner._select_telegram_topic_icon_id(
            adapter,
            first_source,
            "Developer Tools",
            "debug the agent",
        )
        await runner._select_telegram_topic_icon_id(
            adapter,
            second_source,
            "Creative Assets",
            "design a campaign",
        )

    assert choose.call_args_list[0].kwargs["recent_emojis"] == []
    assert choose.call_args_list[1].kwargs["recent_emojis"] == []


@pytest.mark.asyncio
async def test_auto_topic_icon_per_chat_state_is_lru_bounded():
    runner = _make_runner()
    runner._telegram_topic_icon_locks = OrderedDict(
        (str(index), {"lock": asyncio.Lock(), "users": 0})
        for index in range(256)
    )
    runner._telegram_topic_icon_history = OrderedDict(
        (str(index), ["💻"]) for index in range(256)
    )
    adapter = cast(Any, runner.adapters[Platform.TELEGRAM])

    result = await runner._select_telegram_topic_icon_id(
        adapter,
        _make_source(thread_id="42"),
        "Developer Tools",
        "debug the agent",
    )

    assert result is None
    assert len(runner._telegram_topic_icon_locks) == 256
    assert "208214988" in runner._telegram_topic_icon_locks
    assert "0" not in runner._telegram_topic_icon_locks
    assert "0" not in runner._telegram_topic_icon_history


@pytest.mark.asyncio
async def test_auto_topic_icon_lock_cache_recovers_after_concurrent_pressure():
    runner = _make_runner()
    adapter = cast(Any, runner.adapters[Platform.TELEGRAM])
    all_started = asyncio.Event()
    release = asyncio.Event()
    started = 0

    async def blocked_selection(adapter, source, title, user_message):
        nonlocal started
        started += 1
        if started == 300:
            all_started.set()
        await release.wait()
        return None

    runner._select_telegram_topic_icon_id_unlocked = blocked_selection
    tasks = [
        asyncio.create_task(
            runner._select_telegram_topic_icon_id(
                adapter,
                SessionSource(
                    platform=Platform.TELEGRAM,
                    user_id=str(index),
                    chat_id=str(index),
                    user_name="tester",
                    chat_type="dm",
                    thread_id="42",
                ),
                f"Topic {index}",
                f"Opening request {index}",
            )
        )
        for index in range(300)
    ]

    await asyncio.wait_for(all_started.wait(), timeout=2)
    assert len(runner._telegram_topic_icon_locks) == 300
    assert all(
        entry["users"] == 1
        for entry in runner._telegram_topic_icon_locks.values()
    )
    release.set()
    await asyncio.gather(*tasks)

    assert len(runner._telegram_topic_icon_locks) == 256
    assert all(
        entry["users"] == 0
        for entry in runner._telegram_topic_icon_locks.values()
    )


@pytest.mark.asyncio
async def test_auto_topic_icon_history_keeps_only_twelve_recent_unique_choices():
    runner = _make_runner()
    adapter = cast(Any, runner.adapters[Platform.TELEGRAM])
    runner.config.platforms[Platform.TELEGRAM].extra["auto_topic_icons"] = True
    adapter.dm_topic_custom_icon_state.return_value = False
    emojis = [
        "📰", "💡", "⚡️", "🎙", "🔝", "🗣", "🆒",
        "❗️", "📝", "📆", "📁", "🔎", "📣", "🔥",
    ]
    adapter.get_forum_topic_icon_options.return_value = [
        {"emoji": emoji, "custom_emoji_id": f"icon-{index}"}
        for index, emoji in enumerate(emojis)
    ]
    source = _make_source(thread_id="42")

    with patch(
        "agent.title_generator.choose_topic_icon",
        side_effect=emojis,
    ) as choose:
        for index in range(len(emojis)):
            selected_id = await runner._select_telegram_topic_icon_id(
                adapter,
                source,
                f"Topic {index}",
                f"Opening request {index}",
            )
            assert selected_id == f"icon-{index}"

    assert runner._telegram_topic_icon_history["208214988"] == emojis[-12:]
    assert choose.call_args_list[-1].kwargs["recent_emojis"] == emojis[1:13]


@pytest.mark.asyncio
async def test_auto_topic_icons_preserve_manual_custom_icon(tmp_path):
    db = SessionDB(db_path=tmp_path / "state.db")
    db.apply_telegram_topic_migration()
    db.create_session("sess-topic", source="telegram", user_id="208214988")
    db.bind_telegram_topic(
        chat_id="208214988",
        thread_id="42",
        user_id="208214988",
        session_key="agent:main:telegram:dm:208214988:42",
        session_id="sess-topic",
    )
    runner = _make_runner(session_db=db)
    runner._telegram_topic_mode_enabled = lambda source: True
    adapter = cast(Any, runner.adapters[Platform.TELEGRAM])
    runner.config.platforms[Platform.TELEGRAM].extra["auto_topic_icons"] = True
    adapter.dm_topic_custom_icon_state.return_value = True

    await runner._rename_telegram_topic_for_session_title(
        _make_source(thread_id="42"),
        "sess-topic",
        "ProjectAtlas",
        user_message="Improve the onboarding flow",
    )

    adapter.get_forum_topic_icon_options.assert_not_called()
    adapter.rename_dm_topic.assert_awaited_once_with(
        chat_id="208214988",
        thread_id="42",
        name="ProjectAtlas",
    )


@pytest.mark.asyncio
async def test_auto_topic_icons_recheck_manual_icon_after_llm_choice(tmp_path):
    db = SessionDB(db_path=tmp_path / "state.db")
    db.apply_telegram_topic_migration()
    db.create_session("sess-topic", source="telegram", user_id="208214988")
    db.bind_telegram_topic(
        chat_id="208214988",
        thread_id="42",
        user_id="208214988",
        session_key="agent:main:telegram:dm:208214988:42",
        session_id="sess-topic",
    )
    runner = _make_runner(session_db=db)
    runner._telegram_topic_mode_enabled = lambda source: True
    adapter = cast(Any, runner.adapters[Platform.TELEGRAM])
    runner.config.platforms[Platform.TELEGRAM].extra["auto_topic_icons"] = True
    adapter.dm_topic_custom_icon_state.side_effect = [False, True]
    adapter.get_forum_topic_icon_options.return_value = [
        {"emoji": "🔭", "custom_emoji_id": "scope-id"},
    ]

    with patch("agent.title_generator.choose_topic_icon", return_value="🔭"):
        await runner._rename_telegram_topic_for_session_title(
            _make_source(thread_id="42"),
            "sess-topic",
            "ProjectAtlas",
            user_message="Improve the ProjectAtlas planning flow",
        )

    adapter.rename_dm_topic.assert_awaited_once_with(
        chat_id="208214988",
        thread_id="42",
        name="ProjectAtlas",
    )
    assert getattr(runner, "_telegram_topic_icon_history", {}) == {}


@pytest.mark.asyncio
async def test_auto_topic_icon_override_uses_live_matching_sticker_without_llm(tmp_path):
    db = SessionDB(db_path=tmp_path / "state.db")
    db.apply_telegram_topic_migration()
    db.create_session("sess-topic", source="telegram", user_id="208214988")
    db.bind_telegram_topic(
        chat_id="208214988",
        thread_id="42",
        user_id="208214988",
        session_key="agent:main:telegram:dm:208214988:42",
        session_id="sess-topic",
    )
    runner = _make_runner(session_db=db)
    runner._telegram_topic_mode_enabled = lambda source: True
    adapter = cast(Any, runner.adapters[Platform.TELEGRAM])
    extra = runner.config.platforms[Platform.TELEGRAM].extra
    extra["auto_topic_icons"] = True
    extra["topic_icon_overrides"] = {"projectatlas": "🚀"}
    adapter.dm_topic_custom_icon_state.return_value = False
    adapter.get_forum_topic_icon_options.return_value = [
        {"emoji": "🚀", "custom_emoji_id": "rocket-id"},
    ]

    with patch("agent.title_generator.choose_topic_icon") as choose:
        await runner._rename_telegram_topic_for_session_title(
            _make_source(thread_id="42"),
            "sess-topic",
            "ProjectAtlas",
            user_message="Improve onboarding",
        )

    choose.assert_not_called()
    adapter.rename_dm_topic.assert_awaited_once_with(
        chat_id="208214988",
        thread_id="42",
        name="ProjectAtlas",
        icon_custom_emoji_id="rocket-id",
    )


@pytest.mark.asyncio
async def test_auto_topic_icon_override_matches_deduped_lineage_title():
    runner = _make_runner()
    adapter = cast(Any, runner.adapters[Platform.TELEGRAM])
    extra = runner.config.platforms[Platform.TELEGRAM].extra
    extra["auto_topic_icons"] = True
    extra["topic_icon_overrides"] = {"projectatlas": "🔭"}
    adapter.dm_topic_custom_icon_state.return_value = False
    adapter.get_forum_topic_icon_options.return_value = [
        {"emoji": "🔭", "custom_emoji_id": "telescope-id"},
        {"emoji": "🚀", "custom_emoji_id": "rocket-id"},
    ]

    with patch("agent.title_generator.choose_topic_icon") as choose:
        selected = await runner._select_telegram_topic_icon_id(
            adapter,
            _make_source(thread_id="42"),
            "ProjectAtlas #2",
            "Improve onboarding",
        )

    choose.assert_not_called()
    assert selected == "telescope-id"


@pytest.mark.asyncio
async def test_exact_deduped_override_beats_earlier_base_override():
    runner = _make_runner()
    adapter = cast(Any, runner.adapters[Platform.TELEGRAM])
    extra = runner.config.platforms[Platform.TELEGRAM].extra
    extra["auto_topic_icons"] = True
    # Base is intentionally inserted first; the exact title must still win.
    extra["topic_icon_overrides"] = {
        "projectatlas": "🔭",
        "projectatlas #2": "🚀",
    }
    adapter.dm_topic_custom_icon_state.return_value = False
    adapter.get_forum_topic_icon_options.return_value = [
        {"emoji": "🔭", "custom_emoji_id": "telescope-id"},
        {"emoji": "🚀", "custom_emoji_id": "rocket-id"},
    ]

    with patch("agent.title_generator.choose_topic_icon") as choose:
        selected = await runner._select_telegram_topic_icon_id(
            adapter,
            _make_source(thread_id="42"),
            "ProjectAtlas #2",
            "Improve onboarding",
        )

    choose.assert_not_called()
    assert selected == "rocket-id"


@pytest.mark.asyncio
async def test_auto_topic_icon_override_matches_variation_selector_form(tmp_path):
    db = SessionDB(db_path=tmp_path / "state.db")
    db.apply_telegram_topic_migration()
    db.create_session("sess-topic", source="telegram", user_id="208214988")
    db.bind_telegram_topic(
        chat_id="208214988",
        thread_id="42",
        user_id="208214988",
        session_key="agent:main:telegram:dm:208214988:42",
        session_id="sess-topic",
    )
    runner = _make_runner(session_db=db)
    runner._telegram_topic_mode_enabled = lambda source: True
    adapter = cast(Any, runner.adapters[Platform.TELEGRAM])
    extra = runner.config.platforms[Platform.TELEGRAM].extra
    extra["auto_topic_icons"] = True
    extra["topic_icon_overrides"] = {"projectbolt": "⚡"}
    adapter.dm_topic_custom_icon_state.return_value = False
    adapter.get_forum_topic_icon_options.return_value = [
        {"emoji": "⚡️", "custom_emoji_id": "bolt-id"},
    ]

    with patch("agent.title_generator.choose_topic_icon") as choose:
        await runner._rename_telegram_topic_for_session_title(
            _make_source(thread_id="42"),
            "sess-topic",
            "ProjectBolt",
            user_message="Improve performance",
        )

    choose.assert_not_called()
    adapter.rename_dm_topic.assert_awaited_once_with(
        chat_id="208214988",
        thread_id="42",
        name="ProjectBolt",
        icon_custom_emoji_id="bolt-id",
    )


@pytest.mark.asyncio
async def test_auto_generated_title_rechecks_binding_after_icon_selection():
    session_db = MagicMock()
    session_db.get_telegram_topic_binding.side_effect = [
        {"session_id": "sess-topic"},
        {"session_id": "sess-other"},
    ]
    runner = _make_runner(session_db=session_db)
    runner._telegram_topic_mode_enabled = lambda source: True
    adapter = cast(Any, runner.adapters[Platform.TELEGRAM])
    runner.config.platforms[Platform.TELEGRAM].extra["auto_topic_icons"] = True
    adapter.dm_topic_custom_icon_state.return_value = False
    adapter.get_forum_topic_icon_options.return_value = [
        {"emoji": "🔭", "custom_emoji_id": "scope-id"},
    ]

    with patch("agent.title_generator.choose_topic_icon", return_value="🔭"):
        await runner._rename_telegram_topic_for_session_title(
            _make_source(thread_id="42"),
            "sess-topic",
            "ProjectAtlas",
            user_message="Improve planning",
        )

    assert session_db.get_telegram_topic_binding.call_count == 2
    adapter.rename_dm_topic.assert_not_called()


@pytest.mark.asyncio
async def test_manual_icon_change_during_binding_recheck_prevents_icon_overwrite():
    second_binding_started = threading.Event()
    release_second_binding = threading.Event()
    binding_calls = 0

    def get_binding(**kwargs):
        nonlocal binding_calls
        binding_calls += 1
        if binding_calls == 2:
            second_binding_started.set()
            assert release_second_binding.wait(timeout=2)
        return {"session_id": "sess-topic"}

    session_db = MagicMock()
    session_db.get_telegram_topic_binding.side_effect = get_binding
    runner = _make_runner(session_db=session_db)
    runner._telegram_topic_mode_enabled = lambda source: True
    adapter = cast(Any, runner.adapters[Platform.TELEGRAM])
    extra = runner.config.platforms[Platform.TELEGRAM].extra
    extra["auto_topic_icons"] = True
    extra["preserve_manual_topic_icons"] = True
    adapter.dm_topic_custom_icon_state.return_value = False
    adapter.get_forum_topic_icon_options.return_value = [
        {"emoji": "🔭", "custom_emoji_id": "scope-id"},
    ]

    with patch("agent.title_generator.choose_topic_icon", return_value="🔭"):
        rename_task = asyncio.create_task(
            runner._rename_telegram_topic_for_session_title(
                _make_source(thread_id="42"),
                "sess-topic",
                "ProjectAtlas",
                user_message="Improve planning",
            )
        )
        assert await asyncio.to_thread(second_binding_started.wait, 2)
        adapter.dm_topic_custom_icon_state.return_value = True
        release_second_binding.set()
        await rename_task

    assert session_db.get_telegram_topic_binding.call_count == 2
    assert adapter.dm_topic_custom_icon_state.call_count == 3
    adapter.rename_dm_topic.assert_awaited_once_with(
        chat_id="208214988",
        thread_id="42",
        name="ProjectAtlas",
    )

@pytest.mark.asyncio
async def test_message_cached_topic_is_still_auto_renamed(tmp_path):
    """Telegram's incoming topic-name cache must not disable semantic rename."""
    db = SessionDB(db_path=tmp_path / "state.db")
    db.enable_telegram_topic_mode(chat_id="208214988", user_id="208214988")
    db.create_session(session_id="sess-topic", source="telegram", user_id="208214988")
    db.bind_telegram_topic(
        chat_id="208214988",
        thread_id="17585",
        user_id="208214988",
        session_key=build_session_key(_make_source(thread_id="17585")),
        session_id="sess-topic",
    )
    runner = _make_runner(session_db=db)
    runner._telegram_topic_mode_enabled = lambda source: True

    # Exercise the real adapter state that caused the production regression:
    # Telegram supplied a topic name before the first title callback ran.
    from plugins.platforms.telegram.adapter import TelegramAdapter

    adapter = TelegramAdapter(
        PlatformConfig(enabled=True, token="***", extra={})
    )
    bot = SimpleNamespace(edit_forum_topic=AsyncMock())
    adapter._bot = cast(Any, bot)
    adapter._cache_dm_topic_from_message(
        "208214988", "17585", "Can you still or"
    )
    adapter._reload_dm_topics_from_config = lambda: None
    runner.adapters[Platform.TELEGRAM] = adapter

    await runner._rename_telegram_topic_for_session_title(
        _make_source(thread_id="17585"),
        "sess-topic",
        "Grab",
    )

    bot.edit_forum_topic.assert_awaited_once_with(
        chat_id=208214988,
        message_thread_id=17585,
        name="Grab",
    )

def test_background_rename_copy_preserves_transport_adapter_provenance():
    runner = _make_runner()
    default_adapter = runner.adapters[Platform.TELEGRAM]
    secondary_adapter = MagicMock()
    runner._profile_adapters = {
        "work": {Platform.TELEGRAM: secondary_adapter},
    }
    source = _make_source(thread_id="42")
    source.profile = "work"
    setattr(source, "_transport_adapter_ref", lambda: default_adapter)

    copied = runner._copy_source_for_background_rename(source)

    assert copied is not source
    assert copied.profile == "work"
    assert runner._adapter_for_source(copied) is default_adapter

@pytest.mark.asyncio
async def test_operator_declared_topic_is_not_auto_renamed(tmp_path):
    """Operator-configured topic names remain untouched by auto-title."""
    db = SessionDB(db_path=tmp_path / "state.db")
    db.enable_telegram_topic_mode(chat_id="208214988", user_id="208214988")
    db.create_session(session_id="sess-topic", source="telegram", user_id="208214988")
    db.bind_telegram_topic(
        chat_id="208214988",
        thread_id="17585",
        user_id="208214988",
        session_key=build_session_key(_make_source(thread_id="17585")),
        session_id="sess-topic",
    )
    runner = _make_runner(session_db=db)
    runner._telegram_topic_mode_enabled = lambda source: True

    class _FakeAdapter:
        def _is_dm_topic_operator_declared(self, chat_id, thread_id):
            return True

        async def rename_dm_topic(self, **kwargs):
            return None

    fake = _FakeAdapter()
    fake.rename_dm_topic = AsyncMock()
    runner.adapters[Platform.TELEGRAM] = cast(Any, fake)

    await runner._rename_telegram_topic_for_session_title(
        _make_source(thread_id="17585"),
        "sess-topic",
        "Auto-generated title",
    )

    fake.rename_dm_topic.assert_not_called()
