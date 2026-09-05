"""Tests for Telegram private-chat topic-mode routing.

Topic mode makes the root Telegram DM a system lobby while user-created
Telegram topics act as independent Hermes session lanes.
"""

from datetime import datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from agent.context_compressor import (
    HISTORICAL_TASK_HEADING,
    SUMMARY_PREFIX,
    _MERGED_PRIOR_CONTEXT_HEADER,
    _MERGED_SUMMARY_DELIMITER,
    _SUMMARY_END_MARKER,
)
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
    runner.session_store.get_or_create_session.side_effect = lambda source, force_new=False, **_kwargs: SessionEntry(
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
async def test_topic_restore_quote_never_exposes_compaction_scaffolding(tmp_path):
    db = SessionDB(db_path=tmp_path / "state.db")
    db.enable_telegram_topic_mode(chat_id="208214988", user_id="208214988")
    db.create_session(
        session_id="restorable",
        source="telegram",
        user_id="208214988",
    )
    db.set_session_title("restorable", "Browser control")
    db.append_message("restorable", "assistant", "real completed answer")
    summary = (
        f"{SUMMARY_PREFIX}\n\n"
        f"{HISTORICAL_TASK_HEADING}\nold work\n\n"
        f"{_SUMMARY_END_MARKER}"
    )
    db.append_message("restorable", "assistant", summary)
    runner = _make_runner(session_db=db)

    result = await runner._restore_telegram_topic_session(
        _make_event("/topic restorable", thread_id="17585"),
        "restorable",
    )

    assert "Last Hermes message:\nreal completed answer" in result
    assert "CONTEXT COMPACTION" not in result
    assert "Historical Task Snapshot" not in result
    db.close()


@pytest.mark.asyncio
async def test_topic_restore_quote_unwraps_merged_assistant_carrier(tmp_path):
    db = SessionDB(db_path=tmp_path / "state.db")
    db.enable_telegram_topic_mode(chat_id="208214988", user_id="208214988")
    db.create_session(
        session_id="restorable",
        source="telegram",
        user_id="208214988",
    )
    carrier = (
        f"{_MERGED_PRIOR_CONTEXT_HEADER}\n"
        "real completed answer\n\n"
        f"{_MERGED_SUMMARY_DELIMITER}\n\n"
        f"{SUMMARY_PREFIX}\n\n"
        f"{HISTORICAL_TASK_HEADING}\nold work\n\n"
        f"{_SUMMARY_END_MARKER}"
    )
    db.append_message("restorable", "assistant", carrier)
    runner = _make_runner(session_db=db)

    result = await runner._restore_telegram_topic_session(
        _make_event("/topic restorable", thread_id="17585"),
        "restorable",
    )

    assert "Last Hermes message:\nreal completed answer" in result
    assert "PRIOR CONTEXT" not in result
    assert "CONTEXT COMPACTION" not in result
    db.close()


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
    adapter = runner.adapters[Platform.TELEGRAM]
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

