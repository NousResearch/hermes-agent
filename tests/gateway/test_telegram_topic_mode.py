"""Tests for Telegram private-chat topic-mode routing.

Topic mode makes the root Telegram DM a system lobby while user-created
Telegram topics act as independent Hermes session lanes.
"""

from datetime import datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from hermes_state import SessionDB
from gateway.config import GatewayConfig, Platform, PlatformConfig
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
    runner._release_running_agent_state = MagicMock()
    runner._evict_cached_agent = MagicMock()
    runner._clear_session_boundary_security_state = MagicMock()
    runner._set_session_reasoning_override = MagicMock()
    runner._format_session_info = MagicMock(return_value="")
    return runner


@pytest.mark.asyncio
async def test_root_telegram_dm_prompt_is_system_lobby_when_topic_mode_enabled(monkeypatch):
    import gateway.run as gateway_run

    runner = _make_runner()
    runner._telegram_topic_mode_enabled = lambda source: True
    runner._run_agent = AsyncMock(
        side_effect=AssertionError("root Telegram DM prompt leaked to the agent loop")
    )

    monkeypatch.setattr(
        gateway_run, "_resolve_runtime_agent_kwargs", lambda: {"api_key": "***"}
    )

    result = await runner._handle_message(_make_event("hello from root"))

    assert "main chat is reserved for system commands" in result
    assert "All Messages" in result
    runner._run_agent.assert_not_called()
    runner.session_store.get_or_create_session.assert_not_called()


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
async def test_telegram_topic_prompt_still_runs_agent_when_topic_mode_enabled(monkeypatch):
    import gateway.run as gateway_run

    runner = _make_runner()
    runner._telegram_topic_mode_enabled = lambda source: True
    runner._handle_message_with_agent = AsyncMock(return_value="agent response")

    monkeypatch.setattr(
        gateway_run, "_resolve_runtime_agent_kwargs", lambda: {"api_key": "***"}
    )

    result = await runner._handle_message(_make_event("hello in topic", thread_id="17585"))

    assert result == "agent response"
    runner._handle_message_with_agent.assert_awaited_once()


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
async def test_topic_command_is_private_dm_only_and_does_not_enable_group_topic_mode(
    tmp_path, monkeypatch
):
    import gateway.run as gateway_run

    session_db = SessionDB(db_path=tmp_path / "state.db")
    runner = _make_runner(session_db=session_db)
    runner._run_agent = AsyncMock(
        side_effect=AssertionError("group /topic must not enter the agent loop")
    )

    monkeypatch.setattr(
        gateway_run, "_resolve_runtime_agent_kwargs", lambda: {"api_key": "***"}
    )

    result = await runner._handle_message(_make_group_event("/topic", thread_id="555"))

    assert "only available in Telegram private chats" in result
    assert session_db.is_telegram_topic_mode_enabled(chat_id="-100123", user_id="208214988") is False
    runner._run_agent.assert_not_called()


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

    result = await runner._handle_message(_make_group_event("/new", thread_id="555"))

    assert "Started a new Hermes session in this topic" not in result
    assert "parallel work" not in result
    runner.session_store.reset_session.assert_called_once_with(group_key)


@pytest.mark.asyncio
async def test_new_inside_telegram_topic_resets_current_topic_with_parallel_tip(monkeypatch):
    import gateway.run as gateway_run

    runner = _make_runner()
    runner._telegram_topic_mode_enabled = lambda source: True
    topic_source = _make_source(thread_id="17585")
    topic_key = build_session_key(topic_source)
    old_entry = SessionEntry(
        session_key=topic_key,
        session_id="old-topic-session",
        created_at=datetime.now(),
        updated_at=datetime.now(),
        platform=Platform.TELEGRAM,
        chat_type="dm",
        origin=topic_source,
    )
    new_entry = SessionEntry(
        session_key=topic_key,
        session_id="new-topic-session",
        created_at=datetime.now(),
        updated_at=datetime.now(),
        platform=Platform.TELEGRAM,
        chat_type="dm",
        origin=topic_source,
    )
    runner.session_store._entries = {topic_key: old_entry}
    runner.session_store.reset_session.return_value = new_entry
    runner._agent_cache_lock = None

    monkeypatch.setattr(
        gateway_run, "_resolve_runtime_agent_kwargs", lambda: {"api_key": "***"}
    )

    result = await runner._handle_message(_make_event("/new", thread_id="17585"))

    assert "Started a new Hermes session in this topic" in result
    assert "parallel work" in result
    assert "All Messages" in result
    runner.session_store.reset_session.assert_called_once_with(topic_key)


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
async def test_topic_root_command_explicitly_migrates_and_enables_topic_mode(tmp_path, monkeypatch):
    import gateway.run as gateway_run

    session_db = SessionDB(db_path=tmp_path / "state.db")
    runner = _make_runner(session_db=session_db)
    runner._run_agent = AsyncMock(
        side_effect=AssertionError("/topic activation must not enter the agent loop")
    )

    monkeypatch.setattr(
        gateway_run, "_resolve_runtime_agent_kwargs", lambda: {"api_key": "***"}
    )

    result = await runner._handle_message(_make_event("/topic"))

    assert "Telegram multi-session topics are enabled" in result
    assert "All Messages" in result
    assert session_db.get_meta("telegram_dm_topic_schema_version") == "2"
    assert session_db.is_telegram_topic_mode_enabled(chat_id="208214988", user_id="208214988")
    assert runner._telegram_topic_mode_enabled(_make_source()) is True
    runner._run_agent.assert_not_called()

    lobby_result = await runner._handle_message(_make_event("hello after activation"))

    assert "main chat is reserved for system commands" in lobby_result
    runner._run_agent.assert_not_called()


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
async def test_topic_root_command_handles_no_unlinked_sessions(tmp_path, monkeypatch):
    import gateway.run as gateway_run

    session_db = SessionDB(db_path=tmp_path / "state.db")
    runner = _make_runner(session_db=session_db)
    runner._run_agent = AsyncMock(
        side_effect=AssertionError("root /topic status must not enter the agent loop")
    )

    monkeypatch.setattr(
        gateway_run, "_resolve_runtime_agent_kwargs", lambda: {"api_key": "***"}
    )

    result = await runner._handle_message(_make_event("/topic"))

    assert "Telegram multi-session topics are enabled" in result
    assert "No previous unlinked Telegram sessions found" in result
    assert "All Messages" in result
    runner._run_agent.assert_not_called()


@pytest.mark.asyncio
async def test_topic_command_inside_bound_topic_shows_current_session(tmp_path, monkeypatch):
    import gateway.run as gateway_run

    session_db = SessionDB(db_path=tmp_path / "state.db")
    session_db.create_session(
        session_id="sess-topic",
        source="telegram",
        user_id="208214988",
    )
    session_db.set_session_title("sess-topic", "Research notes")
    session_db.bind_telegram_topic(
        chat_id="208214988",
        thread_id="17585",
        user_id="208214988",
        session_key="telegram:dm:208214988:thread:17585",
        session_id="sess-topic",
    )
    runner = _make_runner(session_db=session_db)
    runner._run_agent = AsyncMock(
        side_effect=AssertionError("/topic status must not enter the agent loop")
    )

    monkeypatch.setattr(
        gateway_run, "_resolve_runtime_agent_kwargs", lambda: {"api_key": "***"}
    )

    result = await runner._handle_message(_make_event("/topic", thread_id="17585"))

    assert "This topic is linked to" in result
    assert "Research notes" in result
    assert "sess-topic" in result
    assert "Use /new to replace" in result
    runner._run_agent.assert_not_called()


@pytest.mark.asyncio
async def test_topic_restore_inside_topic_binds_old_session_and_returns_last_assistant_message(
    tmp_path, monkeypatch
):
    import gateway.run as gateway_run

    session_db = SessionDB(db_path=tmp_path / "state.db")
    session_db.enable_telegram_topic_mode(chat_id="208214988", user_id="208214988")
    session_db.create_session(
        session_id="old-session",
        source="telegram",
        user_id="208214988",
    )
    session_db.set_session_title("old-session", "Research notes")
    session_db.append_message("old-session", "user", "summarize this")
    session_db.append_message("old-session", "assistant", "Here is the summary.")
    runner = _make_runner(session_db=session_db)
    runner._run_agent = AsyncMock(
        side_effect=AssertionError("/topic restore must not enter the agent loop")
    )

    monkeypatch.setattr(
        gateway_run, "_resolve_runtime_agent_kwargs", lambda: {"api_key": "***"}
    )

    result = await runner._handle_message(_make_event("/topic old-session", thread_id="17585"))

    assert "Session restored: Research notes" in result
    assert "Last Hermes message:" in result
    assert "Here is the summary." in result
    binding = session_db.get_telegram_topic_binding(chat_id="208214988", thread_id="17585")
    assert binding is not None
    assert binding["session_id"] == "old-session"
    assert binding["user_id"] == "208214988"
    assert binding["session_key"] == build_session_key(_make_source(thread_id="17585"))
    runner._run_agent.assert_not_called()


@pytest.mark.asyncio
async def test_topic_restore_refuses_session_owned_by_another_telegram_user(tmp_path, monkeypatch):
    import gateway.run as gateway_run

    session_db = SessionDB(db_path=tmp_path / "state.db")
    session_db.enable_telegram_topic_mode(chat_id="208214988", user_id="208214988")
    session_db.create_session(
        session_id="other-session",
        source="telegram",
        user_id="someone-else",
    )
    runner = _make_runner(session_db=session_db)

    monkeypatch.setattr(
        gateway_run, "_resolve_runtime_agent_kwargs", lambda: {"api_key": "***"}
    )

    result = await runner._handle_message(_make_event("/topic other-session", thread_id="17585"))

    assert "does not belong to this Telegram user" in result
    assert session_db.get_telegram_topic_binding(chat_id="208214988", thread_id="17585") is None


@pytest.mark.asyncio
async def test_topic_restore_refuses_already_linked_session(tmp_path, monkeypatch):
    import gateway.run as gateway_run

    session_db = SessionDB(db_path=tmp_path / "state.db")
    session_db.enable_telegram_topic_mode(chat_id="208214988", user_id="208214988")
    session_db.create_session(
        session_id="linked-session",
        source="telegram",
        user_id="208214988",
    )
    session_db.bind_telegram_topic(
        chat_id="208214988",
        thread_id="11111",
        user_id="208214988",
        session_key="agent:main:telegram:dm:208214988:11111",
        session_id="linked-session",
    )
    runner = _make_runner(session_db=session_db)

    monkeypatch.setattr(
        gateway_run, "_resolve_runtime_agent_kwargs", lambda: {"api_key": "***"}
    )

    result = await runner._handle_message(_make_event("/topic linked-session", thread_id="17585"))

    assert "already linked to another Telegram topic" in result
    assert session_db.get_telegram_topic_binding(chat_id="208214988", thread_id="17585") is None


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
async def test_topic_root_command_checks_getme_capabilities_before_enabling(tmp_path, monkeypatch):
    import gateway.run as gateway_run

    session_db = SessionDB(db_path=tmp_path / "state.db")
    runner = _make_runner(session_db=session_db)
    bot = AsyncMock()
    bot.get_me.return_value = SimpleNamespace(
        has_topics_enabled=False,
        allows_users_to_create_topics=True,
    )
    runner.adapters[Platform.TELEGRAM]._bot = bot
    runner._run_agent = AsyncMock(
        side_effect=AssertionError("/topic capability failure must not enter the agent loop")
    )

    monkeypatch.setattr(
        gateway_run, "_resolve_runtime_agent_kwargs", lambda: {"api_key": "***"}
    )

    result = await runner._handle_message(_make_event("/topic"))

    assert "topics are not enabled" in result
    assert "Open @BotFather" in result
    assert session_db.is_telegram_topic_mode_enabled(chat_id="208214988", user_id="208214988") is False
    bot.get_me.assert_awaited_once()
    runner.adapters[Platform.TELEGRAM].send_image_file.assert_awaited_once()
    image_kwargs = runner.adapters[Platform.TELEGRAM].send_image_file.await_args.kwargs
    assert image_kwargs["chat_id"] == "208214988"
    assert image_kwargs["image_path"].endswith("telegram-botfather-threads-settings.jpg")
    runner._run_agent.assert_not_called()


@pytest.mark.asyncio
async def test_topic_root_command_creates_and_pins_system_topic(tmp_path, monkeypatch):
    import gateway.run as gateway_run

    session_db = SessionDB(db_path=tmp_path / "state.db")
    runner = _make_runner(session_db=session_db)
    adapter = runner.adapters[Platform.TELEGRAM]
    adapter._create_dm_topic.return_value = 4242
    adapter.send.return_value = SimpleNamespace(success=True, message_id="777")
    bot = AsyncMock()
    bot.get_me.return_value = {
        "has_topics_enabled": True,
        "allows_users_to_create_topics": True,
    }
    adapter._bot = bot

    monkeypatch.setattr(
        gateway_run, "_resolve_runtime_agent_kwargs", lambda: {"api_key": "***"}
    )

    result = await runner._handle_message(_make_event("/topic"))

    assert "Telegram multi-session topics are enabled" in result
    adapter._create_dm_topic.assert_awaited_once_with(208214988, "System")
    adapter.send.assert_awaited_once_with(
        "208214988",
        "System topic for Hermes commands and status.",
        metadata={"thread_id": "4242"},
    )
    bot.pin_chat_message.assert_awaited_once_with(
        chat_id=208214988,
        message_id=777,
        disable_notification=True,
    )


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
async def test_auto_generated_title_does_not_rename_topic_bound_to_other_session(tmp_path):
    db = SessionDB(db_path=tmp_path / "state.db")
    db.apply_telegram_topic_migration()
    db.create_session("sess-other", source="telegram", user_id="208214988")
    db.bind_telegram_topic(
        chat_id="208214988",
        thread_id="42",
        user_id="208214988",
        session_key="agent:main:telegram:dm:208214988:42",
        session_id="sess-other",
    )
    runner = _make_runner(session_db=db)
    runner._telegram_topic_mode_enabled = lambda source: True

    await runner._rename_telegram_topic_for_session_title(
        _make_source(thread_id="42"),
        "sess-topic",
        "Wrong Session Title",
    )

    runner.adapters[Platform.TELEGRAM].rename_dm_topic.assert_not_called()


@pytest.mark.asyncio
async def test_operator_declared_topic_is_not_auto_renamed(tmp_path):
    """Topics registered in extra.dm_topics keep their operator-chosen name."""
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

    # Give the adapter a concrete class with _get_dm_topic_info so the
    # class-based lookup in _rename_telegram_topic_for_session_title
    # actually finds it (a MagicMock auto-attr would be skipped).
    class _FakeAdapter:
        def _get_dm_topic_info(self, chat_id, thread_id):
            return {"name": "Research", "skill": "arxiv"}

        async def rename_dm_topic(self, **kwargs):
            return None

    fake = _FakeAdapter()
    fake.rename_dm_topic = AsyncMock()
    runner.adapters[Platform.TELEGRAM] = fake

    await runner._rename_telegram_topic_for_session_title(
        _make_source(thread_id="17585"),
        "sess-topic",
        "Auto-generated title",
    )

    fake.rename_dm_topic.assert_not_called()


def test_general_topic_is_treated_as_root_lobby(tmp_path):
    """Messages in the Telegram General topic (thread_id=1) route to the lobby, not a lane."""
    db = SessionDB(db_path=tmp_path / "state.db")
    db.enable_telegram_topic_mode(chat_id="208214988", user_id="208214988")
    runner = _make_runner(session_db=db)

    general_source = _make_source(thread_id="1")
    assert runner._is_telegram_topic_root_lobby(general_source) is True
    assert runner._is_telegram_topic_lane(general_source) is False

    no_thread_source = _make_source(thread_id=None)
    assert runner._is_telegram_topic_root_lobby(no_thread_source) is True
    assert runner._is_telegram_topic_lane(no_thread_source) is False

    real_topic = _make_source(thread_id="17585")
    assert runner._is_telegram_topic_root_lobby(real_topic) is False
    assert runner._is_telegram_topic_lane(real_topic) is True


def test_lobby_reminder_is_debounced_per_chat(tmp_path):
    """Consecutive root-DM prompts should only surface one lobby reminder per cooldown."""
    db = SessionDB(db_path=tmp_path / "state.db")
    db.enable_telegram_topic_mode(chat_id="208214988", user_id="208214988")
    runner = _make_runner(session_db=db)

    source = _make_source(thread_id=None)
    assert runner._should_send_telegram_lobby_reminder(source) is True
    # Next call inside the cooldown window must return False.
    assert runner._should_send_telegram_lobby_reminder(source) is False
    assert runner._should_send_telegram_lobby_reminder(source) is False

    # A different chat gets its own window.
    other = _make_source(thread_id=None)
    # Swap chat_id so the debounce key is different.
    from dataclasses import replace
    other = replace(other, chat_id="999999999")
    assert runner._should_send_telegram_lobby_reminder(other) is True


def test_binding_survives_session_deletion_via_cascade(tmp_path):
    """Deleting a session with a topic binding must not raise FK errors."""
    import sqlite3
    db = SessionDB(db_path=tmp_path / "state.db")
    db.enable_telegram_topic_mode(chat_id="208214988", user_id="208214988")
    db.create_session(session_id="sess-to-delete", source="telegram", user_id="208214988")
    db.bind_telegram_topic(
        chat_id="208214988",
        thread_id="17585",
        user_id="208214988",
        session_key="agent:main:telegram:dm:208214988:17585",
        session_id="sess-to-delete",
    )

    # Before: binding exists.
    binding = db.get_telegram_topic_binding(chat_id="208214988", thread_id="17585")
    assert binding is not None

    # Delete the session. Without ON DELETE CASCADE this would raise
    # sqlite3.IntegrityError: FOREIGN KEY constraint failed.
    db._conn.execute("DELETE FROM sessions WHERE id = ?", ("sess-to-delete",))
    db._conn.commit()

    # After: binding row automatically cleared.
    binding_after = db.get_telegram_topic_binding(chat_id="208214988", thread_id="17585")
    assert binding_after is None


def test_migration_rebuilds_v1_binding_table_with_cascade_fk(tmp_path):
    """v1 → v2 migration rebuilds the bindings table when FK lacks ON DELETE CASCADE."""
    import sqlite3
    db_path = tmp_path / "state.db"
    db = SessionDB(db_path=db_path)

    # Simulate a v1-shaped DB: migration ran without ON DELETE CASCADE.
    db.apply_telegram_topic_migration()  # Creates v2 (our new shape)
    # Drop the v2 bindings table and recreate it in the old v1 shape.
    with db._lock:
        db._conn.execute("DROP TABLE telegram_dm_topic_bindings")
        db._conn.execute(
            """
            CREATE TABLE telegram_dm_topic_bindings (
                chat_id TEXT NOT NULL,
                thread_id TEXT NOT NULL,
                user_id TEXT NOT NULL,
                session_key TEXT NOT NULL,
                session_id TEXT NOT NULL REFERENCES sessions(id),
                managed_mode TEXT NOT NULL DEFAULT 'auto',
                linked_at REAL NOT NULL,
                updated_at REAL NOT NULL,
                PRIMARY KEY (chat_id, thread_id)
            )
            """
        )
        # Also rewind the version marker so migration treats this as v1.
        db._conn.execute(
            "UPDATE state_meta SET value = '1' WHERE key = 'telegram_dm_topic_schema_version'"
        )
        db._conn.commit()

    # Sanity check: FK has no CASCADE action yet.
    fk_rows = db._conn.execute(
        "PRAGMA foreign_key_list('telegram_dm_topic_bindings')"
    ).fetchall()
    assert any(row[2] == "sessions" and (row[6] or "") != "CASCADE" for row in fk_rows)

    # Re-run migration — should upgrade to v2 shape.
    db.apply_telegram_topic_migration()

    fk_rows_after = db._conn.execute(
        "PRAGMA foreign_key_list('telegram_dm_topic_bindings')"
    ).fetchall()
    assert any(row[2] == "sessions" and row[6] == "CASCADE" for row in fk_rows_after)

    version = db._conn.execute(
        "SELECT value FROM state_meta WHERE key = 'telegram_dm_topic_schema_version'"
    ).fetchone()
    assert version is not None and version[0] == "2"


@pytest.mark.asyncio
async def test_topic_help_subcommand_returns_usage(tmp_path):
    """/topic help surfaces usage without activating anything."""
    db = SessionDB(db_path=tmp_path / "state.db")
    runner = _make_runner(session_db=db)

    result = await runner._handle_topic_command(_make_event("/topic help"))

    assert "/topic help" in result
    assert "/topic off" in result
    assert "/topic <id>" in result
    # No side effects — topic mode tables should not even exist yet.
    tables = {
        row[0]
        for row in db._conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name LIKE 'telegram_dm%'"
        ).fetchall()
    }
    assert tables == set()


@pytest.mark.asyncio
async def test_topic_off_disables_mode_and_clears_bindings(tmp_path, monkeypatch):
    """/topic off flips the row off AND deletes bindings for this chat."""
    import gateway.run as gateway_run

    db = SessionDB(db_path=tmp_path / "state.db")
    db.enable_telegram_topic_mode(chat_id="208214988", user_id="208214988")
    db.create_session(session_id="topic-sess", source="telegram", user_id="208214988")
    db.bind_telegram_topic(
        chat_id="208214988",
        thread_id="17585",
        user_id="208214988",
        session_key="k",
        session_id="topic-sess",
    )
    runner = _make_runner(session_db=db)

    monkeypatch.setattr(
        gateway_run, "_resolve_runtime_agent_kwargs", lambda: {"api_key": "***"}
    )

    result = await runner._handle_topic_command(_make_event("/topic off"))

    assert "OFF" in result or "off" in result
    assert db.is_telegram_topic_mode_enabled(
        chat_id="208214988", user_id="208214988"
    ) is False
    # Bindings cleared.
    assert db.get_telegram_topic_binding(
        chat_id="208214988", thread_id="17585"
    ) is None


@pytest.mark.asyncio
async def test_topic_off_is_idempotent_when_never_enabled(tmp_path):
    """/topic off against a chat that never ran /topic is a no-op message."""
    db = SessionDB(db_path=tmp_path / "state.db")
    runner = _make_runner(session_db=db)

    result = await runner._handle_topic_command(_make_event("/topic off"))

    assert "not currently enabled" in result


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


def test_capability_hint_is_debounced_per_chat(tmp_path):
    """BotFather screenshot is sent once per cooldown window per chat."""
    db = SessionDB(db_path=tmp_path / "state.db")
    runner = _make_runner(session_db=db)

    source = _make_source()
    assert runner._should_send_telegram_capability_hint(source) is True
    assert runner._should_send_telegram_capability_hint(source) is False
    assert runner._should_send_telegram_capability_hint(source) is False

    from dataclasses import replace
    other = replace(source, chat_id="999999999")
    assert runner._should_send_telegram_capability_hint(other) is True


def test_topic_off_resets_debounce_counters(tmp_path):
    """Disabling topic mode clears per-chat debounce state."""
    db = SessionDB(db_path=tmp_path / "state.db")
    db.enable_telegram_topic_mode(chat_id="208214988", user_id="208214988")
    runner = _make_runner(session_db=db)

    source = _make_source()
    # Prime the debounce counters.
    assert runner._should_send_telegram_lobby_reminder(source) is True
    assert runner._should_send_telegram_capability_hint(source) is True
    assert runner._should_send_telegram_lobby_reminder(source) is False
    assert runner._should_send_telegram_capability_hint(source) is False

    # /topic off resets them.
    result = runner._disable_telegram_topic_mode_for_chat(source)
    assert "OFF" in result or "off" in result

    # Re-enable and verify counters reset (so the first reminder/hint
    # after re-enabling can land immediately).
    db.enable_telegram_topic_mode(chat_id="208214988", user_id="208214988")
    assert runner._should_send_telegram_lobby_reminder(source) is True
    assert runner._should_send_telegram_capability_hint(source) is True
