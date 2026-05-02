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
    assert "+ button" in result
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
    assert "+ button" in result
    assert "Use /new inside a topic" in result
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
    assert "for parallel work" in result
    assert "+ button" in result
    runner.session_store.reset_session.assert_called_once_with(topic_key)


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
    assert "+ button" in result
    assert session_db.get_meta("telegram_dm_topic_schema_version") == "1"
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
    assert "+ button" in result
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
