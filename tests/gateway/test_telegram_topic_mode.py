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
from gateway.platforms.event import MessageEvent
from gateway.session import SessionEntry, SessionSource, SessionStore, build_session_key


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
    def _switch_session(session_key, target_session_id, **_kwargs):
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
    store = SessionStore(tmp_path / "sessions", runner.config)
    store._db = session_db
    runner.session_store = store
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
    # Exercise missing-route recovery through the real store, including tip following.
    store = SessionStore(tmp_path / "sessions", runner.config)
    store._db = session_db
    runner.session_store = store
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
    assert store.lookup_by_session_key(topic_key).session_id == "child-session"
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



@pytest.fixture
def routed_topic(tmp_path):
    from gateway.session import SessionStore
    db = SessionDB(db_path=tmp_path / "state.db")
    db.enable_telegram_topic_mode(chat_id="208214988", user_id="208214988")
    runner = _make_runner(db)
    for name in ("_release_running_agent_state", "_invalidate_session_run_generation",
                 "_begin_session_run_generation", "_is_session_run_current"):
        delattr(runner, name)
    runner._persist_active_agents = MagicMock()
    store = SessionStore(tmp_path / "sessions", runner.config)
    store._db = db
    runner.session_store = store
    runner._hmwa_prepare_turn = AsyncMock(return_value=("prepared", None))
    source = _make_source(thread_id="17585")
    yield SimpleNamespace(db=db, store=store, runner=runner, source=source, tmp=tmp_path)
    runner._shutdown_executor()
    store.close_all_db_handles()
    db.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("paused", [False, True])
async def test_explicit_topic_restore_commits_route_before_success(routed_topic, paused):
    from gateway.session import SessionStore
    e = routed_topic
    old = e.store.get_or_create_session(e.source)
    if paused:
        e.store.set_session_metadata(old.session_key, "compression_exhausted", True, require_primary=True)
    e.runner._record_telegram_topic_binding(e.source, old)
    e.db.create_session("restorable-target", source="telegram", user_id=e.source.user_id)
    e.db.append_message("restorable-target", "assistant", "restored history")
    reply = await e.runner._handle_topic_command(
        _make_event("/topic restorable-target", thread_id=e.source.thread_id), "restorable-target")
    assert "Session restored" in reply
    assert e.store.lookup_by_session_key(old.session_key).session_id == "restorable-target"
    fresh = SessionStore(e.store.sessions_dir, e.store.config)
    fresh._db = e.db
    try:
        assert fresh.lookup_by_session_key(old.session_key).session_id == "restorable-target"
        e.runner.session_store = fresh
        assert await e.runner._handle_message(_make_event("continue restored", thread_id=e.source.thread_id)) == "prepared"
        entry = e.runner._hmwa_prepare_turn.await_args.args[2]
        assert entry.session_id == "restorable-target" and not entry.compression_paused
        assert fresh.load_transcript(entry.session_id)[-1]["content"] == "restored history"
    finally:
        e.runner.session_store = e.store
        fresh.close_all_db_handles()


@pytest.mark.asyncio
@pytest.mark.parametrize("route_lost", [False, True])
async def test_topic_restore_primary_failure_keeps_pause(routed_topic, monkeypatch, route_lost):
    e = routed_topic
    old = e.store.get_or_create_session(e.source)
    e.store.set_session_metadata(old.session_key, "compression_exhausted", True, require_primary=True)
    e.runner._record_telegram_topic_binding(e.source, old)
    binding_args = dict(chat_id=e.source.chat_id, thread_id=e.source.thread_id)
    old_binding = e.db.get_telegram_topic_binding(**binding_args)
    e.db.create_session("restorable-target", source="telegram", user_id=e.source.user_id)
    persist = e.db.replace_gateway_routing_entries
    e.db._conn.execute("""
        CREATE TRIGGER abort_topic_restore BEFORE INSERT ON gateway_routing
        WHEN EXISTS (SELECT 1 FROM telegram_dm_topic_bindings
                     WHERE session_id = 'restorable-target' AND managed_mode = 'restored')
        BEGIN SELECT RAISE(ABORT, 'topic route write aborted after binding'); END
    """)
    monkeypatch.setattr("gateway.session._now", lambda: old.updated_at)
    before_routes = e.db.load_gateway_routing_entries(scope=e.store._routing_scope())
    before_mirror = (e.store.sessions_dir / "sessions.json").read_bytes()
    clear_scope = MagicMock(wraps=e.runner._clear_conversation_scope)
    monkeypatch.setattr(e.runner, "_clear_conversation_scope", clear_scope)
    reply = await e.runner._handle_topic_command(
        _make_event("/topic restorable-target", thread_id=e.source.thread_id), "restorable-target")
    assert "could not be persisted" in reply
    assert e.store.lookup_by_session_key(old.session_key) is old and old.compression_paused
    assert e.db.load_gateway_routing_entries(scope=e.store._routing_scope()) == before_routes
    assert (e.store.sessions_dir / "sessions.json").read_bytes() == before_mirror
    assert e.db.get_session(old.session_id)["end_reason"] is None
    assert e.db.get_session("restorable-target")["session_key"] is None
    clear_scope.assert_not_called()
    e.runner._evict_cached_agent.assert_not_called()
    e.db._conn.execute("DROP TRIGGER abort_topic_restore")
    if not route_lost:
        assert e.db.get_telegram_topic_binding(**binding_args) == old_binding
        assert "paused" in await e.runner._handle_message(_make_event("continue", thread_id=e.source.thread_id))
        e.runner._hmwa_prepare_turn.assert_not_awaited()
        return

    # With both primary routing and its JSON mirror absent after restart, the
    # durable topic hint must not resurrect a restoration that failed to commit.
    persist({}, scope=e.store._routing_scope())
    (e.store.sessions_dir / "sessions.json").unlink(missing_ok=True)
    fresh_db = SessionDB(db_path=e.tmp / "state.db")
    fresh = SessionStore(e.store.sessions_dir, e.store.config)
    fresh._db = fresh_db
    e.runner.session_store = fresh
    try:
        await e.runner._hmwa_resolve_session(
            _make_event("continue", thread_id=e.source.thread_id), e.source)
        recovered = fresh.lookup_by_session_key(old.session_key)
        assert recovered is not None and recovered.session_id == old.session_id
    finally:
        e.runner.session_store = e.store
        fresh.close_all_db_handles()
        fresh_db.close()


@pytest.mark.asyncio
async def test_topic_restore_losing_concurrent_binding_preserves_route(routed_topic, monkeypatch):
    e = routed_topic
    old = e.store.get_or_create_session(e.source)
    e.store.set_session_metadata(old.session_key, "compression_exhausted", True, require_primary=True)
    e.runner._record_telegram_topic_binding(e.source, old)
    e.db.create_session("contended-target", source="telegram", user_id=e.source.user_id)
    linked = e.db.is_telegram_session_linked_to_topic
    def competing_bind(*args, **kwargs):
        result = linked(*args, **kwargs)
        assert not result
        peer = SessionDB(db_path=e.tmp / "state.db")
        try:
            peer.bind_telegram_topic(
                chat_id=e.source.chat_id, thread_id="other-topic", user_id=e.source.user_id,
                session_key="other-topic-key", session_id="contended-target")
        finally:
            peer.close()
        return result
    monkeypatch.setattr(e.db, "is_telegram_session_linked_to_topic", competing_bind)
    reply = await e.runner._handle_topic_command(
        _make_event("/topic contended-target", thread_id=e.source.thread_id), "contended-target")
    assert "already linked" in reply
    assert e.store.lookup_by_session_key(old.session_key) is old and old.compression_paused
    assert e.db.get_telegram_topic_binding(
        chat_id=e.source.chat_id, thread_id=e.source.thread_id)["session_id"] == old.session_id
    assert e.db.get_telegram_topic_binding(
        chat_id=e.source.chat_id, thread_id="other-topic")["session_id"] == "contended-target"
    e.runner._evict_cached_agent.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize("denial", ["source", "user", "linked", "profile_link"])
async def test_topic_restore_preserves_native_ownership_checks(routed_topic, denial):
    e = routed_topic
    old = e.store.get_or_create_session(e.source)
    e.db.create_session("denied-target", source="slack" if denial == "source" else "telegram",
                        user_id="foreign-user" if denial == "user" else e.source.user_id)
    if denial in {"linked", "profile_link"}:
        e.db.bind_telegram_topic(chat_id=e.source.chat_id,
                                thread_id="different" if denial == "linked" else e.source.thread_id,
                                user_id=e.source.user_id, session_key="foreign-key", session_id="denied-target",
                                profile_name="other" if denial == "profile_link" else "default")
    reply = await e.runner._handle_topic_command(
        _make_event("/topic denied-target", thread_id=e.source.thread_id), "denied-target")
    assert "Session restored" not in reply
    assert e.store.lookup_by_session_key(old.session_key).session_id == old.session_id


@pytest.mark.asyncio
@pytest.mark.parametrize("missing", [False, True])
async def test_route_created_during_topic_fallback_lookup_wins(routed_topic, monkeypatch, missing):
    import asyncio
    import threading
    e = routed_topic
    if not missing:
        e.store.get_or_create_session(e.source)
    e.db.create_session("stale-binding", source="telegram", user_id=e.source.user_id)
    e.db.end_session("stale-binding", end_reason="session_reset")
    e.db.bind_telegram_topic(chat_id=e.source.chat_id, thread_id=e.source.thread_id,
                            user_id=e.source.user_id, session_key=build_session_key(e.source), session_id="stale-binding")
    entered, release = threading.Event(), threading.Event()
    original = e.db.get_telegram_topic_binding
    def blocked(*args, **kwargs):
        value = original(*args, **kwargs)
        entered.set()
        assert release.wait(5)
        return value
    monkeypatch.setattr(e.db, "get_telegram_topic_binding", blocked)
    event = _make_event("ordinary message", thread_id=e.source.thread_id)
    task = asyncio.create_task(e.runner._hmwa_resolve_session(event, e.source))
    try:
        assert await asyncio.to_thread(entered.wait, 3)
        winner = await asyncio.to_thread(e.store.get_or_create_session, e.source, force_new=True)
    finally:
        release.set()
    resolved = await task
    assert resolved[1].session_id == winner.session_id
    assert e.store.lookup_by_session_key(winner.session_key).session_id == winner.session_id
    assert e.db.get_session("stale-binding")["ended_at"] is not None


@pytest.mark.asyncio
async def test_missing_route_topic_restore_requires_primary_commit(routed_topic, monkeypatch):
    import json
    e = routed_topic
    e.store._write_sessions_json = False
    key = build_session_key(e.source)
    scope = e.store._routing_scope()
    e.db.create_session("restorable-target", source="telegram", user_id=e.source.user_id,
                        chat_id=e.source.chat_id, chat_type="dm", thread_id=e.source.thread_id,
                        session_key=key)
    e.db.append_message("restorable-target", "assistant", "recoverable history")
    e.db._conn.execute("""
        CREATE TRIGGER abort_topic_restore BEFORE INSERT ON gateway_routing
        BEGIN SELECT RAISE(ABORT, 'primary unavailable'); END
    """)
    reply = await e.runner._handle_topic_command(
        _make_event("/topic restorable-target", thread_id=e.source.thread_id), "restorable-target")
    # Read primary directly: a new SessionStore can import the bootstrap JSON.
    durable = e.db.load_gateway_routing_entries(scope=scope)
    assert key not in durable
    assert "could not be persisted" in reply and "Session restored" not in reply
    recovered = e.store.lookup_by_session_key(key)
    assert recovered.session_id == "restorable-target"
    e.runner._evict_cached_agent.assert_not_called()

    e.db._conn.execute("DROP TRIGGER abort_topic_restore")
    get_current = e.store.get_or_create_session
    def without_primary(*args, **kwargs):
        entry = get_current(*args, **kwargs)
        # Keep the route absent after bootstrap too: the atomic writer must accept
        # an absent primary, while preserving this same-ID entry and its metadata.
        e.db.replace_gateway_routing_entries({}, scope=scope)
        return entry
    monkeypatch.setattr(e.store, "get_or_create_session", without_primary)
    reply = await e.runner._handle_topic_command(
        _make_event("/topic restorable-target", thread_id=e.source.thread_id), "restorable-target")
    durable = e.db.load_gateway_routing_entries(scope=scope)
    assert json.loads(durable[key])["session_id"] == "restorable-target"
    assert "Session restored" in reply and "recoverable history" in reply
    assert e.store.lookup_by_session_key(key) is recovered
    e.runner._evict_cached_agent.assert_not_called()


@pytest.mark.asyncio
async def test_strict_same_id_switch_commits_primary_without_resetting_entry(routed_topic, monkeypatch):
    import copy
    import json
    e = routed_topic
    old = e.store.get_or_create_session(e.source)
    key, target = old.session_key, old.session_id
    old.last_prompt_tokens = 900
    old.cache_read_tokens, old.cache_write_tokens = 300, 40
    e.store.set_session_metadata(key, "compression_exhausted", True, require_primary=True)
    old._compression_pause_pending = True
    before = copy.deepcopy(old.to_dict())
    metadata, origin = old.metadata, old.origin
    scope = e.store._routing_scope()
    persist = e.db.replace_gateway_routing_entries
    persist({}, scope=scope)
    failure = MagicMock(side_effect=OSError("primary unavailable"))
    monkeypatch.setattr(e.db, "replace_gateway_routing_entries", failure)
    clear_scope = MagicMock(wraps=e.runner._clear_conversation_scope)
    monkeypatch.setattr(e.runner, "_clear_conversation_scope", clear_scope)

    # Ordinary same-ID callers retain their no-write fast return.
    assert e.store.switch_session(key, target) is old
    failure.assert_not_called()
    with pytest.raises(OSError, match="primary unavailable"):
        e.store.switch_session(key, target, require_primary=True, expected_session_id=target)
    assert e.db.load_gateway_routing_entries(scope=scope) == {}
    assert e.store.lookup_by_session_key(key) is old
    assert old.to_dict() == before and old._compression_pause_pending

    monkeypatch.setattr(e.db, "replace_gateway_routing_entries", persist)
    assert e.store.switch_session(key, target, require_primary=True, expected_session_id=target) is old
    assert json.loads(e.db.load_gateway_routing_entries(scope=scope)[key]) == before
    assert old.to_dict() == before and old._compression_pause_pending
    assert old.metadata is metadata and old.origin is origin
    # The explicit topic caller must also retain a paused same-ID route and caches.
    # Its get_or_create_session call separately advances the user-activity clock.
    reply = await e.runner._handle_topic_command(
        _make_event(f"/topic {target}", thread_id=e.source.thread_id), target)
    assert "Session restored" in reply
    assert e.store.lookup_by_session_key(key) is old
    assert old.compression_paused and old._compression_pause_pending
    assert old.metadata is metadata and old.origin is origin
    clear_scope.assert_not_called()
    e.runner._evict_cached_agent.assert_not_called()


@pytest.mark.parametrize("same_id", [False, True])
def test_strict_same_id_switch_refuses_reconciled_stale_identity(routed_topic, monkeypatch, same_id):
    import json
    e = routed_topic
    old = e.store.get_or_create_session(e.source)
    key, target = old.session_key, old.session_id
    e.store.set_session_metadata(key, "compression_exhausted", True, require_primary=True)
    scope = e.store._routing_scope()
    loader = e.db.load_gateway_routing_entries
    fallback = SessionStore(e.store.sessions_dir, e.store.config)
    fallback._db = e.db
    try:
        with monkeypatch.context() as outage:
            outage.setattr(e.db, "load_gateway_routing_entries", MagicMock(side_effect=OSError("read unavailable")))
            stale = fallback.lookup_by_session_key(key)
        assert stale is not None and stale.session_id == target and stale.compression_paused
        durable = json.loads(loader(scope=scope)[key])
        durable["metadata"]["primary_marker"] = "authoritative"
        if not same_id:
            durable["session_id"] = "primary-winner"
            e.db.create_session("primary-winner", source="telegram", user_id=e.source.user_id)
        e.db.replace_gateway_routing_entries({key: json.dumps(durable)}, scope=scope)
        before = loader(scope=scope)
        calls = 0
        def recovering_load(*args, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 1:
                # Entry lookup still sees fallback; the snapshot sees recovered primary.
                raise OSError("read still unavailable")
            return loader(*args, **kwargs)
        monkeypatch.setattr(e.db, "load_gateway_routing_entries", recovering_load)
        persist = MagicMock(wraps=e.db.replace_gateway_routing_entries)
        monkeypatch.setattr(e.db, "replace_gateway_routing_entries", persist)
        with pytest.raises(RuntimeError, match="Session route changed"):
            fallback.switch_session(key, target, require_primary=True, expected_session_id=target)
        assert calls == 2
        authoritative = fallback.lookup_by_session_key(key)
        assert authoritative is not None and authoritative is not stale
        assert authoritative.to_dict() == durable
        assert loader(scope=scope) == before
        persist.assert_not_called()
    finally:
        fallback.close_all_db_handles()


@pytest.mark.asyncio
@pytest.mark.parametrize("missing_primary", [False, True])
async def test_topic_restore_route_cas_rejects_independent_writer(routed_topic, monkeypatch, missing_primary):
    import json
    e = routed_topic
    old = e.store.get_or_create_session(e.source)
    key, scope = old.session_key, e.store._routing_scope()
    e.store.set_session_metadata(key, "compression_exhausted", True, require_primary=True)
    e.runner._record_telegram_topic_binding(e.source, old)
    e.db.create_session("cas-target", source="telegram", user_id=e.source.user_id)
    e.db.create_session("route-winner", source="telegram", user_id=e.source.user_id)
    monkeypatch.setattr("gateway.session._now", lambda: old.updated_at)
    if missing_primary:
        get_current = e.store.get_or_create_session
        def lose_primary(*args, **kwargs):
            current = get_current(*args, **kwargs)
            e.db.replace_gateway_routing_entries({}, scope=scope)
            return current
        monkeypatch.setattr(e.store, "get_or_create_session", lose_primary)
    before_binding = e.db.get_telegram_topic_binding(chat_id=e.source.chat_id, thread_id=e.source.thread_id)
    before_mirror = (e.store.sessions_dir / "sessions.json").read_bytes()
    loader = e.db.load_gateway_routing_entries
    observed = []
    winner = {**old.to_dict(), "session_id": "route-winner"}
    peer = SessionDB(db_path=e.db.db_path)
    def competing_load(*args, **kwargs):
        rows = loader(*args, **kwargs)
        observed.append(rows.get(key))
        peer.save_gateway_routing_entry(key, json.dumps(winner), scope=scope)
        return rows
    monkeypatch.setattr(e.db, "load_gateway_routing_entries", competing_load)
    try:
        reply = await e.runner._handle_topic_command(
            _make_event("/topic cas-target", thread_id=e.source.thread_id), "cas-target")
        assert "could not be persisted" in reply
        assert observed and (observed[0] is None) == missing_primary
        assert e.store.lookup_by_session_key(key) is old and old.compression_paused
        assert json.loads(loader(scope=scope)[key]) == winner
        assert e.db.get_telegram_topic_binding(chat_id=e.source.chat_id, thread_id=e.source.thread_id) == before_binding
        assert (e.store.sessions_dir / "sessions.json").read_bytes() == before_mirror
        assert e.db.get_session(old.session_id)["end_reason"] is None
        e.runner._evict_cached_agent.assert_not_called()
    finally:
        peer.close()


@pytest.mark.asyncio
async def test_topic_restore_mirror_failure_keeps_committed_authority(routed_topic, monkeypatch):
    import json
    e = routed_topic
    old = e.store.get_or_create_session(e.source)
    e.store.set_session_metadata(old.session_key, "compression_exhausted", True, require_primary=True)
    e.runner._record_telegram_topic_binding(e.source, old)
    e.db.create_session("mirror-target", source="telegram", user_id=e.source.user_id)
    e.db.end_session("mirror-target", end_reason="session_switch")
    monkeypatch.setattr(e.store, "_save_sessions_json", MagicMock(side_effect=OSError("mirror unavailable")))
    reply = await e.runner._handle_topic_command(
        _make_event("/topic mirror-target", thread_id=e.source.thread_id), "mirror-target")
    assert "Session restored" in reply
    fresh_db = SessionDB(db_path=e.db.db_path)
    fresh = SessionStore(e.store.sessions_dir, e.store.config)
    fresh._db = fresh_db
    try:
        assert fresh.lookup_by_session_key(old.session_key).session_id == "mirror-target"
        assert json.loads(fresh_db.load_gateway_routing_entries(scope=e.store._routing_scope())[old.session_key])["session_id"] == "mirror-target"
        assert fresh_db.get_telegram_topic_binding(chat_id=e.source.chat_id, thread_id=e.source.thread_id)["session_id"] == "mirror-target"
        assert fresh_db.get_session(old.session_id)["end_reason"] == "session_switch"
        assert fresh_db.get_session("mirror-target")["ended_at"] is None
        e.runner._evict_cached_agent.assert_called_once_with(old.session_key)
    finally:
        fresh.close_all_db_handles()
        fresh_db.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("fail", [False, True])
async def test_topic_restore_cancellation_waits_for_atomic_settlement(routed_topic, monkeypatch, fail):
    import asyncio
    import threading
    e = routed_topic
    old = e.store.get_or_create_session(e.source)
    e.store.set_session_metadata(old.session_key, "compression_exhausted", True, require_primary=True)
    e.runner._record_telegram_topic_binding(e.source, old)
    e.db.create_session("cancel-target", source="telegram", user_id=e.source.user_id)
    entered, release = threading.Event(), threading.Event()
    def pause_write():
        entered.set()
        return int(release.wait(5))
    e.db._conn.create_function("pause_topic_restore", 0, pause_write)
    e.db._conn.execute("""
        CREATE TRIGGER hold_topic_restore BEFORE INSERT ON gateway_routing
        WHEN EXISTS (SELECT 1 FROM telegram_dm_topic_bindings
                     WHERE session_id = 'cancel-target' AND managed_mode = 'restored')
        BEGIN SELECT pause_topic_restore();
    """ + ("SELECT RAISE(ABORT, 'restore aborted');" if fail else "") + " END")
    clear_scope = MagicMock(wraps=e.runner._clear_conversation_scope)
    monkeypatch.setattr(e.runner, "_clear_conversation_scope", clear_scope)
    task = asyncio.create_task(e.runner._handle_topic_command(
        _make_event("/topic cancel-target", thread_id=e.source.thread_id), "cancel-target"))
    try:
        assert await asyncio.to_thread(entered.wait, 3)
        for _ in range(3):
            task.cancel()
            await asyncio.sleep(0)
            assert not task.done()
            assert "recovery is still settling" in e.runner._paused_recovery_busy_reply(old.session_key)
        clear_scope.assert_not_called()
        e.runner._evict_cached_agent.assert_not_called()
    finally:
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await task
    assert not e.runner._is_session_running(old.session_key)
    expected = old.session_id if fail else "cancel-target"
    assert e.db.get_telegram_topic_binding(chat_id=e.source.chat_id, thread_id=e.source.thread_id)["session_id"] == expected
    assert e.store.lookup_by_session_key(old.session_key).session_id == expected
    assert e.store.lookup_by_session_key(old.session_key).compression_paused == fail
    assert clear_scope.call_count == (0 if fail else 1)
    assert e.runner._evict_cached_agent.call_count == (0 if fail else 1)


@pytest.mark.asyncio
@pytest.mark.parametrize("topology", ["same_file_alias", "split_existing", "split_missing"])
async def test_topic_restore_checks_actual_database_identity(routed_topic, monkeypatch, topology):
    from hermes_state import AsyncSessionDB
    e = routed_topic
    old = None
    if topology != "split_missing":
        old = e.store.get_or_create_session(e.source)
        e.store.set_session_metadata(old.session_key, "compression_exhausted", True, require_primary=True)
        e.runner._record_telegram_topic_binding(e.source, old)
    alias = e.tmp / "topic-state.db"
    if topology == "same_file_alias":
        alias.symlink_to(e.db.db_path)
    peer = SessionDB(db_path=alias)
    peer.enable_telegram_topic_mode(chat_id=e.source.chat_id, user_id=e.source.user_id)
    peer.create_session("physical-target", source="telegram", user_id=e.source.user_id)
    e.runner._session_db = AsyncSessionDB(peer)
    before_routes = e.db.load_gateway_routing_entries(scope=e.store._routing_scope())
    before_bindings = peer.list_telegram_topic_bindings_for_chat(chat_id=e.source.chat_id)
    mirror = e.store.sessions_dir / "sessions.json"
    before_mirror = mirror.read_bytes() if mirror.exists() else None
    get_current = MagicMock(wraps=e.store.get_or_create_session)
    monkeypatch.setattr(e.store, "get_or_create_session", get_current)
    lookup = MagicMock(wraps=e.store.lookup_by_session_key)
    monkeypatch.setattr(e.store, "lookup_by_session_key", lookup)
    try:
        reply = await e.runner._handle_topic_command(
            _make_event("/topic physical-target", thread_id=e.source.thread_id), "physical-target")
        if topology == "same_file_alias":
            assert "Session restored" in reply
            assert peer.get_telegram_topic_binding(chat_id=e.source.chat_id, thread_id=e.source.thread_id)["session_id"] == "physical-target"
            assert e.store.lookup_by_session_key(old.session_key).session_id == "physical-target"
        else:
            assert "separate databases" in reply and "not supported" in reply
            get_current.assert_not_called()
            lookup.assert_not_called()  # Admission lookup can reconcile and save a route.
            assert e.db.load_gateway_routing_entries(scope=e.store._routing_scope()) == before_routes
            assert peer.list_telegram_topic_bindings_for_chat(chat_id=e.source.chat_id) == before_bindings
            assert (mirror.read_bytes() if mirror.exists() else None) == before_mirror
            assert peer.get_session("physical-target")["session_key"] is None
            if old is not None:
                assert e.store._entries[old.session_key] is old and old.compression_paused
                assert e.db.get_session(old.session_id)["end_reason"] is None
            e.runner._evict_cached_agent.assert_not_called()
    finally:
        peer.close()
