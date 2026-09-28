"""Slack-specific opt-in for autonomous restart recovery."""

import asyncio
from datetime import datetime, timedelta
from unittest.mock import AsyncMock

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig, load_gateway_config
from gateway.platforms.base import SendResult
from gateway.platforms.event import MessageEvent, MessageType
from gateway.profile_routing import parse_profile_routes
from gateway.run import (
    GatewayRunner,
    _AGENT_PENDING_SENTINEL,
    _prepare_resume_pending_message,
)
from gateway.run_turn_runner import TurnRunner
from gateway.session import SessionEntry, SessionSource
from gateway.session_identity import resolve_identity
from gateway.turn_context import TurnContext
from plugins.platforms.slack.adapter import SlackAdapter
from tests.gateway.restart_test_helpers import make_restart_runner


def _make_slack_resume_runner(*, marked_at=None, authorized=True):
    config = PlatformConfig(
        enabled=True,
        token="***",
        extra={"auto_continue_resume_pending": True},
    )
    adapter = SlackAdapter(config)
    runner, _ = make_restart_runner(adapter)
    runner.config = GatewayConfig(platforms={Platform.SLACK: config})
    runner.adapters = {Platform.SLACK: adapter}
    runner._is_user_authorized = lambda _source: authorized

    source = SessionSource(
        platform=Platform.SLACK,
        chat_id="C-RESUME",
        chat_type="group",
        user_id="U-OWNER",
        thread_id="1712345678.000100",
    )
    session_key = runner._session_key_for_source(source)
    now = datetime.now()
    entry = SessionEntry(
        session_key=session_key,
        session_id="sid-slack-resume",
        created_at=now,
        updated_at=now,
        origin=source,
        platform=Platform.SLACK,
        chat_type="group",
        resume_pending=True,
        resume_reason="restart_interrupted",
        last_resume_marked_at=marked_at or now,
    )
    runner.session_store._entries = {session_key: entry}
    return runner, adapter, source, entry


def _wire_real_slack_turn_path(runner, adapter):
    """Use the real scheduler, Slack background pipeline, gateway handler, and TurnRunner."""
    runner._handle_message = GatewayRunner._handle_message.__get__(
        runner, GatewayRunner
    )
    runner._release_running_agent_state = (
        GatewayRunner._release_running_agent_state.__get__(runner, GatewayRunner)
    )
    runner._check_slash_access = lambda *args, **kwargs: None
    runner._begin_session_run_generation = lambda _session_key: 1
    runner._is_session_run_current = lambda _session_key, _generation: True
    runner._invalidate_session_run_generation = lambda *args, **kwargs: 0
    runner._claim_active_session_slot = lambda _session_key, _source: (object(), None)
    runner._active_session_leases = {}
    runner._busy_ack_ts = {}
    runner._post_turn_goal_continuation = AsyncMock()
    runner.session_store.get_or_create_session.return_value = None

    prepared_turns = []

    async def capture_prepared_turn(event, source, session_key, _run_generation):
        ctx = TurnContext(
            source=source,
            message=event.text,
            history=[],
            session_id="sid-slack-resume",
            session_key=session_key,
            persist_user_message=None,
            persist_user_timestamp=None,
            user_config={},
        )
        persisted, _ = TurnRunner(runner, ctx)._prepare_turn_message([])
        prepared_turns.append({
            "message": ctx.message,
            "persisted": persisted,
            "internal": event.internal,
            "original_text": event.text,
        })
        return "RESUMED OK"

    runner._handle_message_with_agent = capture_prepared_turn
    adapter.set_message_handler(runner._handle_message)
    adapter.send = AsyncMock()
    adapter._keep_typing = AsyncMock()
    adapter._stop_typing_refresh = AsyncMock()
    adapter._send_with_retry = AsyncMock(
        return_value=SendResult(success=True, message_id="1")
    )
    adapter._run_processing_hook = AsyncMock()
    return prepared_turns


async def _wait_for_turn(prepared_turns):
    for _ in range(100):
        if prepared_turns:
            return prepared_turns[0]
        await asyncio.sleep(0.01)
    raise AssertionError("Slack resume turn did not reach TurnRunner")


async def _wait_until_idle(runner, session_key):
    for _ in range(100):
        if session_key not in runner._running_agents:
            return
        await asyncio.sleep(0.01)
    raise AssertionError("Slack resume turn leaked its session slot")


def test_slack_resume_policy_is_explicit_profile_local_and_fail_closed(
    tmp_path, monkeypatch
):
    for configured in (True, "true", "1", "yes", "on", " TRUE "):
        adapter = SlackAdapter(
            PlatformConfig(extra={"auto_continue_resume_pending": configured})
        )
        assert adapter.interactive_resume is False

    for configured in (
        None,
        False,
        "false",
        "0",
        "no",
        "off",
        "",
        "enabled",
        1,
        [],
        {},
    ):
        extra = (
            {} if configured is None else {"auto_continue_resume_pending": configured}
        )
        assert SlackAdapter(PlatformConfig(extra=extra)).interactive_resume is True

    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir()
    (hermes_home / "config.yaml").write_text(
        "slack:\n  auto_continue_resume_pending: true\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    slack_config = load_gateway_config().platforms[Platform.SLACK]
    opted_in = SlackAdapter(slack_config)
    defaulted = SlackAdapter(PlatformConfig(extra={}))

    assert slack_config.extra["auto_continue_resume_pending"] is True
    assert opted_in.interactive_resume is False
    assert defaulted.interactive_resume is True
    recovery_message, persisted_message = _prepare_resume_pending_message(
        "restart_interrupted", "", interactive=opted_in.interactive_resume
    )
    assert "CONTINUE the interrupted task to completion" in recovery_message
    assert "Do NOT re-run tool calls whose results already appear" in recovery_message
    assert "ask what they would like to do next" not in recovery_message
    assert persisted_message == recovery_message


@pytest.mark.asyncio
async def test_slack_opt_in_preserves_gateway_recovery_invariants(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(
        "gateway.restart_loop_guard.check_and_record", lambda *args, **kwargs: False
    )

    paused_runner, _, _, paused_entry = _make_slack_resume_runner()
    with monkeypatch.context() as scoped:
        scoped.setattr("agent.estop.check_paused", lambda *_args, **_kwargs: True)
        assert paused_runner._schedule_resume_pending_sessions(Platform.SLACK) == 0
        assert paused_entry.resume_pending is True
        assert paused_runner._is_session_running(paused_entry.session_key) is False

    runner, adapter, _source, entry = _make_slack_resume_runner()
    prepared_turns = _wire_real_slack_turn_path(runner, adapter)
    assert runner._schedule_resume_pending_sessions(platform=Platform.SLACK) == 1
    prepared = await _wait_for_turn(prepared_turns)
    await _wait_until_idle(runner, entry.session_key)

    assert prepared["internal"] is True
    assert prepared["original_text"] == ""
    assert "CONTINUE the interrupted task to completion" in prepared["message"]
    assert "ask what they would like to do next" not in prepared["message"]
    assert prepared["persisted"] == prepared["message"]
    assert len(prepared_turns) == 1

    with monkeypatch.context() as scoped:
        home = tmp_path / "multiplex-home"
        profile_home = home / "profiles" / "engineering-lead"
        profile_home.mkdir(parents=True)
        scoped.setenv("HERMES_HOME", str(home))
        served = [("default", home), ("engineering-lead", profile_home)]
        scoped.setattr(
            "hermes_cli.profiles.profiles_to_serve", lambda **_kwargs: served
        )
        scoped.setattr(
            "hermes_cli.profiles.get_profile_dir",
            lambda name: home if name == "default" else profile_home,
        )
        scoped.setattr("hermes_cli.profiles.profile_exists", lambda _name: True)

        mux_runner, mux_adapter, _source, _entry = _make_slack_resume_runner()
        mux_runner.config.multiplex_profiles = True
        mux_runner.config.profile_routes = parse_profile_routes([
            {
                "name": "engineering-slack",
                "platform": "slack",
                "profile": "engineering-lead",
                "chat_id": "C-RESUME",
            }
        ])
        mux_runner._primary_profile_name = "default"
        mux_runner._profile_adapters = {"engineering-lead": {}}
        mux_source = mux_adapter.build_source(
            chat_id="C-RESUME",
            chat_type="group",
            user_id="U-OWNER",
            thread_id="1712345678.000100",
        )
        mux_identity = resolve_identity(mux_source, runner=mux_runner)
        mux_key = mux_runner._session_key_for_source(mux_source)
        now = datetime.now()
        mux_entry = SessionEntry(
            session_key=mux_key,
            session_id="sid-engineering-resume",
            created_at=now,
            updated_at=now,
            origin=mux_source,
            platform=Platform.SLACK,
            chat_type="group",
            resume_pending=True,
            resume_reason="restart_interrupted",
            last_resume_marked_at=now,
        )
        mux_runner.session_store._entries = {mux_key: mux_entry}
        mux_turns = _wire_real_slack_turn_path(mux_runner, mux_adapter)
        mux_runner._startup_restore_in_progress = True
        mux_runner._startup_restore_queue = []
        mux_runner._startup_restore_tasks = []

        assert mux_identity.runtime_profile == "engineering-lead"
        assert mux_identity.transport_profile == "default"
        assert mux_key.startswith("agent:engineering-lead:slack:")
        assert mux_runner._delivery_adapter_for(mux_source) is mux_adapter
        assert mux_adapter.interactive_resume is False
        assert (
            mux_runner._schedule_resume_pending_sessions(platform=Platform.SLACK) == 0
        )
        await mux_runner._finish_startup_restore()
        mux_prepared = await _wait_for_turn(mux_turns)
        await _wait_until_idle(mux_runner, mux_key)
        assert mux_prepared["internal"] is True
        assert "CONTINUE the interrupted task to completion" in mux_prepared["message"]

    user_runner, user_adapter, user_source, user_entry = _make_slack_resume_runner()
    user_turns = _wire_real_slack_turn_path(user_runner, user_adapter)
    await user_adapter.handle_message(
        MessageEvent(
            text="Handle this new request instead",
            message_type=MessageType.TEXT,
            source=user_source,
        )
    )
    user_prepared = await _wait_for_turn(user_turns)
    await _wait_until_idle(user_runner, user_entry.session_key)
    assert user_prepared["internal"] is False
    assert "Address the user's NEW message below FIRST" in user_prepared["message"]
    assert "CONTINUE the interrupted task to completion" not in user_prepared["message"]
    assert user_prepared["message"].endswith("Handle this new request instead")
    assert user_prepared["persisted"] == "Handle this new request instead"

    startup_runner, startup_adapter, startup_source, startup_entry = (
        _make_slack_resume_runner()
    )
    startup_turns = _wire_real_slack_turn_path(startup_runner, startup_adapter)
    startup_runner._startup_restore_in_progress = True
    startup_runner._startup_restore_queue = []
    startup_runner._startup_restore_tasks = []
    inbound = MessageEvent(
        text="This replaces the interrupted request",
        message_type=MessageType.TEXT,
        source=startup_source,
    )
    await startup_adapter.handle_message(inbound)
    for _ in range(100):
        if startup_runner._startup_restore_queue:
            break
        await asyncio.sleep(0.01)
    assert startup_runner._startup_restore_queue == [inbound]
    assert startup_turns == []

    startup_runner._schedule_resume_pending_sessions(platform=Platform.SLACK)
    await startup_runner._finish_startup_restore()
    await _wait_until_idle(startup_runner, startup_entry.session_key)

    assert len(startup_turns) == 1
    assert startup_turns[0]["internal"] is False
    assert "Address the user's NEW message below FIRST" in startup_turns[0]["message"]
    assert startup_turns[0]["message"].endswith("This replaces the interrupted request")

    blocked_runner, blocked_adapter, blocked_source, blocked_entry = (
        _make_slack_resume_runner()
    )
    blocked_runner._is_user_authorized = lambda source: source.user_name != "blocked"
    blocked_turns = _wire_real_slack_turn_path(blocked_runner, blocked_adapter)
    blocked_runner._startup_restore_in_progress = True
    blocked_runner._startup_restore_queue = []
    blocked_runner._startup_restore_tasks = []
    blocked_inbound_source = SessionSource(
        platform=blocked_source.platform,
        chat_id=blocked_source.chat_id,
        chat_type=blocked_source.chat_type,
        user_id=blocked_source.user_id,
        user_name="blocked",
        thread_id=blocked_source.thread_id,
    )
    blocked_inbound = MessageEvent(
        text="Suppress the owner's recovery",
        message_type=MessageType.TEXT,
        source=blocked_inbound_source,
    )
    await blocked_adapter.handle_message(blocked_inbound)
    for _ in range(100):
        if blocked_runner._startup_restore_queue:
            break
        await asyncio.sleep(0.01)
    assert blocked_runner._startup_restore_queue == [blocked_inbound]

    blocked_runner._schedule_resume_pending_sessions(platform=Platform.SLACK)
    await blocked_runner._finish_startup_restore()
    await _wait_for_turn(blocked_turns)
    await _wait_until_idle(blocked_runner, blocked_entry.session_key)

    assert len(blocked_turns) == 1
    assert blocked_turns[0]["internal"] is True
    assert "CONTINUE the interrupted task to completion" in blocked_turns[0]["message"]

    late_runner, late_adapter, late_source, _late_entry = _make_slack_resume_runner()
    _wire_real_slack_turn_path(late_runner, late_adapter)
    late_runner._startup_restore_in_progress = True
    late_runner._startup_restore_queue = []
    first_started = asyncio.Event()
    release_first = asyncio.Event()
    real_turn = late_runner._handle_message_with_agent

    async def delayed_first(event, *args, **kwargs):
        if event.text == "first queued input":
            first_started.set()
            await release_first.wait()
        return await real_turn(event, *args, **kwargs)

    late_runner._handle_message_with_agent = delayed_first
    first_event = MessageEvent(
        text="first queued input", message_type=MessageType.TEXT, source=late_source
    )
    await late_adapter.handle_message(first_event)
    for _ in range(100):
        if late_runner._startup_restore_queue:
            break
        await asyncio.sleep(0.01)
    drain_task = asyncio.create_task(
        late_runner._drain_startup_restore_queue(
            capture_admission=True,
            admission_deadline=asyncio.get_running_loop().time() + 1.0,
        )
    )
    await asyncio.wait_for(first_started.wait(), timeout=1.0)
    late_source_2 = SessionSource(
        platform=Platform.SLACK,
        chat_id="C-LATE",
        chat_type="group",
        user_id="U-OWNER",
        thread_id="1712345678.000200",
    )
    late_event = MessageEvent(
        text="arrived during admission wait",
        message_type=MessageType.TEXT,
        source=late_source_2,
    )
    await late_adapter.handle_message(late_event)
    for _ in range(100):
        if late_runner._startup_restore_queue:
            break
        await asyncio.sleep(0.01)
    release_first.set()
    assert await asyncio.wait_for(drain_task, timeout=2.0) == 2
    assert late_runner._startup_restore_queue == []

    with monkeypatch.context() as scoped:
        scoped.setenv("HERMES_STARTUP_RESTORE_DRAIN_TIMEOUT", "0.05")
        budget_runner, _budget_adapter, _budget_source, budget_entry = (
            _make_slack_resume_runner()
        )
        budget_runner._startup_restore_in_progress = True
        budget_runner._startup_restore_queue = []
        budget_runner._startup_deferred_resume_session_keys = {budget_entry.session_key}
        completed_restore = asyncio.get_running_loop().create_future()
        completed_restore.set_result(None)
        budget_runner._startup_restore_tasks = [completed_restore]
        clock = [100.0]
        observed = {}

        class FakeLoop:
            def time(self):
                return clock[0]

        async def fake_wait(tasks, timeout, *_args, **_kwargs):
            observed["initial_timeout"] = timeout
            clock[0] += 0.04
            return set(tasks)

        async def fake_drain(*, capture_admission, admission_deadline):
            observed["capture_admission"] = capture_admission
            observed["admission_deadline"] = admission_deadline
            observed["remaining_budget"] = admission_deadline - clock[0]
            return 0

        def fake_schedule(superseded):
            observed["superseded"] = superseded
            return 0

        scoped.setattr(
            "gateway.run_startup.asyncio.get_running_loop", lambda: FakeLoop()
        )
        budget_runner._wait_bounded_or_release = fake_wait
        budget_runner._drain_startup_restore_queue = fake_drain
        budget_runner._schedule_deferred_startup_resumes = fake_schedule
        await budget_runner._finish_startup_restore()

        assert observed["initial_timeout"] == pytest.approx(0.05)
        assert observed["capture_admission"] is True
        assert observed["admission_deadline"] == pytest.approx(100.05)
        assert observed["remaining_budget"] == pytest.approx(0.01)

    store_runner, _store_adapter, _store_source, store_entry = (
        _make_slack_resume_runner()
    )
    store_runner._startup_deferred_resume_session_keys = {store_entry.session_key}

    def fail_store_load():
        raise RuntimeError("store unavailable")

    with monkeypatch.context() as scoped:
        scoped.setattr(
            store_runner.session_store, "_ensure_loaded_locked", fail_store_load
        )
        with pytest.raises(RuntimeError, match="store unavailable"):
            store_runner._schedule_deferred_startup_resumes(set())
    assert store_runner._startup_deferred_resume_session_keys == {
        store_entry.session_key
    }

    lane_runner, lane_adapter, lane_source, lane_entry = _make_slack_resume_runner()
    lane_adapter.handle_message = AsyncMock()
    lane_probe = MessageEvent(
        text="fresh reconnect input",
        message_type=MessageType.TEXT,
        source=lane_source,
    )
    lane_key = lane_adapter._event_session_key(lane_probe)
    lane_task = asyncio.create_task(asyncio.sleep(1.0))
    lane_adapter._session_tasks[lane_key] = lane_task
    lane_adapter._active_sessions[lane_key] = object()
    assert lane_runner._schedule_resume_pending_sessions(Platform.SLACK) == 0
    assert lane_entry.resume_pending is True
    assert lane_runner._is_session_running(lane_entry.session_key) is False
    lane_task.cancel()
    await asyncio.gather(lane_task, return_exceptions=True)
    lane_adapter._session_tasks.pop(lane_key, None)
    lane_adapter._active_sessions.pop(lane_key, None)
    assert lane_runner._schedule_resume_pending_sessions(Platform.SLACK) == 1
    await _wait_until_idle(lane_runner, lane_entry.session_key)
    lane_adapter.handle_message.assert_awaited_once()

    with monkeypatch.context() as scoped:
        scoped.setenv("HERMES_AUTO_CONTINUE_FRESHNESS", "60")
        stale_runner, stale_adapter, _source, stale_entry = _make_slack_resume_runner(
            marked_at=datetime.now() - timedelta(minutes=2)
        )
        stale_adapter.handle_message = AsyncMock()
        assert (
            stale_runner._schedule_resume_pending_sessions(platform=Platform.SLACK) == 0
        )
        stale_adapter.handle_message.assert_not_awaited()
        assert stale_entry.session_key not in stale_runner._running_agents

    denied_runner, denied_adapter, _source, denied_entry = _make_slack_resume_runner(
        authorized=False
    )
    denied_adapter.handle_message = AsyncMock()
    assert denied_runner._schedule_resume_pending_sessions(platform=Platform.SLACK) == 0
    denied_adapter.handle_message.assert_not_awaited()
    assert denied_entry.session_key not in denied_runner._running_agents

    with monkeypatch.context() as scoped:
        scoped.setattr(
            "gateway.restart_loop_guard.check_and_record", lambda *args, **kwargs: True
        )
        loop_runner, loop_adapter, _source, loop_entry = _make_slack_resume_runner()
        loop_adapter.handle_message = AsyncMock()
        assert (
            loop_runner._schedule_resume_pending_sessions(platform=Platform.SLACK) == 0
        )
        loop_adapter.handle_message.assert_not_awaited()
        assert loop_entry.session_key not in loop_runner._running_agents

    claim_runner, claim_adapter, _source, claim_entry = _make_slack_resume_runner()
    release = asyncio.Event()

    async def slow_handle(_event):
        await release.wait()

    claim_adapter.handle_message = slow_handle
    assert claim_runner._schedule_resume_pending_sessions(platform=Platform.SLACK) == 1
    assert claim_runner._schedule_resume_pending_sessions(platform=Platform.SLACK) == 0
    assert (
        claim_runner._running_agents[claim_entry.session_key] is _AGENT_PENDING_SENTINEL
    )
    release.set()
    await _wait_until_idle(claim_runner, claim_entry.session_key)
