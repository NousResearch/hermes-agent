"""Tests for configurable background process notification modes.

The gateway process watcher pushes status updates to users' chats when
background terminal commands run.  ``display.background_process_notifications``
controls verbosity: off | result | error | all (default).

Contributed by @PeterFile (PR #593), reimplemented on current main.
"""

import asyncio
import threading
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import GatewayConfig, Platform
from gateway.run import GatewayRunner, _parse_session_key


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _FakeRegistry:
    """Return pre-canned sessions, then None once exhausted."""

    def __init__(self, sessions):
        self._sessions = list(sessions)

    def get(self, session_id):
        if self._sessions:
            return self._sessions.pop(0)
        return None

    def is_completion_consumed(self, session_id):
        return False


def _build_runner(monkeypatch, tmp_path, mode: str) -> GatewayRunner:
    """Create a GatewayRunner with a fake config for the given mode."""
    (tmp_path / "config.yaml").write_text(
        f"display:\n  background_process_notifications: {mode}\n",
        encoding="utf-8",
    )

    import gateway.run as gateway_run

    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)

    runner = GatewayRunner(GatewayConfig())
    adapter = SimpleNamespace(send=AsyncMock(), handle_message=AsyncMock())
    runner.adapters[Platform.TELEGRAM] = adapter

    async def _capture_process_event(event, **_kwargs):
        return await adapter.handle_message(event)

    runner._run_and_deliver_stamped_process_event = AsyncMock(
        side_effect=_capture_process_event
    )
    return runner


def _watcher_dict(session_id="proc_test", thread_id=""):
    d = {
        "session_id": session_id,
        "check_interval": 0,
        "platform": "telegram",
        "chat_id": "123",
    }
    if thread_id:
        d["thread_id"] = thread_id
    return d


def test_set_session_env_exposes_physical_session_id(monkeypatch, tmp_path):
    """Gateway-bound tools must see the physical session id, not just session_key."""
    from gateway.session import SessionContext, SessionSource
    from gateway.session_context import clear_session_vars, get_session_env

    runner = _build_runner(monkeypatch, tmp_path, "all")
    ctx = SessionContext(
        source=SessionSource(
            platform=Platform.TELEGRAM,
            chat_id="123",
            chat_type="dm",
        ),
        connected_platforms=[Platform.TELEGRAM],
        home_channels={},
        session_key="agent:main:telegram:dm:123",
        session_id="sess-current",
    )

    tokens = runner._set_session_env(ctx)
    try:
        assert get_session_env("HERMES_SESSION_ID") == "sess-current"
    finally:
        clear_session_vars(tokens)


# ---------------------------------------------------------------------------
# _load_background_notifications_mode unit tests
# ---------------------------------------------------------------------------

class TestLoadBackgroundNotificationsMode:

    def test_defaults_to_all(self, monkeypatch, tmp_path):
        import gateway.run as gw
        monkeypatch.setattr(gw, "_hermes_home", tmp_path)
        monkeypatch.delenv("HERMES_BACKGROUND_NOTIFICATIONS", raising=False)
        assert GatewayRunner._load_background_notifications_mode() == "all"

    def test_reads_config_yaml(self, monkeypatch, tmp_path):
        (tmp_path / "config.yaml").write_text(
            "display:\n  background_process_notifications: error\n"
        )
        import gateway.run as gw
        monkeypatch.setattr(gw, "_hermes_home", tmp_path)
        monkeypatch.delenv("HERMES_BACKGROUND_NOTIFICATIONS", raising=False)
        assert GatewayRunner._load_background_notifications_mode() == "error"

    def test_env_var_overrides_config(self, monkeypatch, tmp_path):
        (tmp_path / "config.yaml").write_text(
            "display:\n  background_process_notifications: error\n"
        )
        import gateway.run as gw
        monkeypatch.setattr(gw, "_hermes_home", tmp_path)
        monkeypatch.setenv("HERMES_BACKGROUND_NOTIFICATIONS", "off")
        assert GatewayRunner._load_background_notifications_mode() == "off"

    def test_false_value_maps_to_off(self, monkeypatch, tmp_path):
        (tmp_path / "config.yaml").write_text(
            "display:\n  background_process_notifications: false\n"
        )
        import gateway.run as gw
        monkeypatch.setattr(gw, "_hermes_home", tmp_path)
        monkeypatch.delenv("HERMES_BACKGROUND_NOTIFICATIONS", raising=False)
        assert GatewayRunner._load_background_notifications_mode() == "off"

    def test_invalid_value_defaults_to_all(self, monkeypatch, tmp_path):
        (tmp_path / "config.yaml").write_text(
            "display:\n  background_process_notifications: banana\n"
        )
        import gateway.run as gw
        monkeypatch.setattr(gw, "_hermes_home", tmp_path)
        monkeypatch.delenv("HERMES_BACKGROUND_NOTIFICATIONS", raising=False)
        assert GatewayRunner._load_background_notifications_mode() == "all"


# ---------------------------------------------------------------------------
# _run_process_watcher integration tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("mode", "sessions", "expected_calls", "expected_fragment"),
    [
        # all mode: running output → sends update
        (
            "all",
            [
                SimpleNamespace(output_buffer="building...\n", exited=False, exit_code=None),
                None,  # process disappears → watcher exits
            ],
            1,
            "is still running",
        ),
        # result mode: running output → no update
        (
            "result",
            [
                SimpleNamespace(output_buffer="building...\n", exited=False, exit_code=None),
                None,
            ],
            0,
            None,
        ),
        # off mode: exited process → no notification
        (
            "off",
            [SimpleNamespace(output_buffer="done\n", exited=True, exit_code=0)],
            0,
            None,
        ),
        # result mode: exited → notifies
        (
            "result",
            [SimpleNamespace(output_buffer="done\n", exited=True, exit_code=0)],
            1,
            "finished with exit code 0",
        ),
        # error mode: exit 0 → no notification
        (
            "error",
            [SimpleNamespace(output_buffer="done\n", exited=True, exit_code=0)],
            0,
            None,
        ),
        # error mode: exit 1 → notifies
        (
            "error",
            [SimpleNamespace(output_buffer="traceback\n", exited=True, exit_code=1)],
            1,
            "finished with exit code 1",
        ),
        # all mode: exited → notifies
        (
            "all",
            [SimpleNamespace(output_buffer="ok\n", exited=True, exit_code=0)],
            1,
            "finished with exit code 0",
        ),
    ],
)
async def test_run_process_watcher_respects_notification_mode(
    monkeypatch, tmp_path, mode, sessions, expected_calls, expected_fragment
):
    import tools.process_registry as pr_module

    monkeypatch.setattr(pr_module, "process_registry", _FakeRegistry(sessions))

    # Patch asyncio.sleep to avoid real delays
    async def _instant_sleep(*_a, **_kw):
        pass
    monkeypatch.setattr(asyncio, "sleep", _instant_sleep)

    runner = _build_runner(monkeypatch, tmp_path, mode)
    adapter = runner.adapters[Platform.TELEGRAM]

    await runner._run_process_watcher(_watcher_dict())

    assert adapter.send.await_count == expected_calls, (
        f"mode={mode}: expected {expected_calls} sends, got {adapter.send.await_count}"
    )
    if expected_fragment is not None:
        sent_message = adapter.send.await_args.args[1]
        assert expected_fragment in sent_message


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("sessions", "mode"),
    [
        ([SimpleNamespace(output_buffer="building...\n", exited=False, exit_code=None), None], "all"),
        ([SimpleNamespace(output_buffer="done\n", exited=True, exit_code=0)], "result"),
    ],
)
async def test_process_watcher_uses_secondary_profile_adapter(
    monkeypatch, tmp_path, sessions, mode
):
    import tools.process_registry as pr_module
    from gateway.session import SessionSource

    monkeypatch.setattr(pr_module, "process_registry", _FakeRegistry(sessions))

    async def _instant_sleep(*_a, **_kw):
        pass

    monkeypatch.setattr(asyncio, "sleep", _instant_sleep)
    runner = _build_runner(monkeypatch, tmp_path, mode)
    default_adapter = runner.adapters[Platform.TELEGRAM]
    coder_adapter = SimpleNamespace(send=AsyncMock(), handle_message=AsyncMock())
    runner._profile_adapters = {"coder": {Platform.TELEGRAM: coder_adapter}}
    key = "agent:coder:telegram:dm:123"
    source = SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="123",
        chat_type="dm",
        user_id="123",
        profile="coder",
    )
    runner.session_store._entries[key] = SimpleNamespace(
        session_id="sess-coder",
        origin=source,
    )
    watcher = _watcher_dict()
    watcher.update(
        {
            "session_key": key,
            "conversation_session_id": "sess-coder",
        }
    )

    await runner._run_process_watcher(watcher)

    coder_adapter.send.assert_awaited_once()
    default_adapter.send.assert_not_awaited()


@pytest.mark.asyncio
async def test_injected_process_event_uses_secondary_profile_adapter(
    monkeypatch, tmp_path
):
    from gateway.session import SessionSource

    runner = _build_runner(monkeypatch, tmp_path, "all")
    default_adapter = runner.adapters[Platform.TELEGRAM]
    coder_adapter = SimpleNamespace(send=AsyncMock(), handle_message=AsyncMock())
    runner._profile_adapters = {"coder": {Platform.TELEGRAM: coder_adapter}}
    key = "agent:coder:telegram:dm:123"
    source = SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="123",
        chat_type="dm",
        user_id="123",
        profile="coder",
    )
    runner.session_store._entries[key] = SimpleNamespace(
        session_id="sess-coder",
        origin=source,
    )
    runner._run_and_deliver_stamped_process_event = AsyncMock()

    await runner._inject_watch_notification(
        "[SYSTEM: coder process completion]",
        {
            "session_id": "proc-coder",
            "session_key": key,
            "conversation_session_id": "sess-coder",
        },
    )

    selected = runner._run_and_deliver_stamped_process_event.await_args.kwargs[
        "adapter"
    ]
    assert selected is coder_adapter
    assert selected is not default_adapter


@pytest.mark.asyncio
async def test_thread_id_passed_to_send(monkeypatch, tmp_path):
    """thread_id from watcher dict is forwarded as metadata to adapter.send()."""
    import tools.process_registry as pr_module

    sessions = [SimpleNamespace(output_buffer="done\n", exited=True, exit_code=0)]
    monkeypatch.setattr(pr_module, "process_registry", _FakeRegistry(sessions))

    async def _instant_sleep(*_a, **_kw):
        pass
    monkeypatch.setattr(asyncio, "sleep", _instant_sleep)

    runner = _build_runner(monkeypatch, tmp_path, "all")
    adapter = runner.adapters[Platform.TELEGRAM]

    await runner._run_process_watcher(_watcher_dict(thread_id="42"))

    assert adapter.send.await_count == 1
    _, kwargs = adapter.send.call_args
    assert kwargs["metadata"] == {"thread_id": "42"}

@pytest.mark.asyncio
async def test_no_thread_id_sends_no_metadata(monkeypatch, tmp_path):
    """When thread_id is empty, metadata should be None (general topic)."""
    import tools.process_registry as pr_module

    sessions = [SimpleNamespace(output_buffer="done\n", exited=True, exit_code=0)]
    monkeypatch.setattr(pr_module, "process_registry", _FakeRegistry(sessions))

    async def _instant_sleep(*_a, **_kw):
        pass
    monkeypatch.setattr(asyncio, "sleep", _instant_sleep)

    runner = _build_runner(monkeypatch, tmp_path, "all")
    adapter = runner.adapters[Platform.TELEGRAM]

    await runner._run_process_watcher(_watcher_dict())

    assert adapter.send.await_count == 1
    _, kwargs = adapter.send.call_args
    assert kwargs["metadata"] is None

@pytest.mark.asyncio
async def test_run_process_watcher_drops_stale_text_completion_after_reset(monkeypatch, tmp_path):
    import tools.process_registry as pr_module

    sessions = [SimpleNamespace(output_buffer="done\n", exited=True, exit_code=0)]
    monkeypatch.setattr(pr_module, "process_registry", _FakeRegistry(sessions))

    async def _instant_sleep(*_a, **_kw):
        pass
    monkeypatch.setattr(asyncio, "sleep", _instant_sleep)

    runner = _build_runner(monkeypatch, tmp_path, "all")
    adapter = runner.adapters[Platform.TELEGRAM]
    session_key = "agent:main:telegram:dm:123"
    runner.session_store._entries[session_key] = SimpleNamespace(session_id="sess-new")

    watcher = _watcher_dict()
    watcher.update({
        "session_key": session_key,
        "conversation_session_id": "sess-old",
    })

    await runner._run_process_watcher(watcher)

    adapter.send.assert_not_awaited()

@pytest.mark.asyncio
async def test_run_process_watcher_keeps_text_completion_for_compression_continuation(monkeypatch, tmp_path):
    import tools.process_registry as pr_module

    sessions = [SimpleNamespace(output_buffer="done\n", exited=True, exit_code=0)]
    monkeypatch.setattr(pr_module, "process_registry", _FakeRegistry(sessions))

    async def _instant_sleep(*_a, **_kw):
        pass
    monkeypatch.setattr(asyncio, "sleep", _instant_sleep)

    runner = _build_runner(monkeypatch, tmp_path, "all")
    adapter = runner.adapters[Platform.TELEGRAM]
    session_key = "agent:main:telegram:dm:123"
    runner.session_store._entries[session_key] = SimpleNamespace(session_id="sess-new")
    runner._session_db = SimpleNamespace(
        _db=SimpleNamespace(
            get_compression_tip=lambda sid: {
                "sess-old": "sess-new",
                "sess-new": "sess-new",
            }.get(sid, sid)
        )
    )

    watcher = _watcher_dict()
    watcher.update({
        "session_key": session_key,
        "conversation_session_id": "sess-old",
    })

    await runner._run_process_watcher(watcher)

    adapter.send.assert_awaited_once()
@pytest.mark.asyncio
async def test_run_process_watcher_drops_stale_text_running_update_after_reset(monkeypatch, tmp_path):
    import tools.process_registry as pr_module

    sessions = [SimpleNamespace(output_buffer="building...\n", exited=False, exit_code=None)]
    monkeypatch.setattr(pr_module, "process_registry", _FakeRegistry(sessions))

    async def _instant_sleep(*_a, **_kw):
        pass
    monkeypatch.setattr(asyncio, "sleep", _instant_sleep)

    runner = _build_runner(monkeypatch, tmp_path, "all")
    adapter = runner.adapters[Platform.TELEGRAM]
    session_key = "agent:main:telegram:dm:123"
    runner.session_store._entries[session_key] = SimpleNamespace(session_id="sess-new")

    watcher = _watcher_dict()
    watcher.update({
        "session_key": session_key,
        "conversation_session_id": "sess-old",
    })

    await runner._run_process_watcher(watcher)

    adapter.send.assert_not_awaited()
@pytest.mark.asyncio
async def test_inject_watch_notification_routes_from_session_store_origin(monkeypatch, tmp_path):
    from gateway.session import SessionSource

    runner = _build_runner(monkeypatch, tmp_path, "all")
    adapter = runner.adapters[Platform.TELEGRAM]
    runner.session_store._entries["agent:main:telegram:group:-100:42"] = SimpleNamespace(
        origin=SessionSource(
            platform=Platform.TELEGRAM,
            chat_id="-100",
            chat_type="group",
            thread_id="42",
            user_id="123",
            user_name="Emiliyan",
        )
    )

    evt = {
        "session_id": "proc_watch",
        "session_key": "agent:main:telegram:group:-100:42",
    }

    await runner._inject_watch_notification("[SYSTEM: Background process matched]", evt)

    adapter.handle_message.assert_awaited_once()
    synth_event = adapter.handle_message.await_args.args[0]
    assert synth_event.internal is True
    assert synth_event.source.platform == Platform.TELEGRAM
    assert synth_event.source.chat_id == "-100"
    assert synth_event.source.chat_type == "group"
    assert synth_event.source.thread_id == "42"
    assert synth_event.source.user_id == "123"
    assert synth_event.source.user_name == "Emiliyan"


@pytest.mark.asyncio
async def test_stamped_watch_uses_guarded_production_delivery_rail(
    monkeypatch, tmp_path
):
    from gateway.session import SessionSource

    runner = _build_runner(monkeypatch, tmp_path, "all")
    adapter = runner.adapters[Platform.TELEGRAM]
    session_key = "agent:main:telegram:dm:123"
    runner.session_store._entries[session_key] = SimpleNamespace(
        session_id="sess-current",
        origin=SessionSource(
            platform=Platform.TELEGRAM,
            chat_id="123",
            chat_type="dm",
            user_id="123",
        ),
    )
    runner._run_and_deliver_stamped_process_event = AsyncMock()

    await runner._inject_watch_notification(
        "[SYSTEM: Background process matched]",
        {
            "session_id": "proc_watch",
            "session_key": session_key,
            "conversation_session_id": "sess-current",
        },
    )

    runner._run_and_deliver_stamped_process_event.assert_awaited_once()
    synth_event = runner._run_and_deliver_stamped_process_event.await_args.args[0]
    assert synth_event.metadata["expected_gateway_session_id"] == "sess-current"
    assert runner._run_and_deliver_stamped_process_event.await_args.kwargs == {
        "adapter": adapter,
        "session_key": session_key,
    }


@pytest.mark.asyncio
async def test_stamped_process_event_waits_without_interrupting_foreground_turn(
    monkeypatch, tmp_path
):
    from gateway.platforms.base import MessageEvent, MessageType
    from gateway.run import _STAMPED_PROCESS_EVENT_BUSY
    from gateway.session import SessionSource

    runner = _build_runner(monkeypatch, tmp_path, "all")
    source = SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="123",
        chat_type="dm",
        user_id="123",
    )
    session_key = runner._session_key_for_source(source)
    foreground_agent = SimpleNamespace(
        interrupt=MagicMock(),
        get_activity_summary=lambda: {"seconds_since_activity": 0},
    )
    runner._running_agents[session_key] = foreground_agent
    runner._running_agents_ts[session_key] = time.time()
    event = MessageEvent(
        text="[SYSTEM: process completion]",
        message_type=MessageType.TEXT,
        source=source,
        internal=True,
        metadata={"expected_gateway_session_id": "sess-current"},
    )

    result = await runner._handle_message(event)

    assert result is _STAMPED_PROCESS_EVENT_BUSY
    foreground_agent.interrupt.assert_not_called()


@pytest.mark.asyncio
async def test_stamped_process_event_returns_through_guarded_direct_send(
    monkeypatch, tmp_path
):
    from gateway.platforms.base import MessageEvent, MessageType
    from gateway.session import SessionSource

    runner = _build_runner(monkeypatch, tmp_path, "all")
    runner._run_and_deliver_stamped_process_event = (
        GatewayRunner._run_and_deliver_stamped_process_event.__get__(runner)
    )
    adapter = runner.adapters[Platform.TELEGRAM]
    adapter.strip_media_directives_for_display = lambda text: text
    adapter.send.return_value = SimpleNamespace(success=True, message_id="sent-1")
    source = SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="123",
        chat_type="dm",
        user_id="123",
    )
    session_key = "agent:main:telegram:dm:123"
    runner.session_store._entries[session_key] = SimpleNamespace(
        session_id="sess-current"
    )
    generation = runner._begin_session_run_generation(session_key)
    runner._handle_message = AsyncMock(return_value="process result")
    event = MessageEvent(
        text="[SYSTEM: process completion]",
        message_type=MessageType.TEXT,
        source=source,
        internal=True,
        metadata={
            "expected_gateway_session_id": "sess-current",
            "gateway_run_generation": generation,
        },
    )

    result = await runner._run_and_deliver_stamped_process_event(
        event,
        adapter=adapter,
        session_key=session_key,
    )

    assert result.message_id == "sent-1"
    adapter.handle_message.assert_not_awaited()
    adapter.send.assert_awaited_once()


@pytest.mark.asyncio
async def test_stamped_process_event_suppresses_media_only_response(
    monkeypatch, tmp_path
):
    from gateway.platforms.base import MessageEvent, MessageType
    from gateway.session import SessionSource

    runner = _build_runner(monkeypatch, tmp_path, "all")
    runner._run_and_deliver_stamped_process_event = (
        GatewayRunner._run_and_deliver_stamped_process_event.__get__(runner)
    )
    adapter = runner.adapters[Platform.TELEGRAM]
    adapter.strip_media_directives_for_display = lambda _text: ""
    source = SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="123",
        chat_type="dm",
        user_id="123",
    )
    session_key = runner._session_key_for_source(source)
    runner.session_store._entries[session_key] = SimpleNamespace(
        session_id="sess-current"
    )
    runner._handle_message = AsyncMock(return_value="MEDIA:/tmp/untrusted.png")
    event = MessageEvent(
        text="[SYSTEM: process completion]",
        message_type=MessageType.TEXT,
        source=source,
        internal=True,
        metadata={"expected_gateway_session_id": "sess-current"},
    )

    result = await runner._run_and_deliver_stamped_process_event(
        event,
        adapter=adapter,
        session_key=session_key,
    )

    assert result is None
    adapter.send.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("busy_mode", ["interrupt", "steer"])
async def test_stamped_process_turn_does_not_drain_queued_foreground_message(
    monkeypatch, tmp_path, busy_mode
):
    import gateway.run as gateway_run
    import run_agent
    from gateway.platforms.base import MessageEvent, MessageType
    from gateway.session import SessionSource

    runner = _build_runner(monkeypatch, tmp_path, "all")
    monkeypatch.setattr(runner, "_is_user_authorized", lambda _source: True)
    runner._busy_input_mode = busy_mode
    runner._busy_text_mode = "interrupt"
    adapter = runner.adapters[Platform.TELEGRAM]
    session_key = "agent:main:telegram:dm:123"
    source = SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="123",
        chat_type="dm",
        user_id="123",
    )
    runner.session_store._entries[session_key] = SimpleNamespace(
        session_id="sess-old"
    )
    runner.session_store._loaded = True
    monkeypatch.setattr(runner.session_store, "_save", lambda: None)
    generation = runner._begin_session_run_generation(session_key)
    runner._agent_cache = {}
    runner._session_db = None

    queued = MessageEvent(
        text="real foreground user message",
        message_type=MessageType.TEXT,
        source=source,
    )
    adapter.get_pending_message = MagicMock(return_value=queued)
    adapter._pending_messages = {}
    adapter._active_sessions = {}
    runner._prepare_profile_scoped_inbound_message_text = AsyncMock(
        return_value="real foreground user message"
    )
    runner._refresh_agent_cache_message_count = AsyncMock()

    agent = MagicMock(name="agent")
    agent.model = "test-model"
    agent.session_id = "sess-old"
    agent.tools = []
    agent.context_compressor = SimpleNamespace(
        context_length=100_000,
        last_prompt_tokens=0,
    )
    agent.run_conversation.return_value = {
        "final_response": "MEDIA:/tmp/secret.png",
        "messages": [],
        "api_calls": 1,
        "tools": [],
    }
    monkeypatch.setattr(run_agent, "AIAgent", lambda *_a, **_kw: agent)
    monkeypatch.setattr(
        gateway_run,
        "_resolve_runtime_agent_kwargs",
        lambda: {"provider": "test", "api_key": "test-key"},
    )
    monkeypatch.setattr(
        runner,
        "_resolve_turn_agent_config",
        lambda _message, _model, runtime: {
            "model": "test-model",
            "runtime": runtime,
        },
    )
    monkeypatch.setattr(runner, "_get_proxy_url", lambda: None)
    backend_finished = asyncio.Event()
    release_backend_tail = asyncio.Event()
    original_executor = runner._run_in_executor_with_context

    async def _pause_after_backend(func, *args):
        backend_result = await original_executor(func, *args)
        backend_finished.set()
        await release_backend_tail.wait()
        return backend_result

    monkeypatch.setattr(runner, "_run_in_executor_with_context", _pause_after_backend)

    run_task = asyncio.create_task(
        runner._run_agent(
            message="[SYSTEM: process completion]",
            context_prompt="",
            history=[],
            source=source,
            session_id="sess-old",
            session_key=session_key,
            run_generation=generation,
            expected_process_session_id="sess-old",
        )
    )
    await backend_finished.wait()

    # Backend completion is not the full synthetic-turn boundary. The concrete
    # local agent remains marked until the outer gateway lifecycle releases it.
    assert runner._running_agents[session_key] is agent
    assert runner._stamped_process_running_agents[session_key] is agent
    agent.interrupt.reset_mock()
    agent.steer.reset_mock()
    tail_foreground = MessageEvent(
        text="foreground during local postprocessing tail",
        message_type=MessageType.TEXT,
        source=source,
    )

    handled = await runner._handle_active_session_busy_message(
        tail_foreground, session_key
    )

    assert handled is True
    agent.interrupt.assert_not_called()
    agent.steer.assert_not_called()
    assert adapter._pending_messages[session_key] is tail_foreground
    release_backend_tail.set()
    result = await run_task

    assert result["final_response"] == "MEDIA:/tmp/secret.png"
    adapter.send.assert_not_awaited()
    adapter.get_pending_message.assert_not_called()
    assert agent.run_conversation.call_count == 1
    assert runner._running_agents[session_key] is agent
    assert runner._stamped_process_running_agents[session_key] is agent
    runner._release_running_agent_state(
        session_key, run_generation=generation
    )
    assert session_key not in runner._running_agents
    assert session_key not in runner._stamped_process_running_agents


@pytest.mark.asyncio
async def test_foreground_message_queues_behind_stamped_process_agent(
    monkeypatch, tmp_path
):
    from gateway.platforms.base import MessageEvent, MessageType
    from gateway.session import SessionSource

    runner = _build_runner(monkeypatch, tmp_path, "all")
    monkeypatch.setattr(runner, "_is_user_authorized", lambda _source: True)
    runner._busy_input_mode = "interrupt"
    adapter = runner.adapters[Platform.TELEGRAM]
    adapter._pending_messages = {}
    source = SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="123",
        chat_type="dm",
        user_id="123",
    )
    key = runner._session_key_for_source(source)
    runner.session_store._entries[key] = SimpleNamespace(session_id="sess-old")
    process_agent = SimpleNamespace(
        interrupt=MagicMock(),
        get_activity_summary=lambda: {"seconds_since_activity": 0},
    )
    runner._running_agents[key] = process_agent
    runner._running_agents_ts[key] = time.time() - 10
    runner._stamped_process_running_agents[key] = process_agent
    foreground = MessageEvent(
        text="real foreground request",
        message_type=MessageType.TEXT,
        source=source,
    )

    result = await runner._handle_message(foreground)

    assert result is None
    process_agent.interrupt.assert_not_called()
    assert adapter._pending_messages[key] is foreground


@pytest.mark.asyncio
async def test_busy_handler_queues_behind_stamped_process_agent(
    monkeypatch, tmp_path
):
    from gateway.platforms.base import MessageEvent, MessageType
    from gateway.session import SessionSource

    runner = _build_runner(monkeypatch, tmp_path, "all")
    runner._busy_input_mode = "steer"
    monkeypatch.setattr(runner, "_is_user_authorized", lambda _source: True)
    adapter = runner.adapters[Platform.TELEGRAM]
    adapter._pending_messages = {}
    source = SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="123",
        chat_type="dm",
        user_id="123",
    )
    key = runner._session_key_for_source(source)
    process_agent = SimpleNamespace(interrupt=MagicMock(), steer=MagicMock())
    runner._running_agents[key] = process_agent
    runner._stamped_process_running_agents[key] = process_agent
    foreground = MessageEvent(
        text="real foreground request",
        message_type=MessageType.TEXT,
        source=source,
    )

    handled = await runner._handle_active_session_busy_message(foreground, key)

    assert handled is True
    process_agent.interrupt.assert_not_called()
    process_agent.steer.assert_not_called()
    assert adapter._pending_messages[key] is foreground


@pytest.mark.asyncio
async def test_stamped_process_delivery_retries_after_foreground_turn_finishes(
    monkeypatch, tmp_path
):
    from gateway.platforms.base import MessageEvent, MessageType
    from gateway.run import _STAMPED_PROCESS_EVENT_BUSY
    from gateway.session import SessionSource

    runner = _build_runner(monkeypatch, tmp_path, "all")
    runner._run_and_deliver_stamped_process_event = (
        GatewayRunner._run_and_deliver_stamped_process_event.__get__(runner)
    )
    adapter = runner.adapters[Platform.TELEGRAM]
    adapter.strip_media_directives_for_display = lambda text: text
    adapter.send.return_value = SimpleNamespace(success=True, message_id="sent-queued")
    source = SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="123",
        chat_type="dm",
        user_id="123",
    )
    session_key = runner._session_key_for_source(source)
    runner.session_store._entries[session_key] = SimpleNamespace(
        session_id="sess-current"
    )
    runner._handle_message = AsyncMock(
        side_effect=[_STAMPED_PROCESS_EVENT_BUSY, "process result"]
    )
    monkeypatch.setattr(asyncio, "sleep", AsyncMock())
    event = MessageEvent(
        text="[SYSTEM: process completion]",
        message_type=MessageType.TEXT,
        source=source,
        internal=True,
        metadata={"expected_gateway_session_id": "sess-current"},
    )

    result = await runner._run_and_deliver_stamped_process_event(
        event,
        adapter=adapter,
        session_key=session_key,
    )

    assert result.message_id == "sent-queued"
    assert runner._handle_message.await_count == 2
    adapter.send.assert_awaited_once()


@pytest.mark.asyncio
async def test_agent_notification_carries_message_id_reply_anchor(monkeypatch, tmp_path):
    """notify_on_complete injection carries the triggering message_id so the
    synthetic event can be reply-anchored back into a Telegram DM topic.

    Without an anchor, Telegram private-chat topic sends fall back to the main
    chat (see _thread_kwargs_for_send / telegram_dm_topic_reply_fallback)."""
    import tools.process_registry as pr_module

    sessions = [SimpleNamespace(
        output_buffer="SMOKE_OK\n", exited=True, exit_code=0, command="sleep 1",
    )]
    monkeypatch.setattr(pr_module, "process_registry", _FakeRegistry(sessions))

    async def _instant_sleep(*_a, **_kw):
        pass
    monkeypatch.setattr(asyncio, "sleep", _instant_sleep)

    runner = _build_runner(monkeypatch, tmp_path, "all")
    adapter = runner.adapters[Platform.TELEGRAM]
    runner.session_store._entries["agent:main:telegram:dm:123:24296"] = SimpleNamespace(
        session_id="sess-current"
    )

    watcher = {
        "session_id": "proc_anchor",
        "check_interval": 0,
        "session_key": "agent:main:telegram:dm:123:24296",
        "platform": "telegram",
        "chat_id": "123",
        "thread_id": "24296",
        "message_id": "555",
        "notify_on_complete": True,
        "conversation_session_id": "sess-current",
    }
    await runner._run_process_watcher(watcher)

    adapter.handle_message.assert_awaited_once()
    synth_event = adapter.handle_message.await_args.args[0]
    assert synth_event.internal is True
    assert synth_event.message_id == "555"
    assert synth_event.source.thread_id == "24296"
    assert synth_event.metadata["expected_gateway_session_id"] == "sess-current"

@pytest.mark.asyncio
async def test_agent_notification_no_message_id_is_tolerated(monkeypatch, tmp_path):
    """A watcher dict without message_id (CLI spawn, pre-upgrade checkpoint)
    still injects — message_id is simply None."""
    import tools.process_registry as pr_module

    sessions = [SimpleNamespace(
        output_buffer="done\n", exited=True, exit_code=0, command="sleep 1",
    )]
    monkeypatch.setattr(pr_module, "process_registry", _FakeRegistry(sessions))

    async def _instant_sleep(*_a, **_kw):
        pass
    monkeypatch.setattr(asyncio, "sleep", _instant_sleep)

    runner = _build_runner(monkeypatch, tmp_path, "all")
    adapter = runner.adapters[Platform.TELEGRAM]

    watcher = {
        "session_id": "proc_anchorless",
        "check_interval": 0,
        "session_key": "agent:main:telegram:dm:123:24296",
        "platform": "telegram",
        "chat_id": "123",
        "thread_id": "24296",
        "notify_on_complete": True,
    }
    await runner._run_process_watcher(watcher)

    adapter.handle_message.assert_awaited_once()
    synth_event = adapter.handle_message.await_args.args[0]
    assert synth_event.message_id is None

@pytest.mark.asyncio
async def test_inject_watch_notification_carries_message_id_reply_anchor(monkeypatch, tmp_path):
    from gateway.session import SessionSource

    runner = _build_runner(monkeypatch, tmp_path, "all")
    adapter = runner.adapters[Platform.TELEGRAM]
    runner.session_store._entries["agent:main:telegram:dm:123:24296"] = SimpleNamespace(
        origin=SessionSource(
            platform=Platform.TELEGRAM,
            chat_id="123",
            chat_type="dm",
            thread_id="24296",
            user_id="1",
            user_name="Fabio",
        )
    )

    evt = {
        "session_id": "proc_watch",
        "session_key": "agent:main:telegram:dm:123:24296",
        "message_id": "777",
    }

    await runner._inject_watch_notification("[SYSTEM: Background process matched]", evt)

    adapter.handle_message.assert_awaited_once()
    synth_event = adapter.handle_message.await_args.args[0]
    assert synth_event.message_id == "777"
    assert synth_event.source.thread_id == "24296"


def test_build_process_event_source_falls_back_to_session_key_chat_type(monkeypatch, tmp_path):
    runner = _build_runner(monkeypatch, tmp_path, "all")

    evt = {
        "session_id": "proc_watch",
        "session_key": "agent:main:telegram:group:-100:42",
        "platform": "telegram",
        "chat_id": "-100",
        "thread_id": "42",
        "user_id": "123",
        "user_name": "Emiliyan",
    }

    source = runner._build_process_event_source(evt)

    assert source is not None
    assert source.platform == Platform.TELEGRAM
    assert source.chat_id == "-100"
    assert source.chat_type == "group"
    assert source.thread_id == "42"
    assert source.user_id == "123"
    assert source.user_name == "Emiliyan"


def test_build_process_event_source_uses_cached_live_source_before_session_key_parse(
    monkeypatch, tmp_path
):
    from gateway.session import SessionSource

    runner = _build_runner(monkeypatch, tmp_path, "all")
    runner._cache_session_source(
        "agent:main:telegram:group:-100:42",
        SessionSource(
            platform=Platform.TELEGRAM,
            chat_id="-100",
            chat_type="group",
            thread_id="42",
            user_id="proc_owner",
            user_name="alice",
        ),
    )

    source = runner._build_process_event_source(
        {
            "session_id": "proc_watch",
            "session_key": "agent:main:telegram:group:-100:42",
        }
    )

    assert source is not None
    assert source.platform == Platform.TELEGRAM
    assert source.chat_id == "-100"
    assert source.chat_type == "group"
    assert source.thread_id == "42"
    assert source.user_id == "proc_owner"
    assert source.user_name == "alice"

@pytest.mark.asyncio
async def test_inject_watch_notification_ignores_foreground_event_source(monkeypatch, tmp_path):
    """Negative test: watch notification must NOT route to the foreground thread."""
    from gateway.session import SessionSource

    runner = _build_runner(monkeypatch, tmp_path, "all")
    adapter = runner.adapters[Platform.TELEGRAM]

    # Session store has the process's original thread (thread 42)
    runner.session_store._entries["agent:main:telegram:group:-100:42"] = SimpleNamespace(
        origin=SessionSource(
            platform=Platform.TELEGRAM,
            chat_id="-100",
            chat_type="group",
            thread_id="42",
            user_id="proc_owner",
            user_name="alice",
        )
    )

    # The evt dict carries the correct session_key — NOT a foreground event
    evt = {
        "session_id": "proc_cross_thread",
        "session_key": "agent:main:telegram:group:-100:42",
    }

    await runner._inject_watch_notification("[SYSTEM: watch match]", evt)

    adapter.handle_message.assert_awaited_once()
    synth_event = adapter.handle_message.await_args.args[0]
    # Must route to thread 42 (process origin), NOT some other thread
    assert synth_event.source.thread_id == "42"
    assert synth_event.source.user_id == "proc_owner"

@pytest.mark.asyncio
async def test_inject_watch_notification_drops_stale_boundary_after_reset(monkeypatch, tmp_path):
    from gateway.session import SessionSource

    runner = _build_runner(monkeypatch, tmp_path, "all")
    adapter = runner.adapters[Platform.TELEGRAM]

    runner.session_store._entries["agent:main:telegram:group:-100:42"] = SimpleNamespace(
        session_id="sess-new",
        origin=SessionSource(
            platform=Platform.TELEGRAM,
            chat_id="-100",
            chat_type="group",
            thread_id="42",
            user_id="proc_owner",
            user_name="alice",
        ),
    )

    evt = {
        "session_id": "proc_stale_watch",
        "session_key": "agent:main:telegram:group:-100:42",
        "conversation_session_id": "sess-old",
    }

    await runner._inject_watch_notification("[SYSTEM: stale watch match]", evt)

    adapter.handle_message.assert_not_awaited()


@pytest.mark.asyncio
async def test_inject_watch_notification_drops_stamped_event_when_session_entry_missing(
    monkeypatch, tmp_path
):
    runner = _build_runner(monkeypatch, tmp_path, "all")
    adapter = runner.adapters[Platform.TELEGRAM]

    evt = {
        "session_id": "proc_orphaned_watch",
        "session_key": "agent:main:telegram:group:-100:42",
        "conversation_session_id": "sess-old",
    }

    await runner._inject_watch_notification("[SYSTEM: orphaned watch match]", evt)

    adapter.handle_message.assert_not_awaited()


@pytest.mark.asyncio
async def test_inject_watch_notification_drops_stamped_event_when_store_lookup_fails(
    monkeypatch, tmp_path
):
    runner = _build_runner(monkeypatch, tmp_path, "all")
    adapter = runner.adapters[Platform.TELEGRAM]

    def _fail_lookup():
        raise OSError("session store unavailable")

    monkeypatch.setattr(runner.session_store, "_ensure_loaded", _fail_lookup)
    evt = {
        "session_id": "proc_unverifiable_watch",
        "session_key": "agent:main:telegram:group:-100:42",
        "conversation_session_id": "sess-old",
    }

    await runner._inject_watch_notification("[SYSTEM: unverifiable watch match]", evt)

    adapter.handle_message.assert_not_awaited()


@pytest.mark.asyncio
async def test_inject_watch_notification_keeps_compression_continuation(monkeypatch, tmp_path):
    from gateway.session import SessionSource

    runner = _build_runner(monkeypatch, tmp_path, "all")
    adapter = runner.adapters[Platform.TELEGRAM]
    runner.session_store._entries["agent:main:telegram:group:-100:42"] = SimpleNamespace(
        session_id="sess-new",
        origin=SessionSource(
            platform=Platform.TELEGRAM,
            chat_id="-100",
            chat_type="group",
            thread_id="42",
        ),
    )
    runner._session_db = SimpleNamespace(
        _db=SimpleNamespace(
            get_compression_tip=lambda sid: {
                "sess-old": "sess-new",
                "sess-new": "sess-new",
            }.get(sid, sid)
        )
    )

    evt = {
        "session_id": "proc_compression",
        "session_key": "agent:main:telegram:group:-100:42",
        "conversation_session_id": "sess-old",
    }

    await runner._inject_watch_notification("[SYSTEM: compression continuation watch]", evt)

    adapter.handle_message.assert_awaited_once()
    synth_event = adapter.handle_message.await_args.args[0]
    assert synth_event.metadata["expected_gateway_session_id"] == "sess-old"


@pytest.mark.asyncio
async def test_agent_handoff_drops_when_reset_races_initial_process_check(monkeypatch, tmp_path):
    from gateway.session import SessionSource

    runner = _build_runner(monkeypatch, tmp_path, "all")
    adapter = runner.adapters[Platform.TELEGRAM]
    session_key = "agent:main:telegram:group:-100:42"
    runner.session_store._entries[session_key] = SimpleNamespace(
        session_id="sess-old",
        origin=SessionSource(
            platform=Platform.TELEGRAM,
            chat_id="-100",
            chat_type="group",
            thread_id="42",
        ),
    )
    evt = {
        "session_id": "proc_handoff_race",
        "session_key": session_key,
        "conversation_session_id": "sess-old",
    }

    # Initial queue-drain validation succeeds and creates the synthetic event.
    await runner._inject_watch_notification("[SYSTEM: racing watch match]", evt)
    synth_event = adapter.handle_message.await_args.args[0]

    # /new wins before the adapter hands the event to the gateway agent path.
    runner._async_session_store = SimpleNamespace(
        _store=runner.session_store,
        get_or_create_session=AsyncMock(
            return_value=SimpleNamespace(session_key=session_key, session_id="sess-new")
        ),
    )
    monkeypatch.setattr(runner, "_recover_telegram_topic_thread_id", lambda _source: None)
    runner._cache_session_source = MagicMock()

    await runner._handle_message_with_agent(
        synth_event, synth_event.source, session_key, run_generation=0
    )

    runner._cache_session_source.assert_not_called()


@pytest.mark.asyncio
async def test_agent_handoff_rechecks_after_telegram_topic_binding_switch(
    monkeypatch, tmp_path
):
    from gateway.session import SessionSource

    runner = _build_runner(monkeypatch, tmp_path, "all")
    adapter = runner.adapters[Platform.TELEGRAM]
    session_key = "agent:main:telegram:dm:123:42"
    source = SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="123",
        chat_type="dm",
        thread_id="42",
    )
    runner.session_store._entries[session_key] = SimpleNamespace(
        session_id="sess-old",
        origin=source,
    )
    evt = {
        "session_id": "proc_topic_rebind_race",
        "session_key": session_key,
        "conversation_session_id": "sess-old",
    }

    await runner._inject_watch_notification("[SYSTEM: topic race]", evt)
    synth_event = adapter.handle_message.await_args.args[0]

    old_entry = SimpleNamespace(session_key=session_key, session_id="sess-old")
    new_entry = SimpleNamespace(session_key=session_key, session_id="sess-new")
    runner._async_session_store = SimpleNamespace(
        _store=runner.session_store,
        get_or_create_session=AsyncMock(return_value=old_entry),
        switch_session=AsyncMock(return_value=new_entry),
    )
    runner._session_db = SimpleNamespace(
        _db=SimpleNamespace(get_compression_tip=lambda sid: sid),
        get_telegram_topic_binding=AsyncMock(return_value={"session_id": "sess-new"}),
        get_compression_tip=AsyncMock(side_effect=lambda sid: sid),
    )
    monkeypatch.setattr(runner, "_recover_telegram_topic_thread_id", lambda _source: None)
    monkeypatch.setattr(runner, "_is_telegram_topic_lane", lambda _source: True)
    runner._cache_session_source = MagicMock()
    runner.hooks.emit = AsyncMock()

    await runner._handle_message_with_agent(
        synth_event, synth_event.source, session_key, run_generation=0
    )

    runner._async_session_store.switch_session.assert_awaited_once_with(
        session_key, "sess-new"
    )
    runner.hooks.emit.assert_not_awaited()


@pytest.mark.asyncio
async def test_process_run_aborts_when_reset_wins_during_pre_agent_setup(
    monkeypatch, tmp_path
):
    from datetime import datetime, timedelta, timezone

    from gateway.session import SessionEntry, SessionSource

    runner = _build_runner(monkeypatch, tmp_path, "all")
    session_key = "agent:main:telegram:dm:123"
    source = SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="123",
        chat_type="dm",
    )
    now = datetime.now(timezone.utc)
    old_entry = SessionEntry(
        session_key=session_key,
        session_id="sess-old",
        created_at=now - timedelta(seconds=1),
        updated_at=now,
        origin=source,
    )
    runner.session_store._entries[session_key] = old_entry
    generation = runner._begin_session_run_generation(session_key)

    history_started = asyncio.Event()
    release_history = asyncio.Event()

    async def _blocked_load_transcript(_session_id):
        history_started.set()
        await release_history.wait()
        return []

    runner._async_session_store = SimpleNamespace(
        _store=runner.session_store,
        get_or_create_session=AsyncMock(return_value=old_entry),
        load_transcript=_blocked_load_transcript,
    )
    monkeypatch.setattr(runner, "_recover_telegram_topic_thread_id", lambda _source: None)
    monkeypatch.setattr(runner, "_is_telegram_topic_lane", lambda _source: False)
    runner._run_agent = AsyncMock()
    event = SimpleNamespace(
        text="[SYSTEM: stale process completion]",
        source=source,
        metadata={"expected_gateway_session_id": "sess-old"},
        auto_skill=None,
    )

    task = asyncio.create_task(
        runner._handle_message_with_agent(
            event, source, session_key, run_generation=generation
        )
    )
    await history_started.wait()

    # /new wins while the handler is suspended in awaited pre-agent setup.
    runner.session_store._entries[session_key] = SessionEntry(
        session_key=session_key,
        session_id="sess-new",
        created_at=now,
        updated_at=now,
        origin=source,
    )
    runner._session_run_generation[session_key] = generation + 1
    release_history.set()
    await task

    runner._run_agent.assert_not_awaited()


@pytest.mark.asyncio
async def test_process_turn_uses_narrow_side_effect_free_handler_path(
    monkeypatch, tmp_path
):
    from datetime import datetime, timedelta, timezone

    from gateway.session import SessionEntry, SessionSource

    runner = _build_runner(monkeypatch, tmp_path, "all")
    session_key = "agent:main:telegram:dm:123"
    source = SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="123",
        chat_type="dm",
    )
    now = datetime.now(timezone.utc)
    entry = SessionEntry(
        session_key=session_key,
        session_id="sess-old",
        created_at=now - timedelta(seconds=1),
        updated_at=now,
        origin=source,
    )
    runner.session_store._entries[session_key] = entry
    generation = runner._begin_session_run_generation(session_key)

    update_session = AsyncMock()
    runner._async_session_store = SimpleNamespace(
        _store=runner.session_store,
        get_or_create_session=AsyncMock(return_value=entry),
        load_transcript=AsyncMock(return_value=[]),
        has_any_sessions=AsyncMock(return_value=True),
        update_session=update_session,
    )
    monkeypatch.setattr(runner, "_recover_telegram_topic_thread_id", lambda _source: None)
    monkeypatch.setattr(runner, "_is_telegram_topic_lane", lambda _source: False)
    runner._prepare_profile_scoped_inbound_message_text = AsyncMock(
        side_effect=AssertionError("process turns must skip enrichment")
    )
    runner.hooks.emit = AsyncMock(
        side_effect=AssertionError("process turns must skip lifecycle hooks")
    )
    runner._run_agent = AsyncMock(
        return_value={
            "final_response": "process result",
            "messages": [],
            "api_calls": 1,
            "tools": [],
            "history_offset": 0,
            "session_id": "sess-old",
        }
    )
    event = SimpleNamespace(
        text="[SYSTEM: process completion]",
        source=source,
        metadata={"expected_gateway_session_id": "sess-old"},
        auto_skill=None,
        channel_prompt=None,
        message_id=None,
        timestamp=None,
        reply_to_message_id=None,
        reply_to_text=None,
    )

    response = await runner._handle_message_with_agent(
        event,
        source,
        session_key,
        run_generation=generation,
    )

    assert response == "process result"
    runner._prepare_profile_scoped_inbound_message_text.assert_not_awaited()
    runner.hooks.emit.assert_not_awaited()
    update_session.assert_not_awaited()


@pytest.mark.asyncio
async def test_reset_between_cache_check_and_lock_preserves_replacement_agent(
    monkeypatch, tmp_path
):
    import gateway.run as gateway_run
    import run_agent
    from gateway.session import SessionSource

    runner = _build_runner(monkeypatch, tmp_path, "all")
    runner.adapters[Platform.TELEGRAM].get_pending_message = MagicMock(
        return_value=None
    )
    session_key = "agent:main:telegram:dm:123"
    source = SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="123",
        chat_type="dm",
    )
    runner.session_store._entries[session_key] = SimpleNamespace(
        session_id="sess-old"
    )
    generation = runner._begin_session_run_generation(session_key)
    runner._agent_cache = {}

    publication_waiting = threading.Event()
    release_publication = threading.Event()

    class _PublicationBarrierLock:
        def __init__(self):
            self._lock = threading.RLock()
            self._enters = 0

        def __enter__(self):
            self._enters += 1
            # First entry is cache lookup; second is publication. Pause before
            # acquiring the publication lock so reset/replacement wins exactly
            # in the former check-to-lock TOCTOU window.
            if self._enters == 2:
                publication_waiting.set()
                assert release_publication.wait(timeout=5)
            self._lock.acquire()
            return self

        def __exit__(self, *_exc):
            self._lock.release()

    runner._agent_cache_lock = _PublicationBarrierLock()
    runner._session_db = None

    stale_agent = MagicMock(name="stale-agent")
    stale_agent.model = "test-model"
    stale_agent.tools = []
    stale_agent.context_compressor = SimpleNamespace(
        context_length=100_000,
        last_prompt_tokens=0,
    )
    stale_agent.run_conversation = MagicMock(
        return_value={"final_response": "must not run", "messages": []}
    )

    def _construct_agent(*_args, **_kwargs):
        return stale_agent

    monkeypatch.setattr(run_agent, "AIAgent", _construct_agent)
    monkeypatch.setattr(
        gateway_run,
        "_resolve_runtime_agent_kwargs",
        lambda: {"provider": "test", "api_key": "test-key"},
    )
    monkeypatch.setattr(
        runner,
        "_resolve_turn_agent_config",
        lambda _message, _model, runtime: {
            "model": "test-model",
            "runtime": runtime,
        },
    )
    monkeypatch.setattr(runner, "_get_proxy_url", lambda: None)

    task = asyncio.create_task(
        runner._run_agent(
            message="[SYSTEM: stale process completion]",
            context_prompt="",
            history=[],
            source=source,
            session_id="sess-old",
            session_key=session_key,
            run_generation=generation,
            expected_process_session_id="sess-old",
        )
    )
    assert await asyncio.to_thread(publication_waiting.wait, 5)

    # /new wins after lookup but before stale publication acquires the lock.
    runner._invalidate_session_run_generation(session_key, reason="test-reset")
    runner.session_store._entries[session_key] = SimpleNamespace(
        session_id="sess-new"
    )
    replacement_agent = MagicMock(name="replacement-agent")
    runner._agent_cache[session_key] = (
        replacement_agent,
        "replacement-sig",
        0,
        "sess-new",
    )
    release_publication.set()

    result = await task
    assert result["final_response"] != "must not run"
    stale_agent.run_conversation.assert_not_called()
    assert runner._agent_cache[session_key][0] is replacement_agent
    for callback_name in (
        "tool_progress_callback",
        "tool_start_callback",
        "step_callback",
        "stream_delta_callback",
        "interim_assistant_callback",
        "status_callback",
        "notice_callback",
        "event_callback",
        "background_review_callback",
        "clarify_callback",
    ):
        assert getattr(stale_agent, callback_name) is None


@pytest.mark.asyncio
async def test_stamped_process_run_auto_denies_approval_without_platform_prompt(
    monkeypatch, tmp_path
):
    import gateway.run as gateway_run
    import run_agent
    import tools.approval as approval
    from gateway.session import SessionSource

    runner = _build_runner(monkeypatch, tmp_path, "all")
    adapter = runner.adapters[Platform.TELEGRAM]
    adapter.get_pending_message = MagicMock(return_value=None)
    session_key = "agent:main:telegram:dm:123"
    source = SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="123",
        chat_type="dm",
    )
    runner.session_store._entries[session_key] = SimpleNamespace(
        session_id="sess-old"
    )
    monkeypatch.setattr(runner.session_store, "_save", lambda: None)
    generation = runner._begin_session_run_generation(session_key)
    runner._agent_cache = {}
    runner._session_db = None

    process_agent = MagicMock(name="process-agent")
    process_agent.model = "test-model"
    process_agent.tools = []
    process_agent.context_compressor = SimpleNamespace(
        context_length=100_000,
        last_prompt_tokens=0,
    )
    process_agent.run_conversation.return_value = {
        "final_response": "process result",
        "messages": [],
        "api_calls": 1,
        "tools": [],
    }
    monkeypatch.setattr(run_agent, "AIAgent", lambda *_a, **_kw: process_agent)
    monkeypatch.setattr(
        gateway_run,
        "_resolve_runtime_agent_kwargs",
        lambda: {"provider": "test", "api_key": "test-key"},
    )
    monkeypatch.setattr(
        runner,
        "_resolve_turn_agent_config",
        lambda _message, _model, runtime: {
            "model": "test-model",
            "runtime": runtime,
        },
    )
    monkeypatch.setattr(runner, "_get_proxy_url", lambda: None)

    registered = []
    resolved = MagicMock()
    monkeypatch.setattr(
        approval,
        "register_gateway_notify",
        lambda key, callback: registered.append((key, callback)),
    )
    monkeypatch.setattr(approval, "unregister_gateway_notify", lambda _key: None)
    monkeypatch.setattr(approval, "resolve_gateway_approval", resolved)

    result = await runner._run_agent(
        message="[SYSTEM: process completion]",
        context_prompt="",
        history=[],
        source=source,
        session_id="sess-old",
        session_key=session_key,
        run_generation=generation,
        expected_process_session_id="sess-old",
    )

    assert result["final_response"] == "process result"
    assert len(registered) == 1
    key, callback = registered[0]
    assert key == f"{session_key}:process-run:{generation}:sess-old"
    callback({"command": "dangerous"})
    resolved.assert_called_once_with(
        key,
        "deny",
        reason="Background process notifications cannot request approval.",
    )
    adapter.send.assert_not_awaited()


def test_process_approval_cleanup_cannot_remove_replacement_foreground_state():
    import tools.approval as approval

    session_key = "agent:main:telegram:dm:123"
    process_key = f"{session_key}:process-run:7:sess-old"
    process_callback = MagicMock()
    replacement_callback = MagicMock()
    replacement_entry = approval._ApprovalEntry({"command": "dangerous"})

    try:
        approval.register_gateway_notify(process_key, process_callback)
        approval.register_gateway_notify(session_key, replacement_callback)
        with approval._lock:
            approval._gateway_queues[session_key] = [replacement_entry]

        approval.unregister_gateway_notify(process_key)

        with approval._lock:
            assert approval._gateway_notify_cbs[session_key] is replacement_callback
            assert approval._gateway_queues[session_key] == [replacement_entry]
        assert not replacement_entry.event.is_set()
    finally:
        approval.unregister_gateway_notify(process_key)
        approval.unregister_gateway_notify(session_key)


@pytest.mark.asyncio
async def test_process_delivery_and_reset_are_linearized_when_delivery_wins(
    monkeypatch, tmp_path
):
    runner = _build_runner(monkeypatch, tmp_path, "all")
    session_key = "agent:main:telegram:dm:delivery-first"
    runner.session_store._entries[session_key] = SimpleNamespace(
        session_id="sess-old"
    )
    send_started = asyncio.Event()
    release_send = asyncio.Event()
    reset_called = asyncio.Event()

    async def _send():
        send_started.set()
        await release_send.wait()
        return "sent"

    async def _reset(_session_key):
        reset_called.set()
        runner.session_store._entries[session_key] = SimpleNamespace(
            session_id="sess-new"
        )
        return runner.session_store._entries[session_key]

    runner._async_session_store = SimpleNamespace(
        _store=runner.session_store,
        reset_session=_reset,
    )
    delivery_task = asyncio.create_task(
        runner._deliver_process_notification_guarded(
            expected_session_id="sess-old",
            session_key=session_key,
            run_generation=None,
            delivery=_send,
        )
    )
    await send_started.wait()
    reset_task = asyncio.create_task(
        runner._reset_session_at_process_delivery_boundary(session_key)
    )
    await asyncio.sleep(0)
    assert not reset_called.is_set()

    release_send.set()
    assert await delivery_task == "sent"
    await reset_task
    assert reset_called.is_set()
    assert runner.session_store._entries[session_key].session_id == "sess-new"


@pytest.mark.asyncio
async def test_yuanbao_lookup_waits_for_guarded_process_delivery(
    monkeypatch, tmp_path
):
    from gateway.platforms.yuanbao import QuoteContextMiddleware
    from gateway.session import SessionSource

    runner = _build_runner(monkeypatch, tmp_path, "all")
    session_key = "agent:main:yuanbao:dm:direct:123"
    source = SessionSource(
        platform=Platform.YUANBAO,
        chat_id="direct:123",
        chat_type="dm",
        user_id="123",
    )
    runner.session_store._entries[session_key] = SimpleNamespace(
        session_id="sess-old"
    )
    send_started = asyncio.Event()
    release_send = asyncio.Event()
    lookup_called = asyncio.Event()

    async def _send():
        send_started.set()
        await release_send.wait()
        return "sent"

    async def _get_or_create(_source):
        lookup_called.set()
        runner.session_store._entries[session_key] = SimpleNamespace(
            session_id="sess-new"
        )
        return runner.session_store._entries[session_key]

    runner._async_session_store = SimpleNamespace(
        _store=runner.session_store,
        get_or_create_session=_get_or_create,
    )
    adapter = SimpleNamespace(
        _session_store=SimpleNamespace(load_transcript=lambda _sid: []),
        resolve_session_entry=lambda resolved_source: (
            runner._get_or_create_session_at_process_delivery_boundary(
                resolved_source, session_key
            )
        ),
        name="yuanbao",
    )
    ctx = SimpleNamespace(
        reply_to_message_id="quoted",
        adapter=adapter,
        source=source,
    )

    delivery_task = asyncio.create_task(
        runner._deliver_process_notification_guarded(
            expected_session_id="sess-old",
            session_key=session_key,
            run_generation=None,
            delivery=_send,
        )
    )
    await send_started.wait()
    lookup_task = asyncio.create_task(
        QuoteContextMiddleware()._extract_media_refs_from_transcript(ctx)
    )
    await asyncio.sleep(0)
    assert not lookup_called.is_set()

    release_send.set()
    assert await delivery_task == "sent"
    assert await lookup_task == []
    assert lookup_called.is_set()
    assert runner.session_store._entries[session_key].session_id == "sess-new"


@pytest.mark.asyncio
async def test_cron_seed_waits_for_guarded_process_delivery(monkeypatch, tmp_path):
    from cron.scheduler import _seed_cron_channel_session

    runner = _build_runner(monkeypatch, tmp_path, "all")
    session_key = "agent:coder:slack:group:C123:U_HUMAN"
    runner.session_store._entries[session_key] = SimpleNamespace(
        session_id="sess-old"
    )
    append_to_transcript = MagicMock()
    runner.session_store.append_to_transcript = append_to_transcript
    lookup_called = asyncio.Event()
    send_started = asyncio.Event()
    release_send = asyncio.Event()

    async def _get_or_create(source):
        lookup_called.set()
        assert source.profile == "coder"
        return SimpleNamespace(session_id="sess-old")

    runner._async_session_store = SimpleNamespace(
        _store=runner.session_store,
        get_or_create_session=_get_or_create,
    )
    profile_resolver = runner._make_profile_session_resolver("coder")

    async def _resolve(source, *, operation=None):
        return await profile_resolver(source, operation=operation)

    adapter = SimpleNamespace(
        _session_store=runner.session_store,
        resolve_session_entry=_resolve,
    )

    async def _send():
        send_started.set()
        await release_send.wait()
        return "sent"

    delivery_task = asyncio.create_task(
        runner._deliver_process_notification_guarded(
            expected_session_id="sess-old",
            session_key=session_key,
            run_generation=None,
            delivery=_send,
        )
    )
    await send_started.wait()
    loop = asyncio.get_running_loop()
    seed_task = asyncio.create_task(
        asyncio.to_thread(
            _seed_cron_channel_session,
            {"id": "cron-1"},
            adapter,
            "slack",
            "C123",
            "Daily brief",
            is_dm=False,
            user_id="U_HUMAN",
            loop=loop,
        )
    )
    await asyncio.sleep(0)
    assert not lookup_called.is_set()

    release_send.set()
    assert await delivery_task == "sent"
    assert await seed_task is True
    assert lookup_called.is_set()
    append_to_transcript.assert_called_once()
    assert append_to_transcript.call_args.args[0] == "sess-old"


@pytest.mark.asyncio
async def test_model_switch_warning_lookup_waits_for_guarded_process_delivery(
    monkeypatch, tmp_path
):
    from hermes_cli.context_switch_guard import (
        enrich_model_switch_warnings_for_gateway,
    )
    from gateway.session import SessionSource

    runner = _build_runner(monkeypatch, tmp_path, "all")
    source = SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="123",
        chat_type="dm",
        user_id="123",
    )
    session_key = runner._session_key_for_source(source)
    runner.session_store._entries[session_key] = SimpleNamespace(
        session_id="sess-old"
    )
    runner._agent_cache[session_key] = (
        SimpleNamespace(context_compressor=None),
        "signature",
        0,
    )
    lookup_called = asyncio.Event()
    send_started = asyncio.Event()
    release_send = asyncio.Event()
    read_messages = MagicMock(return_value=[])
    runner._session_db = SimpleNamespace(
        get_messages_as_conversation=read_messages
    )

    async def _get_or_create(resolved_source):
        lookup_called.set()
        assert resolved_source is source
        return SimpleNamespace(session_id="sess-old")

    runner._async_session_store = SimpleNamespace(
        _store=runner.session_store,
        get_or_create_session=_get_or_create,
    )

    async def _send():
        send_started.set()
        await release_send.wait()
        return "sent"

    delivery_task = asyncio.create_task(
        runner._deliver_process_notification_guarded(
            expected_session_id="sess-old",
            session_key=session_key,
            run_generation=None,
            delivery=_send,
        )
    )
    await send_started.wait()
    warning_task = asyncio.create_task(
        enrich_model_switch_warnings_for_gateway(
            SimpleNamespace(success=True),
            runner,
            session_key=session_key,
            source=source,
        )
    )
    await asyncio.sleep(0)
    assert not lookup_called.is_set()

    release_send.set()
    assert await delivery_task == "sent"
    await warning_task
    assert lookup_called.is_set()
    read_messages.assert_called_once_with("sess-old")


@pytest.mark.asyncio
async def test_process_delivery_is_suppressed_when_reset_wins_boundary(
    monkeypatch, tmp_path
):
    runner = _build_runner(monkeypatch, tmp_path, "all")
    session_key = "agent:main:telegram:dm:reset-first"
    runner.session_store._entries[session_key] = SimpleNamespace(
        session_id="sess-old"
    )
    reset_committed = asyncio.Event()
    release_reset = asyncio.Event()
    delivered = False

    async def _reset(_session_key):
        runner.session_store._entries[session_key] = SimpleNamespace(
            session_id="sess-new"
        )
        reset_committed.set()
        await release_reset.wait()
        return runner.session_store._entries[session_key]

    async def _send():
        nonlocal delivered
        delivered = True
        return "sent"

    runner._async_session_store = SimpleNamespace(
        _store=runner.session_store,
        reset_session=_reset,
    )
    reset_task = asyncio.create_task(
        runner._reset_session_at_process_delivery_boundary(session_key)
    )
    await reset_committed.wait()
    delivery_task = asyncio.create_task(
        runner._deliver_process_notification_guarded(
            expected_session_id="sess-old",
            session_key=session_key,
            run_generation=None,
            delivery=_send,
        )
    )
    await asyncio.sleep(0)
    assert not delivery_task.done()

    release_reset.set()
    await reset_task
    assert await delivery_task is None
    assert delivered is False


@pytest.mark.asyncio
async def test_process_delivery_is_suppressed_when_session_lookup_auto_reset_wins(
    monkeypatch, tmp_path
):
    runner = _build_runner(monkeypatch, tmp_path, "all")
    session_key = "agent:main:telegram:dm:auto-reset-first"
    runner.session_store._entries[session_key] = SimpleNamespace(
        session_id="sess-old"
    )
    reset_committed = asyncio.Event()
    release_lookup = asyncio.Event()
    delivered = False

    async def _get_or_create(_source):
        runner.session_store._entries[session_key] = SimpleNamespace(
            session_id="sess-new"
        )
        reset_committed.set()
        await release_lookup.wait()
        return runner.session_store._entries[session_key]

    async def _send():
        nonlocal delivered
        delivered = True
        return "sent"

    runner._async_session_store = SimpleNamespace(
        _store=runner.session_store,
        get_or_create_session=_get_or_create,
    )
    lookup_task = asyncio.create_task(
        runner._get_or_create_session_at_process_delivery_boundary(
            SimpleNamespace(), session_key
        )
    )
    await reset_committed.wait()
    delivery_task = asyncio.create_task(
        runner._deliver_process_notification_guarded(
            expected_session_id="sess-old",
            session_key=session_key,
            run_generation=None,
            delivery=_send,
        )
    )
    await asyncio.sleep(0)
    assert not delivery_task.done()

    release_lookup.set()
    await lookup_task
    assert await delivery_task is None
    assert delivered is False


@pytest.mark.asyncio
@pytest.mark.parametrize("exited", [False, True])
async def test_text_delivery_rechecks_after_initial_session_validation(
    monkeypatch, tmp_path, exited
):
    import tools.process_registry as pr_module

    session = SimpleNamespace(
        output_buffer="done\n" if exited else "building\n",
        exited=exited,
        exit_code=0 if exited else None,
        command="echo test",
    )
    sessions = [session] if exited else [session, None]
    monkeypatch.setattr(pr_module, "process_registry", _FakeRegistry(sessions))

    async def _instant_sleep(*_a, **_kw):
        pass

    monkeypatch.setattr(asyncio, "sleep", _instant_sleep)
    runner = _build_runner(monkeypatch, tmp_path, "all")
    adapter = runner.adapters[Platform.TELEGRAM]
    session_key = "agent:main:telegram:group:-100:42"
    runner.session_store._entries[session_key] = SimpleNamespace(session_id="sess-old")
    watcher = _watcher_dict(session_id="proc_text_race")
    watcher.update(
        {
            "session_key": session_key,
            "conversation_session_id": "sess-old",
        }
    )

    original_check = runner._is_stale_process_notification
    checks = 0

    def _reset_after_initial_check(payload):
        nonlocal checks
        checks += 1
        result = original_check(payload)
        if checks == 1:
            runner.session_store._entries[session_key] = SimpleNamespace(
                session_id="sess-new"
            )
        return result

    monkeypatch.setattr(runner, "_is_stale_process_notification", _reset_after_initial_check)

    await runner._run_process_watcher(watcher)

    assert checks >= 2
    adapter.send.assert_not_awaited()


@pytest.mark.asyncio
async def test_run_process_watcher_drops_completion_for_reset_session_boundary(monkeypatch, tmp_path):
    import tools.process_registry as pr_module

    sessions = [SimpleNamespace(output_buffer="done\n", exited=True, exit_code=0, command="echo hi")]
    monkeypatch.setattr(pr_module, "process_registry", _FakeRegistry(sessions))

    async def _instant_sleep(*_a, **_kw):
        pass
    monkeypatch.setattr(asyncio, "sleep", _instant_sleep)

    runner = _build_runner(monkeypatch, tmp_path, "all")
    adapter = runner.adapters[Platform.TELEGRAM]
    runner.session_store._entries["agent:main:telegram:group:-100:42"] = SimpleNamespace(
        session_id="sess-new",
    )

    watcher = _watcher_dict(session_id="proc_done")
    watcher.update({
        "session_key": "agent:main:telegram:group:-100:42",
        "notify_on_complete": True,
        "conversation_session_id": "sess-old",
    })

    await runner._run_process_watcher(watcher)

    adapter.handle_message.assert_not_awaited()


def test_build_process_event_source_returns_none_for_empty_evt(monkeypatch, tmp_path):
    """Missing session_key and no platform metadata → None (drop notification)."""
    runner = _build_runner(monkeypatch, tmp_path, "all")

    source = runner._build_process_event_source({"session_id": "proc_orphan"})
    assert source is None


def test_build_process_event_source_returns_none_for_invalid_platform(monkeypatch, tmp_path):
    """Invalid platform string → None."""
    runner = _build_runner(monkeypatch, tmp_path, "all")

    evt = {
        "session_id": "proc_bad",
        "platform": "not_a_real_platform",
        "chat_type": "dm",
        "chat_id": "123",
    }
    source = runner._build_process_event_source(evt)
    assert source is None


def test_build_process_event_source_returns_none_for_short_session_key(monkeypatch, tmp_path):
    """Session key with <5 parts doesn't parse, falls through to empty metadata → None."""
    runner = _build_runner(monkeypatch, tmp_path, "all")

    evt = {
        "session_id": "proc_short",
        "session_key": "agent:main:telegram",  # Too few parts
    }
    source = runner._build_process_event_source(evt)
    assert source is None


# ---------------------------------------------------------------------------
# _parse_session_key helper
# ---------------------------------------------------------------------------

def test_parse_session_key_valid():
    result = _parse_session_key("agent:main:telegram:group:-100")
    assert result == {"platform": "telegram", "chat_type": "group", "chat_id": "-100"}


def test_parse_session_key_named_profile():
    result = _parse_session_key("agent:coder:telegram:dm:123:topic42")
    assert result == {
        "platform": "telegram",
        "chat_type": "dm",
        "chat_id": "123",
        "profile": "coder",
        "thread_id": "topic42",
    }


def test_parse_session_key_with_extra_parts():
    """6th part in a group key may be a user_id, not a thread_id — omit it."""
    result = _parse_session_key("agent:main:discord:group:chan123:thread456")
    assert result == {"platform": "discord", "chat_type": "group", "chat_id": "chan123"}


def test_parse_session_key_with_user_id_part():
    """Group keys with per-user isolation have user_id as 6th part — don't return as thread_id."""
    result = _parse_session_key("agent:main:telegram:group:chat1:user99")
    assert result == {"platform": "telegram", "chat_type": "group", "chat_id": "chat1"}


def test_parse_session_key_dm_with_thread():
    """DM keys use parts[5] as thread_id unambiguously."""
    result = _parse_session_key("agent:main:telegram:dm:chat1:topic42")
    assert result == {"platform": "telegram", "chat_type": "dm", "chat_id": "chat1", "thread_id": "topic42"}


def test_parse_session_key_thread_chat_type():
    """Thread-typed keys use parts[5] as thread_id unambiguously."""
    result = _parse_session_key("agent:main:discord:thread:chan1:thread99")
    assert result == {"platform": "discord", "chat_type": "thread", "chat_id": "chan1", "thread_id": "thread99"}


def test_parse_session_key_too_short():
    assert _parse_session_key("agent:main:telegram") is None
    assert _parse_session_key("") is None


def test_parse_session_key_wrong_prefix():
    assert _parse_session_key("cron:main:telegram:dm:123") is None
    assert _parse_session_key("agent::telegram:dm:123") is None
