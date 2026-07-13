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
@pytest.mark.parametrize("process_exit", ["complete", "cancel", "reset"])
async def test_foreground_adapter_task_waits_once_for_stamped_process_owner(
    monkeypatch, tmp_path, process_exit
):
    from gateway.platforms.base import MessageEvent, MessageType
    from gateway.session import SessionSource
    from tests.gateway.restart_test_helpers import RestartTestAdapter

    runner = _build_runner(monkeypatch, tmp_path, "all")
    runner._run_and_deliver_stamped_process_event = (
        GatewayRunner._run_and_deliver_stamped_process_event.__get__(
            runner, GatewayRunner
        )
    )
    adapter = RestartTestAdapter()
    adapter.config.typing_indicator = False
    runner.adapters[Platform.TELEGRAM] = adapter
    monkeypatch.setattr(runner, "_is_user_authorized", lambda _source: True)
    monkeypatch.setattr(
        runner, "_process_notification_run_is_current", lambda **_kwargs: True
    )
    session_key = "agent:main:telegram:dm:process-owner"
    source = SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="process-owner",
        chat_type="dm",
        user_id="user",
    )
    process_event = MessageEvent(
        text="[SYSTEM: process completion]",
        message_type=MessageType.TEXT,
        source=source,
        internal=True,
        metadata={"expected_gateway_session_id": "sess-old"},
    )
    foreground_event = MessageEvent(
        text="normal foreground request",
        message_type=MessageType.TEXT,
        source=source,
    )
    process_started = asyncio.Event()
    release_process = asyncio.Event()
    drain_started = asyncio.Event()
    release_drain = asyncio.Event()
    owner = SimpleNamespace(interrupt=MagicMock())

    async def _synthetic_agent_handler(_event, _source, key, generation):
        nonlocal dispatches_while_blocked
        assert key == session_key
        assert generation == process_event.metadata["gateway_run_generation"]
        runner._running_agents[session_key] = owner
        runner._stamped_process_running_agents[session_key] = owner
        process_started.set()
        try:
            await release_process.wait()
        finally:
            dispatches_while_blocked = foreground_dispatches
        return None

    foreground_dispatches = 0
    normal_turns = 0
    dispatches_while_blocked = 0

    async def _foreground_handler(event):
        nonlocal foreground_dispatches, normal_turns
        foreground_dispatches += 1
        if foreground_dispatches >= 20:
            # Bound the known-bad hot loop so the RED regression terminates.
            release_process.set()
        if session_key in runner._running_agents:
            await runner._handle_active_session_busy_message(event, session_key)
        else:
            normal_turns += 1
        return None

    monkeypatch.setattr(
        runner, "_handle_message_with_agent", _synthetic_agent_handler
    )
    original_flush = adapter._flush_text_debounce_now

    async def _flush_with_reset_gap(key):
        if (
            process_exit == "reset"
            and release_process.is_set()
            and not drain_started.is_set()
        ):
            drain_started.set()
            await release_drain.wait()
        await original_flush(key)

    monkeypatch.setattr(adapter, "_flush_text_debounce_now", _flush_with_reset_gap)
    adapter.set_message_handler(_foreground_handler)
    adapter.set_busy_session_handler(runner._handle_active_session_busy_message)

    process_task = asyncio.create_task(
        runner._run_and_deliver_stamped_process_event(
            process_event,
            adapter=adapter,
            session_key=session_key,
        )
    )
    await process_started.wait()
    release_claim = runner._stamped_process_release_events[session_key]
    assert release_claim.generation == process_event.metadata[
        "gateway_run_generation"
    ]
    try:
        await adapter.handle_message(foreground_event)
        for _ in range(50):
            if foreground_dispatches == 1:
                break
            await asyncio.sleep(0.01)
        assert foreground_dispatches == 1
        interrupt_event = adapter._active_sessions[session_key]
        assert (
            getattr(
                interrupt_event,
                "_hermes_stamped_process_release_event",
                None,
            )
            is release_claim.event
        )
        assert not process_task.done()
        if process_exit == "cancel":
            process_task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await process_task
        elif process_exit == "reset":
            release_process.set()
            await process_task
            await asyncio.wait_for(drain_started.wait(), timeout=2)
            await runner._interrupt_and_clear_session(
                session_key,
                source,
                interrupt_reason="test reset",
                invalidation_reason="test reset",
            )
            release_drain.set()
        else:
            release_process.set()
            await process_task
        expected_normal_turns = 0 if process_exit == "reset" else 1
        for _ in range(50):
            if (
                normal_turns == expected_normal_turns
                and not adapter._background_tasks
            ):
                break
            await asyncio.sleep(0.01)
        assert dispatches_while_blocked == 1
        assert foreground_dispatches == 1 + expected_normal_turns
        assert normal_turns == expected_normal_turns

        assert session_key not in adapter._pending_messages
    finally:
        release_process.set()
        release_drain.set()
        if not process_task.done():
            process_task.cancel()
        await asyncio.gather(process_task, return_exceptions=True)
        await adapter.cancel_background_tasks()
        adapter._pending_messages.clear()
        runner._release_running_agent_state(session_key)


@pytest.mark.asyncio
async def test_new_stamped_generation_rebinds_existing_adapter_guard(
    monkeypatch, tmp_path
):
    from gateway.platforms.base import MessageEvent, MessageType
    from gateway.session import SessionSource
    from tests.gateway.restart_test_helpers import RestartTestAdapter

    runner = _build_runner(monkeypatch, tmp_path, "all")
    runner._run_and_deliver_stamped_process_event = (
        GatewayRunner._run_and_deliver_stamped_process_event.__get__(
            runner, GatewayRunner
        )
    )
    adapter = RestartTestAdapter()
    adapter.config.typing_indicator = False
    runner.adapters[Platform.TELEGRAM] = adapter
    monkeypatch.setattr(runner, "_is_user_authorized", lambda _source: True)
    monkeypatch.setattr(
        runner, "_process_notification_run_is_current", lambda **_kwargs: True
    )
    source = SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="overlapping-process-owner",
        chat_type="dm",
        user_id="user",
    )
    session_key = runner._session_key_for_source(source)
    process_event = MessageEvent(
        text="[SYSTEM: second process completion]",
        message_type=MessageType.TEXT,
        source=source,
        internal=True,
        metadata={"expected_gateway_session_id": "sess-old"},
    )
    adapter_guard = asyncio.Event()
    stale_release_event = asyncio.Event()
    stale_release_event.set()
    setattr(
        adapter_guard,
        "_hermes_stamped_process_release_event",
        stale_release_event,
    )
    adapter._active_sessions[session_key] = adapter_guard

    async def _assert_rebound(_event, _source, key, generation):
        assert key == session_key
        claim = runner._stamped_process_release_events[key]
        assert claim.generation == generation
        assert (
            getattr(
                adapter_guard,
                "_hermes_stamped_process_release_event",
                None,
            )
            is claim.event
        )
        return None

    monkeypatch.setattr(runner, "_handle_message_with_agent", _assert_rebound)

    await runner._run_and_deliver_stamped_process_event(
        process_event,
        adapter=adapter,
        session_key=session_key,
    )

    assert session_key not in runner._stamped_process_release_events
    assert getattr(
        process_event, "_hermes_stamped_process_release_claim"
    ).event.is_set()


@pytest.mark.asyncio
async def test_busy_handler_snapshots_stamped_owner_under_publication_lock(
    monkeypatch, tmp_path
):
    from gateway.platforms.base import MessageEvent, MessageType
    from gateway.session import SessionSource
    from tests.gateway.restart_test_helpers import RestartTestAdapter

    runner = _build_runner(monkeypatch, tmp_path, "all")
    monkeypatch.setattr(runner, "_is_user_authorized", lambda _source: True)
    adapter = RestartTestAdapter()
    adapter.config.typing_indicator = False
    runner.adapters[Platform.TELEGRAM] = adapter
    runner._busy_input_mode = "steer"
    source = SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="publication-lock",
        chat_type="dm",
        user_id="user",
    )
    session_key = runner._session_key_for_source(source)
    event = MessageEvent(
        text="must be retained",
        message_type=MessageType.TEXT,
        source=source,
    )
    pending_owner = SimpleNamespace(interrupt=MagicMock())
    concrete_owner = SimpleNamespace(
        interrupt=MagicMock(),
        steer=MagicMock(return_value=True),
    )
    runner._running_agents[session_key] = pending_owner
    runner._stamped_process_running_agents[session_key] = pending_owner
    publication_halfway = threading.Event()
    publication_finished = threading.Event()
    release_publication = threading.Event()

    def _publish_concrete_owner():
        with runner._running_agent_state_lock:
            runner._running_agents[session_key] = concrete_owner
            publication_halfway.set()
            assert release_publication.wait(timeout=5)
            runner._stamped_process_running_agents[session_key] = concrete_owner
        publication_finished.set()

    publisher = threading.Thread(target=_publish_concrete_owner)
    publisher.start()
    assert await asyncio.to_thread(publication_halfway.wait, 5)
    release_timer = threading.Timer(0.1, release_publication.set)
    release_timer.start()
    try:
        result = await runner._handle_active_session_busy_message(
            event, session_key
        )
        assert publication_finished.is_set()
    finally:
        release_publication.set()
        release_timer.cancel()
        await asyncio.to_thread(publisher.join, 5)

    assert result is True
    concrete_owner.interrupt.assert_not_called()
    concrete_owner.steer.assert_not_called()
    assert adapter._pending_messages[session_key] is event


@pytest.mark.asyncio
async def test_explicit_steer_is_queued_behind_stamped_process_owner(
    monkeypatch, tmp_path
):
    from gateway.platforms.base import MessageEvent, MessageType
    from gateway.session import SessionSource

    runner = _build_runner(monkeypatch, tmp_path, "all")
    monkeypatch.setattr(runner, "_is_user_authorized", lambda _source: True)
    adapter = runner.adapters[Platform.TELEGRAM]
    adapter._pending_messages = {}
    source = SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="explicit-steer-process-owner",
        chat_type="dm",
        user_id="user",
    )
    session_key = runner._session_key_for_source(source)
    process_owner = SimpleNamespace(
        interrupt=MagicMock(),
        steer=MagicMock(return_value=True),
    )
    runner._running_agents[session_key] = process_owner
    runner._stamped_process_running_agents[session_key] = process_owner
    event = MessageEvent(
        text="/steer retain this",
        message_type=MessageType.TEXT,
        source=source,
    )

    response = await runner._handle_message(event)

    assert "queued" in str(response).lower()
    process_owner.interrupt.assert_not_called()
    process_owner.steer.assert_not_called()
    assert adapter._pending_messages[session_key].text == "retain this"

    second = MessageEvent(
        text="second",
        message_type=MessageType.TEXT,
        source=source,
    )
    runner._queue_or_replace_pending_event(session_key, second)
    assert adapter._pending_messages[session_key].text == "retain this"
    assert [item.text for item in runner._queued_events[session_key]] == ["second"]


@pytest.mark.asyncio
async def test_stamped_owner_remains_published_through_guarded_send(
    monkeypatch, tmp_path
):
    from gateway.platforms.base import MessageEvent, MessageType
    from gateway.session import SessionSource
    from tests.gateway.restart_test_helpers import RestartTestAdapter

    runner = _build_runner(monkeypatch, tmp_path, "all")
    runner._run_and_deliver_stamped_process_event = (
        GatewayRunner._run_and_deliver_stamped_process_event.__get__(
            runner, GatewayRunner
        )
    )
    adapter = RestartTestAdapter()
    adapter.config.typing_indicator = False
    runner.adapters[Platform.TELEGRAM] = adapter
    monkeypatch.setattr(runner, "_is_user_authorized", lambda _source: True)
    monkeypatch.setattr(
        runner, "_process_notification_run_is_current", lambda **_kwargs: True
    )
    source = SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="guarded-send-owner",
        chat_type="dm",
        user_id="user",
    )
    session_key = runner._session_key_for_source(source)
    event = MessageEvent(
        text="[SYSTEM: process completion]",
        message_type=MessageType.TEXT,
        source=source,
        internal=True,
        metadata={"expected_gateway_session_id": "sess-old"},
    )
    owner = SimpleNamespace(interrupt=MagicMock())
    send_started = asyncio.Event()
    release_send = asyncio.Event()

    async def _synthetic_agent(_event, _source, key, _generation):
        runner._running_agents[key] = owner
        runner._stamped_process_running_agents[key] = owner
        return {"final_response": "done"}

    async def _blocked_send(*_args, **_kwargs):
        send_started.set()
        await release_send.wait()
        return SimpleNamespace(success=True)

    monkeypatch.setattr(runner, "_handle_message_with_agent", _synthetic_agent)
    monkeypatch.setattr(adapter, "send", _blocked_send)

    task = asyncio.create_task(
        runner._run_and_deliver_stamped_process_event(
            event,
            adapter=adapter,
            session_key=session_key,
        )
    )
    await asyncio.wait_for(send_started.wait(), timeout=2)
    try:
        assert runner._running_agents[session_key] is owner
        assert runner._stamped_process_running_agents[session_key] is owner
    finally:
        release_send.set()
        await task

    assert session_key not in runner._running_agents
    assert session_key not in runner._stamped_process_running_agents


@pytest.mark.asyncio
async def test_adapter_waiter_follows_rebound_stamped_release_event():
    from gateway.platforms.base import MessageEvent, MessageType
    from gateway.session import SessionSource
    from tests.gateway.restart_test_helpers import RestartTestAdapter

    adapter = RestartTestAdapter()
    adapter.config.typing_indicator = False
    source = SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="release-chain",
        chat_type="dm",
        user_id="user",
    )
    session_key = "agent:main:telegram:dm:release-chain"
    first = MessageEvent(
        text="first",
        message_type=MessageType.TEXT,
        source=source,
    )
    second = MessageEvent(
        text="second",
        message_type=MessageType.TEXT,
        source=source,
    )
    first_dispatch = asyncio.Event()
    dispatches = 0
    release_a = asyncio.Event()
    release_b = asyncio.Event()

    async def _handler(_event):
        nonlocal dispatches
        dispatches += 1
        if dispatches == 1:
            guard = adapter._active_sessions[session_key]
            setattr(
                guard,
                "_hermes_stamped_process_release_event",
                release_a,
            )
            adapter._pending_messages[session_key] = second
            first_dispatch.set()
        return None

    adapter.set_message_handler(_handler)
    await adapter.handle_message(first)
    await asyncio.wait_for(first_dispatch.wait(), timeout=2)
    guard = adapter._active_sessions[session_key]
    setattr(guard, "_hermes_stamped_process_release_event", release_b)

    release_a.set()
    await asyncio.sleep(0.05)
    assert dispatches == 1

    release_b.set()
    for _ in range(50):
        if dispatches == 2 and not adapter._background_tasks:
            break
        await asyncio.sleep(0.01)
    assert dispatches == 2
    await adapter.cancel_background_tasks()


@pytest.mark.asyncio
async def test_in_band_drain_rechecks_release_rebound_during_typing_stop():
    from gateway.platforms.base import MessageEvent, MessageType
    from gateway.session import SessionSource
    from tests.gateway.restart_test_helpers import RestartTestAdapter

    adapter = RestartTestAdapter()
    adapter.config.typing_indicator = False
    source = SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="typing-release-rebind",
        chat_type="dm",
        user_id="user",
    )
    session_key = "agent:main:telegram:dm:typing-release-rebind"
    first = MessageEvent(
        text="first", message_type=MessageType.TEXT, source=source
    )
    second = MessageEvent(
        text="second", message_type=MessageType.TEXT, source=source
    )
    first_dispatch = asyncio.Event()
    stop_rebound = asyncio.Event()
    release_a = asyncio.Event()
    release_b = asyncio.Event()
    dispatches = 0
    stop_calls = 0

    async def _handler(_event):
        nonlocal dispatches
        dispatches += 1
        if dispatches == 1:
            guard = adapter._active_sessions[session_key]
            setattr(
                guard,
                "_hermes_stamped_process_release_event",
                release_a,
            )
            adapter._pending_messages[session_key] = second
            first_dispatch.set()
        return None

    async def _stop_typing(*_args, **_kwargs):
        nonlocal stop_calls
        stop_calls += 1
        if stop_calls == 1:
            guard = adapter._active_sessions[session_key]
            setattr(
                guard,
                "_hermes_stamped_process_release_event",
                release_b,
            )
            stop_rebound.set()

    adapter.set_message_handler(_handler)
    adapter._stop_typing_refresh = _stop_typing
    await adapter.handle_message(first)
    await asyncio.wait_for(first_dispatch.wait(), timeout=2)
    release_a.set()
    await asyncio.wait_for(stop_rebound.wait(), timeout=2)
    await asyncio.sleep(0.05)
    assert dispatches == 1
    assert adapter._pending_messages[session_key] is second

    release_b.set()
    for _ in range(100):
        if dispatches == 2 and not adapter._background_tasks:
            break
        await asyncio.sleep(0.01)
    assert dispatches == 2
    await adapter.cancel_background_tasks()


@pytest.mark.asyncio
@pytest.mark.parametrize("handler_raises", [False, True])
async def test_late_drain_waits_for_rebound_stamped_release_event(handler_raises):
    from gateway.platforms.base import MessageEvent, MessageType
    from gateway.session import SessionSource
    from tests.gateway.restart_test_helpers import RestartTestAdapter

    adapter = RestartTestAdapter()
    adapter.config.typing_indicator = False
    source = SessionSource(
        platform=Platform.TELEGRAM,
        chat_id=f"late-release-chain-{handler_raises}",
        chat_type="dm",
        user_id="user",
    )
    session_key = (
        f"agent:main:telegram:dm:late-release-chain-{handler_raises}"
    )
    first = MessageEvent(
        text="first",
        message_type=MessageType.TEXT,
        source=source,
    )
    second = MessageEvent(
        text="second",
        message_type=MessageType.TEXT,
        source=source,
    )
    dispatches = 0
    stop_calls = 0
    injected = asyncio.Event()
    release_b = asyncio.Event()

    async def _handler(_event):
        nonlocal dispatches
        dispatches += 1
        if handler_raises and dispatches == 1:
            raise RuntimeError("synthetic handler failure")
        return None

    async def _stop_typing(*_args, **_kwargs):
        nonlocal stop_calls
        stop_calls += 1
        if stop_calls == 2:
            guard = adapter._active_sessions[session_key]
            setattr(
                guard,
                "_hermes_stamped_process_release_event",
                release_b,
            )
            adapter._pending_messages[session_key] = second
            injected.set()

    adapter.set_message_handler(_handler)
    adapter._stop_typing_refresh = _stop_typing
    await adapter.handle_message(first)
    await asyncio.wait_for(injected.wait(), timeout=2)
    await asyncio.sleep(0.05)
    assert dispatches == 1
    assert adapter._pending_messages[session_key] is second

    release_b.set()
    for _ in range(100):
        if dispatches == 2 and not adapter._background_tasks:
            break
        await asyncio.sleep(0.01)
    assert dispatches == 2
    await adapter.cancel_background_tasks()


@pytest.mark.asyncio
async def test_stale_finalizer_leaves_pending_for_replacement_command_guard():
    from gateway.platforms.base import MessageEvent, MessageType
    from gateway.session import SessionSource
    from tests.gateway.restart_test_helpers import RestartTestAdapter

    adapter = RestartTestAdapter()
    adapter.config.typing_indicator = False
    source = SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="replacement-command-guard",
        chat_type="dm",
        user_id="user",
    )
    session_key = "agent:main:telegram:dm:replacement-command-guard"
    first = MessageEvent(
        text="first",
        message_type=MessageType.TEXT,
        source=source,
    )
    second = MessageEvent(
        text="second",
        message_type=MessageType.TEXT,
        source=source,
    )
    command_guard = asyncio.Event()
    command_guard.set()
    dispatches = 0
    stop_calls = 0
    swapped = asyncio.Event()

    async def _handler(_event):
        nonlocal dispatches
        dispatches += 1
        return None

    async def _stop_typing(*_args, **_kwargs):
        nonlocal stop_calls
        stop_calls += 1
        if stop_calls == 2:
            adapter._active_sessions[session_key] = command_guard
            adapter._session_tasks.pop(session_key, None)
            adapter._pending_messages[session_key] = second
            swapped.set()

    adapter.set_message_handler(_handler)
    adapter._stop_typing_refresh = _stop_typing
    await adapter.handle_message(first)
    await asyncio.wait_for(swapped.wait(), timeout=2)
    await asyncio.sleep(0.05)

    assert dispatches == 1
    assert adapter._active_sessions[session_key] is command_guard
    assert command_guard.is_set()
    assert adapter._pending_messages[session_key] is second

    await adapter._drain_pending_after_session_command(
        session_key, command_guard
    )
    for _ in range(100):
        if dispatches == 2 and not adapter._background_tasks:
            break
        await asyncio.sleep(0.01)
    assert dispatches == 2
    await adapter.cancel_background_tasks()


@pytest.mark.asyncio
async def test_stale_command_drain_leaves_replacement_pending_owned():
    from gateway.platforms.base import MessageEvent, MessageType
    from gateway.session import SessionSource
    from tests.gateway.restart_test_helpers import RestartTestAdapter

    adapter = RestartTestAdapter()
    source = SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="stale-command-drain",
        chat_type="dm",
        user_id="user",
    )
    session_key = "agent:main:telegram:dm:stale-command-drain"
    pending = MessageEvent(
        text="replacement pending",
        message_type=MessageType.TEXT,
        source=source,
    )
    guard_a = asyncio.Event()
    guard_b = asyncio.Event()
    flush_entered = asyncio.Event()
    release_flush = asyncio.Event()
    started = []

    async def _blocked_flush(_session_key):
        flush_entered.set()
        await release_flush.wait()

    adapter._active_sessions[session_key] = guard_a
    adapter._flush_text_debounce_now = _blocked_flush
    adapter._start_session_processing = lambda event, key: started.append(
        (event, key)
    )

    drain_a = asyncio.create_task(
        adapter._drain_pending_after_session_command(session_key, guard_a)
    )
    await asyncio.wait_for(flush_entered.wait(), timeout=2)
    adapter._active_sessions[session_key] = guard_b
    adapter._pending_messages[session_key] = pending
    release_flush.set()
    await drain_a

    assert adapter._active_sessions[session_key] is guard_b
    assert adapter._pending_messages[session_key] is pending
    assert started == []


@pytest.mark.asyncio
async def test_superseded_session_command_does_not_cancel_replacement_owner():
    from gateway.platforms.base import MessageEvent, MessageType
    from gateway.session import SessionSource
    from tests.gateway.restart_test_helpers import RestartTestAdapter

    adapter = RestartTestAdapter()
    source = SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="superseded-command",
        chat_type="dm",
        user_id="user",
    )
    session_key = "agent:main:telegram:dm:superseded-command"
    event = MessageEvent(
        text="/new",
        message_type=MessageType.TEXT,
        source=source,
    )
    old_guard = asyncio.Event()
    replacement_guard = asyncio.Event()
    handler_entered = asyncio.Event()
    release_handler = asyncio.Event()

    async def _handler(_event):
        handler_entered.set()
        await release_handler.wait()
        return None

    adapter.set_message_handler(_handler)
    adapter._active_sessions[session_key] = old_guard
    adapter.cancel_session_processing = AsyncMock()
    adapter._drain_pending_after_session_command = AsyncMock()

    command_a = asyncio.create_task(
        adapter._dispatch_active_session_command(event, session_key, "new")
    )
    await asyncio.wait_for(handler_entered.wait(), timeout=2)
    adapter._active_sessions[session_key] = replacement_guard
    release_handler.set()
    await command_a

    assert adapter._active_sessions[session_key] is replacement_guard
    adapter.cancel_session_processing.assert_not_awaited()
    adapter._drain_pending_after_session_command.assert_not_awaited()


@pytest.mark.asyncio
async def test_reset_like_commands_serialize_through_response_delivery():
    from gateway.platforms.base import MessageEvent, MessageType
    from gateway.session import SessionSource
    from tests.gateway.restart_test_helpers import RestartTestAdapter

    adapter = RestartTestAdapter()
    source = SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="serialized-commands",
        chat_type="dm",
        user_id="user",
    )
    session_key = "agent:main:telegram:dm:serialized-commands"
    command_a = MessageEvent(
        text="/new",
        message_type=MessageType.TEXT,
        source=source,
    )
    command_b = MessageEvent(
        text="/reset",
        message_type=MessageType.TEXT,
        source=source,
    )
    send_a_entered = asyncio.Event()
    release_send_a = asyncio.Event()
    handler_calls = []
    deliveries = []

    async def _handler(event):
        handler_calls.append(event.text)
        return event.text.upper()

    async def _send(*_args, **kwargs):
        content = kwargs["content"]
        if content == "/NEW":
            send_a_entered.set()
            await release_send_a.wait()
        deliveries.append(content)
        return SimpleNamespace(success=True, message_id=None)

    adapter.set_message_handler(_handler)
    adapter._send_with_retry = _send
    adapter._active_sessions[session_key] = asyncio.Event()

    task_a = asyncio.create_task(
        adapter._dispatch_active_session_command(
            command_a, session_key, "new"
        )
    )
    await asyncio.wait_for(send_a_entered.wait(), timeout=2)
    task_b = asyncio.create_task(
        adapter._dispatch_active_session_command(
            command_b, session_key, "reset"
        )
    )
    await asyncio.sleep(0.05)
    assert handler_calls == ["/new"]
    assert deliveries == []

    release_send_a.set()
    await asyncio.gather(task_a, task_b)
    assert handler_calls == ["/new", "/reset"]
    assert deliveries == ["/NEW", "/RESET"]


@pytest.mark.asyncio
async def test_cancelled_session_command_keeps_guard_until_live_owner_unwinds():
    from gateway.platforms.base import MessageEvent, MessageType
    from gateway.session import SessionSource
    from tests.gateway.restart_test_helpers import RestartTestAdapter

    adapter = RestartTestAdapter()
    source = SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="cancelled-command",
        chat_type="dm",
        user_id="user",
    )
    session_key = "agent:main:telegram:dm:cancelled-command"
    event = MessageEvent(
        text="/new",
        message_type=MessageType.TEXT,
        source=source,
    )
    old_guard = asyncio.Event()
    handler_entered = asyncio.Event()
    never_release = asyncio.Event()

    async def _handler(_event):
        handler_entered.set()
        await never_release.wait()

    adapter.set_message_handler(_handler)
    adapter._active_sessions[session_key] = old_guard
    adapter._session_tasks[session_key] = asyncio.create_task(
        asyncio.sleep(60)
    )

    command_task = asyncio.create_task(
        adapter._dispatch_active_session_command(event, session_key, "new")
    )
    await asyncio.wait_for(handler_entered.wait(), timeout=2)
    command_guard = adapter._active_sessions[session_key]
    assert command_guard is not old_guard
    command_task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await command_task

    assert adapter._active_sessions[session_key] is command_guard
    owner = adapter._session_tasks[session_key]
    owner.cancel()
    with pytest.raises(asyncio.CancelledError):
        await owner
    for _ in range(100):
        if (
            session_key not in adapter._active_sessions
            and session_key not in adapter._session_tasks
            and not adapter._background_tasks
        ):
            break
        await asyncio.sleep(0.01)
    assert session_key not in adapter._active_sessions
    assert session_key not in adapter._session_tasks


@pytest.mark.asyncio
async def test_cancelled_session_command_drains_when_previous_owner_is_done():
    from gateway.platforms.base import MessageEvent, MessageType
    from gateway.session import SessionSource
    from tests.gateway.restart_test_helpers import RestartTestAdapter

    adapter = RestartTestAdapter()
    source = SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="cancelled-command-done-owner",
        chat_type="dm",
        user_id="user",
    )
    session_key = "agent:main:telegram:dm:cancelled-command-done-owner"
    command_event = MessageEvent(
        text="/new",
        message_type=MessageType.TEXT,
        source=source,
    )
    pending_event = MessageEvent(
        text="pending-follow-up",
        message_type=MessageType.TEXT,
        source=source,
    )
    old_guard = asyncio.Event()
    handler_entered = asyncio.Event()
    pending_started = asyncio.Event()
    owner_release = asyncio.Event()
    flush_entered = asyncio.Event()
    release_flush = asyncio.Event()

    async def _blocked_flush(session_key):
        del session_key
        flush_entered.set()
        await release_flush.wait()
        return False

    adapter._flush_text_debounce_now = _blocked_flush

    async def _handler(event):
        if event.text == "/new":
            handler_entered.set()
            await asyncio.Future()
        else:
            pending_started.set()

    async def _owner():
        await owner_release.wait()

    adapter.set_message_handler(_handler)
    adapter._active_sessions[session_key] = old_guard
    owner_task = asyncio.create_task(_owner())
    adapter._session_tasks[session_key] = owner_task

    command_task = asyncio.create_task(
        adapter._dispatch_active_session_command(
            command_event, session_key, "new"
        )
    )
    await handler_entered.wait()
    command_guard = adapter._active_sessions[session_key]
    adapter._pending_messages[session_key] = pending_event
    owner_release.set()
    await owner_task
    assert adapter._session_tasks[session_key] is owner_task
    assert owner_task.done()

    command_task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await command_task
    await asyncio.wait_for(flush_entered.wait(), timeout=2)
    # Repeated caller cancellation cannot reach the separately tracked drain.
    command_task.cancel()
    release_flush.set()
    await asyncio.wait_for(pending_started.wait(), timeout=2)

    assert adapter._active_sessions.get(session_key) is not old_guard
    assert session_key not in adapter._pending_messages
    for task in list(adapter._background_tasks):
        if not task.done():
            await task


@pytest.mark.asyncio
@pytest.mark.parametrize("owner_state", ["done", "live"])
@pytest.mark.parametrize(
    "publication_failure", ["raise", "sentinel", "callback"]
)
async def test_cancelled_command_drain_publication_failure_is_transactional(
    monkeypatch, owner_state, publication_failure
):
    from gateway.platforms.base import MessageEvent, MessageType
    from gateway.session import SessionSource
    from tests.gateway.restart_test_helpers import RestartTestAdapter

    adapter = RestartTestAdapter()
    source = SessionSource(
        platform=Platform.TELEGRAM,
        chat_id=f"cancelled-command-publish-failure-{owner_state}",
        chat_type="dm",
        user_id="user",
    )
    session_key = (
        "agent:main:telegram:dm:cancelled-command-publish-failure-"
        f"{owner_state}"
    )
    command_event = MessageEvent(
        text="/new",
        message_type=MessageType.TEXT,
        source=source,
    )
    pending_event = MessageEvent(
        text="preserve-on-publication-failure",
        message_type=MessageType.TEXT,
        source=source,
    )
    handler_entered = asyncio.Event()
    owner_release = asyncio.Event()

    async def _handler(_event):
        handler_entered.set()
        await asyncio.Future()

    async def _owner():
        if owner_state == "live":
            await owner_release.wait()

    adapter.set_message_handler(_handler)
    old_guard = asyncio.Event()
    adapter._active_sessions[session_key] = old_guard
    owner_task = asyncio.create_task(_owner())
    if owner_state == "done":
        await owner_task
    adapter._session_tasks[session_key] = owner_task

    command_task = asyncio.create_task(
        adapter._dispatch_active_session_command(
            command_event, session_key, "new"
        )
    )
    await handler_entered.wait()
    command_guard = adapter._active_sessions[session_key]
    adapter._pending_messages[session_key] = pending_event

    create_attempts = 0

    class _CallbackFailTask(asyncio.Task):
        def add_done_callback(self, *_args, **_kwargs):
            raise RuntimeError("forced add_done_callback failure")

    def _fail_create_task(coroutine):
        nonlocal create_attempts
        create_attempts += 1
        if publication_failure == "sentinel":
            return object()
        if publication_failure == "callback":
            return _CallbackFailTask(
                coroutine, loop=asyncio.get_running_loop()
            )
        raise RuntimeError("forced continuation publication failure")

    monkeypatch.setattr(asyncio, "create_task", _fail_create_task)
    command_task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await command_task

    if owner_state == "live":
        assert adapter._active_sessions[session_key] is command_guard
        owner_release.set()
        await owner_task
        await asyncio.sleep(0)
    elif publication_failure == "callback":
        await asyncio.sleep(0)

    assert create_attempts == 1
    assert session_key not in adapter._active_sessions
    assert adapter._pending_messages[session_key] is pending_event
    assert adapter._session_tasks[session_key] is owner_task
    assert owner_task.done()
    assert not adapter._background_tasks


@pytest.mark.asyncio
async def test_cancellation_waiters_keep_owner_published_until_unwind():
    from tests.gateway.restart_test_helpers import RestartTestAdapter

    adapter = RestartTestAdapter()
    session_key = "agent:main:telegram:dm:cancel-waiters"
    unwind_entered = asyncio.Event()
    release_unwind = asyncio.Event()

    async def _owner():
        try:
            await asyncio.Event().wait()
        finally:
            unwind_entered.set()
            await release_unwind.wait()

    owner = asyncio.create_task(_owner())
    adapter._session_tasks[session_key] = owner
    first = asyncio.create_task(
        adapter.cancel_session_processing(
            session_key,
            release_guard=False,
            discard_pending=False,
        )
    )
    await asyncio.wait_for(unwind_entered.wait(), timeout=2)
    assert adapter._session_tasks[session_key] is owner

    second = asyncio.create_task(
        adapter.cancel_session_processing(
            session_key,
            release_guard=False,
            discard_pending=False,
        )
    )
    await asyncio.sleep(0.05)
    assert adapter._session_tasks[session_key] is owner
    assert not first.done()
    assert not second.done()

    release_unwind.set()
    await asyncio.gather(first, second)
    assert session_key not in adapter._session_tasks


@pytest.mark.asyncio
async def test_command_cancellation_during_owner_unwind_defers_exact_drain():
    from gateway.platforms.base import MessageEvent, MessageType
    from gateway.session import SessionSource
    from tests.gateway.restart_test_helpers import RestartTestAdapter

    adapter = RestartTestAdapter()
    source = SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="cancel-during-unwind",
        chat_type="dm",
        user_id="user",
    )
    session_key = "agent:main:telegram:dm:cancel-during-unwind"
    event = MessageEvent(
        text="/new",
        message_type=MessageType.TEXT,
        source=source,
    )
    pending_event = MessageEvent(
        text="pending-after-cancelled-command",
        message_type=MessageType.TEXT,
        source=source,
    )
    old_guard = asyncio.Event()
    unwind_entered = asyncio.Event()
    release_unwind = asyncio.Event()
    dispatches = []

    async def _owner():
        try:
            await asyncio.Event().wait()
        finally:
            unwind_entered.set()
            await release_unwind.wait()

    async def _handler(message_event):
        dispatches.append(message_event.text)
        return None

    owner = asyncio.create_task(_owner())
    adapter.set_message_handler(_handler)
    adapter._active_sessions[session_key] = old_guard
    adapter._session_tasks[session_key] = owner
    command = asyncio.create_task(
        adapter._dispatch_active_session_command(event, session_key, "new")
    )
    await asyncio.wait_for(unwind_entered.wait(), timeout=2)
    command_guard = adapter._active_sessions[session_key]
    adapter._pending_messages[session_key] = pending_event
    command.cancel()
    with pytest.raises(asyncio.CancelledError):
        await command

    assert adapter._active_sessions[session_key] is command_guard
    assert adapter._session_tasks[session_key] is owner
    release_unwind.set()
    with pytest.raises(asyncio.CancelledError):
        await owner
    for _ in range(100):
        if (
            dispatches == ["/new", "pending-after-cancelled-command"]
            and session_key not in adapter._pending_messages
            and session_key not in adapter._active_sessions
            and session_key not in adapter._session_tasks
        ):
            break
        await asyncio.sleep(0.01)
    assert dispatches == ["/new", "pending-after-cancelled-command"]
    assert session_key not in adapter._pending_messages
    assert session_key not in adapter._active_sessions
    assert session_key not in adapter._session_tasks


@pytest.mark.asyncio
async def test_command_timeout_defers_pending_drain_until_owner_unwinds(
    monkeypatch,
):
    import gateway.platforms.base as base_module
    from gateway.platforms.base import MessageEvent, MessageType
    from gateway.session import SessionSource
    from tests.gateway.restart_test_helpers import RestartTestAdapter

    monkeypatch.setattr(
        base_module, "_SESSION_CANCEL_TIMEOUT_SECONDS", 0.01, raising=False
    )
    adapter = RestartTestAdapter()
    adapter.config.typing_indicator = False
    source = SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="timeout-owner-unwind",
        chat_type="dm",
        user_id="user",
    )
    session_key = "agent:main:telegram:dm:timeout-owner-unwind"
    command_event = MessageEvent(
        text="/new",
        message_type=MessageType.TEXT,
        source=source,
    )
    pending_event = MessageEvent(
        text="pending",
        message_type=MessageType.TEXT,
        source=source,
    )
    old_guard = asyncio.Event()
    unwind_entered = asyncio.Event()
    release_unwind = asyncio.Event()
    dispatches = []

    async def _owner():
        try:
            await asyncio.Event().wait()
        finally:
            unwind_entered.set()
            await release_unwind.wait()

    async def _handler(event):
        dispatches.append(event.text)
        return None

    owner = asyncio.create_task(_owner())
    adapter.set_message_handler(_handler)
    adapter._active_sessions[session_key] = old_guard
    adapter._session_tasks[session_key] = owner
    adapter._pending_messages[session_key] = pending_event
    await asyncio.sleep(0)

    await adapter._dispatch_active_session_command(
        command_event, session_key, "new"
    )
    assert unwind_entered.is_set()
    assert not owner.done()
    assert adapter._session_tasks[session_key] is owner
    assert adapter._pending_messages[session_key] is pending_event
    assert dispatches == ["/new"]

    release_unwind.set()
    with pytest.raises(asyncio.CancelledError):
        await owner
    for _ in range(100):
        if dispatches == ["/new", "pending"]:
            break
        await asyncio.sleep(0.01)
    assert dispatches == ["/new", "pending"]
    await adapter.cancel_background_tasks()


@pytest.mark.asyncio
async def test_session_command_lock_registry_evicts_idle_entries():
    from gateway.platforms.base import MessageEvent, MessageType
    from gateway.session import SessionSource
    from tests.gateway.restart_test_helpers import RestartTestAdapter

    adapter = RestartTestAdapter()

    async def _handler(_event):
        return None

    adapter.set_message_handler(_handler)
    for index in range(250):
        source = SessionSource(
            platform=Platform.TELEGRAM,
            chat_id=f"lock-churn-{index}",
            chat_type="dm",
            user_id="user",
        )
        session_key = f"agent:main:telegram:dm:lock-churn-{index}"
        adapter._active_sessions[session_key] = asyncio.Event()
        await adapter._dispatch_active_session_command(
            MessageEvent(
                text="/new",
                message_type=MessageType.TEXT,
                source=source,
            ),
            session_key,
            "new",
        )

    assert getattr(adapter, "_session_command_locks", {}) == {}


@pytest.mark.asyncio
async def test_deferred_drain_failure_recovers_exact_guard_and_pending():
    from gateway.platforms.base import MessageEvent, MessageType
    from gateway.session import SessionSource
    from tests.gateway.restart_test_helpers import RestartTestAdapter

    adapter = RestartTestAdapter()
    source = SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="deferred-drain-failure",
        chat_type="dm",
        user_id="user",
    )
    session_key = "agent:main:telegram:dm:deferred-drain-failure"
    pending = MessageEvent(
        text="pending",
        message_type=MessageType.TEXT,
        source=source,
    )
    command_guard = asyncio.Event()
    owner = asyncio.create_task(asyncio.sleep(0))
    await owner
    started = []

    adapter._active_sessions[session_key] = command_guard
    adapter._pending_messages[session_key] = pending
    adapter._drain_pending_after_session_command = AsyncMock(
        side_effect=RuntimeError("forced deferred drain failure")
    )

    def _start_recovered(event, session_key_arg):
        started.append((event, session_key_arg))
        return True

    adapter._start_session_processing = _start_recovered

    adapter._defer_session_command_drain_until_owner_done(
        session_key, command_guard, owner
    )
    for _ in range(100):
        if started:
            break
        await asyncio.sleep(0.01)

    assert started == [(pending, session_key)]
    assert session_key not in adapter._active_sessions
    assert session_key not in adapter._pending_messages
    assert not adapter._background_tasks


@pytest.mark.asyncio
@pytest.mark.parametrize("startup_failure", ["false", "raise"])
async def test_command_drain_restores_pending_when_startup_fails(
    startup_failure,
):
    from gateway.platforms.base import MessageEvent, MessageType
    from gateway.session import SessionSource
    from tests.gateway.restart_test_helpers import RestartTestAdapter

    adapter = RestartTestAdapter()
    source = SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="command-drain-startup-failure",
        chat_type="dm",
        user_id="user",
    )
    session_key = "agent:main:telegram:dm:command-drain-startup-failure"
    pending = MessageEvent(
        text="pending",
        message_type=MessageType.TEXT,
        source=source,
    )
    command_guard = asyncio.Event()
    adapter._active_sessions[session_key] = command_guard
    adapter._pending_messages[session_key] = pending
    adapter._flush_text_debounce_now = AsyncMock()
    if startup_failure == "raise":
        adapter._start_session_processing = MagicMock(
            side_effect=RuntimeError("forced startup failure")
        )
        with pytest.raises(RuntimeError, match="forced startup failure"):
            await adapter._drain_pending_after_session_command(
                session_key, command_guard
            )
    else:
        adapter._start_session_processing = MagicMock(return_value=False)
        await adapter._drain_pending_after_session_command(
            session_key, command_guard
        )

    assert adapter._pending_messages[session_key] is pending
    assert session_key not in adapter._active_sessions


@pytest.mark.asyncio
async def test_start_session_processing_rolls_back_when_create_task_raises(
    monkeypatch,
):
    from gateway.platforms.base import MessageEvent, MessageType
    from gateway.session import SessionSource
    from tests.gateway.restart_test_helpers import RestartTestAdapter

    adapter = RestartTestAdapter()
    source = SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="create-task-failure",
        chat_type="dm",
        user_id="user",
    )
    session_key = "agent:main:telegram:dm:create-task-failure"
    event = MessageEvent(
        text="pending",
        message_type=MessageType.TEXT,
        source=source,
    )

    def _raise_create_task(coro):
        coro.close()
        raise RuntimeError("forced create_task failure")

    monkeypatch.setattr(asyncio, "create_task", _raise_create_task)
    with pytest.raises(RuntimeError, match="forced create_task failure"):
        adapter._start_session_processing(event, session_key)

    assert session_key not in adapter._active_sessions
    assert session_key not in adapter._session_tasks


@pytest.mark.asyncio
async def test_start_session_processing_rejects_hashable_non_task(
    monkeypatch,
):
    import inspect

    from gateway.platforms.base import MessageEvent, MessageType
    from gateway.session import SessionSource
    from tests.gateway.restart_test_helpers import RestartTestAdapter

    adapter = RestartTestAdapter()
    source = SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="create-task-hashable-sentinel",
        chat_type="dm",
        user_id="user",
    )
    session_key = "agent:main:telegram:dm:create-task-hashable-sentinel"
    event = MessageEvent(
        text="pending",
        message_type=MessageType.TEXT,
        source=source,
    )
    created_coroutines = []

    def _return_hashable_sentinel(coroutine):
        created_coroutines.append(coroutine)
        return object()

    monkeypatch.setattr(asyncio, "create_task", _return_hashable_sentinel)
    assert not adapter._start_session_processing(event, session_key)

    assert session_key not in adapter._active_sessions
    assert session_key not in adapter._session_tasks
    assert not adapter._background_tasks
    assert len(created_coroutines) == 1
    assert inspect.getcoroutinestate(created_coroutines[0]) == inspect.CORO_CLOSED


@pytest.mark.asyncio
async def test_start_session_processing_rolls_back_callback_failure(
    monkeypatch,
):
    from gateway.platforms.base import MessageEvent, MessageType
    from gateway.session import SessionSource
    from tests.gateway.restart_test_helpers import RestartTestAdapter

    adapter = RestartTestAdapter()
    source = SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="create-task-callback-failure",
        chat_type="dm",
        user_id="user",
    )
    session_key = "agent:main:telegram:dm:create-task-callback-failure"
    event = MessageEvent(
        text="pending",
        message_type=MessageType.TEXT,
        source=source,
    )

    class _CallbackFailTask(asyncio.Task):
        def add_done_callback(self, *_args, **_kwargs):
            raise RuntimeError("forced callback registration failure")

    def _return_callback_fail_task(coroutine):
        return _CallbackFailTask(
            coroutine, loop=asyncio.get_running_loop()
        )

    monkeypatch.setattr(asyncio, "create_task", _return_callback_fail_task)
    with pytest.raises(RuntimeError, match="forced callback registration failure"):
        adapter._start_session_processing(event, session_key)
    await asyncio.sleep(0)

    assert session_key not in adapter._active_sessions
    assert session_key not in adapter._session_tasks
    assert not adapter._background_tasks


@pytest.mark.asyncio
async def test_in_band_drain_preserves_replacement_command_ownership():
    from gateway.platforms.base import MessageEvent, MessageType
    from gateway.session import SessionSource
    from tests.gateway.restart_test_helpers import RestartTestAdapter

    adapter = RestartTestAdapter()
    adapter.config.typing_indicator = False
    source = SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="in-band-command-replacement",
        chat_type="dm",
        user_id="user",
    )
    session_key = "agent:main:telegram:dm:in-band-command-replacement"
    first_event = MessageEvent(
        text="first",
        message_type=MessageType.TEXT,
        source=source,
    )
    pending_event = MessageEvent(
        text="second-must-stay-command-owned",
        message_type=MessageType.TEXT,
        source=source,
    )
    handler_entered = asyncio.Event()
    release_handler = asyncio.Event()
    stop_entered = asyncio.Event()
    release_stop = asyncio.Event()
    replacement_release = asyncio.Event()
    dispatches = []

    async def _handler(event):
        dispatches.append(event.text)
        if event is first_event:
            handler_entered.set()
            await release_handler.wait()
        return None

    async def _blocked_stop(*_args, **_kwargs):
        if not stop_entered.is_set():
            stop_entered.set()
            await release_stop.wait()

    adapter.set_message_handler(_handler)
    adapter._stop_typing_refresh = _blocked_stop
    assert adapter._start_session_processing(first_event, session_key)
    first_task = adapter._session_tasks[session_key]
    await handler_entered.wait()
    adapter._pending_messages[session_key] = pending_event
    release_handler.set()
    await stop_entered.wait()

    command_guard = asyncio.Event()
    replacement_task = asyncio.create_task(replacement_release.wait())
    adapter._active_sessions[session_key] = command_guard
    adapter._session_tasks[session_key] = replacement_task
    release_stop.set()
    await first_task
    await asyncio.sleep(0)

    assert dispatches == ["first"]
    assert adapter._pending_messages[session_key] is pending_event
    assert adapter._active_sessions[session_key] is command_guard
    assert adapter._session_tasks[session_key] is replacement_task

    replacement_release.set()
    await replacement_task
    adapter._active_sessions.pop(session_key, None)
    adapter._session_tasks.pop(session_key, None)


@pytest.mark.asyncio
async def test_adapter_teardown_fences_deferred_command_drain():
    from gateway.platforms.base import MessageEvent, MessageType
    from gateway.session import SessionSource
    from tests.gateway.restart_test_helpers import RestartTestAdapter

    adapter = RestartTestAdapter()
    session_key = "agent:main:telegram:dm:teardown-command-drain"
    source = SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="teardown-command-drain",
        chat_type="dm",
        user_id="user",
    )
    pending = MessageEvent(
        text="must-not-start",
        message_type=MessageType.TEXT,
        source=source,
    )
    handler_calls = []
    owner_cancelled = asyncio.Event()
    release_owner = asyncio.Event()

    async def _handler(event):
        handler_calls.append(event.text)

    async def _owner():
        try:
            await asyncio.Future()
        except asyncio.CancelledError:
            owner_cancelled.set()
            await release_owner.wait()

    adapter.set_message_handler(_handler)
    owner_task = asyncio.create_task(_owner())
    adapter._background_tasks.add(owner_task)
    adapter._session_tasks[session_key] = owner_task
    command_guard = asyncio.Event()
    adapter._active_sessions[session_key] = command_guard
    adapter._pending_messages[session_key] = pending
    adapter._defer_session_command_drain_until_owner_done(
        session_key, command_guard, owner_task
    )

    teardown_task = asyncio.create_task(adapter.cancel_background_tasks())
    await owner_cancelled.wait()
    release_owner.set()
    await teardown_task
    await asyncio.sleep(0)

    assert handler_calls == []
    assert adapter._background_tasks == set()
    assert adapter._session_tasks == {}
    assert adapter._pending_messages == {}
    assert adapter._active_sessions == {}


@pytest.mark.asyncio
async def test_fresh_trackerless_process_owner_is_not_stale_evicted(
    monkeypatch, tmp_path
):
    from gateway.platforms.base import MessageEvent, MessageType
    from gateway.session import SessionSource

    runner = _build_runner(monkeypatch, tmp_path, "all")
    monkeypatch.setattr(runner, "_is_user_authorized", lambda _source: True)
    adapter = runner.adapters[Platform.TELEGRAM]
    adapter._pending_messages = {}
    source = SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="fresh-trackerless-owner",
        chat_type="dm",
        user_id="user",
    )
    session_key = runner._session_key_for_source(source)
    owner = SimpleNamespace(interrupt=MagicMock())
    runner._running_agents[session_key] = owner
    runner._stamped_process_running_agents[session_key] = owner
    runner._running_agents_ts[session_key] = time.time()
    generation = runner._begin_session_run_generation(session_key)
    event = MessageEvent(
        text="retain this",
        message_type=MessageType.TEXT,
        source=source,
    )

    await runner._handle_message(event)

    assert runner._session_run_generation[session_key] == generation
    assert runner._running_agents[session_key] is owner
    assert runner._stamped_process_running_agents[session_key] is owner
    assert adapter._pending_messages[session_key] is event


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
@pytest.mark.parametrize("transition", ["reset", "forced_shutdown"])
async def test_process_run_aborts_when_boundary_wins_during_pre_agent_setup(
    monkeypatch, tmp_path, transition
):
    from datetime import datetime, timedelta, timezone

    from gateway.run import _AGENT_PENDING_SENTINEL
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
    runner._running_agents[session_key] = _AGENT_PENDING_SENTINEL
    runner._running_agents_ts[session_key] = time.time()
    active_lease = MagicMock()
    runner._active_session_leases[session_key] = active_lease

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

    if transition == "reset":
        # /new wins while the handler is suspended in awaited pre-agent setup.
        runner.session_store._entries[session_key] = SessionEntry(
            session_key=session_key,
            session_id="sess-new",
            created_at=now,
            updated_at=now,
            origin=source,
        )
        runner._session_run_generation[session_key] = generation + 1
    else:
        # Forced shutdown must invalidate and identity-remove the early generic
        # sentinel before a cancellable stamped handle has been published.
        runner._interrupt_running_agents("test forced shutdown")
        assert runner._session_run_generation[session_key] > generation
        assert session_key not in runner._running_agents
        assert session_key not in runner._running_agents_ts
        assert session_key not in runner._active_session_leases
        active_lease.release.assert_called_once_with()
    release_history.set()
    await task

    runner._run_agent.assert_not_awaited()
    if transition == "reset":
        runner._release_running_agent_state(session_key)
    else:
        assert session_key not in runner._running_agents
        assert session_key not in runner._running_agents_ts


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
@pytest.mark.parametrize(
    "replacement_stage",
    [
        None,
        "reset_before_recursive",
        "shutdown_before_recursive",
        "worker",
        "publication",
    ],
)
async def test_fallback_eviction_rebuild_runs_queued_ordinary_followup(
    monkeypatch, tmp_path, replacement_stage
):
    import gateway.run as gateway_run
    import run_agent
    from gateway.platforms.base import MessageEvent, MessageType
    from gateway.session import SessionSource
    from tests.gateway.test_queued_native_image_session_key import (
        CaptureAdapter,
        _make_runner,
    )

    class _FallbackRebuildAgent:
        instances = []
        calls = []

        def __init__(self, **kwargs):
            type(self).instances.append(self)
            self.tools = []
            self.tool_progress_callback = kwargs.get(
                "tool_progress_callback"
            )
            self.model = "fallback-model"
            self.provider = "openrouter"

        def run_conversation(
            self, message, conversation_history=None, task_id=None
        ):
            type(self).calls.append((self, message))
            return {
                "final_response": f"done-{len(type(self).calls)}",
                "messages": [],
                "api_calls": 1,
            }

    monkeypatch.setattr(run_agent, "AIAgent", _FallbackRebuildAgent)
    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
    monkeypatch.setattr(
        gateway_run,
        "_resolve_runtime_agent_kwargs",
        lambda: {"api_key": "***"},
    )
    monkeypatch.setattr(
        gateway_run,
        "_resolve_gateway_model",
        lambda _config=None: "primary-model",
    )

    adapter = CaptureAdapter()
    runner = _make_runner(adapter)
    runner._model = "primary-model"
    source = SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="fallback-queue",
        chat_type="dm",
        user_id="user",
    )
    session_key = "agent:main:telegram:dm:fallback-queue"
    generation = runner._begin_session_run_generation(session_key)
    from gateway.run import _AGENT_PENDING_SENTINEL

    runner._running_agents[session_key] = _AGENT_PENDING_SENTINEL
    replacement_owner = MagicMock(name="boundary-replacement-owner")
    original_transfer = runner._transfer_running_agent_to_pending_rebuild
    original_publish = runner._publish_running_agent_at_execution_boundary
    original_executor = runner._run_in_executor_with_context
    executor_calls = 0

    def _transfer_or_replace(key, **kwargs):
        transferred = original_transfer(key, **kwargs)
        if transferred and replacement_stage in {
            "reset_before_recursive",
            "shutdown_before_recursive",
        }:
            runner._session_run_generation[key] = generation + 1
            if replacement_stage == "reset_before_recursive":
                runner._running_agents[key] = replacement_owner
            else:
                runner._running_agents.pop(key, None)
        return transferred

    def _publish_or_replace(key, **kwargs):
        concrete_owner = kwargs.get("concrete_owner")
        if (
            replacement_stage == "publication"
            and len(_FallbackRebuildAgent.instances) >= 2
            and concrete_owner is _FallbackRebuildAgent.instances[1]
        ):
            runner._session_run_generation[key] = generation + 1
            runner._running_agents[key] = replacement_owner
            return False
        return original_publish(key, **kwargs)

    async def _execute_or_replace(func, *args):
        nonlocal executor_calls
        executor_calls += 1
        if replacement_stage == "worker" and executor_calls == 2:
            runner._session_run_generation[session_key] = generation + 1
            runner._running_agents[session_key] = replacement_owner
        return await original_executor(func, *args)

    monkeypatch.setattr(
        runner,
        "_transfer_running_agent_to_pending_rebuild",
        _transfer_or_replace,
    )
    monkeypatch.setattr(
        runner,
        "_publish_running_agent_at_execution_boundary",
        _publish_or_replace,
    )
    monkeypatch.setattr(
        runner,
        "_run_in_executor_with_context",
        _execute_or_replace,
    )
    pending_event = MessageEvent(
        text="queued-after-fallback",
        message_type=MessageType.TEXT,
        source=source,
    )
    adapter._pending_messages[session_key] = pending_event

    result = await runner._run_agent(
        message="first-turn",
        context_prompt="",
        history=[],
        source=source,
        session_id="fallback-queue-session",
        session_key=session_key,
        run_generation=generation,
    )

    if replacement_stage is not None:
        assert result["final_response"] == "done-1"
        assert [message for _agent, message in _FallbackRebuildAgent.calls] == [
            "first-turn"
        ]
        assert adapter._pending_messages[session_key] is pending_event
        if replacement_stage == "shutdown_before_recursive":
            assert session_key not in runner._running_agents
        else:
            assert runner._running_agents[session_key] is replacement_owner
        expected_instances = (
            2 if replacement_stage == "publication" else 1
        )
        assert len(_FallbackRebuildAgent.instances) == expected_instances
    else:
        assert result["final_response"] == "done-2"
        assert len(_FallbackRebuildAgent.instances) == 2
        assert [message for _agent, message in _FallbackRebuildAgent.calls] == [
            "first-turn",
            "queued-after-fallback",
        ]
        assert _FallbackRebuildAgent.calls[0][0] is not _FallbackRebuildAgent.calls[1][0]
        assert session_key not in adapter._pending_messages


@pytest.mark.asyncio
async def test_proxy_publication_rejection_signals_no_execution(
    monkeypatch, tmp_path
):
    from gateway.run import _AGENT_PENDING_SENTINEL
    from gateway.session import SessionSource

    runner = _build_runner(monkeypatch, tmp_path, "all")
    runner.config.streaming.enabled = True
    runner._get_proxy_url = lambda: "http://proxy.invalid"
    adapter = SimpleNamespace(send_typing=AsyncMock())
    monkeypatch.setattr(runner, "_adapter_for_source", lambda _source: adapter)

    import gateway.stream_consumer as stream_consumer_module

    class _ObservedStreamConsumer:
        instances = []

        def __init__(self, **_kwargs):
            self.run_calls = 0
            self.finish_calls = 0
            self.__class__.instances.append(self)

        async def run(self):
            self.run_calls += 1
            await asyncio.Event().wait()

        def finish(self):
            self.finish_calls += 1

    monkeypatch.setattr(
        stream_consumer_module,
        "GatewayStreamConsumer",
        _ObservedStreamConsumer,
    )
    session_key = "agent:main:telegram:dm:proxy-publication-reject"
    generation = runner._begin_session_run_generation(session_key)
    runner._running_agents[session_key] = _AGENT_PENDING_SENTINEL
    runner._publish_running_agent_at_execution_boundary = MagicMock(
        return_value=False
    )
    source = SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="proxy-publication-reject",
        chat_type="dm",
        user_id="user",
    )

    result = await runner._run_agent_via_proxy(
        message="queued-turn",
        context_prompt="",
        history=[],
        source=source,
        session_id="proxy-publication-reject-session",
        session_key=session_key,
        run_generation=generation,
    )

    assert result["_execution_boundary_dropped"] is True
    assert session_key not in runner._running_agents
    assert len(_ObservedStreamConsumer.instances) == 1
    assert _ObservedStreamConsumer.instances[0].run_calls == 0
    assert _ObservedStreamConsumer.instances[0].finish_calls == 0


@pytest.mark.asyncio
async def test_proxy_rejection_does_not_release_newer_ordinary_pending_owner(
    monkeypatch, tmp_path
):
    from gateway.run import _AGENT_PENDING_SENTINEL
    from gateway.session import SessionSource

    runner = _build_runner(monkeypatch, tmp_path, "all")
    runner.config.streaming.enabled = False
    runner._get_proxy_url = lambda: "http://proxy.invalid"
    session_key = "agent:main:telegram:dm:proxy-aba-replacement"
    generation = runner._begin_session_run_generation(session_key)
    runner._running_agents[session_key] = _AGENT_PENDING_SENTINEL
    replacement_lease = MagicMock(name="proxy-replacement-lease")
    replacement_generation = None

    def _replace_then_reject(*_args, **_kwargs):
        nonlocal replacement_generation
        replacement_generation = runner._begin_session_run_generation(
            session_key
        )
        runner._running_agents[session_key] = _AGENT_PENDING_SENTINEL
        runner._running_agents_ts[session_key] = time.time()
        runner._active_session_leases[session_key] = replacement_lease
        return False

    monkeypatch.setattr(
        runner,
        "_publish_running_agent_at_execution_boundary",
        _replace_then_reject,
    )
    source = SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="proxy-aba-replacement",
        chat_type="dm",
        user_id="user",
    )

    result = await runner._run_agent_via_proxy(
        message="queued-turn",
        context_prompt="",
        history=[],
        source=source,
        session_id="proxy-aba-session",
        session_key=session_key,
        run_generation=generation,
    )

    assert result["_execution_boundary_dropped"] is True
    assert replacement_generation is not None
    assert replacement_generation > generation
    assert runner._running_agents[session_key] is _AGENT_PENDING_SENTINEL
    assert runner._running_agents_ts[session_key] > 0
    assert runner._active_session_leases[session_key] is replacement_lease
    replacement_lease.release.assert_not_called()


def test_execution_boundary_publication_is_single_use(monkeypatch, tmp_path):
    from gateway.run import _AGENT_PENDING_SENTINEL

    runner = _build_runner(monkeypatch, tmp_path, "all")
    session_key = "agent:main:telegram:dm:single-publication"
    generation = runner._begin_session_run_generation(session_key)
    concrete_owner = MagicMock(name="single-concrete-owner")
    runner._running_agents[session_key] = _AGENT_PENDING_SENTINEL

    assert runner._publish_running_agent_at_execution_boundary(
        session_key,
        run_generation=generation,
        pending_owner=_AGENT_PENDING_SENTINEL,
        concrete_owner=concrete_owner,
    )
    assert not runner._publish_running_agent_at_execution_boundary(
        session_key,
        run_generation=generation,
        pending_owner=_AGENT_PENDING_SENTINEL,
        concrete_owner=concrete_owner,
    )
    assert runner._running_agents[session_key] is concrete_owner


def test_restore_dequeued_event_preserves_newer_fifo(monkeypatch, tmp_path):
    from gateway.platforms.base import MessageEvent

    runner = _build_runner(monkeypatch, tmp_path, "all")
    session_key = "agent:main:telegram:dm:restore-fifo"
    old_event = MagicMock(spec=MessageEvent)
    newer_event = MagicMock(spec=MessageEvent)
    tail_event = MagicMock(spec=MessageEvent)
    adapter = SimpleNamespace(
        _pending_messages={session_key: newer_event}
    )
    runner._queued_events[session_key] = [tail_event, newer_event, old_event]

    assert runner._restore_dequeued_event_ahead_of_newer(
        session_key, adapter, old_event
    )
    assert adapter._pending_messages[session_key] is old_event
    assert runner._queued_events[session_key] == [newer_event, tail_event]

    assert runner._restore_dequeued_event_ahead_of_newer(
        session_key, adapter, old_event
    )
    assert adapter._pending_messages[session_key] is old_event
    assert runner._queued_events[session_key] == [newer_event, tail_event]


@pytest.mark.parametrize("replacement_wins", [False, True])
def test_recursive_rebuild_transfers_only_exact_completed_owner(
    monkeypatch, tmp_path, replacement_wins
):
    from gateway.run import _AGENT_PENDING_SENTINEL

    runner = _build_runner(monkeypatch, tmp_path, "all")
    session_key = "agent:main:telegram:dm:recursive-fallback-rebuild"
    generation = runner._begin_session_run_generation(session_key)
    old_owner = MagicMock(name="fallback-old-owner")
    new_owner = MagicMock(name="fallback-rebuilt-owner")
    replacement_owner = MagicMock(name="replacement-owner")
    runner._running_agents[session_key] = (
        replacement_owner if replacement_wins else old_owner
    )

    transferred = runner._transfer_running_agent_to_pending_rebuild(
        session_key,
        run_generation=generation,
        expected_owner=old_owner,
    )

    if replacement_wins:
        assert transferred is False
        assert runner._running_agents[session_key] is replacement_owner
    else:
        assert transferred is True
        assert runner._running_agents[session_key] is _AGENT_PENDING_SENTINEL
        assert runner._publish_running_agent_at_execution_boundary(
            session_key,
            run_generation=generation,
            pending_owner=_AGENT_PENDING_SENTINEL,
            concrete_owner=new_owner,
        )
        assert runner._running_agents[session_key] is new_owner
        runner._release_running_agent_if_identity(
            session_key, new_owner
        )


@pytest.mark.parametrize("winner", ["shutdown", "publication"])
def test_forced_shutdown_linearizes_ordinary_pending_publication(
    monkeypatch, tmp_path, winner
):
    from gateway.run import _AGENT_PENDING_SENTINEL

    runner = _build_runner(monkeypatch, tmp_path, "all")
    session_key = "agent:main:telegram:dm:ordinary-shutdown-publication"
    generation = runner._begin_session_run_generation(session_key)
    concrete_owner = MagicMock(name="ordinary-concrete-owner")
    runner._running_agents[session_key] = _AGENT_PENDING_SENTINEL
    runner._running_agents_ts[session_key] = time.time()

    if winner == "shutdown":
        captured = runner._capture_running_agents_for_forced_shutdown(
            "test forced shutdown"
        )
        published = runner._publish_running_agent_at_execution_boundary(
            session_key,
            run_generation=generation,
            pending_owner=_AGENT_PENDING_SENTINEL,
            concrete_owner=concrete_owner,
        )
        assert captured == []
        assert published is False
        assert not runner._process_notification_run_is_current(
            expected_session_id="",
            session_key=session_key,
            run_generation=generation,
        )
        assert session_key not in runner._running_agents
        concrete_owner.interrupt.assert_not_called()
    else:
        published = runner._publish_running_agent_at_execution_boundary(
            session_key,
            run_generation=generation,
            pending_owner=_AGENT_PENDING_SENTINEL,
            concrete_owner=concrete_owner,
        )
        captured = runner._capture_running_agents_for_forced_shutdown(
            "test forced shutdown"
        )
        assert published is True
        assert captured == [(session_key, concrete_owner)]
        runner._interrupt_captured_running_agents(
            captured, "test forced shutdown"
        )
        concrete_owner.interrupt.assert_called_once_with(
            "test forced shutdown"
        )
        runner._release_running_agent_if_identity(
            session_key, concrete_owner
        )



def test_forced_shutdown_linearizes_pending_to_concrete_publication(
    monkeypatch, tmp_path
):
    from gateway.run import _StampedProcessPendingHandle

    runner = _build_runner(monkeypatch, tmp_path, "all")
    session_key = "agent:main:telegram:dm:shutdown-publication-cas"
    generation = runner._begin_session_run_generation(session_key)
    pending_owner = _StampedProcessPendingHandle()
    concrete_owner = MagicMock(name="concrete-owner")
    publication_attempted = threading.Event()
    publication_finished = threading.Event()
    publisher = None
    original_interrupt = pending_owner.interrupt

    runner._running_agents[session_key] = pending_owner
    runner._stamped_process_running_agents[session_key] = pending_owner

    def _publish_concrete():
        publication_attempted.set()
        with runner._running_agent_state_lock:
            if (
                not pending_owner.cancelled
                and runner._is_session_run_current(session_key, generation)
                and runner._running_agents.get(session_key) is pending_owner
            ):
                runner._running_agents[session_key] = concrete_owner
                runner._stamped_process_running_agents[session_key] = (
                    concrete_owner
                )
        publication_finished.set()

    def _interrupt_after_publication_attempt(reason):
        nonlocal publisher
        publisher = threading.Thread(target=_publish_concrete)
        publisher.start()
        assert publication_attempted.wait(timeout=5)
        original_interrupt(reason)

    pending_owner.interrupt = _interrupt_after_publication_attempt
    runner._interrupt_running_agents("test forced shutdown")
    assert publisher is not None
    publisher.join(timeout=5)

    assert publication_finished.is_set()
    assert pending_owner.cancelled
    assert runner._session_run_generation[session_key] > generation
    assert session_key not in runner._running_agents
    assert session_key not in runner._stamped_process_running_agents
    concrete_owner.interrupt.assert_not_called()


@pytest.mark.asyncio
async def test_forced_shutdown_concrete_owner_finalizer_uses_exact_identity(
    monkeypatch, tmp_path
):
    runner = _build_runner(monkeypatch, tmp_path, "all")
    session_key = "agent:main:telegram:dm:forced-owner-finalizer"
    concrete_owner = MagicMock(name="forced-concrete-owner")
    lease = MagicMock()
    event = SimpleNamespace()
    identity_release_calls = []
    original_identity_release = runner._release_running_agent_if_identity

    def _observe_identity_release(key, owner):
        result = original_identity_release(key, owner)
        identity_release_calls.append((key, owner, result))
        return result

    monkeypatch.setattr(
        runner,
        "_release_running_agent_if_identity",
        _observe_identity_release,
    )

    async def _force_during_inner(event_arg, **_kwargs):
        claim = event_arg._hermes_stamped_process_release_claim
        generation = runner._begin_session_run_generation(session_key)
        claim.generation = generation
        runner._running_agents[session_key] = concrete_owner
        runner._stamped_process_running_agents[session_key] = concrete_owner
        runner._running_agents_ts[session_key] = time.time()
        runner._active_session_leases[session_key] = lease
        runner._stamped_process_release_events[session_key] = claim
        runner._interrupt_running_agents("test forced shutdown")
        assert claim.forced_shutdown_owner is concrete_owner
        assert runner._running_agents[session_key] is concrete_owner
        return None

    monkeypatch.setattr(
        runner,
        "_run_and_deliver_stamped_process_event_inner",
        _force_during_inner,
    )

    await GatewayRunner._run_and_deliver_stamped_process_event(
        runner,
        event,
        adapter=SimpleNamespace(),
        session_key=session_key,
    )

    assert session_key not in runner._running_agents
    assert session_key not in runner._stamped_process_running_agents
    assert session_key not in runner._running_agents_ts
    assert session_key not in runner._active_session_leases
    assert identity_release_calls == [
        (session_key, concrete_owner, True)
    ]
    lease.release.assert_called_once_with()


@pytest.mark.asyncio
async def test_shutdown_does_not_mark_stamped_pending_handle_resume_pending(
    monkeypatch,
    tmp_path,
):
    from unittest.mock import patch

    import gateway.run as gateway_run
    from gateway.run import _StampedProcessPendingHandle
    from tests.gateway.test_restart_resume_pending import make_restart_runner

    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
    runner, adapter = make_restart_runner()
    adapter.disconnect = AsyncMock()
    runner._restart_drain_timeout = 0.01
    session_key = "agent:main:telegram:dm:stamped-pending"
    pending_owner = _StampedProcessPendingHandle()
    runner._running_agents = {session_key: pending_owner}
    runner._stamped_process_running_agents = {
        session_key: pending_owner
    }
    session_store = MagicMock()
    session_store.mark_resume_pending = MagicMock(return_value=True)
    runner.session_store = session_store
    runner._finalize_shutdown_agents = AsyncMock()
    runner._increment_restart_failure_counts = MagicMock()

    with patch("gateway.status.remove_pid_file"), patch(
        "gateway.status.write_runtime_status"
    ):
        await runner.stop()

    session_store.mark_resume_pending.assert_not_called()
    runner._finalize_shutdown_agents.assert_awaited_once_with({})
    runner._increment_restart_failure_counts.assert_not_called()
    assert session_key not in runner._running_agents
    assert session_key not in runner._stamped_process_running_agents
    assert (tmp_path / ".clean_shutdown").exists()


@pytest.mark.asyncio
async def test_clean_concrete_to_pending_shutdown_does_not_count_restart_failure(
    monkeypatch,
    tmp_path,
):
    from unittest.mock import patch

    import gateway.run as gateway_run
    from gateway.run import _AGENT_PENDING_SENTINEL
    from tests.gateway.test_restart_resume_pending import make_restart_runner

    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
    runner, adapter = make_restart_runner()
    adapter.disconnect = AsyncMock()
    session_key = "agent:main:telegram:dm:concrete-to-pending-clean"
    concrete_owner = MagicMock(name="completed-before-capture")
    runner._running_agents = {session_key: concrete_owner}
    runner._running_agents_ts[session_key] = time.time()
    runner._finalize_shutdown_agents = AsyncMock()
    runner._increment_restart_failure_counts = MagicMock()

    async def _transition_during_drain(_timeout):
        runner._running_agents[session_key] = _AGENT_PENDING_SENTINEL
        return {session_key: concrete_owner}, True

    monkeypatch.setattr(
        runner, "_drain_active_agents", _transition_during_drain
    )

    with patch("gateway.status.remove_pid_file"), patch(
        "gateway.status.write_runtime_status"
    ):
        await runner.stop()

    runner._increment_restart_failure_counts.assert_not_called()
    runner._finalize_shutdown_agents.assert_awaited_once_with(
        {session_key: concrete_owner}
    )
    assert (tmp_path / ".clean_shutdown").exists()


@pytest.mark.asyncio
async def test_forced_shutdown_marks_owner_promoted_at_capture_epoch(
    monkeypatch,
):
    from unittest.mock import patch

    from gateway.run import _StampedProcessPendingHandle
    from tests.gateway.test_restart_resume_pending import make_restart_runner

    runner, adapter = make_restart_runner()
    adapter.disconnect = AsyncMock()
    runner._restart_drain_timeout = 0.01
    session_key = "agent:main:telegram:dm:promote-at-forced-epoch"
    generation = runner._begin_session_run_generation(session_key)
    pending_owner = _StampedProcessPendingHandle()
    concrete_owner = MagicMock(name="promoted-concrete-owner")
    runner._running_agents = {session_key: pending_owner}
    runner._stamped_process_running_agents = {
        session_key: pending_owner
    }
    session_store = MagicMock()
    session_store.mark_resume_pending = MagicMock(return_value=True)
    runner.session_store = session_store
    runner._finalize_shutdown_agents = AsyncMock()
    runner._increment_restart_failure_counts = MagicMock()
    original_capture = runner._capture_running_agents_for_forced_shutdown
    promoted = False

    def _promote_then_capture(reason):
        nonlocal promoted
        if not promoted:
            promoted = True
            with runner._running_agent_state_lock:
                assert runner._is_session_run_current(
                    session_key, generation
                )
                runner._running_agents[session_key] = concrete_owner
                runner._stamped_process_running_agents[session_key] = (
                    concrete_owner
                )
        return original_capture(reason)

    def _interrupt_and_unwind(_reason):
        runner._release_running_agent_if_identity(
            session_key, concrete_owner
        )

    concrete_owner.interrupt.side_effect = _interrupt_and_unwind
    monkeypatch.setattr(
        runner,
        "_capture_running_agents_for_forced_shutdown",
        _promote_then_capture,
    )

    with patch("gateway.status.remove_pid_file"), patch(
        "gateway.status.write_runtime_status"
    ):
        await runner.stop()

    assert promoted
    marked_keys = {
        call.args[0]
        for call in session_store.mark_resume_pending.call_args_list
    }
    assert session_key in marked_keys
    concrete_owner.interrupt.assert_called_once()
    finalized_agents = (
        runner._finalize_shutdown_agents.await_args.args[0]
    )
    assert finalized_agents[session_key] is concrete_owner
    runner._increment_restart_failure_counts.assert_called_once_with(
        {session_key}
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("transition", ["replacement", "cancel", "shutdown"])
async def test_reset_between_cache_check_and_lock_preserves_replacement_agent(
    monkeypatch, tmp_path, transition
):
    from datetime import datetime, timezone

    import gateway.run as gateway_run
    import run_agent
    from gateway.session import SessionEntry, SessionSource

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
    runner.session_store._entries[session_key] = SessionEntry(
        session_key=session_key,
        session_id="sess-old",
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
        origin=source,
    )
    runner.session_store._save = MagicMock()
    generation = runner._begin_session_run_generation(session_key)
    runner._agent_cache = {}

    publication_waiting = threading.Event()
    publication_finished = threading.Event()
    release_publication = threading.Event()

    class _ObservedRunningStateLock:
        def __init__(self):
            self._lock = threading.RLock()

        def __enter__(self):
            self._lock.acquire()
            return self

        def __exit__(self, *_exc):
            self._lock.release()
            if threading.current_thread() is not threading.main_thread():
                publication_finished.set()

    runner._running_agent_state_lock = _ObservedRunningStateLock()
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
        publication_waiting.set()
        assert release_publication.wait(timeout=5)
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

    if transition == "cancel":
        task.cancel()
        runner._release_running_agent_state(
            session_key, run_generation=generation
        )
        release_publication.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert await asyncio.to_thread(publication_finished.wait, 5)
        assert session_key not in runner._running_agents
        assert session_key not in runner._stamped_process_running_agents
    elif transition == "shutdown":
        runner._running = False
        runner._draining = True
        runner._interrupt_running_agents("test shutdown timeout")
        release_publication.set()

        result = await task
        assert result["final_response"] != "must not run"
        assert session_key not in runner._running_agents
        assert session_key not in runner._stamped_process_running_agents
    else:
        # /new wins after lookup but before stale publication acquires the lock.
        runner._invalidate_session_run_generation(session_key, reason="test-reset")
        runner.session_store._entries[session_key] = SimpleNamespace(
            session_id="sess-new"
        )
        replacement_agent = MagicMock(name="replacement-agent")
        replacement_running_agent = MagicMock(name="replacement-running-agent")
        runner._running_agents[session_key] = replacement_running_agent
        runner._stamped_process_running_agents[session_key] = replacement_running_agent
        runner._agent_cache[session_key] = (
            replacement_agent,
            "replacement-sig",
            0,
            "sess-new",
        )
        release_publication.set()

        result = await task
        assert result["final_response"] != "must not run"
        assert runner._agent_cache[session_key][0] is replacement_agent
        assert runner._running_agents[session_key] is replacement_running_agent
        assert (
            runner._stamped_process_running_agents[session_key]
            is replacement_running_agent
        )
    stale_agent.run_conversation.assert_not_called()
    if transition == "cancel":
        assert runner._agent_cache[session_key][0] is stale_agent
        stale_agent.close.assert_not_called()
    else:
        stale_agent.close.assert_called_once_with()
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
async def test_reused_cached_agent_is_retained_when_caller_cancel_removes_pending_owner(
    monkeypatch, tmp_path
):
    from datetime import datetime, timezone

    import gateway.run as gateway_run
    import run_agent
    from gateway.run import _AGENT_PENDING_SENTINEL
    from gateway.session import SessionEntry, SessionSource

    runner = _build_runner(monkeypatch, tmp_path, "all")
    runner.adapters[Platform.TELEGRAM].get_pending_message = MagicMock(
        return_value=None
    )
    session_key = "agent:main:telegram:dm:reused-cancel"
    session_id = "sess-reused-cancel"
    source = SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="reused-cancel",
        chat_type="dm",
    )
    runner.session_store._entries[session_key] = SessionEntry(
        session_key=session_key,
        session_id=session_id,
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
        origin=source,
    )
    runner.session_store._save = MagicMock()
    runner._session_db = None
    generation = runner._begin_session_run_generation(session_key)
    runner._running_agents[session_key] = _AGENT_PENDING_SENTINEL

    cached_agent = MagicMock(name="reused-cached-agent")
    cached_agent.model = "test-model"
    cached_agent.tools = []
    cached_agent._fallback_activated = False
    cached_agent._rate_limited_until = 0.0
    cached_agent._fallback_chain = []
    cached_agent.context_compressor = SimpleNamespace(
        context_length=100_000,
        last_prompt_tokens=0,
    )
    cached_agent.run_conversation.return_value = {
        "final_response": "must not run",
        "messages": [],
    }
    runner._agent_cache[session_key] = (
        cached_agent,
        "cached-sig",
        None,
        session_id,
    )
    monkeypatch.setattr(
        runner, "_agent_config_signature", lambda *_a, **_kw: "cached-sig"
    )
    monkeypatch.setattr(
        run_agent,
        "AIAgent",
        lambda *_a, **_kw: pytest.fail("cached agent should be reused"),
    )
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

    publication_entered = threading.Event()
    release_publication = threading.Event()
    retention_checked = threading.Event()
    original_publish = runner._publish_running_agent_at_execution_boundary
    original_retained = runner._agent_is_retained_after_execution_rejection

    def _blocked_publish(*args, **kwargs):
        publication_entered.set()
        assert release_publication.wait(timeout=5)
        return original_publish(*args, **kwargs)

    def _observe_retention(*args, **kwargs):
        result = original_retained(*args, **kwargs)
        retention_checked.set()
        return result

    monkeypatch.setattr(
        runner, "_publish_running_agent_at_execution_boundary", _blocked_publish
    )
    monkeypatch.setattr(
        runner, "_agent_is_retained_after_execution_rejection", _observe_retention
    )

    task = asyncio.create_task(
        runner._run_agent(
            message="cancel before execution",
            context_prompt="",
            history=[],
            source=source,
            session_id=session_id,
            session_key=session_key,
            run_generation=generation,
        )
    )
    assert await asyncio.to_thread(publication_entered.wait, 5)
    task.cancel()
    assert runner._release_running_agent_state(
        session_key, run_generation=generation
    )
    release_publication.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert session_key not in runner._running_agents
    assert await asyncio.to_thread(retention_checked.wait, 5)

    assert runner._agent_cache[session_key][0] is cached_agent
    assert cached_agent._gateway_execution_rejected_pending_cleanup is True
    cached_agent.run_conversation.assert_not_called()
    cached_agent.close.assert_not_called()
    cached_agent.shutdown_memory_provider.assert_not_called()

    next_generation = runner._begin_session_run_generation(session_key)
    runner._running_agents[session_key] = _AGENT_PENDING_SENTINEL
    cached_agent.run_conversation.return_value = {
        "final_response": "reused successfully",
        "messages": [],
        "api_calls": 1,
        "tools": [],
    }
    next_result = await runner._run_agent(
        message="next real turn",
        context_prompt="",
        history=[],
        source=source,
        session_id=session_id,
        session_key=session_key,
        run_generation=next_generation,
    )

    assert next_result["final_response"] == "reused successfully"
    cached_agent.run_conversation.assert_called_once()
    assert cached_agent._gateway_execution_rejected_pending_cleanup is False
    cached_agent.close.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize("replacement_kind", ["ordinary", "stamped"])
@pytest.mark.parametrize("eviction_timing", ["before_retention", "after_retention"])
async def test_reset_soft_eviction_hard_cleans_fresh_rejected_agent(
    monkeypatch, tmp_path, replacement_kind, eviction_timing
):
    from datetime import datetime, timezone

    import gateway.run as gateway_run
    import run_agent
    from gateway.run import (
        _AGENT_PENDING_SENTINEL,
        _StampedProcessPendingHandle,
    )
    from gateway.session import SessionEntry, SessionSource

    runner = _build_runner(monkeypatch, tmp_path, "all")
    runner.adapters[Platform.TELEGRAM].get_pending_message = MagicMock(
        return_value=None
    )
    session_key = "agent:main:telegram:dm:fresh-reset-after-cache"
    session_id = "sess-fresh-reset-after-cache"
    source = SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="fresh-reset-after-cache",
        chat_type="dm",
    )
    runner.session_store._entries[session_key] = SessionEntry(
        session_key=session_key,
        session_id=session_id,
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
        origin=source,
    )
    runner.session_store._save = MagicMock()
    runner._session_db = None
    generation = runner._begin_session_run_generation(session_key)
    runner._running_agents[session_key] = _AGENT_PENDING_SENTINEL

    fresh_agent = MagicMock(name="fresh-reset-rejected-agent")
    fresh_agent._gateway_execution_rejected_pending_cleanup = False
    fresh_agent.model = "test-model"
    fresh_agent.tools = []
    fresh_agent.context_compressor = SimpleNamespace(
        context_length=100_000,
        last_prompt_tokens=0,
    )
    fresh_agent.run_conversation.return_value = {
        "final_response": "must not run",
        "messages": [],
    }
    monkeypatch.setattr(run_agent, "AIAgent", lambda *_a, **_kw: fresh_agent)
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

    publication_entered = threading.Event()
    release_publication = threading.Event()
    retention_seen = threading.Event()
    release_retention = threading.Event()
    cleanup_done = threading.Event()
    original_publish = runner._publish_running_agent_at_execution_boundary
    original_retained = runner._agent_is_retained_after_execution_rejection
    original_cleanup = runner._cleanup_agent_resources

    def _blocked_publish(*args, **kwargs):
        publication_entered.set()
        assert release_publication.wait(timeout=5)
        return original_publish(*args, **kwargs)

    def _observe_retention(*args, **kwargs):
        result = original_retained(*args, **kwargs)
        if eviction_timing == "after_retention":
            assert result is True
            retention_seen.set()
            assert release_retention.wait(timeout=5)
        return result

    def _observe_cleanup(agent):
        try:
            return original_cleanup(agent)
        finally:
            if agent is fresh_agent:
                cleanup_done.set()

    monkeypatch.setattr(
        runner, "_publish_running_agent_at_execution_boundary", _blocked_publish
    )
    monkeypatch.setattr(
        runner, "_agent_is_retained_after_execution_rejection", _observe_retention
    )
    monkeypatch.setattr(runner, "_cleanup_agent_resources", _observe_cleanup)

    task = asyncio.create_task(
        runner._run_agent(
            message="reset before execution",
            context_prompt="",
            history=[],
            source=source,
            session_id=session_id,
            session_key=session_key,
            run_generation=generation,
        )
    )
    assert await asyncio.to_thread(publication_entered.wait, 5)
    assert runner._agent_cache[session_key][0] is fresh_agent
    runner._invalidate_session_run_generation(
        session_key, reason="test-reset-after-cache"
    )
    runner._release_running_agent_state(session_key)

    replacement_agent = MagicMock(name="replacement-cache-agent")
    replacement_lease = MagicMock(name="replacement-pending-lease")
    replacement_generation = None
    replacement_owner = None

    def _install_replacement():
        nonlocal replacement_generation, replacement_owner
        replacement_generation = runner._begin_session_run_generation(
            session_key
        )
        replacement_owner = (
            _AGENT_PENDING_SENTINEL
            if replacement_kind == "ordinary"
            else _StampedProcessPendingHandle()
        )
        runner._running_agents[session_key] = replacement_owner
        runner._running_agents_ts[session_key] = time.time()
        runner._active_session_leases[session_key] = replacement_lease
        runner._agent_cache[session_key] = (
            replacement_agent,
            "replacement-sig",
            None,
            "sess-replacement",
        )
        if replacement_kind == "stamped":
            runner._stamped_process_running_agents[session_key] = (
                replacement_owner
            )

    if eviction_timing == "before_retention":
        runner._evict_cached_agent(session_key)
        _install_replacement()
        release_publication.set()
    else:
        release_publication.set()
        assert await asyncio.to_thread(retention_seen.wait, 5)
        runner._evict_cached_agent(session_key)
        _install_replacement()
        release_retention.set()

    result = await task
    assert await asyncio.to_thread(cleanup_done.wait, 5)
    assert result["_execution_boundary_dropped"] is True
    assert replacement_generation is not None
    assert replacement_generation > generation
    fresh_agent.run_conversation.assert_not_called()
    fresh_agent.shutdown_memory_provider.assert_called_once()
    fresh_agent.close.assert_called_once_with()
    assert runner._agent_cache[session_key][0] is replacement_agent
    assert runner._running_agents[session_key] is replacement_owner
    assert runner._running_agents_ts[session_key] > 0
    assert runner._active_session_leases[session_key] is replacement_lease
    replacement_lease.release.assert_not_called()
    if replacement_kind == "stamped":
        assert (
            runner._stamped_process_running_agents[session_key]
            is replacement_owner
        )


@pytest.mark.asyncio
async def test_ordinary_outer_finalizer_preserves_newer_stamped_owner(
    monkeypatch, tmp_path
):
    from gateway.platforms.base import MessageEvent, MessageType
    from gateway.session import SessionSource

    runner = _build_runner(monkeypatch, tmp_path, "all")
    monkeypatch.setattr(runner, "_is_user_authorized", lambda _source: True)
    source = SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="ordinary-tail-stamped-replacement",
        chat_type="dm",
        user_id="user",
    )
    session_key = runner._session_key_for_source(source)
    replacement_owner = MagicMock(name="newer-stamped-owner")
    replacement_lease = MagicMock(name="newer-stamped-lease")
    observed_generations = []

    async def _replace_after_inner_release(
        _event, _source, key, old_generation
    ):
        observed_generations.append(old_generation)
        assert runner._release_running_agent_state(
            key, run_generation=old_generation
        )
        replacement_generation = runner._begin_session_run_generation(key)
        observed_generations.append(replacement_generation)
        runner._running_agents[key] = replacement_owner
        runner._stamped_process_running_agents[key] = replacement_owner
        runner._running_agents_ts[key] = time.time()
        runner._active_session_leases[key] = replacement_lease
        return {"final_response": "", "messages": []}

    monkeypatch.setattr(
        runner, "_handle_message_with_agent", _replace_after_inner_release
    )
    event = MessageEvent(
        text="ordinary turn",
        message_type=MessageType.TEXT,
        source=source,
    )

    await runner._handle_message(event)

    assert observed_generations[1] > observed_generations[0]
    assert runner._running_agents[session_key] is replacement_owner
    assert (
        runner._stamped_process_running_agents[session_key]
        is replacement_owner
    )
    assert runner._active_session_leases[session_key] is replacement_lease
    replacement_lease.release.assert_not_called()


@pytest.mark.asyncio
async def test_local_process_agent_waits_for_paired_publication(
    monkeypatch, tmp_path
):
    import gateway.run as gateway_run
    import run_agent
    from gateway.session import SessionSource

    runner = _build_runner(monkeypatch, tmp_path, "all")
    adapter = runner.adapters[Platform.TELEGRAM]
    adapter.get_pending_message = MagicMock(return_value=None)
    session_key = "agent:main:telegram:dm:paired-publication"
    source = SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="paired-publication",
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

    history_entered = threading.Event()
    release_history = threading.Event()
    original_build_history = gateway_run._build_gateway_agent_history

    def _blocked_build_history(*args, **kwargs):
        history_entered.set()
        assert release_history.wait(timeout=5)
        return original_build_history(*args, **kwargs)

    monkeypatch.setattr(
        gateway_run, "_build_gateway_agent_history", _blocked_build_history
    )

    task = asyncio.create_task(
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
    assert await asyncio.to_thread(history_entered.wait, 5)
    await asyncio.sleep(0.12)
    running_owner, stamped_owner = runner._running_agent_ownership_snapshot(
        session_key
    )
    try:
        assert running_owner is stamped_owner
        assert running_owner is not process_agent
    finally:
        release_history.set()

    result = await task
    assert result["final_response"] == "process result"
    assert runner._running_agents[session_key] is process_agent
    assert runner._stamped_process_running_agents[session_key] is process_agent


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
