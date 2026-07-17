from __future__ import annotations

import io
import sys
import threading
import types
import uuid
from unittest.mock import Mock, call

import pytest

import hermes_cli.oneshot as oneshot


class RecordingStdout(io.StringIO):
    def __init__(self, events: list[tuple[str, object]]) -> None:
        super().__init__()
        self.events = events

    def write(self, value: str) -> int:
        self.events.append(("write", value))
        return super().write(value)

    def flush(self) -> None:
        self.events.append(("flush", None))
        return super().flush()


def test_delivery_orders_write_flush_and_callback_and_is_idempotent():
    events: list[tuple[str, object]] = []
    delivery = oneshot._FinalDelivery(
        RecordingStdout(events),
        on_delivered=lambda: events.append(("on_delivered", None)),
    )

    assert delivery.deliver("final answer") is True
    assert delivery.deliver("duplicate") is False
    assert delivery.delivered is True
    assert events == [
        ("write", "final answer"),
        ("write", "\n"),
        ("flush", None),
        ("on_delivered", None),
    ]


def test_delivery_preserves_existing_newline_and_rejects_blank_without_claiming():
    events: list[tuple[str, object]] = []
    stream = RecordingStdout(events)
    delivery = oneshot._FinalDelivery(stream, lambda: events.append(("delivered", None)))

    assert delivery.deliver(" \n\t") is False
    assert delivery.deliver("ready\n") is True
    assert stream.getvalue() == "ready\n"
    assert events == [("write", "ready\n"), ("flush", None), ("delivered", None)]


def test_delivery_concurrent_calls_print_exactly_once():
    events: list[tuple[str, object]] = []
    stream = RecordingStdout(events)
    delivery = oneshot._FinalDelivery(stream, lambda: None)
    barrier = threading.Barrier(8)
    results: list[bool] = []

    def worker(index: int) -> None:
        barrier.wait()
        results.append(delivery.deliver(f"answer-{index}"))

    threads = [threading.Thread(target=worker, args=(index,)) for index in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert results.count(True) == 1
    assert results.count(False) == 7
    assert sum(1 for event, _ in events if event == "flush") == 1
    assert stream.getvalue().count("answer-") == 1


@pytest.mark.parametrize("failure_point", ["write", "flush"])
def test_delivery_io_failure_is_retained_and_never_retried(failure_point):
    failure = OSError(failure_point)

    class BrokenStream(RecordingStdout):
        def __init__(self):
            super().__init__([])
            self.write_calls = 0
            self.flush_calls = 0

        def write(self, value):
            self.write_calls += 1
            if failure_point == "write":
                raise failure
            return super().write(value)

        def flush(self):
            self.flush_calls += 1
            if failure_point == "flush":
                raise failure
            return super().flush()

    stream = BrokenStream()
    delivery = oneshot._FinalDelivery(stream, lambda: None)

    with pytest.raises(OSError) as first:
        delivery.deliver("answer\n")
    first_counts = (stream.write_calls, stream.flush_calls)
    with pytest.raises(OSError) as second:
        delivery.deliver("fallback")

    assert first.value is failure
    assert second.value is failure
    assert (stream.write_calls, stream.flush_calls) == first_counts
    assert delivery.delivered is False


def test_delivery_swallows_ordinary_callback_failure_after_flush(caplog):
    stream = RecordingStdout([])

    def fail():
        raise RuntimeError("private detail")

    delivery = oneshot._FinalDelivery(stream, fail)
    assert delivery.deliver("answer") is True
    assert delivery.delivered is True
    assert stream.getvalue() == "answer\n"
    assert "completion callback failed" in caplog.text


@pytest.mark.parametrize("failure", [KeyboardInterrupt(), SystemExit(19)])
def test_delivery_propagates_control_flow_callback_after_flush_without_retry(failure):
    events: list[tuple[str, object]] = []
    stream = RecordingStdout(events)

    def fail():
        raise failure

    delivery = oneshot._FinalDelivery(stream, fail)
    with pytest.raises(type(failure)) as raised:
        delivery.deliver("answer")
    assert raised.value is failure
    assert delivery.delivered is True
    assert stream.getvalue() == "answer\n"
    assert delivery.deliver("fallback") is False
    assert sum(1 for event, _ in events if event == "flush") == 1


@pytest.mark.parametrize("raised", [None, RuntimeError("boom"), KeyboardInterrupt(), SystemExit(4)])
def test_transient_hook_is_first_and_removes_only_it_in_place(monkeypatch, raised):
    events: list[tuple[str, object]] = []
    stream = RecordingStdout(events)
    resources = oneshot._OneshotResources()
    delivery = oneshot._FinalDelivery(stream, lambda: events.append(("delivered", None)))

    def existing(**_kwargs):
        events.append(("existing", None))

    callbacks = [existing]
    manager = types.SimpleNamespace(_hooks={"post_llm_call": callbacks})
    monkeypatch.setattr("hermes_cli.plugins.get_plugin_manager", lambda: manager)

    def execute_body():
        with oneshot._install_final_delivery_hook(delivery, resources):
            assert manager._hooks["post_llm_call"] is callbacks
            assert callbacks[0] is not existing
            for callback in list(callbacks):
                callback(assistant_response="transformed final")
            if raised is not None:
                raise raised

    if raised is None:
        execute_body()
    else:
        with pytest.raises(type(raised)) as caught:
            execute_body()
        assert caught.value is raised

    assert manager._hooks["post_llm_call"] is callbacks
    assert callbacks == [existing]
    assert events.index(("flush", None)) < events.index(("existing", None))


def _install_cleanup_modules(monkeypatch, events):
    def module(name, **attrs):
        value = types.ModuleType(name)
        for key, item in attrs.items():
            setattr(value, key, item)
        monkeypatch.setitem(sys.modules, name, value)
        return value

    interrupt_all = Mock(side_effect=lambda **_kwargs: events.append("interrupt"))
    kill_all = Mock(side_effect=lambda **_kwargs: events.append("processes"))
    shutdown_mcp = Mock(side_effect=lambda: events.append("mcp"))
    shutdown_aux = Mock(side_effect=lambda: events.append("aux"))
    module("tools.async_delegation", interrupt_all=interrupt_all)
    module("tools.process_registry", process_registry=types.SimpleNamespace(kill_all=kill_all))
    module("tools.mcp_tool", shutdown_mcp_servers=shutdown_mcp)
    module("agent.auxiliary_client", shutdown_cached_clients=shutdown_aux)
    return interrupt_all, kill_all, shutdown_mcp, shutdown_aux


def test_resources_use_explicit_task_id_and_close_in_exact_order_once(monkeypatch):
    events: list[str] = []
    task_id = str(uuid.uuid4())
    interrupt_all, kill_all, shutdown_mcp, shutdown_aux = _install_cleanup_modules(
        monkeypatch, events
    )
    monkeypatch.setattr(
        oneshot._OneshotResources,
        "_arm_watchdog_once",
        lambda self, exit_code=70: events.append("watchdog"),
    )

    memory = types.SimpleNamespace(
        flush_pending=Mock(side_effect=lambda **_kwargs: events.append("memory_flush"))
    )
    agent = types.SimpleNamespace(
        session_id="compression-child",
        _session_messages=[{"role": "user", "content": "hello"}],
        _memory_manager=memory,
        shutdown_memory_provider=Mock(side_effect=lambda _messages: events.append("memory_stop")),
        _cleanup_task_resources=Mock(side_effect=lambda _task_id: events.append("task_resources")),
        close=Mock(side_effect=lambda: events.append("agent")),
    )
    db = types.SimpleNamespace(close=Mock(side_effect=lambda: events.append("db")))
    resources = oneshot._OneshotResources()
    resources.agent = agent
    resources.session_db = db
    resources.task_id = task_id

    resources.close()
    resources.close()

    assert events == [
        "watchdog",
        "interrupt",
        "memory_flush",
        "memory_stop",
        "processes",
        "task_resources",
        "agent",
        "mcp",
        "aux",
        "db",
    ]
    assert interrupt_all.call_args_list == [call(reason="oneshot shutdown")]
    assert memory.flush_pending.call_args_list == [call(timeout=10)]
    assert kill_all.call_args_list == [call(task_id=task_id)]
    assert agent._cleanup_task_resources.call_args_list == [call(task_id)]
    assert agent.shutdown_memory_provider.call_args_list == [
        call([{"role": "user", "content": "hello"}])
    ]
    assert agent.close.call_count == shutdown_mcp.call_count == shutdown_aux.call_count == 1
    assert db.close.call_count == 1


def test_resources_continue_after_failures_and_preserve_control_flow(monkeypatch):
    events: list[str] = []
    _install_cleanup_modules(monkeypatch, events)
    monkeypatch.setattr(
        oneshot._OneshotResources,
        "_arm_watchdog_once",
        lambda self, exit_code=70: events.append("watchdog"),
    )
    agent = types.SimpleNamespace(
        session_id="child",
        _session_messages=[],
        _memory_manager=types.SimpleNamespace(
            flush_pending=Mock(side_effect=KeyboardInterrupt())
        ),
        shutdown_memory_provider=Mock(side_effect=lambda _messages: events.append("memory_stop")),
        _cleanup_task_resources=Mock(side_effect=RuntimeError("secret")),
        close=Mock(side_effect=lambda: events.append("agent")),
    )
    resources = oneshot._OneshotResources()
    resources.agent = agent
    resources.session_db = types.SimpleNamespace(close=Mock(side_effect=lambda: events.append("db")))
    resources.task_id = str(uuid.uuid4())

    with pytest.raises(KeyboardInterrupt):
        resources.close()
    assert "memory_stop" in events
    assert "agent" in events
    assert "mcp" in events
    assert "aux" in events
    assert "db" in events


def test_resources_primary_failure_wins_over_cleanup_control_flow(monkeypatch):
    events: list[str] = []
    _install_cleanup_modules(monkeypatch, events)
    monkeypatch.setattr(
        oneshot._OneshotResources,
        "_arm_watchdog_once",
        Mock(side_effect=SystemExit(70)),
    )
    resources = oneshot._OneshotResources()
    resources.task_id = str(uuid.uuid4())
    primary = RuntimeError("primary")

    resources.close(primary_failure=primary)
    assert "interrupt" in events
    assert "mcp" in events
    assert "aux" in events


def test_mark_final_delivered_arms_then_ends_current_session_once(monkeypatch):
    events: list[str] = []
    resources = oneshot._OneshotResources()
    agent = types.SimpleNamespace(session_id="compression-child", _end_session_on_close=True)
    db = types.SimpleNamespace(
        end_session=Mock(side_effect=lambda *_args: events.append("session"))
    )
    resources.agent = agent
    resources.session_db = db
    resources.task_id = str(uuid.uuid4())
    monkeypatch.setattr(
        resources,
        "_arm_watchdog_once",
        lambda exit_code=70: events.append("watchdog"),
    )

    resources.mark_final_delivered()
    resources.mark_final_delivered()

    assert events == ["watchdog", "session"]
    assert db.end_session.call_args_list == [call("compression-child", "oneshot_complete")]
    assert agent._end_session_on_close is False


def test_mark_final_delivered_ends_session_even_when_watchdog_fails(monkeypatch):
    resources = oneshot._OneshotResources()
    agent = types.SimpleNamespace(session_id="current-child", _end_session_on_close=True)
    db = types.SimpleNamespace(end_session=Mock())
    resources.agent = agent
    resources.session_db = db
    monkeypatch.setattr(resources, "_arm_watchdog_once", Mock(side_effect=RuntimeError("boom")))

    resources.mark_final_delivered()

    db.end_session.assert_called_once_with("current-child", "oneshot_complete")
    assert agent._end_session_on_close is False


def test_mark_final_delivered_ends_session_before_propagating_control_flow(monkeypatch):
    resources = oneshot._OneshotResources()
    agent = types.SimpleNamespace(session_id="current-child", _end_session_on_close=True)
    db = types.SimpleNamespace(end_session=Mock())
    resources.agent = agent
    resources.session_db = db
    failure = KeyboardInterrupt()
    monkeypatch.setattr(resources, "_arm_watchdog_once", Mock(side_effect=failure))

    with pytest.raises(KeyboardInterrupt) as raised:
        resources.mark_final_delivered()

    assert raised.value is failure
    db.end_session.assert_called_once_with("current-child", "oneshot_complete")
    assert agent._end_session_on_close is False


def test_resources_without_task_id_never_issue_global_process_cleanup(monkeypatch):
    events: list[str] = []
    _, kill_all, _, _ = _install_cleanup_modules(monkeypatch, events)
    monkeypatch.setattr(oneshot._OneshotResources, "_arm_watchdog_once", Mock())
    cleanup = Mock()
    resources = oneshot._OneshotResources()
    resources.agent = types.SimpleNamespace(
        _memory_manager=None,
        _session_messages=[],
        shutdown_memory_provider=Mock(),
        _cleanup_task_resources=cleanup,
        close=Mock(),
        session_id="session",
    )

    resources.close()

    kill_all.assert_not_called()
    cleanup.assert_not_called()


def test_mark_final_delivered_leaves_agent_fallback_when_db_end_fails(monkeypatch):
    resources = oneshot._OneshotResources()
    agent = types.SimpleNamespace(session_id="current-child", _end_session_on_close=True)
    resources.agent = agent
    resources.session_db = types.SimpleNamespace(
        end_session=Mock(side_effect=RuntimeError("db unavailable"))
    )
    monkeypatch.setattr(resources, "_arm_watchdog_once", Mock())

    resources.mark_final_delivered()

    assert agent._end_session_on_close is True


def test_run_agent_passes_explicit_uuid_task_id_and_installs_hook(monkeypatch):
    captured = {}
    sentinel_db = types.SimpleNamespace()
    manager = types.SimpleNamespace(_hooks={"post_llm_call": []})

    class FakeAgent:
        def __init__(self, **kwargs):
            captured.update(kwargs)
            self.session_id = "session"
            self.suppress_status_output = False
            self.stream_delta_callback = object()
            self.tool_gen_callback = object()

        def run_conversation(self, prompt, task_id):
            captured["prompt"] = prompt
            captured["task_id"] = task_id
            captured["hook_present"] = bool(manager._hooks["post_llm_call"])
            manager._hooks["post_llm_call"][0](assistant_response="transformed")
            return {"final_response": "transformed", "failed": False}

    def mod(name, **attrs):
        module = types.ModuleType(name)
        for key, value in attrs.items():
            setattr(module, key, value)
        monkeypatch.setitem(sys.modules, name, module)

    mod("run_agent", AIAgent=FakeAgent)
    mod("hermes_cli.config", load_config=lambda: {"model": {"default": "m"}})
    mod("hermes_cli.models", detect_provider_for_model=lambda *_a, **_k: None)
    mod(
        "hermes_cli.runtime_provider",
        resolve_runtime_provider=lambda **_k: {
            "api_key": "k",
            "base_url": "u",
            "provider": "p",
            "api_mode": "chat_completions",
            "credential_pool": None,
        },
    )
    mod("hermes_cli.tools_config", _get_platform_tools=lambda *_a, **_k: set())
    monkeypatch.setattr(oneshot, "_create_session_db_for_oneshot", lambda: sentinel_db)
    monkeypatch.setattr("hermes_cli.plugins.get_plugin_manager", lambda: manager)

    resources = oneshot._OneshotResources()
    delivery = oneshot._FinalDelivery(io.StringIO(), lambda: None)
    text, result = oneshot._run_agent(
        "prompt",
        None,
        None,
        None,
        True,
        delivery,
        resources,
    )

    assert text == "transformed"
    assert result["failed"] is False
    assert resources.agent is not None
    assert resources.session_db is sentinel_db
    assert resources.task_id == captured["task_id"]
    uuid.UUID(resources.task_id)
    assert captured["prompt"] == "prompt"
    assert captured["hook_present"] is True
    assert manager._hooks["post_llm_call"] == []
