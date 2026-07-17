from __future__ import annotations

import io
import json
import logging
import os
import subprocess
import sys
import threading
import time
import types
import uuid
from pathlib import Path
from unittest.mock import MagicMock, Mock, call, patch

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


def _cleanup_real_oneshot_fixture(state: dict) -> None:
    """Best-effort teardown for every real resource seeded by integration tests."""

    def attempt(callback):
        try:
            callback()
        except BaseException:
            pass

    resources = state.get("resources")
    if resources is not None:
        attempt(resources.close)

    agent = state.get("agent")
    if agent is not None:
        attempt(agent.close)

    task_id = state.get("task_id")
    terminal_tool = state.get("terminal_tool")
    if terminal_tool is not None and task_id:
        def cleanup_terminal():
            try:
                terminal_tool.cleanup_vm(task_id)
            finally:
                terminal_tool._active_environments.pop(task_id, None)
                terminal_tool._last_activity.pop(task_id, None)
                terminal_tool._creation_locks.pop(task_id, None)

        attempt(cleanup_terminal)

    browser_tool = state.get("browser_tool")
    if browser_tool is not None and task_id:
        def cleanup_browser():
            try:
                browser_tool.cleanup_browser(task_id)
            finally:
                browser_tool._active_sessions.pop(task_id, None)
                browser_tool._session_last_activity.pop(task_id, None)
                browser_tool._last_active_session_key.pop(task_id, None)

        attempt(cleanup_browser)

    db = state.get("db")
    if db is not None:
        attempt(db.close)

    process = state.get("process")
    if process is not None:
        def reap_process():
            if process.poll() is not None:
                return
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)

        attempt(reap_process)


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


@pytest.mark.parametrize("short_phase", ["content", "newline"])
def test_delivery_retains_short_write_failure_without_duplicate(short_phase):
    class ShortWriteStream(RecordingStdout):
        def __init__(self):
            super().__init__([])
            self.calls = 0

        def write(self, value):
            self.calls += 1
            if short_phase == "content" and self.calls == 1:
                super().write(value[:2])
                return 2
            if short_phase == "newline" and value == "\n":
                return 0
            return super().write(value)

    stream = ShortWriteStream()
    delivery = oneshot._FinalDelivery(stream, lambda: None)

    with pytest.raises(oneshot._IncompleteFinalWriteError) as first:
        delivery.deliver("answer")
    calls_after_first = stream.calls
    with pytest.raises(oneshot._IncompleteFinalWriteError) as second:
        delivery.deliver("fallback")

    assert first.value is second.value
    assert "answer" not in str(first.value)
    assert stream.calls == calls_after_first
    assert delivery.delivered is False


def test_delivery_rejects_non_integer_write_result_and_retains_failure():
    class InvalidWriteStream(RecordingStdout):
        def write(self, value):
            self.events.append(("write", value))
            return None

    stream = InvalidWriteStream([])
    delivery = oneshot._FinalDelivery(stream, lambda: None)

    with pytest.raises(oneshot._IncompleteFinalWriteError) as first:
        delivery.deliver("private response")
    with pytest.raises(oneshot._IncompleteFinalWriteError) as second:
        delivery.deliver("fallback")

    assert first.value is second.value
    assert "private response" not in str(first.value)
    assert stream.events == [("write", "private response")]


@pytest.mark.parametrize("short_phase", ["content", "newline"])
def test_run_oneshot_short_write_fails_nonzero_without_fallback_duplicate(
    monkeypatch,
    capsys,
    short_phase,
):
    class ShortWriteStream(io.StringIO):
        def __init__(self):
            super().__init__()
            self.calls = 0

        def write(self, value):
            self.calls += 1
            if short_phase == "content" and self.calls == 1:
                super().write(value[:2])
                return 2
            if short_phase == "newline" and value == "\n":
                return 0
            return super().write(value)

    stream = ShortWriteStream()
    monkeypatch.setattr(sys, "stdout", stream)
    monkeypatch.setattr(
        oneshot,
        "_run_agent",
        lambda *_args, **_kwargs: ("private answer", {}),
    )
    monkeypatch.setattr(oneshot._OneshotResources, "close", lambda *_args, **_kwargs: None)

    try:
        assert oneshot.run_oneshot("prompt") == 1
    finally:
        logging.disable(logging.NOTSET)

    assert stream.calls == (1 if short_phase == "content" else 2)
    assert stream.getvalue() in {"pr", "private answer"}
    assert "private answer" not in capsys.readouterr().err


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
    resources.task_id = "task-current"
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
                callback(
                    assistant_response="transformed final",
                    task_id="task-current",
                )
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


def test_overlapping_transient_hooks_route_only_matching_task_and_leave_no_residue(monkeypatch):
    callbacks = [lambda **_kwargs: None]
    original_callbacks = callbacks
    manager = types.SimpleNamespace(_hooks={"post_llm_call": callbacks})
    monkeypatch.setattr("hermes_cli.plugins.get_plugin_manager", lambda: manager)
    events_a: list[tuple[str, object]] = []
    events_b: list[tuple[str, object]] = []
    delivery_a = oneshot._FinalDelivery(RecordingStdout(events_a), lambda: None)
    delivery_b = oneshot._FinalDelivery(RecordingStdout(events_b), lambda: None)
    resources_a = oneshot._OneshotResources()
    resources_b = oneshot._OneshotResources()
    resources_a.task_id = "task-a"
    resources_b.task_id = "task-b"

    with oneshot._install_final_delivery_hook(delivery_a, resources_a):
        dispatcher = callbacks[0]
        with oneshot._install_final_delivery_hook(delivery_b, resources_b):
            assert callbacks[0] is dispatcher
            assert callbacks.count(dispatcher) == 1
            dispatcher(task_id="task-a", assistant_response="answer-a")
            assert delivery_a.delivered is True
            assert delivery_b.delivered is False
            dispatcher(task_id="unknown", assistant_response="wrong")
            dispatcher(task_id="task-b", assistant_response="answer-b")
            assert delivery_b.delivered is True
        assert callbacks[0] is dispatcher

    assert manager._hooks["post_llm_call"] is original_callbacks
    assert len(callbacks) == 1
    assert events_a[0] == ("write", "answer-a")
    assert events_b[0] == ("write", "answer-b")


def test_dispatcher_snapshot_survives_matching_context_removal_during_invocation(monkeypatch):
    callbacks = []
    manager = types.SimpleNamespace(_hooks={"post_llm_call": callbacks})
    monkeypatch.setattr("hermes_cli.plugins.get_plugin_manager", lambda: manager)
    resources = oneshot._OneshotResources()
    resources.task_id = "task-a"
    entered = threading.Event()
    release = threading.Event()
    stream = RecordingStdout([])

    class BlockingDelivery(oneshot._FinalDelivery):
        def deliver(self, text):
            entered.set()
            assert release.wait(timeout=2)
            return super().deliver(text)

    delivery = BlockingDelivery(stream, lambda: None)

    with oneshot._install_final_delivery_hook(delivery, resources):
        dispatcher = callbacks[0]
        worker = threading.Thread(
            target=dispatcher,
            kwargs={"task_id": "task-a", "assistant_response": "answer"},
        )
        worker.start()
        assert entered.wait(timeout=2)
    assert callbacks == []
    release.set()
    worker.join(timeout=2)

    assert not worker.is_alive()
    assert delivery.delivered is True
    assert stream.getvalue() == "answer\n"


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


def test_watchdog_failed_delivery_arm_is_retried_by_cleanup(monkeypatch):
    events: list[str] = []
    _install_cleanup_modules(monkeypatch, events)
    attempts = Mock(side_effect=[RuntimeError("unavailable"), None])
    monkeypatch.setattr("hermes_cli.exit_watchdog.arm_exit_watchdog", attempts)
    resources = oneshot._OneshotResources()
    resources.task_id = str(uuid.uuid4())

    resources.mark_final_delivered()
    resources.close()

    assert attempts.call_args_list == [call(exit_code=70), call(exit_code=70)]


@pytest.mark.parametrize("failure", [KeyboardInterrupt(), SystemExit(70)])
def test_watchdog_baseexception_does_not_poison_retry(monkeypatch, failure):
    attempts = Mock(side_effect=[failure, None])
    monkeypatch.setattr("hermes_cli.exit_watchdog.arm_exit_watchdog", attempts)
    resources = oneshot._OneshotResources()

    with pytest.raises(type(failure)):
        resources._arm_watchdog_once()
    resources._arm_watchdog_once()
    resources._arm_watchdog_once()

    assert attempts.call_count == 2


def test_watchdog_concurrent_success_arms_exactly_once(monkeypatch):
    entered = threading.Event()
    release = threading.Event()
    attempts = Mock(side_effect=lambda **_kwargs: (entered.set(), release.wait(timeout=2)))
    monkeypatch.setattr("hermes_cli.exit_watchdog.arm_exit_watchdog", attempts)
    resources = oneshot._OneshotResources()
    threads = [threading.Thread(target=resources._arm_watchdog_once) for _ in range(6)]

    for thread in threads:
        thread.start()
    assert entered.wait(timeout=2)
    release.set()
    for thread in threads:
        thread.join(timeout=2)

    assert attempts.call_count == 1


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


@pytest.mark.parametrize(
    ("flush_result", "expect_success_log"),
    [(False, False), (None, True), (True, True)],
)
def test_memory_flush_false_is_cleanup_failure_but_none_and_true_succeed(
    monkeypatch,
    flush_result,
    expect_success_log,
):
    events: list[str] = []
    _install_cleanup_modules(monkeypatch, events)
    monkeypatch.setattr(oneshot._OneshotResources, "_arm_watchdog_once", Mock())
    info = Mock()
    warning = Mock()
    monkeypatch.setattr(oneshot.logger, "info", info)
    monkeypatch.setattr(oneshot.logger, "warning", warning)
    resources = oneshot._OneshotResources()
    resources.task_id = str(uuid.uuid4())
    resources.agent = types.SimpleNamespace(
        session_id="session",
        _session_messages=[],
        _memory_manager=types.SimpleNamespace(
            flush_pending=Mock(return_value=flush_result)
        ),
        shutdown_memory_provider=Mock(side_effect=lambda _messages: events.append("memory_stop")),
        _cleanup_task_resources=Mock(side_effect=lambda _task_id: events.append("task_resources")),
        close=Mock(side_effect=lambda: events.append("agent")),
    )
    resources.session_db = types.SimpleNamespace(
        close=Mock(side_effect=lambda: events.append("db"))
    )

    resources.close()

    assert "memory_stop" in events
    assert "agent" in events
    assert "mcp" in events
    assert "aux" in events
    assert "db" in events
    success_calls = [
        item for item in info.call_args_list if item.args and item.args[0].startswith("oneshot cleanup completed")
    ]
    assert bool(success_calls) is expect_success_log
    if flush_result is False:
        assert any(
            item.args == (
                "oneshot lifecycle operation failed operation=%s exception_type=%s",
                "memory_flush",
                "_IncompleteCleanupError",
            )
            for item in warning.call_args_list
        )


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
            manager._hooks["post_llm_call"][0](
                assistant_response="transformed",
                task_id=task_id,
            )
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


def test_terminal_delivery_resolves_rotated_session_after_flush_and_ends_it_once(
    monkeypatch,
):
    events: list[str] = []

    class OrderedStream(io.StringIO):
        def write(self, value):
            events.append("stdout_write")
            return super().write(value)

        def flush(self):
            events.append("stdout_flush")
            return super().flush()

    stream = OrderedStream()
    resources = oneshot._OneshotResources()
    resources.task_id = str(uuid.uuid4())

    db = types.SimpleNamespace(
        end_session=Mock(side_effect=lambda *_args: events.append("session_end")),
        close=Mock(side_effect=lambda: events.append("db_close")),
    )

    def close_agent():
        events.append("agent_close")
        if agent._end_session_on_close:
            db.end_session(agent.session_id, "agent_close")

    agent = types.SimpleNamespace(
        session_id="compression-parent",
        _end_session_on_close=True,
        _session_messages=[],
        _memory_manager=None,
        shutdown_memory_provider=Mock(side_effect=lambda _messages: events.append("memory_stop")),
        _cleanup_task_resources=Mock(side_effect=lambda _task_id: events.append("task_cleanup")),
        close=Mock(side_effect=close_agent),
    )
    resources.agent = agent
    resources.session_db = db
    _install_cleanup_modules(monkeypatch, events)
    monkeypatch.setattr(
        resources,
        "_arm_watchdog_once",
        Mock(side_effect=lambda exit_code=70: events.append("watchdog")),
    )
    delivery = oneshot._FinalDelivery(
        stream,
        lambda: (events.append("delivered_callback"), resources.mark_final_delivered()),
    )

    agent.session_id = "compression-child"
    assert delivery.deliver("final") is True
    resources.mark_final_delivered()
    resources.close()
    resources.close()

    assert stream.getvalue() == "final\n"
    assert events.index("stdout_flush") < events.index("watchdog")
    assert events.index("watchdog") < events.index("session_end")
    assert db.end_session.call_args_list == [
        call("compression-child", "oneshot_complete")
    ]
    assert agent._end_session_on_close is False
    assert agent.close.call_count == 1
    assert db.close.call_count == 1


def test_memory_shutdown_receives_a_copy_of_session_messages(monkeypatch):
    events: list[str] = []
    _install_cleanup_modules(monkeypatch, events)
    monkeypatch.setattr(oneshot._OneshotResources, "_arm_watchdog_once", Mock())
    original = [{"role": "user", "content": "first"}]
    captured = []

    def shutdown(messages):
        captured.append(messages)
        original.append({"role": "assistant", "content": "late mutation"})

    resources = oneshot._OneshotResources()
    resources.task_id = str(uuid.uuid4())
    resources.agent = types.SimpleNamespace(
        session_id="child",
        _session_messages=original,
        _memory_manager=None,
        shutdown_memory_provider=shutdown,
        _cleanup_task_resources=Mock(),
        close=Mock(),
    )
    resources.session_db = types.SimpleNamespace(close=Mock())

    resources.close()

    assert captured == [[{"role": "user", "content": "first"}]]
    assert captured[0] is not original


@pytest.mark.parametrize(
    "failing_owner",
    [
        "watchdog",
        "async_delegation",
        "memory_flush",
        "memory_shutdown",
        "process_cleanup",
        "task_cleanup",
        "agent_close",
        "mcp_shutdown",
        "auxiliary_shutdown",
        "session_db_close",
    ],
)
def test_each_cleanup_owner_failure_still_runs_every_later_owner(
    monkeypatch,
    failing_owner,
):
    events: list[str] = []

    def action(owner):
        def run(*_args, **_kwargs):
            events.append(owner)
            if owner == failing_owner:
                raise RuntimeError("private cleanup detail")
            return True

        return Mock(side_effect=run)

    def module(name, **attrs):
        value = types.ModuleType(name)
        for key, item in attrs.items():
            setattr(value, key, item)
        monkeypatch.setitem(sys.modules, name, value)

    watchdog = action("watchdog")
    monkeypatch.setattr(
        oneshot._OneshotResources,
        "_arm_watchdog_once",
        lambda self, exit_code=70: watchdog(exit_code=exit_code),
    )
    module("tools.async_delegation", interrupt_all=action("async_delegation"))
    module(
        "tools.process_registry",
        process_registry=types.SimpleNamespace(kill_all=action("process_cleanup")),
    )
    module("tools.mcp_tool", shutdown_mcp_servers=action("mcp_shutdown"))
    module(
        "agent.auxiliary_client",
        shutdown_cached_clients=action("auxiliary_shutdown"),
    )
    resources = oneshot._OneshotResources()
    resources.task_id = str(uuid.uuid4())
    resources.agent = types.SimpleNamespace(
        session_id="child",
        _session_messages=[],
        _memory_manager=types.SimpleNamespace(flush_pending=action("memory_flush")),
        shutdown_memory_provider=action("memory_shutdown"),
        _cleanup_task_resources=action("task_cleanup"),
        close=action("agent_close"),
    )
    resources.session_db = types.SimpleNamespace(close=action("session_db_close"))

    resources.close()

    owners = [
        "watchdog",
        "async_delegation",
        "memory_flush",
        "memory_shutdown",
        "process_cleanup",
        "task_cleanup",
        "agent_close",
        "mcp_shutdown",
        "auxiliary_shutdown",
        "session_db_close",
    ]
    assert events == owners


@pytest.mark.parametrize(
    ("run_failure", "delivers", "expected"),
    [
        (None, True, 0),
        (RuntimeError("before"), False, 1),
        (RuntimeError("after"), True, 1),
        (KeyboardInterrupt(), False, KeyboardInterrupt),
        (SystemExit(23), False, SystemExit),
    ],
)
def test_outer_lifecycle_closes_once_for_success_error_and_interrupt(
    monkeypatch,
    run_failure,
    delivers,
    expected,
):
    stdout = io.StringIO()
    stderr = io.StringIO()
    monkeypatch.setattr(sys, "stdout", stdout)
    monkeypatch.setattr(sys, "stderr", stderr)
    close_calls = Mock()
    monkeypatch.setattr(oneshot._OneshotResources, "close", close_calls)

    def run(*_args, delivery, resources, **_kwargs):
        resources.task_id = "task"
        if delivers:
            delivery.deliver("final")
        if run_failure is not None:
            raise run_failure
        return "final", {"failed": False, "session_id": "child"}

    monkeypatch.setattr(oneshot, "_run_agent", run)
    try:
        if isinstance(expected, type) and issubclass(expected, BaseException):
            with pytest.raises(expected) as raised:
                oneshot.run_oneshot("prompt")
            assert raised.value is run_failure
        else:
            assert oneshot.run_oneshot("prompt") == expected
    finally:
        logging.disable(logging.NOTSET)

    assert close_calls.call_count == 1
    assert stdout.getvalue() == ("final\n" if delivers else "")


def test_outer_lifecycle_retains_flush_failure_without_retry_and_still_closes(
    monkeypatch,
):
    failure = OSError("flush failed")

    class FlushFailure(io.StringIO):
        def __init__(self):
            super().__init__()
            self.flush_calls = 0

        def flush(self):
            self.flush_calls += 1
            raise failure

    stream = FlushFailure()
    monkeypatch.setattr(sys, "stdout", stream)
    close_calls = Mock()
    monkeypatch.setattr(oneshot._OneshotResources, "close", close_calls)
    monkeypatch.setattr(
        oneshot,
        "_run_agent",
        lambda *_args, **_kwargs: ("private final", {"failed": False}),
    )

    try:
        assert oneshot.run_oneshot("prompt") == 1
    finally:
        logging.disable(logging.NOTSET)

    assert stream.getvalue() == "private final\n"
    assert stream.flush_calls == 1
    assert close_calls.call_count == 1


def test_outer_lifecycle_reports_ordinary_cleanup_failure_after_all_owners(
    monkeypatch,
):
    stdout = io.StringIO()
    stderr = io.StringIO()
    monkeypatch.setattr(sys, "stdout", stdout)
    monkeypatch.setattr(sys, "stderr", stderr)
    events: list[str] = []
    _install_cleanup_modules(monkeypatch, events)
    monkeypatch.setattr(oneshot._OneshotResources, "_arm_watchdog_once", Mock())

    def run(*_args, delivery, resources, **_kwargs):
        resources.task_id = str(uuid.uuid4())
        resources.agent = types.SimpleNamespace(
            session_id="child",
            _session_messages=[],
            _memory_manager=types.SimpleNamespace(
                flush_pending=Mock(side_effect=RuntimeError("private memory detail"))
            ),
            shutdown_memory_provider=Mock(side_effect=lambda _messages: events.append("memory")),
            _cleanup_task_resources=Mock(side_effect=lambda _task_id: events.append("task")),
            close=Mock(side_effect=lambda: events.append("agent")),
        )
        resources.session_db = types.SimpleNamespace(
            close=Mock(side_effect=lambda: events.append("db"))
        )
        delivery.deliver("final")
        return "final", {"failed": False, "session_id": "child"}

    monkeypatch.setattr(oneshot, "_run_agent", run)
    try:
        assert oneshot.run_oneshot("prompt") == 1
    finally:
        logging.disable(logging.NOTSET)

    assert stdout.getvalue() == "final\n"
    assert "private memory detail" not in stderr.getvalue()
    assert events == [
        "interrupt",
        "memory",
        "processes",
        "task",
        "agent",
        "mcp",
        "aux",
        "db",
    ]


def test_cleanup_terminal_chatter_stays_inside_redirected_streams(monkeypatch):
    stdout = io.StringIO()
    stderr = io.StringIO()
    monkeypatch.setattr(sys, "stdout", stdout)
    monkeypatch.setattr(sys, "stderr", stderr)

    def run(*_args, delivery, **_kwargs):
        delivery.deliver("final")
        return "final", {"failed": False}

    def noisy_close(*_args, **_kwargs):
        print("cleanup stdout chatter")
        print("cleanup stderr chatter", file=sys.stderr)

    monkeypatch.setattr(oneshot, "_run_agent", run)
    monkeypatch.setattr(oneshot._OneshotResources, "close", noisy_close)
    try:
        assert oneshot.run_oneshot("prompt") == 0
    finally:
        logging.disable(logging.NOTSET)

    assert stdout.getvalue() == "final\n"
    assert stderr.getvalue() == ""


def test_real_fixture_cleanup_survives_failures_and_force_kills_stuck_process():
    events: list[str] = []

    class StuckProcess:
        def __init__(self):
            self.wait_calls = 0

        def poll(self):
            return None

        def terminate(self):
            events.append("terminate")

        def wait(self, timeout):
            self.wait_calls += 1
            events.append(("wait", timeout))
            if self.wait_calls == 1:
                raise subprocess.TimeoutExpired("fixture", timeout)
            return -9

        def kill(self):
            events.append("kill")

    task_id = str(uuid.uuid4())
    terminal = types.SimpleNamespace(
        _active_environments={task_id: object()},
        _last_activity={task_id: 1.0},
        _creation_locks={task_id: object()},
        cleanup_vm=Mock(side_effect=RuntimeError("early terminal failure")),
    )
    browser = types.SimpleNamespace(
        _active_sessions={task_id: object()},
        _session_last_activity={task_id: 1.0},
        _last_active_session_key={task_id: task_id},
        cleanup_browser=Mock(side_effect=RuntimeError("early browser failure")),
    )
    resources = types.SimpleNamespace(
        close=Mock(side_effect=RuntimeError("primary cleanup failure"))
    )
    agent = types.SimpleNamespace(close=Mock(side_effect=lambda: events.append("agent")))
    db = types.SimpleNamespace(close=Mock(side_effect=lambda: events.append("db")))
    process = StuckProcess()

    _cleanup_real_oneshot_fixture(
        {
            "task_id": task_id,
            "resources": resources,
            "agent": agent,
            "db": db,
            "process": process,
            "terminal_tool": terminal,
            "browser_tool": browser,
        }
    )

    resources.close.assert_called_once_with()
    agent.close.assert_called_once_with()
    db.close.assert_called_once_with()
    assert task_id not in terminal._active_environments
    assert task_id not in terminal._last_activity
    assert task_id not in terminal._creation_locks
    assert task_id not in browser._active_sessions
    assert task_id not in browser._session_last_activity
    assert task_id not in browser._last_active_session_key
    assert events == ["agent", "db", "terminate", ("wait", 5), "kill", ("wait", 5)]


def test_real_sessiondb_compression_child_is_the_only_oneshot_terminal_session(
    tmp_path: Path,
    monkeypatch,
    request,
):
    from agent.turn_finalizer import finalize_turn
    from hermes_state import SessionDB
    from run_agent import AIAgent
    from tools import browser_tool, process_registry as process_registry_module
    from tools import terminal_tool
    from tools.process_registry import ProcessRegistry, ProcessSession

    task_id = str(uuid.uuid4())
    cleanup_state = {"task_id": task_id}
    request.addfinalizer(lambda: _cleanup_real_oneshot_fixture(cleanup_state))
    db = SessionDB(db_path=tmp_path / "state.db")
    cleanup_state["db"] = db
    parent = "oneshot-parent"
    db.create_session(parent, source="cli", model="test/model")
    end_calls: list[tuple[str, str]] = []
    real_end_session = db.end_session

    def tracked_end_session(session_id, reason):
        end_calls.append((session_id, reason))
        return real_end_session(session_id, reason)

    monkeypatch.setattr(db, "end_session", tracked_end_session)
    with patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"}):
        agent = AIAgent(
            api_key="test-key",
            base_url="https://openrouter.ai/api/v1",
            model="test/model",
            platform="cli",
            quiet_mode=True,
            session_db=db,
            session_id=parent,
            skip_context_files=True,
            skip_memory=True,
        )
    cleanup_state["agent"] = agent
    compressor = MagicMock()
    compressor.compress.return_value = [
        {"role": "user", "content": "[CONTEXT COMPACTION] summary"},
        {"role": "user", "content": "tail"},
    ]
    compressor.compression_count = 1
    compressor.last_prompt_tokens = 0
    compressor.last_completion_tokens = 0
    compressor._last_summary_error = None
    compressor._last_compress_aborted = False
    compressor._last_summary_auth_failure = False
    compressor._last_aux_model_failure_model = None
    compressor._last_aux_model_failure_error = None
    agent.context_compressor = compressor
    agent.compression_in_place = False
    messages = [
        {"role": "user" if i % 2 == 0 else "assistant", "content": f"m{i}"}
        for i in range(20)
    ]
    compression_task_ids: list[str] = []
    monkeypatch.setattr(
        "tools.file_tools.reset_file_dedup",
        lambda compression_task_id: compression_task_ids.append(compression_task_id),
    )

    agent._compress_context(
        messages,
        "sys",
        approx_tokens=120_000,
        task_id=task_id,
    )
    child = agent.session_id
    assert child != parent
    assert compression_task_ids == [task_id]

    registry = ProcessRegistry()
    monkeypatch.setattr(process_registry_module, "process_registry", registry)
    process = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(60)"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    cleanup_state["process"] = process
    process_session = ProcessSession(
        id=f"proc_{uuid.uuid4().hex[:12]}",
        command="python sleep fixture",
        task_id=task_id,
        pid=process.pid,
        process=process,
        cwd=str(tmp_path),
        started_at=time.time(),
        host_start_time=registry._safe_host_start_time(process.pid),
    )
    registry._running[process_session.id] = process_session

    terminal_cleaned: list[str] = []

    class FixtureEnvironment:
        _persistent = False

        def cleanup(self):
            terminal_cleaned.append(task_id)

    monkeypatch.setattr(terminal_tool, "_active_environments", {task_id: FixtureEnvironment()})
    monkeypatch.setattr(terminal_tool, "_last_activity", {task_id: time.time()})
    monkeypatch.setattr(terminal_tool, "_creation_locks", {task_id: threading.Lock()})
    cleanup_state["terminal_tool"] = terminal_tool

    browser_session_name = f"oneshot-{task_id}"
    monkeypatch.setattr(
        browser_tool,
        "_active_sessions",
        {task_id: {"session_name": browser_session_name, "bb_session_id": None}},
    )
    monkeypatch.setattr(browser_tool, "_session_last_activity", {task_id: time.time()})
    monkeypatch.setattr(browser_tool, "_last_active_session_key", {task_id: task_id})
    monkeypatch.setattr(browser_tool, "_run_browser_command", lambda *_a, **_kw: {"success": True})
    monkeypatch.setattr(browser_tool, "_maybe_stop_recording", lambda *_a, **_kw: None)
    monkeypatch.setattr(browser_tool, "_is_camofox_mode", lambda: False)
    cleanup_state["browser_tool"] = browser_tool

    manager = types.SimpleNamespace(_hooks={})
    hook_task_ids: list[str] = []

    def invoke_hook(name, **kwargs):
        if name in {"post_llm_call", "on_session_end"}:
            hook_task_ids.append(kwargs["task_id"])
        return [callback(**kwargs) for callback in list(manager._hooks.get(name, []))]

    monkeypatch.setattr("hermes_cli.plugins.get_plugin_manager", lambda: manager)
    monkeypatch.setattr("hermes_cli.plugins.invoke_hook", invoke_hook)
    monkeypatch.setattr("tools.async_delegation.interrupt_all", lambda **_kwargs: None)
    monkeypatch.setattr("tools.mcp_tool.shutdown_mcp_servers", lambda: None)
    monkeypatch.setattr("agent.auxiliary_client.shutdown_cached_clients", lambda: None)

    resources = oneshot._OneshotResources()
    resources.agent = agent
    resources.session_db = db
    resources.task_id = task_id
    cleanup_state["resources"] = resources
    monkeypatch.setattr(resources, "_arm_watchdog_once", Mock())
    delivery = oneshot._FinalDelivery(io.StringIO(), resources.mark_final_delivered)

    with oneshot._install_final_delivery_hook(delivery, resources):
        result = finalize_turn(
            agent,
            final_response="compressed final",
            api_call_count=1,
            interrupted=False,
            failed=False,
            messages=messages,
            conversation_history=[],
            effective_task_id=task_id,
            turn_id="oneshot-turn",
            user_message="prompt",
            original_user_message="prompt",
            _should_review_memory=False,
            _turn_exit_reason="text_response(finish_reason=stop)",
        )
    usage_path = tmp_path / "usage.json"
    oneshot._write_usage_file(str(usage_path), result)
    _cleanup_real_oneshot_fixture(cleanup_state)

    reopened = SessionDB(db_path=tmp_path / "state.db")
    parent_row = reopened.get_session(parent)
    child_row = reopened.get_session(child)
    assert parent_row["end_reason"] == "compression"
    assert child_row["end_reason"] == "oneshot_complete"
    assert reopened.get_conversation_root(child) == parent
    assert reopened.get_compression_lineage(child) == [parent, child]
    assert result["session_id"] == child
    assert json.loads(usage_path.read_text())["session_id"] == child
    assert end_calls == [
        (parent, "compression"),
        (child, "oneshot_complete"),
    ]
    assert hook_task_ids == [task_id, task_id]
    assert resources.task_id == task_id
    assert registry.has_active_processes(task_id) is False
    assert process_session.id not in registry._running
    assert process.poll() is not None
    assert task_id not in terminal_tool._active_environments
    assert task_id not in browser_tool._active_sessions
    assert terminal_cleaned == [task_id]
    assert agent._end_session_on_close is False
    reopened.close()
