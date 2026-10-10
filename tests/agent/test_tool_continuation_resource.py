"""Actual middleware handoff/factory, with external authorized tool body isolated."""
from contextvars import ContextVar
import sqlite3
import threading
from types import SimpleNamespace

import pytest

from agent import tool_executor as te
from agent.chat_execution_context import bind_chat_execution
from gateway.chat_execution_recovery import ChatExecutionStore
from tests.gateway.test_chat_execution_tool_dispatch import started


def producer(monkeypatch, plugin, leaf):
    monkeypatch.setattr("hermes_cli.middleware.apply_tool_request_middleware",
                        lambda name, args, **kw: SimpleNamespace(payload=args, trace=[]))
    monkeypatch.setattr("hermes_cli.middleware.run_tool_execution_middleware", plugin)
    monkeypatch.setattr("agent.relay_tools.execute", lambda name, args, pipeline, **kw: (pipeline(args), args))
    monkeypatch.setattr(te, "_dispatch_authorized_once", lambda *a, **kw: leaf())
    agent = SimpleNamespace(session_id="session")
    return lambda: te._run_agent_tool_execution_middleware(
        agent, function_name="fake", function_args={}, effective_task_id="task",
        tool_call_id="call", execute=lambda args: None)


def test_actual_common_middleware_return_does_not_finish_callback_thread(tmp_path, monkeypatch):
    store = started(tmp_path)
    entered, release = threading.Event(), threading.Event()
    threads, errors, observed = [], [], []
    probe = ContextVar("continuation_test_profile", default="absent")
    def leaf():
        try:
            rows = ChatExecutionStore(tmp_path).inspect_resources("key")
            assert len(rows) == 1
            assert rows[0]["state"] == "started"
            assert probe.get() == "factory-context"
            observed.append("body")
            entered.set()
            release.wait()
        finally:
            entered.set()
    def plugin(name, args, next_call, **kw):
        def target():
            try:
                next_call(args)
            except BaseException as exc:
                errors.append(exc)
        thread = threading.Thread(target=target)
        threads.append(thread)
        thread.start()
        assert entered.wait(5)
        return "handoff-returned"
    run = producer(monkeypatch, plugin, leaf)
    token = probe.set("factory-context")
    try:
        with bind_chat_execution(tmp_path, "key", 1):
            result = run()
        assert errors == []
        assert result.result == "handoff-returned"
        assert store.inspect_resources("key")[0]["state"] == "started"
        assert store.worker_finished("key", 1)
        assert store.begin_recovery("key", 1, "fence")
        assert store.inspect_resources("key")[0]["state"] == "started"
    finally:
        release.set()
        probe.reset(token)
        for thread in threads:
            thread.join(timeout=5)
            assert not thread.is_alive()
    assert errors == []
    assert observed == ["body"]
    assert store.inspect_resources("key")[0]["state"] == "finished"
    assert store.inspect("key", 1)["effects"] == "unknown"


@pytest.mark.parametrize("phase", ["reserve", "start", "finish", "unused", "replay", "normal"])
def test_actual_middleware_refusal_unused_and_replay_boundaries(tmp_path, monkeypatch, phase):
    store = started(tmp_path)
    handoffs, bodies = [], []
    def leaf():
        bodies.append("body")
        return "ok"
    def plugin(name, args, next_call, **kw):
        committed = ChatExecutionStore(tmp_path).inspect_resources("key")
        assert len(committed) == 1
        assert committed[0]["kind"] == "middleware_tool_continuation"
        assert committed[0]["state"] == "reserved"
        handoffs.append("handoff")
        if phase == "unused":
            return "unused"
        result = next_call(args)
        if phase == "replay":
            with pytest.raises(RuntimeError, match="more than once"):
                next_call(args)
        return result
    run = producer(monkeypatch, plugin, leaf)
    methods = {"reserve": "reserve_concurrent_tool_resource", "start": "start_concurrent_tool_resource",
               "finish": "finish_concurrent_tool_resource"}
    def broken(*a, **kw):
        raise sqlite3.OperationalError("injected continuation persistence failure")
    if phase in methods:
        monkeypatch.setattr(ChatExecutionStore, methods[phase], broken)
    with bind_chat_execution(tmp_path, "key", 1):
        if phase in {"reserve", "start"}:
            with pytest.raises(sqlite3.OperationalError):
                run()
        else:
            run()
    assert handoffs == ([] if phase == "reserve" else ["handoff"])
    assert bodies == ([] if phase in {"reserve", "start", "unused"} else ["body"])
    expected = {"reserve": [], "start": ["reserved"], "finish": ["started"],
                "unused": ["reserved"], "replay": ["finished"], "normal": ["finished"]}
    assert [row["state"] for row in store.inspect_resources("key")] == expected[phase]


def test_actual_prepared_slot_can_finish_before_its_middleware_callback(tmp_path, monkeypatch):
    from tests.gateway.test_prepared_terminal_resource_producer import prepared_batch
    store = started(tmp_path)
    batch = prepared_batch()
    batch.agent.session_id = "session"
    entered, release = threading.Event(), threading.Event()
    threads, errors = [], []
    def leaf():
        rows = store.inspect_resources("key")
        assert next(row for row in rows if row["kind"] == "middleware_tool_continuation")["state"] == "started"
        entered.set()
        release.wait()
    def plugin(name, args, next_call, **kw):
        def target():
            try:
                next_call(args)
            except BaseException as exc:
                errors.append(exc)
                entered.set()
        thread = threading.Thread(target=target)
        threads.append(thread)
        thread.start()
        assert entered.wait(5)
        return "middleware-returned"
    producer(monkeypatch, plugin, leaf)
    monkeypatch.setattr(te, "_resolve_sequential_dispatch", lambda *a: SimpleNamespace(execute=lambda args: None))
    try:
        with bind_chat_execution(tmp_path, "key", 1):
            batch.start()
        batch.slots[0].future.result(timeout=5)
        rows = {row["kind"]: row["state"] for row in store.inspect_resources("key")}
        assert rows == {"prepared_terminal_tool_thread": "finished", "middleware_tool_continuation": "started"}
        assert errors == []
        batch.close()
        assert store.worker_finished("key", 1)
        assert store.begin_recovery("key", 1, "fence")
    finally:
        release.set()
        batch.executor.shutdown(wait=True)
        for thread in threads:
            thread.join(timeout=5)
            assert not thread.is_alive()
    assert errors == []
    assert all(row["state"] == "finished" for row in store.inspect_resources("key"))
    assert store.inspect("key", 1)["effects"] == "unknown"


@pytest.mark.parametrize("later", ["different_profile", "fence", "successor"])
def test_deferred_callback_uses_captured_profile_and_rejects_stale_admission(tmp_path, monkeypatch, later):
    original, other = tmp_path / "original", tmp_path / "other"
    store = started(original)
    started(other)
    callbacks, bodies = [], []
    def plugin(name, args, next_call, **kw):
        callbacks.append(lambda: next_call(args))
        return "not-invoked-yet"
    run = producer(monkeypatch, plugin, lambda: bodies.append("body"))
    with bind_chat_execution(original, "key", 1):
        run()
    if later == "fence":
        assert store.begin_recovery("key", 1, "fence")
    if later == "successor":
        assert store.reserve("key", 1, 1) == 2
    with bind_chat_execution(other, "key", 1):
        if later == "different_profile":
            callbacks[0]()
        else:
            with pytest.raises(RuntimeError, match="admission refused"):
                callbacks[0]()
    assert bodies == (["body"] if later == "different_profile" else [])
    assert store.inspect_resources("key")[0]["state"] == ("finished" if later == "different_profile" else "reserved")
    assert ChatExecutionStore(other).inspect_resources("key") == []


def test_replay_while_callback_active_rejects_before_entering_copied_context(tmp_path, monkeypatch):
    store = started(tmp_path)
    entered, release = threading.Event(), threading.Event()
    callbacks, errors, bodies = [], [], []
    def plugin(name, args, next_call, **kw):
        callbacks.append(lambda: next_call(args))
        return "deferred"
    def leaf():
        bodies.append("body")
        entered.set()
        release.wait()
    run = producer(monkeypatch, plugin, leaf)
    with bind_chat_execution(tmp_path, "key", 1):
        run()
    def target():
        try:
            callbacks[0]()
        except BaseException as exc:
            errors.append(exc)
            entered.set()
    thread = threading.Thread(target=target)
    try:
        thread.start()
        assert entered.wait(5)
        with pytest.raises(RuntimeError, match="more than once"):
            callbacks[0]()
        assert store.inspect_resources("key")[0]["state"] == "started"
    finally:
        release.set()
        thread.join(timeout=5)
        assert not thread.is_alive()
    assert errors == []
    assert bodies == ["body"]
    assert store.inspect_resources("key")[0]["state"] == "finished"


def test_synchronous_continuation_preserves_parent_prompt_callback(tmp_path, monkeypatch):
    from tools import thread_context
    started(tmp_path)
    callbacks = SimpleNamespace(value="parent-prompt")
    def setter(value):
        callbacks.value = value
    monkeypatch.setattr(thread_context, "_callback_api", lambda: ((lambda: callbacks.value, setter),))
    seen = []
    run = producer(monkeypatch, lambda name, args, next_call, **kw: next_call(args),
                   lambda: seen.append(callbacks.value))
    with bind_chat_execution(tmp_path, "key", 1):
        run()
    assert seen == ["parent-prompt"]
    assert callbacks.value == "parent-prompt"


def test_distinct_factories_with_same_provider_call_id_get_unique_resources(tmp_path, monkeypatch):
    store = started(tmp_path)
    run = producer(monkeypatch, lambda *a, **kw: "unused", lambda: None)
    with bind_chat_execution(tmp_path, "key", 1):
        run()
        run()
    rows = store.inspect_resources("key")
    assert len(rows) == 2
    assert rows[0]["resource_id"] != rows[1]["resource_id"]
    assert all(row["state"] == "reserved" for row in rows)


def test_legacy_unbound_middleware_still_dispatches_without_resource(tmp_path, monkeypatch):
    store = ChatExecutionStore(tmp_path)
    bodies = []
    run = producer(monkeypatch, lambda name, args, next_call, **kw: next_call(args), lambda: bodies.append("legacy"))
    run()
    assert bodies == ["legacy"]
    assert store.inspect_resources("key") == []
