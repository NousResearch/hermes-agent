"""Async registry terminal outcomes must reach the originating ACP child."""

import threading
from concurrent.futures import Future
from contextvars import Context, copy_context
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from acp_adapter.events import make_step_cb, make_tool_progress_cb
from tools import async_delegation as ad
from tools.delegate_tool_dispatch import _dispatch_unit
from tools.delegate_tool_progress import _build_child_progress_callback
from tools.process_registry import process_registry


class _ManualExecutor:
    """Registry pool whose workers run only when a test says so (real Futures keep retirement balanced)."""

    def __init__(self):
        self.jobs = {}

    def submit(self, fn):
        future = Future()
        self.jobs[len(self.jobs)] = (fn, future)
        return future

    def run(self, index):
        fn, future = self.jobs[index]
        try:
            fn()
        finally:
            future.set_result(None)

    def release(self):
        for _, future in self.jobs.values():
            if not future.done():
                future.set_result(None)


@pytest.fixture
def registry(monkeypatch, tmp_path):
    ad._reset_for_tests()
    monkeypatch.setattr(ad, "_db_path", lambda: tmp_path / "state.db")
    executor = _ManualExecutor()
    monkeypatch.setattr(ad, "_get_executor", lambda _: executor)
    real_monitor = ad._ensure_stale_monitor
    monkeypatch.setattr(ad, "_ensure_stale_monitor", lambda: None)
    executor.start_monitor = real_monitor
    while not process_registry.completion_queue.empty():
        process_registry.completion_queue.get_nowait()
    yield executor
    executor.release()
    ad._reset_for_tests()
    while not process_registry.completion_queue.empty():
        process_registry.completion_queue.get_nowait()


@pytest.fixture
def progress():
    ids, meta = {}, {}
    callback = make_tool_progress_cb(None, "session", None, ids, meta)
    with patch("acp_adapter.events._send_update") as send:
        yield callback, make_step_cb(None, "session", None, ids, meta), ids, send


def dispatch(callback, batch_id, indexes=(0, 1)):
    """One real delegate invocation context: parent tool.started, child relays, then registry dispatch."""
    def start():
        tasks = [{"goal": "Same goal"} for _ in range(max(indexes) + 1)]
        callback("tool.started", "delegate_task", None, {"tasks": tasks})
        parent = SimpleNamespace(tool_progress_callback=callback, _delegate_spinner=None)
        children = []
        for index in indexes:
            relay = _build_child_progress_callback(
                index, tasks[index]["goal"], parent, len(tasks), depth=0,
                subagent_id=f"{batch_id}-child-{index}", model=f"test/model-{index}",
                session_ref={"delegation_id": batch_id, "session_id": f"session-{batch_id}-{index}"},
            )
            child = SimpleNamespace(tool_progress_callback=relay, model=f"test/model-{index}")
            children.append((index, tasks[index], child))
        unit = SimpleNamespace(children=children, task_list=tasks, context=None, top_role="leaf",
                               creds={"model": "test/parent"}, live_writers=[])
        assert _dispatch_unit(unit, batch_id, None, {"session_key": ""})["status"] == "dispatched"
        return copy_context(), children

    return Context().run(start)


def updates(send):
    return [call.args[3] for call in send.call_args_list]


def fast_stall(monkeypatch):
    monkeypatch.setattr(ad, "_STALE_CHECK_INTERVAL", 0.01)
    monkeypatch.setattr(ad, "_STALE_IDLE_SECONDS", 0.0)
    monkeypatch.setattr(ad, "_STALL_GRACE_SECONDS", 0.0)


def test_forced_stall_reaches_original_parent_and_only_unfinished_children(registry, progress, monkeypatch):
    callback, step, ids, send = progress
    first_context, children = dispatch(callback, "batch-a")
    dispatch(callback, "batch-b")  # Identical, overlapping, never started: must stay untouched.
    first_id = ids["delegate_task"][0]
    first_context.run(children[0][2].tool_progress_callback, "subagent.complete", status="completed",
                      summary="Already completed", duration_seconds=1)
    step(1, [{"name": "delegate_task", "result": '{"status":"dispatched"}'}] * 2)
    send.reset_mock()
    terminal_sent = threading.Event()
    send.side_effect = lambda *_: terminal_sent.set()
    fast_stall(monkeypatch)
    with ad._records_lock:
        ad._records["batch-a"]["_started"] = True  # The worker began; its token then freezes.
    # The real monitor is one shared thread started with an empty Context, not the dispatcher's.
    Context().run(registry.start_monitor)
    assert terminal_sent.wait(5)
    ad._monitor_stop.set()
    terminal, = updates(send)
    payload = terminal.raw_output["hermesDelegation"]
    assert terminal.tool_call_id == first_id and terminal.status == "in_progress"
    assert payload["event"] == "subagent.complete" and payload["status"] == "stalled"
    assert payload["task_index"] == 1 and payload["goal"] == "Same goal"
    assert payload["model"] == "test/model-1" and payload["subagent_id"] == "batch-a-child-1"
    assert payload["delegation_id"] == "batch-a"
    assert "stopped responding" in payload["error"] and payload["summary"] == payload["error"]
    assert payload["duration_seconds"] >= 0
    assert process_registry.completion_queue.get(timeout=5)["status"] == "stalled"
    assert "_on_finalize" not in ad._records["batch-a"]
    send.reset_mock()
    # The ignored worker finally returns: neither its relays nor its registry result reopen the child.
    first_context.run(children[1][2].tool_progress_callback, "subagent.text", preview="late text")
    first_context.run(children[1][2].tool_progress_callback, "subagent.complete", status="completed")
    with patch("tools.delegate_tool_dispatch._execute_and_aggregate", return_value={"results": []}):
        registry.run(0)
    assert not send.called and process_registry.completion_queue.empty()


def test_crashing_registry_worker_finishes_all_dispatched_children(registry, progress):
    callback, _, ids, send = progress
    dispatch(callback, "crashed", indexes=(1, 3))
    parent_id = ids["delegate_task"][0]
    send.reset_mock()
    with patch("tools.delegate_tool_dispatch._execute_and_aggregate", side_effect=RuntimeError("runner died")):
        registry.run(0)
    terminal = updates(send)
    assert [u.raw_output["hermesDelegation"]["task_index"] for u in terminal] == [1, 3]
    assert all(u.tool_call_id == parent_id and u.status == "in_progress" for u in terminal)
    assert all(u.raw_output["hermesDelegation"]["status"] == "error" for u in terminal)
    assert all("runner died" in u.raw_output["hermesDelegation"]["error"] for u in terminal)
    assert process_registry.completion_queue.get_nowait()["status"] == "error"


def test_registry_result_entries_keep_child_specific_status_and_summary(registry, progress):
    callback, _, _, send = progress
    dispatch(callback, "mixed")
    send.reset_mock()
    with patch("tools.delegate_tool_dispatch._execute_and_aggregate", return_value={"results": [
        {"task_index": 0, "status": "completed", "summary": "Good", "duration_seconds": 1},
        {"task_index": 1, "status": "error", "error": "Bad", "duration_seconds": 2},
    ]}):
        registry.run(0)
    first, second = [u.raw_output["hermesDelegation"] for u in updates(send)]
    assert first["status"] == "completed" and first["summary"] == "Good" and first["duration_seconds"] == 1
    assert second["status"] == "error" and second["error"] == "Bad" and second["duration_seconds"] == 2


def test_observer_failure_does_not_break_durable_completion(registry):
    def observer(result, status):
        assert not ad._records_lock.locked()
        assert ad._records[delegation_id]["status"] == "finalizing"
        raise RuntimeError("observer failed")

    handle = ad.dispatch_async_delegation_batch(
        goals=["A"], context=None, toolsets=None, role="leaf", model="test/model", session_key="",
        runner=lambda: {"results": []}, on_finalize=observer,
    )
    delegation_id = handle["delegation_id"]
    assert all("_on_finalize" not in row for row in ad.list_async_delegations())
    registry.run(0)
    assert ad._records[delegation_id]["status"] == "completed"
    assert "_on_finalize" not in ad._records[delegation_id]
    assert process_registry.completion_queue.get_nowait()["delegation_id"] == delegation_id
    assert ad.get_durable_delegation(delegation_id)["state"] == "completed"


def test_durable_write_failure_still_reports_terminal_child(registry, progress):
    callback, _, _, send = progress
    dispatch(callback, "broken-ledger", indexes=(0,))
    send.reset_mock()
    with patch("tools.async_delegation._persist_completion", side_effect=OSError("ledger unavailable")), \
         patch("tools.delegate_tool_dispatch._execute_and_aggregate", side_effect=RuntimeError("Worker died")):
        registry.run(0)
    terminal, = updates(send)
    assert terminal.raw_output["hermesDelegation"]["status"] == "error"
    assert "Worker died" in terminal.raw_output["hermesDelegation"]["error"]
    assert process_registry.completion_queue.get_nowait()["delegation_id"] == "broken-ledger"


def test_nested_terminal_does_not_finish_ancestor_or_sibling(progress):
    callback, _, _, send = progress

    def run():
        callback("tool.started", "delegate_task", None, {"goal": "Parent child"})
        relay = _build_child_progress_callback(
            0, "Parent child", SimpleNamespace(tool_progress_callback=callback, _delegate_spinner=None),
            subagent_id="outer", depth=0, model="test/model",
        )
        relay("subagent.complete", task_index=0, subagent_id="inner", parent_id="outer", depth=1,
              status="completed", summary="Grandchild")
        relay("subagent.text", preview="Ancestor remains live")
        relay("subagent.complete", status="completed", summary="Ancestor done")
        relay("subagent.complete", status="error", summary="Late fallback")

    Context().run(run)
    payloads = [u.raw_output["hermesDelegation"] for u in updates(send)[1:]]
    assert [(p["subagent_id"], p["event"]) for p in payloads] == [
        ("inner", "subagent.complete"), ("outer", "subagent.text"), ("outer", "subagent.complete"),
    ]
