"""State-aware task clocks and recording admission contracts."""
import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest

from plugins.platforms.a2a import protocol
from tests.plugins import test_a2a_completion_lifecycle as lifecycle
from tests.waiting import wait_until

adapter = lifecycle.adapter


def test_terminal_retention_uses_terminal_transition(monkeypatch):
    clock = [100.0]
    monkeypatch.setattr(protocol.time, "time", lambda: clock[0])
    store = protocol.TaskStore(task_retention_seconds=10)
    store.create("t", "c", "peer")
    clock[0] = 1000
    store.complete("t", protocol.STATE_COMPLETED, "answer")
    clock[0] = 1009
    assert store.get("t")["reply"] == "answer"
    clock[0] = 1010
    assert store.get("t") is None


def test_interrupted_clock_is_transition_not_creation_or_polling(adapter, monkeypatch):
    clock = [100.0]
    monkeypatch.setattr(protocol.time, "time", lambda: clock[0])
    adapter.tasks.input_wait_timeout_seconds = 10
    adapter.tasks.create("t", "c", "peer")
    clock[0] = 1000
    assert adapter._resolve_task("t", protocol.STATE_INPUT_REQUIRED, "Which?")
    clock[0] = 1009
    assert adapter._rpc_tasks_get(1, {"id": "t"})["result"]["status"]["state"] == protocol.STATE_INPUT_REQUIRED
    adapter._fail_orphans_once()
    assert adapter.tasks.get("t")["state"] == protocol.STATE_INPUT_REQUIRED
    clock[0] = 1010
    adapter._fail_orphans_once()
    assert adapter.tasks.get("t")["state"] == protocol.STATE_CANCELED
    assert "input wait expired" in adapter.tasks.get("t")["reply"]
    asyncio.run(adapter.disconnect())
    assert protocol.load_conversation("c")[-1]["text"] == "[input wait expired]"


def test_recorded_questions_release_execution_capacity(adapter):
    try:
        adapter.tasks.create("t", "c", "peer")
        assert adapter._add_pending("t", "c") is not None
        assert adapter._resolve_task("t", protocol.STATE_INPUT_REQUIRED, "Which?")
        assert wait_until(lambda: not adapter._completion_io._reserved, timeout=5)
        assert adapter._completion_io._record_slots.acquire(blocking=False)
        adapter._completion_io._record_slots.release()
    finally:
        asyncio.run(adapter.disconnect())


def test_delayed_question_record_cannot_release_resumed_execution_reservation(adapter, monkeypatch):
    from plugins.platforms.a2a.continuation import _admit
    entered, release = threading.Event(), threading.Event()
    original = adapter._outcome_io

    def record(rec):
        assert adapter.tasks.get(rec["task_id"])["state"] != protocol.STATE_SUBMITTED
        entered.set()
        assert release.wait(5)
        original(rec)

    monkeypatch.setattr(adapter, "_outcome_io", record)
    adapter.tasks.create("t", "c", "peer")
    adapter._add_pending("t", "c")
    try:
        assert adapter._resolve_task("t", protocol.STATE_INPUT_REQUIRED, "Which?")
        assert entered.wait(5)
        rec, future, error, retry = _admit(adapter, "t", None, ("", ""), adapter._generation, "answer")
        assert error is None and not retry
        release.set()
        assert wait_until(lambda: len(protocol.load_conversation("c")) == 1, timeout=5)
        assert rec["attempt_id"] in adapter._completion_io._reserved
        assert "t" not in adapter._completion_io._reserved
        assert not future.done()
        assert adapter._resolve_delivery(rec["attempt_id"], protocol.STATE_COMPLETED, "done")
    finally:
        release.set()
        asyncio.run(adapter.disconnect())
    assert [row["text"] for row in protocol.load_conversation("c")] == ["Which?", "done"]


def test_expiry_retries_after_recording_saturation_and_covers_entire_store(adapter, monkeypatch):
    entered, release = threading.Event(), threading.Event()
    original = adapter._outcome_io

    def record(rec):
        entered.set()
        assert release.wait(10)
        original(rec)

    monkeypatch.setattr(adapter, "_outcome_io", record)
    adapter._completion_io._record_slots = threading.BoundedSemaphore(1)
    adapter.tasks.input_wait_timeout_seconds = 10
    try:
        adapter.tasks.create("t", "c", "peer")
        assert adapter._resolve_task("t", protocol.STATE_INPUT_REQUIRED, "Which?")
        assert entered.wait(5)
        for i in range(110):
            adapter.tasks.create(str(i), "c", "peer")
        with adapter.tasks._lock:
            adapter.tasks._tasks["t"]["interrupted_at"] -= 11
        adapter._fail_orphans_once()
        assert adapter.tasks.get("t")["state"] == protocol.STATE_INPUT_REQUIRED
        release.set()
        assert wait_until(lambda: len(protocol.load_conversation("c")) == 1, timeout=5)
        assert wait_until(lambda: adapter._completion_io._record_slots._value == 1, timeout=5)
        adapter._fail_orphans_once()
        assert adapter.tasks.get("t")["state"] == protocol.STATE_CANCELED
    finally:
        release.set()
        asyncio.run(adapter.disconnect())
    assert [row["text"] for row in protocol.load_conversation("c")] == ["Which?", "[input wait expired]"]


@pytest.mark.parametrize("expire", [False, True])
def test_cancel_resume_expiry_arbitrate_without_resurrection(adapter, expire):
    from plugins.platforms.a2a.continuation import _admit
    adapter.tasks.create("t", "c", "peer")
    adapter._resolve_task("t", protocol.STATE_INPUT_REQUIRED, "Which?")
    if expire:
        with adapter.tasks._lock:
            adapter.tasks._tasks["t"]["interrupted_at"] -= 86401
    barrier = threading.Barrier(3)

    def resume():
        barrier.wait(5)
        return _admit(adapter, "t", None, ("", ""), adapter._generation, "answer")

    def cancel():
        barrier.wait(5)
        return adapter._rpc_tasks_cancel(1, {"id": "t"})

    def expiry():
        barrier.wait(5)
        adapter._expire_waits()

    try:
        with ThreadPoolExecutor(max_workers=3) as pool:
            runs = [pool.submit(fn) for fn in (resume, cancel, expiry)]
            admission, _, _ = [run.result(5) for run in runs]
        rec = adapter.tasks.get("t")
        assert rec["state"] == protocol.STATE_CANCELED
        if admission[1] is not None:
            assert admission[1].result(0)[0] == protocol.STATE_CANCELED
        assert not adapter._resolve_delivery("t", protocol.STATE_COMPLETED, "stale")
    finally:
        asyncio.run(adapter.disconnect())
    rows = protocol.load_conversation("c")
    assert len(rows) == 2
    assert rows[0]["text"] == "Which?"
    assert rows[1]["text"] in ("", "[input wait expired]")


@pytest.mark.parametrize("key", ["input_wait_timeout_seconds", "task_retention_seconds"])
@pytest.mark.parametrize("value", [0, -1, "bad", float("nan"), float("inf"), True, None])
def test_invalid_lifecycle_configuration_fails_at_adapter_boundary(monkeypatch, key, value):
    from gateway.config import PlatformConfig
    from plugins.platforms.a2a.adapter import A2AAdapter
    monkeypatch.setenv("A2A_PORT", "0")
    with pytest.raises(ValueError, match=key):
        A2AAdapter(PlatformConfig(enabled=True, extra={key: value}))


def test_cancellation_capacity_refusal_is_retryable_and_eventually_recorded(adapter, monkeypatch):
    entered, release = threading.Event(), threading.Event()
    original = adapter._outcome_io

    def record(rec):
        entered.set()
        assert release.wait(5)
        original(rec)

    monkeypatch.setattr(adapter, "_outcome_io", record)
    adapter._completion_io._record_slots = threading.BoundedSemaphore(1)
    adapter.tasks.create("t", "c", "peer")
    try:
        assert adapter._resolve_task("t", protocol.STATE_INPUT_REQUIRED, "Which?")
        assert entered.wait(5)
        refused = adapter._rpc_tasks_cancel(1, {"id": "t"})
        assert refused["error"]["code"] == protocol.ERR_RATE_LIMITED
        assert adapter.tasks.get("t")["state"] == protocol.STATE_INPUT_REQUIRED
        release.set()
        assert wait_until(lambda: adapter._completion_io._record_slots._value == 1, timeout=5)
        assert "result" in adapter._rpc_tasks_cancel(2, {"id": "t"})
    finally:
        release.set()
        asyncio.run(adapter.disconnect())
    assert [row["text"] for row in protocol.load_conversation("c")] == ["Which?", ""]


def test_hundreds_of_recorded_interruptions_do_not_consume_execution_slots(adapter):
    adapter._completion_io._record_slots = threading.BoundedSemaphore(1)
    try:
        for i in range(520):
            tid = str(i)
            adapter.tasks.create(tid, "questions", "peer")
            assert adapter._add_pending(tid, "questions") is not None
            assert adapter._resolve_task(tid, protocol.STATE_INPUT_REQUIRED, "Which?")
            assert wait_until(lambda: adapter._completion_io._record_slots._value == 1, timeout=5)
        assert not adapter._completion_io._reserved
        assert adapter.tasks.list(with_total=True)[2] == 520
    finally:
        asyncio.run(adapter.disconnect())
    assert len(protocol.load_conversation("questions", limit=0)) == 520


def test_yaml_lifecycle_settings_reach_adapter_without_environment_bridge(tmp_path, monkeypatch):
    import os
    from gateway.config import Platform, load_gateway_config
    from hermes_constants import get_hermes_home
    from plugins.platforms.a2a.adapter import A2AAdapter

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    assert get_hermes_home() == tmp_path
    (tmp_path / "config.yaml").write_text(
        "gateway:\n  platforms:\n    a2a:\n      enabled: true\n      extra:\n"
        "        input_wait_timeout_seconds: 17\n        task_retention_seconds: 29\n")
    monkeypatch.setenv("A2A_PORT", "0")
    config = load_gateway_config()
    before = dict(os.environ)
    scoped = A2AAdapter(config.platforms[Platform("a2a")])
    try:
        assert scoped.tasks.input_wait_timeout_seconds == 17
        assert scoped.tasks.task_retention_seconds == 29
        assert dict(os.environ) == before
    finally:
        asyncio.run(scoped.disconnect())


def test_purged_task_cannot_resume_and_polling_preserves_status_timestamp(adapter, monkeypatch):
    clock = [100.0]
    monkeypatch.setattr(protocol.time, "time", lambda: clock[0])
    adapter.tasks.task_retention_seconds = 10
    adapter.tasks.create("t", "c", "peer")
    adapter._resolve_task("t", protocol.STATE_COMPLETED, "done")
    timestamp = adapter._rpc_tasks_get(1, {"id": "t"})["result"]["status"]["timestamp"]
    clock[0] = 109
    assert adapter._rpc_tasks_get(2, {"id": "t"})["result"]["status"]["timestamp"] == timestamp
    clock[0] = 110
    arguments = lifecycle.params("answer")
    arguments["message"]["taskId"] = "t"
    try:
        assert adapter._rpc_message_send(3, arguments, "peer")["error"]["code"] == protocol.ERR_TASK_NOT_FOUND
        assert adapter._rpc_tasks_get(4, {"id": "t"})["error"]["code"] == protocol.ERR_TASK_NOT_FOUND
    finally:
        asyncio.run(adapter.disconnect())


def test_stale_orphan_sweep_cannot_fail_newly_resumed_attempt(adapter, monkeypatch):
    from plugins.platforms.a2a.continuation import _admit
    adapter.tasks.create("t", "c", "peer")
    with adapter.tasks._lock:
        adapter.tasks._tasks["t"]["created_at"] -= 86401
    original = adapter._resolve_task

    def race(task_id, state, text, **kwargs):
        assert original(task_id, protocol.STATE_INPUT_REQUIRED, "Which?")
        _, future, error, _ = _admit(adapter, task_id, None, ("", ""), adapter._generation, "answer")
        assert error is None and not future.done()
        return original(task_id, state, text, **kwargs)

    monkeypatch.setattr(adapter, "_resolve_task", race)
    try:
        assert adapter._fail_orphans_once() == []
        assert adapter.tasks.get("t")["state"] == protocol.STATE_WORKING
    finally:
        monkeypatch.setattr(adapter, "_resolve_task", original)
        asyncio.run(adapter.disconnect())
