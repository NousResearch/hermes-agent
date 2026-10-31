"""Retention, admission and queued execution contracts for A2A completion."""
import asyncio
import threading
import json
import io
from types import SimpleNamespace

import pytest

from plugins.platforms.a2a import protocol
from tests.plugins import test_a2a_completion_lifecycle as lifecycle
from tests.waiting import async_wait_until, wait_until

adapter = lifecycle.adapter
params = lifecycle.params


def test_retention_follows_completion_not_creation():
    store = protocol.TaskStore()
    store.create("old-running", "c", "peer")
    for i in range(500):
        store.create(str(i), "c", "peer")
        store.complete(str(i), protocol.STATE_COMPLETED, str(i))
    store.complete("old-running", protocol.STATE_COMPLETED, "late result")
    assert store.get("old-running")["reply"] == "late result"
    assert store.get("0") is None
    assert store.list(page_size=100, with_total=True)[2] == 500


def test_blocking_response_owns_result_after_retention_churn(adapter, monkeypatch):
    adapter.tasks.create("t", "c", "peer")
    future = adapter._add_pending("t", "c")
    pending = {"task_id": "t", "context_id": "c", "future": future, "created_iso": ""}
    monkeypatch.setattr(adapter, "_prepare_task", lambda *a, **kw: (None, pending))

    def observe(*args):
        adapter._resolve_task("t", protocol.STATE_COMPLETED, "owned result")
        for i in range(501):
            adapter.tasks.create(str(i), "c", "peer")
            adapter.tasks.complete(str(i), protocol.STATE_COMPLETED, str(i))
        return future.result()

    monkeypatch.setattr(adapter, "_await_reply", observe)
    try:
        result = adapter._rpc_message_send(1, params(), "peer")["result"]
        assert protocol.extract_text(result["artifacts"][0]) == "owned result"
    finally:
        asyncio.run(adapter.disconnect())


@pytest.mark.parametrize("reconnect", [False, True])
def test_shutdown_closes_admission_before_pending_registration(adapter, monkeypatch, reconnect):
    entered, release = threading.Event(), threading.Event()
    original = adapter._register_inline_push

    def gate(*args, **kwargs):
        entered.set()
        assert release.wait(5)
        return original(*args, **kwargs)

    monkeypatch.setattr(adapter, "_register_inline_push", gate)

    async def run():
        seen = []

        async def handle(event):
            seen.append(event.message_id)

        adapter.set_message_handler(handle)
        assert await adapter.connect()
        waiting = asyncio.create_task(asyncio.to_thread(adapter._rpc_message_send, 1,
                                      params(returnImmediately=True), "peer"))
        try:
            assert await asyncio.to_thread(entered.wait, 5)
            await adapter.disconnect()
            if reconnect:
                assert await adapter.connect()
            release.set()
            result = await asyncio.wait_for(waiting, 5)
            assert result["result"]["status"]["state"] == protocol.STATE_FAILED
            assert not adapter._pending
            assert not seen
        finally:
            release.set()
            await waiting
            await adapter.disconnect()

    asyncio.run(run())


@pytest.mark.parametrize("cancel_index", [None, 0, 1])
def test_three_same_context_requests_execute_individually(adapter, cancel_index):
    async def run():
        entered, release = asyncio.Event(), asyncio.Event()
        seen = []

        async def handle(event):
            seen.append(event.message_id)
            entered.set()
            await release.wait()
            if cancel_index == 0:
                await adapter.send(event.source.chat_id, event.message_id, metadata={"notify": True})
                return None
            return event.message_id

        adapter.set_message_handler(handle)
        assert await adapter.connect()
        ids = []
        try:
            for _ in range(3):
                arguments = params(returnImmediately=True)
                arguments["message"]["contextId"] = "same"
                result = await asyncio.to_thread(adapter._rpc_message_send, 1, arguments, "peer")
                ids.append(result["result"]["id"])
            await asyncio.wait_for(entered.wait(), 5)
            if cancel_index is not None:
                adapter._rpc_tasks_cancel(1, {"id": ids[cancel_index]})
            release.set()
            expected = [tid for i, tid in enumerate(ids) if i != cancel_index]
            await async_wait_until(lambda: all(adapter.tasks.get(t)["state"] == protocol.STATE_COMPLETED
                                              for t in expected), timeout=2)
            assert seen == ([ids[0], ids[2]] if cancel_index == 1 else ids)
            assert [adapter.tasks.get(t)["reply"] for t in expected] == expected
        finally:
            release.set()
            await adapter.cancel_background_tasks()
            await adapter.disconnect()

    asyncio.run(run())


def test_watch_snapshot_remains_coherent_after_interruption_and_cancel():
    store = protocol.TaskStore()
    store.create("t", "c", "peer")
    store.complete("t", protocol.STATE_INPUT_REQUIRED, "Question?")
    snapshot, future = store.watch_snapshot("t")
    store.complete("t", protocol.STATE_CANCELED)
    assert snapshot["state"] == protocol.STATE_INPUT_REQUIRED
    assert future.result() == (snapshot["state"], snapshot["reply"])


def test_slow_push_does_not_drop_local_records_on_shutdown(adapter, monkeypatch):
    from hermes_constants import get_hermes_home
    entered, release = threading.Event(), threading.Event()

    def slow_push(*args):
        entered.set()
        assert release.wait(10)

    monkeypatch.setattr(adapter, "_send_push_notification", slow_push)
    try:
        for i in range(140):
            tid = str(i)
            adapter.tasks.create(tid, "records", "peer")
            adapter.tasks.set_push_config(tid, "http://localhost/hook")
            adapter._resolve_task(tid, protocol.STATE_COMPLETED, tid)
        assert entered.wait(5)
        assert wait_until(lambda: len(protocol.load_conversation("records", limit=0)) == 140, timeout=5)
        assert not release.is_set()
    finally:
        release.set()
        asyncio.run(adapter.disconnect())
    records = protocol.load_conversation("records", limit=0)
    assert {r["task_id"] for r in records} == {str(i) for i in range(140)}
    audit = [json.loads(line) for line in (get_hermes_home() / "a2a_audit.jsonl").read_text().splitlines()]
    assert {r["task_id"] for r in audit if r["direction"] == "outbound"} == {str(i) for i in range(140)}


def test_subscription_uses_atomic_snapshot_during_cancel_and_eviction(adapter, monkeypatch):
    adapter.tasks.create("t", "c", "peer")
    adapter._resolve_task("t", protocol.STATE_INPUT_REQUIRED, "Question?")
    original = adapter.tasks.watch_snapshot

    def churn(*args):
        observation = original(*args)
        adapter._rpc_tasks_cancel(1, {"id": "t"})
        for i in range(501):
            adapter.tasks.create(str(i), "c", "peer")
            adapter.tasks.complete(str(i), protocol.STATE_COMPLETED, str(i))
        return observation

    monkeypatch.setattr(adapter.tasks, "watch_snapshot", churn)
    output = io.BytesIO()
    handler = SimpleNamespace(wfile=output, send_response=lambda *a: None,
                              send_header=lambda *a: None, end_headers=lambda: None)
    try:
        adapter._rpc_tasks_subscribe(handler, 1, {"id": "t"})
        frames = [json.loads(line[6:])["result"] for line in output.getvalue().decode().splitlines()
                  if line.startswith("data: ")]
        assert frames[0]["task"]["status"]["state"] == protocol.STATE_INPUT_REQUIRED
        assert frames[1]["statusUpdate"]["status"]["state"] == protocol.STATE_INPUT_REQUIRED
        assert not adapter.tasks._watchers
    finally:
        asyncio.run(adapter.disconnect())


def test_recording_capacity_refuses_execution_and_shutdown_drains_accepted_records(adapter):
    async def run():
        async def handle(event):
            pytest.fail("capacity refusal must not execute")

        adapter.set_message_handler(handle)
        assert await adapter.connect()
        try:
            for i in range(512):
                tid = str(i)
                adapter.tasks.create(tid, "capacity", "peer")
                assert adapter._add_pending(tid, "capacity") is not None
            result = await asyncio.to_thread(adapter._rpc_message_send, 1, params(), "peer")
            task = result["result"]
            assert task["status"]["state"] == protocol.STATE_FAILED
            assert "capacity" in protocol.extract_text(task["status"]["message"])
        finally:
            await adapter.disconnect()
        records = protocol.load_conversation("capacity", limit=0)
        assert len(records) == 512
        assert {r["task_id"] for r in records} == {str(i) for i in range(512)}

    asyncio.run(run())
