"""A2A task lifecycle regression tests for long-running inbound turns."""

from __future__ import annotations

import asyncio
import threading

import pytest

from plugins.platforms.a2a import adapter as adapter_module
from plugins.platforms.a2a import protocol
from tests.plugins.test_a2a_plugin import _make_live_adapter, _post_json
from tests.waiting import async_wait_until


class TestImmediateSendLifecycle:
    def test_cancel_does_not_cross_talk_future(self, monkeypatch):
        adapter, base = _make_live_adapter(monkeypatch)

        async def run():
            assert await adapter.connect() is True
            # Create task
            adapter.tasks.create("t-cancel", "ctx-c", "peer")
            fut = adapter._add_pending("t-cancel", "ctx-c")
            cancel_resp = await asyncio.to_thread(_post_json, base + "/", {
                "jsonrpc": "2.0", "id": "can-1", "method": "CancelTask", "params": {"id": "t-cancel"}
            })
            assert cancel_resp["result"]["status"]["state"] == protocol.STATE_CANCELED
            assert fut.result(timeout=0) == (protocol.STATE_CANCELED, "")
            assert "t-cancel" not in adapter._pending

            next_future = adapter._add_pending("t-next", "ctx-c")
            await adapter.send("ctx-c", "late canceled reply", reply_to="t-cancel", metadata={"notify": True})
            assert not next_future.done()
            await adapter.send("ctx-c", "next reply", reply_to="t-next", metadata={"notify": True})
            assert next_future.result(timeout=0) == (protocol.STATE_COMPLETED, "next reply")
            adapter._pop_pending("t-next")
            await adapter.disconnect()

        asyncio.run(run())

    @pytest.mark.parametrize("return_immediately", [True, None])
    def test_long_turn_stays_working_then_persists_late_reply(self, monkeypatch, return_immediately):
        """Both an explicit immediate send and a blocking wait expiry keep a live turn queryable."""
        monkeypatch.delenv("A2A_BEARER_TOKEN", raising=False)
        monkeypatch.delenv("A2A_PEER_TOKENS", raising=False)
        monkeypatch.setattr(adapter_module, "_reply_timeout", lambda: 0.0)
        adapter, base = _make_live_adapter(monkeypatch)
        started = threading.Event()

        async def run():
            release = asyncio.Event()

            async def held_reply(event):
                started.set()
                await release.wait()
                await adapter.send(event.source.chat_id, "late but valid reply", metadata={"notify": True})

            adapter.handle_message = held_reply
            try:
                assert await adapter.connect() is True
                params = {"message": protocol.text_message(protocol.ROLE_USER, "long task")}
                if return_immediately is True:
                    params["configuration"] = {"returnImmediately": True}
                body = {"jsonrpc": "2.0", "id": "req-1", "method": "SendMessage", "params": params}
                response = await asyncio.to_thread(_post_json, base + "/", body)
                task = response["result"]["task"]
                assert task["status"]["state"] == protocol.STATE_WORKING
                assert started.wait(timeout=2)

                get_body = {"jsonrpc": "2.0", "id": "get-1", "method": "GetTask",
                            "params": {"id": task["id"]}}
                current = await asyncio.to_thread(_post_json, base + "/", get_body)
                assert current["result"]["status"]["state"] == protocol.STATE_WORKING

                release.set()

                def completed_task():
                    result = _post_json(base + "/", get_body)["result"]
                    return result if result["status"]["state"] == protocol.STATE_COMPLETED else None

                completed = await async_wait_until(completed_task, timeout=5, what="late task completion")
                assert protocol.extract_text(completed["artifacts"][0]) == "late but valid reply"
                assert adapter.tasks.get(task["id"])["reply"] == "late but valid reply"
            finally:
                release.set()
                await adapter.disconnect()

        asyncio.run(run())

    def test_cancelled_task_ignores_late_completion(self, monkeypatch):
        adapter, _base = _make_live_adapter(monkeypatch)
        settlement_jobs = []
        monkeypatch.setattr(adapter_module, "_daemon_thread", lambda target, _name: settlement_jobs.append(target))
        pushes = []
        monkeypatch.setattr(adapter, "_send_push_notification", lambda *args: pushes.append(args))
        context_id, task_id = "ctx-cancel-late", "task-cancel-late"
        rec = adapter.tasks.create(task_id, context_id, "peer")
        future = adapter._add_pending(task_id, context_id)
        pending = {"task_id": task_id, "context_id": context_id, "peer": "peer", "future": future,
                   "created_iso": rec["created_iso"], "started": rec["created_at"]}
        adapter._watch_pending_task(pending)
        metrics = protocol.metrics
        before = (metrics.outbound_total, metrics.tasks_completed, metrics.tasks_failed)

        assert adapter.tasks.complete(task_id, protocol.STATE_CANCELED, "") is not None
        adapter._resolve_task(task_id, protocol.STATE_CANCELED, "")
        adapter._pop_pending(task_id)
        for job in settlement_jobs:
            job()
        adapter._finalize_task(pending, protocol.STATE_COMPLETED, "late result")

        assert adapter.tasks.get(task_id)["state"] == protocol.STATE_CANCELED
        assert not any(m.get("task_id") == task_id and m.get("role") == "agent"
                       for m in protocol.load_conversation(context_id))
        assert (metrics.outbound_total, metrics.tasks_completed, metrics.tasks_failed) == before
        assert pushes == []
        assert task_id not in adapter._pending

    def test_hard_timed_out_task_ignores_late_completion(self, monkeypatch):
        adapter, _base = _make_live_adapter(monkeypatch)
        settlement_jobs = []
        monkeypatch.setattr(adapter_module, "_daemon_thread", lambda target, _name: settlement_jobs.append(target))
        pushes = []
        monkeypatch.setattr(adapter, "_send_push_notification", lambda *args: pushes.append(args))
        monkeypatch.setattr(adapter, "_task_timeout", lambda: 1.0)
        context_id, task_id = "ctx-timeout-late", "task-timeout-late"
        rec = adapter.tasks.create(task_id, context_id, "peer")
        future = adapter._add_pending(task_id, context_id)
        pending = {"task_id": task_id, "context_id": context_id, "peer": "peer", "future": future,
                   "created_iso": rec["created_iso"], "started": rec["created_at"]}
        adapter._watch_pending_task(pending)
        with adapter.tasks._lock:
            adapter.tasks._tasks[task_id]["created_at"] = 0.0

        assert adapter._fail_orphans_once() == [task_id]
        assert task_id not in adapter._pending
        before = (protocol.metrics.outbound_total, protocol.metrics.tasks_completed, protocol.metrics.tasks_failed)
        for job in settlement_jobs:
            job()
        adapter._finalize_task(pending, protocol.STATE_COMPLETED, "late result")

        assert adapter.tasks.get(task_id)["state"] == protocol.STATE_FAILED
        assert any(m.get("task_id") == task_id and m.get("role") == "agent"
                   and m.get("text") == "[task orphaned — no reply produced]"
                   for m in protocol.load_conversation("ctx-timeout-late"))
        assert (protocol.metrics.outbound_total, protocol.metrics.tasks_completed, protocol.metrics.tasks_failed) == before
        assert any(p[0] == task_id and p[3] == protocol.STATE_FAILED for p in pushes)

    def test_send_correlates_reply_to_task_id_before_context_fifo(self, monkeypatch):
        adapter, _base = _make_live_adapter(monkeypatch)
        context_id = "ctx-concurrent"
        first = adapter._add_pending("task-first", context_id)
        second = adapter._add_pending("task-second", context_id)

        asyncio.run(adapter.send(context_id, "answer for second", reply_to="task-second", metadata={"notify": True}))

        assert not first.done()
        assert second.result(timeout=0) == (protocol.STATE_COMPLETED, "answer for second")
        adapter._pop_pending("task-first")
        adapter._pop_pending("task-second")

    def test_a2a_settles_all_ids_absorbed_into_one_base_event(self, monkeypatch):
        from gateway.platforms.base import merge_pending_message_event
        from gateway.platforms.event import MessageEvent, MessageType, ProcessingOutcome

        adapter, _base = _make_live_adapter(monkeypatch)
        context_id = "ctx-merged"
        first_id, second_id, third_id = "task-merged-first", "task-merged-second", "task-merged-third"
        first_future = adapter._add_pending(first_id, context_id)
        second_future = adapter._add_pending(second_id, context_id)
        third_future = adapter._add_pending(third_id, context_id)
        first_event = MessageEvent(text="first part", message_type=MessageType.TEXT, message_id=first_id)
        second_event = MessageEvent(text="second part", message_type=MessageType.TEXT, message_id=second_id)
        third_event = MessageEvent(text="third part", message_type=MessageType.TEXT, message_id=third_id)
        pending = {}
        merge_pending_message_event(pending, context_id, first_event, merge_text=True)
        merge_pending_message_event(pending, context_id, second_event, merge_text=True)
        merge_pending_message_event(pending, context_id, third_event, merge_text=True)
        merged = pending[context_id]
        merged.source = type("Source", (), {"chat_id": context_id})()

        # Gateway executes turn and calls send() on the delivery adapter with notify=True:
        asyncio.run(adapter.send(context_id, "combined non-streamed answer", reply_to=first_id, metadata={"notify": True}))
        # Then gateway calls on_processing_complete:
        asyncio.run(adapter.on_processing_complete(merged, ProcessingOutcome.SUCCESS))

        assert first_future.result(timeout=0) == (protocol.STATE_COMPLETED, "combined non-streamed answer")
        assert second_future.done()
        assert second_future.result(timeout=0) == (protocol.STATE_COMPLETED, "combined non-streamed answer")
        assert third_future.done()
        assert third_future.result(timeout=0) == (protocol.STATE_COMPLETED, "combined non-streamed answer")
        adapter._pop_pending(first_id)
        adapter._pop_pending(second_id)
        adapter._pop_pending(third_id)

        # Subsequent streamed turn must not reuse the consumed non-streamed answer:
        streamed_id = "task-streamed-next"
        streamed_fut = adapter._add_pending(streamed_id, context_id)
        streamed_event = MessageEvent(text="streamed turn", message_type=MessageType.TEXT, message_id=streamed_id)
        streamed_event.source = type("Source", (), {"chat_id": context_id})()
        streamed_event._streamed_final_response = "fresh streamed answer"
        asyncio.run(adapter.on_processing_complete(streamed_event, ProcessingOutcome.SUCCESS))
        assert streamed_fut.result(timeout=0) == (protocol.STATE_COMPLETED, "fresh streamed answer")
        adapter._pop_pending(streamed_id)

    def test_stream_disconnect_leaves_task_working_and_disconnect_drains_pending(self, monkeypatch):
        from tests.waiting import wait_until
        import time as _t
        adapter, _base = _make_live_adapter(monkeypatch)
        context_id = "ctx-stream"
        task_id = "task-stream-disconn"
        adapter.tasks.create(task_id, context_id, "peer-caller")
        adapter.tasks.set_state(task_id, protocol.STATE_WORKING)
        fut = adapter._add_pending(task_id, context_id)
        pending = {"task_id": task_id, "context_id": context_id, "peer": "peer-caller", "future": fut, "started": _t.time()}

        # Simulate BrokenPipeError in stream handler:
        with pytest.raises(BrokenPipeError):
            try:
                raise BrokenPipeError("client hung up")
            except (BrokenPipeError, ConnectionResetError):
                adapter._watch_pending_task(pending)
                raise

        # Task in store must remain working immediately after disconnect:
        rec = adapter.tasks.get(task_id)
        assert rec["state"] == protocol.STATE_WORKING

        # When the agent finishes late, watch callback must settle it to COMPLETED!
        adapter._resolve_task(task_id, protocol.STATE_COMPLETED, "eventual streaming answer")
        wait_until(lambda: (adapter.tasks.get(task_id) or {}).get("state") == protocol.STATE_COMPLETED, timeout=2.0)

        rec = adapter.tasks.get(task_id)
        assert rec["state"] == protocol.STATE_COMPLETED
        assert rec["reply"] == "eventual streaming answer"

    def test_rpc_message_stream_disconnect_hands_off_to_watcher(self, monkeypatch):
        import time as _t
        from tests.waiting import wait_until
        import plugins.platforms.a2a.adapter as adapter_module

        # Keepalive runs frequently so BrokenPipeError triggers immediately during the wait loop
        monkeypatch.setattr(adapter_module, "_SSE_KEEPALIVE", 0.05)
        adapter, _base = _make_live_adapter(monkeypatch)
        adapter.tasks.create("t-live-stream", "ctx-live-stream", "peer")
        adapter.tasks.set_state("t-live-stream", protocol.STATE_WORKING)
        fut = adapter._add_pending("t-live-stream", "ctx-live-stream")
        pending = {"task_id": "t-live-stream", "context_id": "ctx-live-stream", "peer": "peer", "future": fut,
                   "created_iso": "now", "started": _t.time()}

        class BrokenHandler:
            def __init__(self):
                self.flushes = 0
            def _json(self, *a, **k): pass
            def send_response(self, code): pass
            def send_header(self, k, v): pass
            def end_headers(self): pass
            @property
            def wfile(self):
                class WFile:
                    def write(self_inner, data): pass
                    def flush(self_inner):
                        self.flushes += 1
                        # Initial task and status updates flush (1, 2); the keepalive in _await_future is 3rd:
                        if self.flushes >= 3:
                            raise BrokenPipeError("client disconnected on keepalive")
                return WFile()

        handler = BrokenHandler()
        monkeypatch.setattr(adapter, "_prepare_task", lambda *a, **k: (None, pending))
        adapter._rpc_message_stream(handler, "req-1", {"message": {}}, "peer")

        # Proven: broken pipe occurred on keepalive flush
        assert handler.flushes >= 3
        # Task must still be in _pending (not prematurely popped!)
        assert "t-live-stream" in adapter._pending
        assert adapter.tasks.get("t-live-stream")["state"] == protocol.STATE_WORKING

        # Late completion must resolve the task correctly through the watcher:
        adapter._resolve_task("t-live-stream", protocol.STATE_COMPLETED, "eventual answer via real rpc stream")
        wait_until(lambda: (adapter.tasks.get("t-live-stream") or {}).get("state") == protocol.STATE_COMPLETED, timeout=2.0)
        assert adapter.tasks.get("t-live-stream")["reply"] == "eventual answer via real rpc stream"
        wait_until(lambda: "t-live-stream" not in adapter._pending, timeout=2.0)

    def test_task_store_retention_and_profile_scope(self):
        store = protocol.TaskStore()
        # Create tasks across two different profiles / agent slugs
        store.create("task-a1", "ctx-1", "peer", agent_slug="agent-alpha")
        store.create("task-b1", "ctx-1", "peer", agent_slug="agent-beta")
        store.complete("task-a1", protocol.STATE_COMPLETED, "alpha reply")
        store.complete("task-b1", protocol.STATE_COMPLETED, "beta reply")

        # Scoped access isolation
        assert store.get("task-a1", agent_slug="agent-alpha") is not None
        assert store.get("task-a1", agent_slug="agent-beta") is None
        assert store.get("task-b1", agent_slug="agent-beta") is not None
        assert store.get("task-b1", agent_slug="agent-alpha") is None

        # Retention limit trimming
        original_max = store._MAX_TERMINAL
        try:
            store._MAX_TERMINAL = 3
            store.create("t1", "ctx", "p")
            store.create("t2", "ctx", "p")
            store.create("t3", "ctx", "p")
            store.create("t4", "ctx", "p")
            store.complete("t1", protocol.STATE_COMPLETED)
            store.complete("t2", protocol.STATE_INPUT_REQUIRED)
            store.complete("t3", protocol.STATE_COMPLETED)
            store.complete("t4", protocol.STATE_COMPLETED)
            # t1 should be trimmed as terminal/input-required count exceeded 3
            assert store.get("t1") is None
            assert store.get("t4") is not None
            # Duplicate completion on input-required task must be rejected:
            assert store.complete("t2", protocol.STATE_COMPLETED, "duplicate") is None
            # Force terminal transition (e.g. CancelTask or hard timeout) on INPUT_REQUIRED is permitted:
            cancelled = store.complete("t2", protocol.STATE_CANCELED, "force cancel", force_terminal=True)
            assert cancelled is not None
            assert store.get("t2")["state"] == protocol.STATE_CANCELED
        finally:
            store._MAX_TERMINAL = original_max
