"""Connection-independent A2A completion, exercised without model calls."""
import asyncio
import http.client
import json
import socket
import threading
from pathlib import Path
from typing import Any
from types import SimpleNamespace
from concurrent.futures import ThreadPoolExecutor

import pytest

from gateway.config import PlatformConfig
from gateway.platforms.event import MessageEvent, ProcessingOutcome
from plugins.platforms.a2a import adapter as module, protocol
from tests.waiting import async_wait_until


def params(text: str = "hello", **configuration: object) -> dict[str, Any]:
    return {"message": protocol.text_message(protocol.ROLE_USER, text),
            "configuration": configuration}


def request(port: int, method: str, arguments: dict[str, Any]) -> tuple[http.client.HTTPConnection, http.client.HTTPResponse]:
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
    conn.request("POST", "/", json.dumps({"jsonrpc": "2.0", "id": 1,
                 "method": method, "params": arguments}), {"Content-Type": "application/json"})
    return conn, conn.getresponse()


def frame(response: http.client.HTTPResponse) -> str:
    lines: list[bytes] = []
    while line := response.readline():
        if line == b"\n":
            return b"".join(lines).decode()
        lines.append(line)
    return ""


@pytest.fixture
def adapter(monkeypatch: pytest.MonkeyPatch) -> module.A2AAdapter:
    monkeypatch.setenv("A2A_PORT", "0")
    monkeypatch.delenv("A2A_BEARER_TOKEN", raising=False)
    monkeypatch.delenv("A2A_PEER_TOKENS", raising=False)
    monkeypatch.setattr(module, "_SSE_KEEPALIVE", .05)
    monkeypatch.setattr(module, "_reply_timeout", lambda: .1)
    return module.A2AAdapter(PlatformConfig(enabled=True))


@pytest.mark.integration
def test_closed_initial_socket_keeps_result_and_releases_observer(adapter, monkeypatch):
    exited = threading.Event()
    original = adapter._rpc_message_stream

    def observe(*args, **kwargs):
        try:
            return original(*args, **kwargs)
        finally:
            exited.set()

    monkeypatch.setattr(adapter, "_rpc_message_stream", observe)
    async def run():
        entered, release = asyncio.Event(), asyncio.Event()

        async def handle(event: MessageEvent):
            entered.set()
            await release.wait()
            await adapter.send(event.source.chat_id, "eventual exact result", metadata={"notify": True})

        adapter.handle_message = handle
        adapter._message_handler = object()
        assert await adapter.connect()
        port = adapter._httpd.server_port
        try:
            conn, response = await asyncio.to_thread(request, port, "SendStreamingMessage", params())
            first = await asyncio.to_thread(frame, response)
            task = json.loads(first.removeprefix("data: "))["result"]["task"]
            await entered.wait()
            conn.sock.shutdown(socket.SHUT_RDWR) if conn.sock else response.fp.raw._sock.shutdown(socket.SHUT_RDWR)
            response.close()
            conn.close()
            assert await asyncio.to_thread(exited.wait, 5)
            release.set()
            await async_wait_until(lambda: adapter.tasks.get(task["id"])["state"] == protocol.STATE_COMPLETED, timeout=5)
            connection, result = await asyncio.to_thread(request, port, "GetTask", {"id": task["id"]})
            stored = json.loads(await asyncio.to_thread(result.read))["result"]
            connection.close()
            assert stored["contextId"] == task["contextId"]
            assert protocol.extract_text(stored["artifacts"][0]) == "eventual exact result"
        finally:
            release.set()
            await adapter.disconnect()

    asyncio.run(run())


@pytest.mark.integration
@pytest.mark.parametrize("reply,state", [("result", protocol.STATE_COMPLETED),
                                        ("[INPUT_REQUIRED] Which?", protocol.STATE_INPUT_REQUIRED)])
def test_healthy_stream_survives_old_timer(adapter, reply, state):
    async def run():
        release = asyncio.Event()

        async def handle(event: MessageEvent):
            await release.wait()
            await adapter.send(event.source.chat_id, reply, metadata={"notify": True})

        adapter.handle_message = handle
        adapter._message_handler = object()
        assert await adapter.connect()
        conn = None
        try:
            conn, response = await asyncio.to_thread(request, adapter._httpd.server_port, "SendStreamingMessage", params())
            assert '"task"' in await asyncio.to_thread(frame, response)
            assert "WORKING" in await asyncio.to_thread(frame, response)
            for _ in range(4):
                assert "keepalive" in await asyncio.to_thread(frame, response)
            release.set()
            body = (await asyncio.to_thread(response.read)).decode()
            assert state in body
            assert ("Which?" if state == protocol.STATE_INPUT_REQUIRED else "result") in body
        finally:
            release.set()
            if conn:
                conn.close()
            await adapter.disconnect()

    asyncio.run(run())


def test_completion_stores_before_waking_and_only_winner_counts(adapter, monkeypatch):
    adapter.tasks.create("t", "c", "peer")
    future = adapter._add_pending("t", "c")
    seen = []
    future.add_done_callback(lambda _: seen.append(adapter.tasks.get("t")["state"]))
    before = protocol.metrics.tasks_completed
    asyncio.run(adapter.send("c", "answer", metadata={"notify": True}))
    assert adapter.tasks.get("t")["reply"] == "answer"
    assert seen == [protocol.STATE_COMPLETED]
    asyncio.run(adapter.on_processing_complete(SimpleNamespace(message_id="t"), ProcessingOutcome.FAILURE))
    assert protocol.metrics.tasks_completed == before + 1
    assert adapter.tasks.get("t")["reply"] == "answer"
    asyncio.run(adapter.disconnect())


def test_nonblocking_send_continues_without_waiter_thread(adapter):
    async def run():
        release = asyncio.Event()

        async def handle(event: MessageEvent):
            await release.wait()
            event._streamed_final_response = "streamed fallback"
            await adapter.on_processing_complete(event, ProcessingOutcome.SUCCESS)

        adapter.handle_message = handle
        adapter._message_handler = object()
        assert await adapter.connect()
        try:
            baseline = set(threading.enumerate())
            tasks = []
            for _ in range(12):
                result = await asyncio.to_thread(adapter._rpc_message_send, 1, params(returnImmediately=True), "peer")
                task = result["result"]
                assert task["status"]["state"] == protocol.STATE_WORKING
                tasks.append(task)
            assert len(set(threading.enumerate()) - baseline) <= 2
            release.set()
            await async_wait_until(lambda: all(adapter.tasks.get(t["id"])["reply"] == "streamed fallback" for t in tasks), timeout=5)
        finally:
            release.set()
            await adapter.disconnect()

    asyncio.run(run())


def test_blocking_send_survives_timer_and_shutdown_releases_it(adapter):
    async def run():
        entered = asyncio.Event()

        async def handle(event: MessageEvent):
            event._gateway_accepted = True
            entered.set()

        adapter.handle_message = handle
        adapter._message_handler = object()
        assert await adapter.connect()
        waiting = asyncio.create_task(asyncio.to_thread(adapter._rpc_message_send, 1, params(), "peer"))
        try:
            await entered.wait()
            # An event-loop timer advances past the former request deadline.
            timer = asyncio.Event()
            asyncio.get_running_loop().call_later(.2, timer.set)
            await timer.wait()
            assert not waiting.done()
            assert next(iter(adapter.tasks._tasks.values()))["state"] == protocol.STATE_WORKING
        finally:
            await adapter.disconnect()
        result = await asyncio.wait_for(waiting, 5)
        assert result["result"]["status"]["state"] == protocol.STATE_FAILED

    asyncio.run(run())


@pytest.mark.integration
def test_subscribers_get_snapshot_and_one_disconnect_is_independent(adapter):
    async def run():
        adapter.tasks.create("t", "c", "peer")
        adapter.tasks.set_state("t", protocol.STATE_WORKING)
        assert await adapter.connect()
        connections = []
        try:
            for _ in range(2):
                conn, response = await asyncio.to_thread(request, adapter._httpd.server_port, "SubscribeToTask", {"id": "t"})
                connections.append((conn, response))
                snapshot = json.loads((await asyncio.to_thread(frame, response)).removeprefix("data: "))["result"]["task"]
                assert snapshot["id"] == "t"
                assert snapshot["status"]["state"] == protocol.STATE_WORKING
            assert len(adapter.tasks._watchers["t"]) == 2
            conn, response = connections[0]
            response.fp.raw._sock.shutdown(socket.SHUT_RDWR)
            response.close()
            conn.close()
            await async_wait_until(lambda: len(adapter.tasks._watchers.get("t", [])) == 1, timeout=5)
            adapter._resolve_task("t", protocol.STATE_COMPLETED, "shared result")
            body = (await asyncio.to_thread(connections[1][1].read)).decode()
            assert "shared result" in body and protocol.STATE_COMPLETED in body
            assert not adapter.tasks._watchers
        finally:
            for conn, response in connections:
                response.close()
                conn.close()
            await adapter.disconnect()

    asyncio.run(run())


def test_slow_failing_push_does_not_block_completion_or_gateway(adapter, monkeypatch):
    entered, release = threading.Event(), threading.Event()

    def slow_open(*args, **kwargs):
        entered.set()
        assert release.wait(timeout=5)
        raise OSError("controlled webhook failure")

    monkeypatch.setattr(module.urllib.request, "urlopen", slow_open)
    monkeypatch.setattr(module.security, "is_safe_callback_url", lambda *args, **kwargs: True)
    adapter.tasks.create("t", "c", "peer")
    adapter.tasks.set_push_config("t", "http://127.0.0.1/callback")
    adapter._add_pending("t", "c")

    async def run():
        before = protocol.metrics.push_failed
        try:
            await adapter.send("c", "ready now", metadata={"notify": True})
            assert await asyncio.to_thread(entered.wait, 5)
            assert adapter._rpc_tasks_get(1, {"id": "t"})["result"]["status"]["state"] == protocol.STATE_COMPLETED
            tick = asyncio.Event()
            asyncio.get_running_loop().call_soon(tick.set)
            await asyncio.wait_for(tick.wait(), 2)
            assert not release.is_set()
        finally:
            release.set()
            await adapter.disconnect()
        assert protocol.metrics.push_failed == before + 1

    asyncio.run(run())


def test_cancel_reply_and_shutdown_have_one_winner(adapter, monkeypatch):
    adapter.tasks.create("t", "c", "peer")
    future = adapter._add_pending("t", "c")
    watcher = adapter.tasks.watch("t")
    watcher.add_done_callback(lambda _: adapter.tasks.get("t"))
    barrier = threading.Barrier(3)
    before = protocol.metrics.tasks_completed + protocol.metrics.tasks_failed

    def reply():
        barrier.wait(timeout=5)
        adapter._resolve_task("t", protocol.STATE_COMPLETED, "winner")

    def cancel():
        barrier.wait(timeout=5)
        adapter._rpc_tasks_cancel(1, {"id": "t"})

    def shutdown():
        barrier.wait(timeout=5)
        asyncio.run(adapter.disconnect())

    with ThreadPoolExecutor(max_workers=3) as pool:
        runs = [pool.submit(fn) for fn in (reply, cancel, shutdown)]
        for run in runs:
            run.result(timeout=10)
    rec = adapter.tasks.get("t")
    assert future.result(timeout=0) == watcher.result(timeout=0) == (rec["state"], rec["reply"])
    assert protocol.metrics.tasks_completed + protocol.metrics.tasks_failed - before == (rec["state"] != protocol.STATE_CANCELED)
    assert not adapter._resolve_task("t", protocol.STATE_COMPLETED, "too late")
    assert adapter.tasks.get("t") == rec


def test_cancelled_turn_cannot_deliver_to_next_task(adapter):
    for tid in ("first", "second"):
        adapter.tasks.create(tid, "c", "peer")
    adapter._add_pending("first", "c")
    second = adapter._add_pending("second", "c")
    adapter._rpc_tasks_cancel(1, {"id": "first"})

    async def run():
        await adapter.send("c", "late first", reply_to="first", metadata={"notify": True})
        assert not second.done()
        await adapter.on_processing_complete(SimpleNamespace(message_id="first"), ProcessingOutcome.SUCCESS)
        await adapter.send("c", "second reply", reply_to="second", metadata={"notify": True})
        assert second.result(timeout=0) == (protocol.STATE_COMPLETED, "second reply")
        await adapter.disconnect()

    asyncio.run(run())


def test_shutdown_during_store_notification_still_releases_send_waiter(adapter):
    adapter.tasks.create("t", "c", "peer")
    future = adapter._add_pending("t", "c")
    notifying, release = threading.Event(), threading.Event()

    def observer(_future):
        notifying.set()
        assert release.wait(timeout=5)

    adapter.tasks.watch("t").add_done_callback(observer)
    completion = threading.Thread(target=lambda: adapter._resolve_task("t", protocol.STATE_COMPLETED, "stored"))
    completion.start()
    try:
        assert notifying.wait(timeout=5)
        asyncio.run(adapter.disconnect())
    finally:
        release.set()
        completion.join(timeout=5)
    assert not completion.is_alive()
    assert future.result(timeout=0) == (protocol.STATE_COMPLETED, "stored")


def test_forwarded_nonblocking_uses_owner_admission(adapter, monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    from gateway import session_a2a
    entered = []

    async def run():
        release = asyncio.Event()

        async def owner(home, **kwargs):
            entered.append(kwargs["input_id"])
            await release.wait()
            return {"outcome": "completed", "result": {"final_response": "owner result"}}

        monkeypatch.setattr(session_a2a, "forward_to_owner", owner)
        monkeypatch.setattr(module, "_profile_home", lambda profile: str(tmp_path))
        assert await adapter.connect()
        agent = {"slug": "dev", "tenant": "dev", "profile": "dev", "local": False}
        try:
            result = await asyncio.to_thread(adapter._rpc_message_send, 1, params(returnImmediately=True), "peer", agent)
            task = result["result"]
            assert task["status"]["state"] == protocol.STATE_WORKING
            await async_wait_until(lambda: bool(entered), timeout=5)
            release.set()
            await async_wait_until(lambda: adapter.tasks.get(task["id"], "dev", "dev")["reply"] == "owner result", timeout=5)
            assert entered[0].startswith("a2a-msg:")
        finally:
            release.set()
            await adapter.disconnect()

    asyncio.run(run())


def test_completion_io_saturation_preserves_results_and_bounds_workers(adapter, monkeypatch):
    entered, release = threading.Event(), threading.Event()

    def slow_open(*args, **kwargs):
        entered.set()
        assert release.wait(timeout=10)
        raise OSError("controlled failure")

    monkeypatch.setattr(module.urllib.request, "urlopen", slow_open)
    monkeypatch.setattr(module.security, "is_safe_callback_url", lambda *args, **kwargs: True)
    before = protocol.metrics.push_failed
    baseline = set(threading.enumerate())
    try:
        for i in range(140):
            tid = f"t-{i}"
            adapter.tasks.create(tid, tid, "peer")
            adapter.tasks.set_push_config(tid, "http://127.0.0.1/hook")
            assert adapter._resolve_task(tid, protocol.STATE_COMPLETED, tid)
        assert entered.wait(timeout=5)
        assert len(set(threading.enumerate()) - baseline) <= 3
        assert protocol.metrics.push_failed == before + 12
        assert all(adapter.tasks.get(f"t-{i}")["reply"] == f"t-{i}" for i in range(140))
    finally:
        # Keep the running workers gated while queued jobs are canceled.
        shutdown = threading.Thread(target=lambda: asyncio.run(adapter.disconnect()))
        shutdown.start()
        try:
            from tests.waiting import wait_until
            assert wait_until(lambda: protocol.metrics.push_failed >= before + 138, timeout=5)
        finally:
            release.set()
            shutdown.join(timeout=5)
    assert not shutdown.is_alive()
    assert protocol.metrics.push_failed == before + 140


@pytest.mark.integration
def test_http_dispatch_and_completion_io_keep_adapter_profile(adapter, monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    from hermes_constants import get_hermes_home, set_hermes_home_override, reset_hermes_home_override

    home = tmp_path / "served-profile"
    token = set_hermes_home_override(home)
    try:
        scoped = module.A2AAdapter(PlatformConfig(enabled=True))
    finally:
        reset_hermes_home_override(token)
    homes = []
    monkeypatch.setattr(module.security, "audit", lambda *args: homes.append(get_hermes_home()))

    async def handle(event: MessageEvent):
        homes.append(get_hermes_home())
        await scoped.send(event.source.chat_id, "profile result", metadata={"notify": True})

    scoped.handle_message = handle
    scoped._message_handler = object()

    async def run():
        assert await scoped.connect()
        try:
            conn, response = await asyncio.to_thread(request, scoped._httpd.server_port, "SendMessage", params())
            task = json.loads(await asyncio.to_thread(response.read))["result"]["task"]
            conn.close()
            await async_wait_until(lambda: len(homes) == 3, timeout=5)
            assert homes == [home, home, home]
            saved = home / "a2a_conversations" / (task["contextId"] + ".jsonl")
            assert "profile result" in saved.read_text()
        finally:
            await scoped.disconnect()

    asyncio.run(run())


def test_watchdog_does_not_fail_a_live_accepted_turn(adapter):
    async def run():
        entered, release = asyncio.Event(), asyncio.Event()

        async def handle(event: MessageEvent):
            entered.set()
            await release.wait()
            await adapter.send(event.source.chat_id, "long-running result", metadata={"notify": True})

        adapter.handle_message = handle
        adapter._message_handler = handle
        assert await adapter.connect()
        try:
            result = await asyncio.to_thread(adapter._rpc_message_send, 1, params(returnImmediately=True), "peer")
            task_id = result["result"]["id"]
            await entered.wait()
            with adapter.tasks._lock:
                adapter.tasks._tasks[task_id]["created_at"] -= module._orphan_timeout() + 1
            assert adapter._fail_orphans_once() == []
            assert adapter.tasks.get(task_id)["state"] == protocol.STATE_WORKING
            release.set()
            await async_wait_until(lambda: adapter.tasks.get(task_id)["state"] == protocol.STATE_COMPLETED)
            assert adapter.tasks.get(task_id)["reply"] == "long-running result"
        finally:
            release.set()
            await adapter.disconnect()

    asyncio.run(run())


@pytest.mark.parametrize("state", [protocol.STATE_INPUT_REQUIRED, "TASK_STATE_AUTH_REQUIRED"])
def test_interrupted_task_can_cancel_but_late_success_cannot_replace_question(adapter, state):
    adapter.tasks.create("t", "c", "peer")
    adapter._add_pending("t", "c")
    assert adapter._resolve_task("t", state, "Which input?")

    async def run():
        event = SimpleNamespace(message_id="t", _streamed_final_response="late success")
        await adapter.on_processing_complete(event, ProcessingOutcome.SUCCESS)
        assert adapter.tasks.get("t")["state"] == state
        assert adapter.tasks.get("t")["reply"] == "Which input?"
        result = adapter._rpc_tasks_cancel(1, {"id": "t"})
        assert result["result"]["status"]["state"] == protocol.STATE_CANCELED
        assert adapter.tasks.watch("t").result(timeout=0) == (protocol.STATE_CANCELED, "")
        await adapter.on_processing_complete(event, ProcessingOutcome.SUCCESS)
        assert adapter.tasks.get("t")["state"] == protocol.STATE_CANCELED
        assert "error" in adapter._rpc_tasks_cancel(1, {"id": "t"})
        await adapter.disconnect()

    asyncio.run(run())


@pytest.mark.integration
@pytest.mark.parametrize("unanchored_send", [False, True])
def test_real_base_queued_turns_keep_task_identity_and_watchdog_ownership(adapter, unanchored_send):
    async def run():
        first_entered, second_entered = asyncio.Event(), asyncio.Event()
        release_first, release_second = asyncio.Event(), asyncio.Event()
        seen = []

        async def handle(event: MessageEvent):
            seen.append(event.message_id)
            if len(seen) == 1:
                first_entered.set()
                await release_first.wait()
                return "first exact result"
            second_entered.set()
            await release_second.wait()
            if unanchored_send:
                await adapter.send(event.source.chat_id, "second exact result", metadata={"notify": True})
                return None
            return "second exact result"

        adapter.set_message_handler(handle)
        assert await adapter.connect()
        connections = []

        async def send(text: str):
            arguments = params(text, returnImmediately=True)
            arguments["message"]["contextId"] = "same-context"
            conn, response = await asyncio.to_thread(request, adapter._httpd.server_port, "SendMessage", arguments)
            connections.append(conn)
            return json.loads(await asyncio.to_thread(response.read))["result"]["task"]["id"]

        try:
            first = await send("first")
            await asyncio.wait_for(first_entered.wait(), 5)
            second = await send("second")
            await async_wait_until(lambda: len(adapter._owned_tasks) == 2, timeout=5)
            for tid in (first, second):
                with adapter.tasks._lock:
                    adapter.tasks._tasks[tid]["created_at"] -= module._orphan_timeout() + 1
            assert adapter._fail_orphans_once() == []
            release_first.set()
            await asyncio.wait_for(second_entered.wait(), 5)
            assert adapter._fail_orphans_once() == []
            release_second.set()
            await async_wait_until(lambda: all(adapter.tasks.get(tid)["state"] == protocol.STATE_COMPLETED
                                              for tid in (first, second)), timeout=5)
            assert seen == [first, second]
            assert adapter.tasks.get(first)["reply"] == "first exact result"
            assert adapter.tasks.get(second)["reply"] == "second exact result"
            await async_wait_until(lambda: not adapter._session_tasks, timeout=5)
        finally:
            release_first.set()
            release_second.set()
            await adapter.cancel_background_tasks()
            for conn in connections:
                conn.close()
            await adapter.disconnect()

    asyncio.run(run())


def test_real_base_refusal_releases_execution_ownership(adapter, monkeypatch):
    async def handle(event: MessageEvent):
        pytest.fail("refused input must not reach the message handler")

    adapter.set_message_handler(handle)
    monkeypatch.setattr(adapter, "_drop_unresolved", lambda event: True)

    async def run():
        assert await adapter.connect()
        try:
            result = await asyncio.to_thread(adapter._rpc_message_send, 1, params(returnImmediately=True), "peer")
            tid = result["result"]["id"]
            await async_wait_until(lambda: adapter.tasks.get(tid)["state"] == protocol.STATE_FAILED, timeout=2)
            assert tid not in adapter._owned_tasks
            assert tid not in adapter._pending
        finally:
            await adapter.disconnect()

    asyncio.run(run())
