"""Explicit same-task continuation through HTTP and stale delivery callbacks."""
import asyncio
import json
from pathlib import Path

import pytest

from gateway.platforms.event import MessageEvent, ProcessingOutcome
from plugins.platforms.a2a import protocol
from tests.plugins import test_a2a_completion_lifecycle as lifecycle
from tests.waiting import async_wait_until

adapter = lifecycle.adapter
params = lifecycle.params
request = lifecycle.request


@pytest.mark.integration
@pytest.mark.parametrize("stream", [False, True])
def test_http_same_task_continues_with_distinct_attempt_and_old_observer(adapter, stream: bool):
    async def run():
        seen: list[MessageEvent] = []
        resumed, release = asyncio.Event(), asyncio.Event()

        async def handle(event: MessageEvent):
            seen.append(event)
            if len(seen) == 1:
                return "[INPUT_REQUIRED] Which colour?"
            resumed.set()
            await release.wait()
            return "Blue result"

        adapter.set_message_handler(handle)
        assert await adapter.connect()
        port = adapter._httpd.server_port
        try:
            conn, response = await asyncio.to_thread(request, port, "SendMessage", params())
            first = json.loads(await asyncio.to_thread(response.read))["result"]["task"]
            conn.close()
            tid = first["id"]
            original = adapter.tasks.get(tid)
            old = adapter.tasks.watch(tid)
            arguments = params("Blue", returnImmediately=not stream)
            arguments["message"]["taskId"] = tid
            method = "SendStreamingMessage" if stream else "SendMessage"
            conn, response = await asyncio.to_thread(request, port, method, arguments)
            await asyncio.wait_for(resumed.wait(), 5)
            assert adapter.tasks.get(tid)["state"] == protocol.STATE_WORKING
            assert adapter.tasks.get(tid)["created_at"] == original["created_at"]
            assert seen[1].message_id != seen[0].message_id
            await adapter.send(first["contextId"], "stale send", reply_to=seen[0].message_id, metadata={"notify": True})
            seen[0]._streamed_final_response = "stale fallback"
            await adapter.on_processing_complete(seen[0], ProcessingOutcome.SUCCESS)
            assert adapter.tasks.get(tid)["state"] == protocol.STATE_WORKING
            release.set()
            body = (await asyncio.to_thread(response.read)).decode()
            conn.close()
            await async_wait_until(lambda: adapter.tasks.get(tid)["state"] == protocol.STATE_COMPLETED, timeout=5)
            final = adapter.tasks.get(tid)
            assert final["reply"] == "Blue result"
            assert final["context_id"] == first["contextId"]
            assert old.result() == (protocol.STATE_INPUT_REQUIRED, "Which colour?")
            if stream:
                assert tid in body and "Blue result" in body
            else:
                assert json.loads(body)["result"]["task"]["id"] == tid
        finally:
            release.set()
            await adapter.cancel_background_tasks()
            await adapter.disconnect()
        history = protocol.load_conversation(first["contextId"])
        assert [r["text"] for r in history if r["role"] == "agent"] == ["Which colour?", "Blue result"]

    asyncio.run(run())


def test_unavailable_gateway_after_resume_records_failure_and_releases_observer(adapter):
    adapter.tasks.create("t", "c", "peer")
    adapter._resolve_task("t", protocol.STATE_INPUT_REQUIRED, "Which?")
    arguments = params("answer")
    arguments["message"]["taskId"] = "t"
    try:
        result = adapter._rpc_message_send(1, arguments, "peer")
        assert result["result"]["status"]["state"] == protocol.STATE_FAILED
        assert not adapter._pending
        assert not adapter._owned_tasks
        assert not adapter.tasks._watchers
    finally:
        asyncio.run(adapter.disconnect())
    assert protocol.load_conversation("c")[-1]["text"] == "Agent gateway not ready to accept A2A tasks."


def test_unanchored_legacy_send_cannot_complete_resumed_attempt(adapter):
    from plugins.platforms.a2a.continuation import _admit
    adapter.tasks.create("t", "c", "peer")
    adapter._resolve_task("t", protocol.STATE_INPUT_REQUIRED, "Which?")
    try:
        rec, future, error, retry = _admit(adapter, "t", None, ("", ""), adapter._generation, "answer")
        assert error is None and not retry
        asyncio.run(adapter.send("c", "unbound stale answer", metadata={"notify": True}))
        assert not future.done()
        assert adapter.tasks.get("t")["attempt_id"] == rec["attempt_id"]
    finally:
        asyncio.run(adapter.disconnect())


def test_continuation_respects_context_turn_cap_without_stranding_question(adapter, monkeypatch):
    monkeypatch.setenv("A2A_MAX_PINGPONG_TURNS", "1")
    adapter.tasks.create("t", "c", "peer")
    adapter._turns.track("c")
    adapter._resolve_task("t", protocol.STATE_INPUT_REQUIRED, "Which?")
    arguments = params("answer")
    arguments["message"]["taskId"] = "t"
    try:
        result = adapter._rpc_message_send(1, arguments, "peer")
        assert result["error"]["code"] == protocol.ERR_UNSUPPORTED_OPERATION
        assert adapter.tasks.get("t")["state"] == protocol.STATE_INPUT_REQUIRED
        assert not adapter._pending
        assert not adapter.tasks._watchers
    finally:
        asyncio.run(adapter.disconnect())


def test_context_only_creates_new_task_and_continuation_retry_after_terminal_is_rejected(adapter):
    async def run():
        async def handle(event: MessageEvent):
            return "[INPUT_REQUIRED] Which?" if "hello" in event.text else "done"

        adapter.set_message_handler(handle)
        assert await adapter.connect()
        try:
            first = await asyncio.to_thread(adapter._rpc_message_send, 1, params(), "peer")
            tid, cid = first["result"]["id"], first["result"]["contextId"]
            separate = params("other work")
            separate["message"]["contextId"] = cid
            result = await asyncio.to_thread(adapter._rpc_message_send, 2, separate, "peer")
            assert result["result"]["id"] != tid
            assert result["result"]["contextId"] == cid
            answer = params("answer")
            answer["message"]["taskId"] = tid
            finished = await asyncio.to_thread(adapter._rpc_message_send, 3, answer, "peer")
            assert finished["result"]["status"]["state"] == protocol.STATE_COMPLETED
            repeated = await asyncio.to_thread(adapter._rpc_message_send, 4, answer, "peer")
            assert repeated["error"]["code"] == protocol.ERR_UNSUPPORTED_OPERATION
            assert adapter.tasks.list(with_total=True)[2] == 2
        finally:
            await adapter.cancel_background_tasks()
            await adapter.disconnect()

    asyncio.run(run())


@pytest.mark.parametrize("state", [protocol.STATE_INPUT_REQUIRED, "TASK_STATE_AUTH_REQUIRED"])
def test_push_configuration_survives_interruption_and_notifies_resumed_outcome(adapter, monkeypatch, state):
    from plugins.platforms.a2a.continuation import _admit
    import threading
    pushes, notified = [], threading.Event()

    def push(task_id: str, context_id: str, reply: str, outcome: str, url: str) -> None:
        pushes.append((task_id, outcome, url))
        if len(pushes) == 2:
            notified.set()

    monkeypatch.setattr(adapter, "_send_push_notification", push)
    adapter.tasks.create("t", "c", "peer")
    adapter.tasks.set_push_config("t", "http://localhost/hook")
    try:
        adapter._resolve_task("t", state, "Which?")
        assert adapter.tasks.get_push_config("t")["pushNotificationConfig"]["url"] == "http://localhost/hook"
        rec, future, error, retry = _admit(adapter, "t", None, ("", ""), adapter._generation, "answer")
        assert error is None and not retry
        adapter._resolve_delivery(rec["attempt_id"], protocol.STATE_COMPLETED, "done")
        assert future.result(0) == (protocol.STATE_COMPLETED, "done")
        assert notified.wait(5)
        assert sorted(outcome for _, outcome, _ in pushes) == sorted([state, protocol.STATE_COMPLETED])
        assert {url for _, _, url in pushes} == {"http://localhost/hook"}
    finally:
        asyncio.run(adapter.disconnect())


def test_shutdown_between_resume_admission_and_dispatch_releases_attempt(adapter, monkeypatch):
    import threading
    entered, release = threading.Event(), threading.Event()
    original = adapter._register_inline_push

    def gate(*args, **kwargs):
        entered.set()
        assert release.wait(5)
        return original(*args, **kwargs)

    async def run():
        seen = []

        async def handle(event: MessageEvent):
            seen.append(event.message_id)
            return "unexpected"

        adapter.set_message_handler(handle)
        assert await adapter.connect()
        adapter.tasks.create("t", "c", "peer")
        adapter._resolve_task("t", protocol.STATE_INPUT_REQUIRED, "Which?")
        monkeypatch.setattr(adapter, "_register_inline_push", gate)
        arguments = params("answer")
        arguments["message"]["taskId"] = "t"
        waiting = asyncio.create_task(asyncio.to_thread(adapter._rpc_message_send, 1, arguments, "peer"))
        try:
            assert await asyncio.to_thread(entered.wait, 5)
            await adapter.disconnect()
            release.set()
            result = await asyncio.wait_for(waiting, 5)
            assert result["result"]["status"]["state"] == protocol.STATE_FAILED
            assert not adapter._pending
            assert not adapter._owned_tasks
            assert not adapter.tasks._watchers
            assert not adapter._completion_io._reserved
            assert not seen
        finally:
            release.set()
            await waiting
            await adapter.cancel_background_tasks()
            await adapter.disconnect()

    asyncio.run(run())


def test_root_scope_does_not_read_list_cancel_or_mutate_foreign_tasks(adapter):
    adapter.tasks.create("foreign", "c", "peer", "other", "tenant")
    adapter.tasks.set_push_config("foreign", "http://localhost/hook")
    try:
        assert adapter._rpc_tasks_get(1, {"id": "foreign"})["error"]["code"] == protocol.ERR_TASK_NOT_FOUND
        assert adapter._rpc_tasks_list(2, {})["result"]["tasks"] == []
        assert adapter._rpc_tasks_cancel(3, {"id": "foreign"})["error"]["code"] == protocol.ERR_TASK_NOT_FOUND
        assert adapter._rpc_push_config_get(4, {"taskId": "foreign"})["error"]["code"] == protocol.ERR_TASK_NOT_FOUND
        assert adapter._rpc_push_config_create(5, {"taskId": "foreign", "pushNotificationConfig": {
            "url": "http://localhost/other"}})["error"]["code"] == protocol.ERR_TASK_NOT_FOUND
        assert adapter.tasks.get("foreign")["state"] == protocol.STATE_SUBMITTED
        assert adapter.tasks.get_push_config("foreign")["pushNotificationConfig"]["url"] == "http://localhost/hook"
    finally:
        asyncio.run(adapter.disconnect())


def test_sync_forwarding_returns_committed_clarifications_and_final_result(adapter, monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    from gateway import session_a2a
    from plugins.platforms.a2a import adapter as module
    calls = []

    async def owner(home, **kwargs):
        calls.append(kwargs)
        reply = ["[INPUT_REQUIRED] Which?", "[INPUT_REQUIRED] Anything else?", "final answer"][len(calls) - 1]
        return {"outcome": "completed", "result": {"final_response": reply}}

    monkeypatch.setattr(session_a2a, "forward_to_owner", owner)
    monkeypatch.setattr(module, "_profile_home", lambda _: str(tmp_path))
    agent = {"slug": "dev", "tenant": "dev", "profile": "dev", "local": False}
    assert adapter._loop is None
    try:
        first = adapter._rpc_message_send(1, params(), "peer", agent)["result"]
        tid, cid = first["id"], first["contextId"]
        created_at = adapter.tasks.get(tid)["created_at"]
        assert first["status"]["state"] == protocol.STATE_INPUT_REQUIRED
        assert protocol.extract_text(first["status"]["message"]) == "Which?"
        for text, state, reply in [("answer", protocol.STATE_INPUT_REQUIRED, "Anything else?"),
                                   ("last input", protocol.STATE_COMPLETED, "final answer")]:
            arguments = params(text)
            arguments["message"]["taskId"] = tid
            result = adapter._rpc_message_send(2, arguments, "peer", agent)["result"]
            assert result["id"] == tid and result["contextId"] == cid
            assert result["status"]["state"] == state
            assert protocol.extract_text(result["status"]["message"]) == reply
            stored = adapter.tasks.get(tid)
            assert stored["state"] == state and stored["reply"] == reply
            assert stored["created_at"] == created_at
        assert len({call["input_id"] for call in calls}) == 3
    finally:
        asyncio.run(adapter.disconnect())


def test_sync_forwarding_observer_keeps_cancel_outcome_after_retention_purge(adapter, monkeypatch):
    agent = {"slug": "dev", "tenant": "dev", "profile": "dev", "local": False}

    def owner(*args, **kwargs):
        tid = next(iter(adapter._pending))
        assert "result" in adapter._rpc_tasks_cancel(1, {"id": tid}, agent)
        for i in range(501):
            adapter.tasks.create(str(i), "other", "peer")
            adapter.tasks.complete(str(i), protocol.STATE_COMPLETED, "churn")
        assert adapter.tasks.get(tid) is None
        return "late reply", protocol.STATE_COMPLETED

    monkeypatch.setattr(adapter, "_forward_to_profile", owner)
    try:
        result = adapter._rpc_message_send(1, params(), "peer", agent)["result"]
        assert result["status"]["state"] == protocol.STATE_CANCELED
        assert "artifacts" not in result
        assert not adapter._pending
    finally:
        asyncio.run(adapter.disconnect())


@pytest.mark.parametrize("target,context,code", [
    ("missing", None, -32001), ("t", "wrong", -32602),
    ("terminal", None, -32004), ("foreign", None, -32001),
    (123, None, -32602), (" t ", None, -32001),
])
def test_invalid_continuation_does_not_create_or_execute(adapter, target: str | int, context: str | None, code: int):
    adapter.tasks.create("t", "c", "peer")
    adapter.tasks.complete("t", protocol.STATE_INPUT_REQUIRED, "Which?")
    adapter.tasks.create("terminal", "c", "peer")
    adapter.tasks.complete("terminal", protocol.STATE_COMPLETED, "done")
    adapter.tasks.create("foreign", "c", "peer", "other", "other")
    arguments = params()
    arguments["message"]["taskId"] = target
    if context is not None:
        arguments["message"]["contextId"] = context
    try:
        result = adapter._rpc_message_send(1, arguments, "peer")
        assert result["error"]["code"] == code
        assert adapter.tasks.list(with_total=True)[2] == 3
        assert not adapter._pending
    finally:
        asyncio.run(adapter.disconnect())


def test_forwarded_repeated_interruptions_and_retry_have_distinct_owner_admissions(adapter, monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    from gateway import session_a2a
    from plugins.platforms.a2a import adapter as module

    async def run():
        calls = []
        entered, release = asyncio.Event(), asyncio.Event()

        async def owner(home, **kwargs):
            calls.append(kwargs)
            if len(calls) == 2:
                entered.set()
                await release.wait()
            text = "[INPUT_REQUIRED] Another?" if len(calls) < 3 else "done"
            return {"outcome": "completed", "result": {"final_response": text}}

        monkeypatch.setattr(session_a2a, "forward_to_owner", owner)
        monkeypatch.setattr(module, "_profile_home", lambda _: str(tmp_path))
        assert await adapter.connect()
        agent = {"slug": "dev", "tenant": "dev", "profile": "dev", "local": False}
        try:
            first = await asyncio.to_thread(adapter._rpc_message_send, 1, params(), "peer", agent)
            tid, cid = first["result"]["id"], first["result"]["contextId"]
            answer = params("answer", returnImmediately=True)
            answer["message"]["taskId"] = tid
            resumed = await asyncio.to_thread(adapter._rpc_message_send, 2, answer, "peer", agent)
            await asyncio.wait_for(entered.wait(), 5)
            retry = await asyncio.to_thread(adapter._rpc_message_send, 3, answer, "peer", agent)
            assert "error" not in retry
            assert retry["result"]["id"] == resumed["result"]["id"] == tid
            assert len(calls) == 2
            release.set()
            await async_wait_until(lambda: adapter.tasks.get(tid)["state"] == protocol.STATE_INPUT_REQUIRED, timeout=5)
            final = params("last answer")
            final["message"]["taskId"] = tid
            result = await asyncio.to_thread(adapter._rpc_message_send, 4, final, "peer", agent)
            assert result["result"]["status"]["state"] == protocol.STATE_COMPLETED
            assert result["result"]["contextId"] == cid
            assert len({call["input_id"] for call in calls}) == 3
            assert {call["context_id"] for call in calls} == {cid}
        finally:
            release.set()
            await adapter.disconnect()

    asyncio.run(run())
