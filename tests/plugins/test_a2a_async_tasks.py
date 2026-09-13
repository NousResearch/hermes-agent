"""Behavioral tests for non-blocking A2A task submission and retrieval."""

from __future__ import annotations

import asyncio
import threading
import urllib.error
from concurrent.futures import Future

import pytest

from gateway.config import PlatformConfig
from plugins.platforms.a2a import adapter as adapter_module
from plugins.platforms.a2a import protocol, security, tools
from plugins.platforms.a2a.adapter import A2AAdapter


def _bare_adapter() -> A2AAdapter:
    """Create an adapter without opening a socket or starting the gateway."""
    return A2AAdapter(PlatformConfig(enabled=True))


def _configured_peer(monkeypatch) -> None:
    monkeypatch.setattr(
        tools,
        "_load_config",
        lambda: {"a2a_agents": {"peer": {"url": "http://localhost:9999"}}},
    )
    monkeypatch.setattr(tools, "_http_get_json", lambda url, headers, timeout: None)
    monkeypatch.setattr(protocol, "persist_message", lambda *args, **kwargs: None)
    monkeypatch.setattr(security, "audit", lambda *args, **kwargs: None)


def _isolate_server_side_effects(monkeypatch, tmp_path) -> None:
    """Keep task persistence and audit writes inside pytest-owned state."""
    monkeypatch.setattr(protocol, "_hermes_home", lambda: tmp_path)
    monkeypatch.setattr(security, "audit", lambda *args, **kwargs: None)


def _status_text(task: dict) -> str:
    return protocol.extract_text((task.get("status") or {}).get("message") or {})


def _send_then_get(adapter: A2AAdapter, params: dict) -> tuple[dict, dict]:
    sent = adapter._rpc_message_send("send", params, "peer", v1_response=True)
    sent_task = protocol.unwrap_send_message_response(sent["result"])
    fetched = adapter._rpc_tasks_get("get", {"id": sent_task["id"]})
    return sent_task, fetched["result"]


def test_anti_loop_explanation_survives_send_then_get(monkeypatch):
    """Anti-loop rejection keeps the same explanation in the stored Task."""
    monkeypatch.setenv("A2A_MAX_PINGPONG_TURNS", "1")
    adapter = _bare_adapter()
    adapter._turns.track("ctx-loop")

    sent, fetched = _send_then_get(
        adapter,
        {"message": protocol.text_message(protocol.ROLE_USER, "loop", "ctx-loop")},
    )

    assert sent["status"]["state"] == protocol.STATE_REJECTED
    assert _status_text(sent) == _status_text(fetched)
    assert "Anti-loop protection" in _status_text(fetched)


def test_empty_message_explanation_survives_send_then_get():
    """Empty-message rejection keeps its explanation in the stored Task."""
    adapter = _bare_adapter()

    sent, fetched = _send_then_get(
        adapter,
        {"message": protocol.text_message(protocol.ROLE_USER, "", "ctx-empty")},
    )

    assert sent["status"]["state"] == protocol.STATE_REJECTED
    assert _status_text(sent) == _status_text(fetched) == "Empty task — nothing to do."


def test_not_ready_explanation_survives_send_then_get():
    """A not-ready gateway failure keeps its explanation in the stored Task."""
    adapter = _bare_adapter()

    sent, fetched = _send_then_get(
        adapter,
        {"message": protocol.text_message(protocol.ROLE_USER, "work", "ctx-not-ready")},
    )

    assert sent["status"]["state"] == protocol.STATE_FAILED
    assert _status_text(sent) == _status_text(fetched)
    assert "gateway not ready" in _status_text(fetched)


def test_finalizer_thread_start_failure_returns_failed_task_and_cleans_pending(
    monkeypatch, tmp_path
):
    """A detached finalizer that cannot start must not leave WORKING ownership."""
    _isolate_server_side_effects(monkeypatch, tmp_path)

    class StartFails:
        def __init__(self, *args, **kwargs):
            pass

        def start(self):
            raise RuntimeError("finalizer thread start failed")

    monkeypatch.setattr(adapter_module.threading, "Thread", StartFails)
    adapter = _bare_adapter()
    record = adapter.tasks.create("task-finalizer", "ctx-finalizer", "peer")
    adapter.tasks.set_state("task-finalizer", protocol.STATE_WORKING)
    future: Future = adapter._add_pending("task-finalizer", "ctx-finalizer")
    pending = {
        "task_id": "task-finalizer",
        "context_id": "ctx-finalizer",
        "peer": "peer",
        "future": future,
        "created_iso": record["created_iso"],
        "started": 0.0,
    }
    monkeypatch.setattr(adapter, "_prepare_task", lambda *args, **kwargs: (None, pending))

    response = adapter._rpc_message_send(
        "rpc-finalizer",
        {"configuration": {"returnImmediately": True}},
        "peer",
        v1_response=True,
    )
    task = protocol.unwrap_send_message_response(response["result"])
    stored = adapter.tasks.get("task-finalizer")

    assert task["status"]["state"] == protocol.STATE_FAILED
    assert stored["state"] == protocol.STATE_FAILED
    assert "finalizer thread start failed" in stored["reply"]
    assert "task-finalizer" not in adapter._pending


def test_forward_thread_start_failure_returns_failed_task_and_cleans_pending(
    monkeypatch, tmp_path
):
    """A detached profile forwarder that cannot start follows the failed outcome path."""
    _isolate_server_side_effects(monkeypatch, tmp_path)

    class StartFails:
        def __init__(self, *args, **kwargs):
            pass

        def start(self):
            raise RuntimeError("forward thread start failed")

    monkeypatch.setattr(adapter_module.threading, "Thread", StartFails)
    adapter = A2AAdapter(
        PlatformConfig(
            enabled=True,
            extra={"agents": {"dev": {"profile": "dev", "tenant": "dev"}}},
        )
    )
    agent = adapter._agents["dev"]

    response = adapter._rpc_message_send(
        "rpc-forward",
        {
            "tenant": "dev",
            "message": protocol.text_message(
                protocol.ROLE_USER, "slow profile work", "ctx-forward"
            ),
            "configuration": {"returnImmediately": True},
        },
        "peer",
        agent=agent,
        v1_response=True,
    )
    task = protocol.unwrap_send_message_response(response["result"])
    stored = adapter.tasks.get(task["id"], "dev", "dev")

    assert task["status"]["state"] == protocol.STATE_FAILED
    assert stored["state"] == protocol.STATE_FAILED
    assert "forward thread start failed" in stored["reply"]
    assert task["id"] not in adapter._pending


def test_call_can_submit_nonblocking_task(monkeypatch):
    """Non-blocking calls request immediate return and expose the task id."""
    _configured_peer(monkeypatch)
    captured = {}

    def fake_post(url, body, headers, timeout):
        captured["body"] = body
        return protocol.jsonrpc_result(
            body["id"],
            protocol.build_task(
                "task-async",
                "ctx-async",
                protocol.STATE_WORKING,
                "",
            ),
        )

    monkeypatch.setattr(tools, "_http_post_json", fake_post)
    output = tools.a2a_call(
        {"agent": "peer", "message": "slow work", "wait": False}
    )

    configuration = captured["body"]["params"]["configuration"]
    assert configuration["returnImmediately"] is True
    assert "task-async" in output
    assert "working" in output


def test_get_task_fetches_terminal_reply(monkeypatch):
    """A submitted task remains retrievable after the send request returns."""
    _configured_peer(monkeypatch)
    captured = {}

    def fake_post(url, body, headers, timeout):
        captured["body"] = body
        return protocol.jsonrpc_result(
            body["id"],
            protocol.build_task(
                "task-async",
                "ctx-async",
                protocol.STATE_COMPLETED,
                "finished result",
            ),
        )

    monkeypatch.setattr(tools, "_http_post_json", fake_post)
    output = tools.a2a_get_task({"agent": "peer", "task_id": "task-async"})

    assert captured["body"]["method"] == "GetTask"
    assert captured["body"]["params"]["id"] == "task-async"
    assert "completed" in output
    assert "finished result" in output


def test_wait_task_polls_until_terminal(monkeypatch):
    """Waiting polls the detached task without controlling its execution."""
    _configured_peer(monkeypatch)
    states = iter(
        [
            {
                "reply": "",
                "context_id": "ctx-async",
                "state": protocol.STATE_WORKING,
                "task_id": "task-async",
            },
            {
                "reply": "finished result",
                "context_id": "ctx-async",
                "state": protocol.STATE_COMPLETED,
                "task_id": "task-async",
            },
        ]
    )
    calls = []

    def fake_get(agent, peer, task_id, request_timeout=None):
        calls.append((agent, task_id))
        return next(states)

    monkeypatch.setattr(tools, "_get_task_result", fake_get)
    monkeypatch.setattr(tools.time, "sleep", lambda _seconds: None)

    output = tools.a2a_wait(
        {
            "agent": "peer",
            "task_id": "task-async",
            "timeout": 1,
            "poll_interval": 0.01,
        }
    )

    assert calls == [("peer", "task-async"), ("peer", "task-async")]
    assert "completed" in output
    assert "finished result" in output


def test_wait_timeout_does_not_cancel_remote_task(monkeypatch):
    """A local wait deadline leaves the detached remote task untouched."""
    _configured_peer(monkeypatch)
    calls = []

    def fake_get(agent, peer, task_id, request_timeout=None):
        calls.append((agent, task_id))
        return {
            "reply": "",
            "context_id": "ctx-async",
            "state": protocol.STATE_WORKING,
            "task_id": task_id,
        }

    monkeypatch.setattr(tools, "_get_task_result", fake_get)
    output = tools.a2a_wait(
        {"agent": "peer", "task_id": "task-async", "timeout": 0}
    )

    assert calls == [("peer", "task-async")]
    assert "not canceled" in output
    assert "a2a_get_task" in output


def test_wait_bounds_get_request_by_remaining_deadline(monkeypatch):
    """One slow HTTP lookup cannot overrun the caller's local wait budget."""
    _configured_peer(monkeypatch)
    request_timeouts = []

    def fake_get(agent, peer, task_id, request_timeout=None):
        request_timeouts.append(request_timeout)
        return {
            "reply": "finished result",
            "context_id": "ctx-async",
            "state": protocol.STATE_COMPLETED,
            "task_id": task_id,
        }

    monkeypatch.setattr(tools, "_get_task_result", fake_get)
    monkeypatch.setattr(tools.time, "monotonic", lambda: 100.0)

    output = tools.a2a_wait(
        {"agent": "peer", "task_id": "task-async", "timeout": 0.25}
    )

    assert 0 < request_timeouts[0] <= 0.25
    assert "completed" in output


def test_wait_budget_covers_card_fallback_and_get_task(monkeypatch):
    """Agent-card fallback and GetTask share one fake monotonic budget."""
    _configured_peer(monkeypatch)
    clock = [0.0]
    card_timeouts = []
    task_timeouts = []

    def fake_monotonic():
        return clock[0]

    def fake_get(url, headers, timeout):
        card_timeouts.append(timeout)
        if url.endswith("/.well-known/agent-card.json"):
            clock[0] += 0.6
            raise urllib.error.HTTPError(url, 404, "missing", {}, None)
        clock[0] += 0.2
        return {
            "supportedInterfaces": [
                {"protocolBinding": "JSONRPC", "url": "http://peer/rpc"}
            ]
        }

    def fake_post(url, body, headers, timeout):
        task_timeouts.append(timeout)
        clock[0] += timeout
        raise TimeoutError("GetTask timed out")

    monkeypatch.setattr(tools.time, "monotonic", fake_monotonic)
    monkeypatch.setattr(tools, "_http_get_json", fake_get)
    monkeypatch.setattr(tools, "_http_post_json", fake_post)

    output = tools.a2a_wait(
        {
            "agent": "peer",
            "task_id": "task-async",
            "timeout": 1,
            "poll_interval": 0.1,
        }
    )

    assert card_timeouts == [pytest.approx(1.0), pytest.approx(0.4)]
    assert task_timeouts == [pytest.approx(0.2)]
    assert clock[0] == pytest.approx(1.0)
    assert "Local wait timed out; the remote task was not canceled." in output
    assert "task lookup" not in output


def test_repeated_get_does_not_duplicate_persisted_reply(monkeypatch, tmp_path):
    """A terminal reply stays idempotent after more than 50 newer messages."""
    monkeypatch.setattr(protocol, "_hermes_home", lambda: tmp_path)
    task = {
        "reply": "finished result",
        "context_id": "ctx-async",
        "state": protocol.STATE_COMPLETED,
        "task_id": "task-async",
    }
    protocol.persist_message(
        task["context_id"], "agent", task["reply"], task["task_id"]
    )
    for index in range(60):
        protocol.persist_message(
            task["context_id"], "user", f"newer-{index}", f"newer-{index}"
        )

    tools._persist_task_reply(task)
    tools._persist_task_reply(task)

    messages = protocol.load_conversation(task["context_id"], limit=200)
    matches = [
        message
        for message in messages
        if message.get("role") == "agent"
        and message.get("task_id") == task["task_id"]
    ]
    assert len(matches) == 1


def test_concurrent_task_reply_persistence_is_single_write(monkeypatch, tmp_path):
    """Concurrent polls append one record for the same task identity."""
    monkeypatch.setattr(protocol, "_hermes_home", lambda: tmp_path)
    task = {
        "reply": "finished result",
        "context_id": "ctx-concurrent",
        "state": protocol.STATE_COMPLETED,
        "task_id": "task-concurrent",
    }
    ready = threading.Barrier(2)
    errors = []

    def poll_task():
        try:
            ready.wait()
            tools._persist_task_reply(task)
        except BaseException as exc:  # pragma: no cover - diagnostic only
            errors.append(exc)

    workers = [threading.Thread(target=poll_task) for _ in range(2)]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join(2.0)

    messages = protocol.load_conversation(task["context_id"], limit=20)
    matches = [
        message
        for message in messages
        if message.get("role") == "agent"
        and message.get("task_id") == task["task_id"]
    ]
    assert not errors
    assert all(not worker.is_alive() for worker in workers)
    assert len(matches) == 1


def test_async_tools_are_registered_with_wait_schema():
    """The runtime exposes submit/get/wait through the A2A toolset."""
    from types import SimpleNamespace
    registered = {}
    tools.register_tools(SimpleNamespace(register_tool=lambda **item: registered.update({item["name"]: item})))
    call_properties = registered["a2a_call"]["schema"]["parameters"]["properties"]
    assert call_properties["wait"]["type"] == "boolean"
    assert registered["a2a_get_task"]["handler"] is tools.a2a_get_task
    assert registered["a2a_wait"]["handler"] is tools.a2a_wait
    assert registered["a2a_get_task"]["schema"]["parameters"][
        "required"
    ] == ["agent", "task_id"]
    assert registered["a2a_wait"]["schema"]["parameters"][
        "required"
    ] == ["agent", "task_id"]


def test_http_detached_task_round_trip(monkeypatch, tmp_path):
    """Real HTTP submit returns before gateway completion and remains queryable."""
    monkeypatch.delenv("A2A_PORT", raising=False)
    monkeypatch.delenv("A2A_TOKEN", raising=False)
    monkeypatch.setenv("A2A_BIND_HOST", "127.0.0.1")
    _isolate_server_side_effects(monkeypatch, tmp_path)

    async def scenario():
        adapter = _bare_adapter()
        adapter.port = 0
        release = asyncio.Event()
        finalized = threading.Event()

        async def delayed_handle(event):
            await release.wait()
            await adapter.send(
                event.source.chat_id,
                "finished over HTTP",
                metadata={"notify": True},
            )

        adapter.set_message_handler(delayed_handle)
        original_finalize = adapter._finalize_task

        def finalize_task(*args, **kwargs):
            try:
                return original_finalize(*args, **kwargs)
            finally:
                finalized.set()

        adapter._finalize_task = finalize_task  # type: ignore[method-assign]
        assert await adapter.connect()
        try:
            assert adapter._httpd is not None
            port = adapter._httpd.server_address[1]
            peer = {"url": f"http://127.0.0.1:{port}", "auth": {}, "timeout": 2}

            submitted = await asyncio.wait_for(
                asyncio.to_thread(
                    tools._send_task_result,
                    "peer",
                    peer,
                    "slow task",
                    "ctx-http",
                    return_immediately=True,
                ),
                timeout=2.0,
            )
            assert submitted["task_id"]
            assert submitted["state"] == protocol.STATE_WORKING

            working = await asyncio.to_thread(
                tools._get_task_result,
                "peer",
                peer,
                submitted["task_id"],
            )
            assert working["state"] == protocol.STATE_WORKING

            release.set()
            assert await asyncio.to_thread(finalized.wait, 2.0)
            final = await asyncio.to_thread(
                tools._get_task_result,
                "peer",
                peer,
                submitted["task_id"],
            )
            assert final["state"] == protocol.STATE_COMPLETED
            assert final["reply"] == "finished over HTTP"
        finally:
            release.set()
            await adapter.disconnect()

    asyncio.run(scenario())


def test_send_return_immediately_finalizes_in_background(monkeypatch, tmp_path):
    """The HTTP response returns WORKING while task completion stays live."""
    _isolate_server_side_effects(monkeypatch, tmp_path)
    adapter = _bare_adapter()
    record = adapter.tasks.create("task-async", "ctx-async", "peer")
    adapter.tasks.set_state("task-async", protocol.STATE_WORKING)
    future: Future = adapter._add_pending("task-async", "ctx-async")
    pending = {
        "task_id": "task-async",
        "context_id": "ctx-async",
        "peer": "peer",
        "future": future,
        "created_iso": record["created_iso"],
        "started": 0.0,
    }
    monkeypatch.setattr(adapter, "_prepare_task", lambda *args, **kwargs: (None, pending))
    monkeypatch.setattr(
        adapter,
        "_await_reply",
        lambda _pending: pytest.fail("non-blocking send must not await the reply"),
    )

    finalized = threading.Event()
    original_finalize = adapter._finalize_task

    def finalize_task(*args, **kwargs):
        try:
            return original_finalize(*args, **kwargs)
        finally:
            finalized.set()

    monkeypatch.setattr(adapter, "_finalize_task", finalize_task)
    response = adapter._rpc_message_send(
        "rpc-1",
        {
            "message": protocol.text_message(
                protocol.ROLE_USER,
                "slow work",
                context_id="ctx-async",
            ),
            "configuration": {"returnImmediately": True},
        },
        "peer",
        v1_response=True,
    )
    task = protocol.unwrap_send_message_response(response["result"])

    assert task["id"] == "task-async"
    assert task["status"]["state"] == protocol.STATE_WORKING
    assert not future.done()

    adapter.tasks._tasks["task-async"]["created_at"] = 0.0
    assert adapter._fail_orphaned_tasks(timeout_seconds=1) == []

    future.set_result((protocol.STATE_COMPLETED, "finished result"))
    assert finalized.wait(2.0)
    stored = adapter.tasks.get("task-async")
    assert stored["state"] == protocol.STATE_COMPLETED
    assert stored["reply"] == "finished result"
    assert "task-async" not in adapter._pending


def test_forwarded_profile_return_immediately_runs_in_background(monkeypatch, tmp_path):
    """Profile-backed agents also detach only for non-blocking requests."""
    _isolate_server_side_effects(monkeypatch, tmp_path)
    adapter = A2AAdapter(
        PlatformConfig(
            enabled=True,
            extra={"agents": {"dev": {"profile": "dev", "tenant": "dev"}}},
        )
    )
    agent = adapter._agents["dev"]
    release = threading.Event()
    finalized = threading.Event()

    def fake_forward(agent_arg, peer, context_id, framed_text):
        assert agent_arg["slug"] == "dev"
        assert release.wait(2.0)
        return "profile result", protocol.STATE_COMPLETED

    adapter._forward_to_profile = fake_forward  # type: ignore[method-assign]
    original_finalize = adapter._finalize_task

    def finalize_task(*args, **kwargs):
        try:
            return original_finalize(*args, **kwargs)
        finally:
            finalized.set()

    monkeypatch.setattr(adapter, "_finalize_task", finalize_task)
    response = adapter._rpc_message_send(
        "rpc-profile",
        {
            "tenant": "dev",
            "message": protocol.text_message(
                protocol.ROLE_USER,
                "slow profile work",
                context_id="ctx-profile",
            ),
            "configuration": {"returnImmediately": True},
        },
        "peer",
        agent=agent,
        v1_response=True,
    )
    task = protocol.unwrap_send_message_response(response["result"])

    assert task["status"]["state"] == protocol.STATE_WORKING
    assert not finalized.is_set()
    release.set()
    assert finalized.wait(2.0)
    stored = adapter.tasks.get(task["id"], "dev", "dev")
    assert stored["state"] == protocol.STATE_COMPLETED
    assert stored["reply"] == "profile result"


def test_orphan_sweep_preserves_active_pending_task():
    """Elapsed wall time alone must not kill an actively executing task."""
    adapter = _bare_adapter()
    adapter.tasks.create("task-active", "ctx-active", "peer")
    adapter.tasks.create("task-orphan", "ctx-orphan", "peer")
    adapter.tasks.set_state("task-active", protocol.STATE_WORKING)
    adapter.tasks.set_state("task-orphan", protocol.STATE_WORKING)
    adapter.tasks._tasks["task-active"]["created_at"] = 0.0
    adapter.tasks._tasks["task-orphan"]["created_at"] = 0.0
    adapter._add_pending("task-active", "ctx-active")

    try:
        failed = adapter._fail_orphaned_tasks(timeout_seconds=1)
    finally:
        adapter._pop_pending("task-active")

    assert failed == ["task-orphan"]
    assert adapter.tasks.get("task-active")["state"] == protocol.STATE_WORKING
    assert adapter.tasks.get("task-orphan")["state"] == protocol.STATE_FAILED


def test_task_store_orphan_sweep_can_exclude_active_task():
    """TaskStore keeps an active old task while failing an equally old orphan."""
    store = protocol.TaskStore()
    store.create("task-active", "ctx-active", "peer")
    store.create("task-orphan", "ctx-orphan", "peer")
    store._tasks["task-active"]["created_at"] = 0.0
    store._tasks["task-orphan"]["created_at"] = 0.0

    failed = store.fail_orphans(
        timeout_seconds=1,
        active_task_ids={"task-active"},
    )

    assert failed == ["task-orphan"]
    assert store.get("task-active")["state"] == protocol.STATE_SUBMITTED
    assert store.get("task-orphan")["state"] == protocol.STATE_FAILED
