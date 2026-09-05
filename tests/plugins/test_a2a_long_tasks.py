"""A2A long-running task behavior."""

from __future__ import annotations

import asyncio
import io
import json
import multiprocessing
import threading
import time
from email.message import Message
from pathlib import Path

from gateway.config import PlatformConfig
from plugins.platforms.a2a import adapter as adapter_module
from plugins.platforms.a2a import protocol, security, tools
from plugins.platforms.a2a.adapter import A2AAdapter


def _adapter() -> A2AAdapter:
    return A2AAdapter(PlatformConfig(enabled=True))


def _pending(adapter: A2AAdapter, task_id: str = "task-long", context_id: str = "ctx-long"):
    adapter.tasks.create(task_id, context_id, "doctor")
    adapter.tasks.set_state(task_id, protocol.STATE_WORKING)
    future = adapter._add_pending(task_id, context_id)
    assert future is not None
    return {
        "task_id": task_id,
        "context_id": context_id,
        "peer": "doctor",
        "future": future,
        "created_iso": protocol.now_iso(),
        "started": time.time(),
    }, future


def _persist_once_worker(directory: str, start) -> None:
    protocol._conv_dir = lambda: Path(directory)
    start.wait(5)
    protocol.persist_message_once(
        "ctx-shared-process",
        "agent",
        "finished",
        "task-shared-process",
    )


def test_return_immediately_acknowledges_work_before_reply() -> None:
    adapter = _adapter()
    pending, future = _pending(adapter)
    adapter._prepare_task = lambda params, peer, agent=None: (None, pending)  # type: ignore

    def must_not_block(*_args, **_kwargs):
        raise AssertionError("returnImmediately must not use the blocking reply path")

    adapter._await_reply = must_not_block  # type: ignore

    response = adapter._rpc_message_send(
        1,
        {"configuration": {"returnImmediately": True}, "message": {}},
        "doctor",
    )

    task = response["result"]
    assert task["id"] == "task-long"
    assert task["status"]["state"] == protocol.STATE_WORKING
    assert future.done() is False


def test_return_immediately_finalizes_after_the_agent_replies(monkeypatch) -> None:
    adapter = _adapter()
    pending, future = _pending(adapter)
    adapter._prepare_task = lambda params, peer, agent=None: (None, pending)  # type: ignore
    monkeypatch.setattr(protocol, "persist_message", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(security, "audit", lambda *_args, **_kwargs: None)

    adapter._rpc_message_send(
        1,
        {"configuration": {"returnImmediately": True}, "message": {}},
        "doctor",
    )
    future.set_result((protocol.STATE_COMPLETED, "finished"))

    deadline = time.time() + 1
    record = adapter.tasks.get("task-long")
    assert record is not None
    while record["state"] != protocol.STATE_COMPLETED and time.time() < deadline:
        time.sleep(0.01)
        record = adapter.tasks.get("task-long")
        assert record is not None

    assert record["state"] == protocol.STATE_COMPLETED
    assert record["reply"] == "finished"
    assert "task-long" not in adapter._pending


def test_nonlocal_return_immediately_does_not_block_forwarding(monkeypatch) -> None:
    adapter = _adapter()
    release = threading.Event()
    response: dict = {}
    agent = {
        "local": False,
        "slug": "finance",
        "profile": "finance",
        "tenant": "finance",
    }

    def forward(*_args, **_kwargs):
        release.wait(1)
        return "forwarded result", protocol.STATE_COMPLETED

    monkeypatch.setattr(adapter, "_forward_to_profile", forward)
    monkeypatch.setattr(protocol, "persist_message", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(security, "audit", lambda *_args, **_kwargs: None)

    def call() -> None:
        response.update(
            adapter._rpc_message_send(
                1,
                {
                    "configuration": {"returnImmediately": True},
                    "message": protocol.text_message(
                        protocol.ROLE_USER,
                        "run the long task",
                        context_id="ctx-forwarded",
                    ),
                },
                "doctor",
                agent=agent,
            )
        )

    caller = threading.Thread(target=call)
    caller.start()
    caller.join(0.1)
    returned_immediately = not caller.is_alive()
    release.set()
    caller.join(1)

    assert returned_immediately is True
    task = response["result"]
    assert task["status"]["state"] == protocol.STATE_WORKING

    deadline = time.time() + 1
    record = adapter.tasks.get(task["id"])
    assert record is not None
    while record["state"] != protocol.STATE_COMPLETED and time.time() < deadline:
        time.sleep(0.01)
        record = adapter.tasks.get(task["id"])
        assert record is not None
    assert record["reply"] == "forwarded result"


def test_immediate_background_rejects_a_second_task_in_the_same_context(monkeypatch) -> None:
    adapter = _adapter()
    release = threading.Event()
    agent = {
        "local": False,
        "slug": "finance",
        "profile": "finance",
        "tenant": "finance",
    }

    def forward(*_args, **_kwargs):
        release.wait(1)
        return "finished", protocol.STATE_COMPLETED

    monkeypatch.setattr(adapter, "_forward_to_profile", forward)
    monkeypatch.setattr(protocol, "persist_message", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(security, "audit", lambda *_args, **_kwargs: None)
    params = {
        "configuration": {"returnImmediately": True},
        "message": protocol.text_message(
            protocol.ROLE_USER,
            "run this",
            context_id="ctx-capacity",
        ),
    }

    first = adapter._rpc_message_send(1, params, "doctor", agent=agent)["result"]
    second = adapter._rpc_message_send(2, params, "doctor", agent=agent)["result"]
    release.set()

    assert first["status"]["state"] == protocol.STATE_WORKING
    assert second["status"]["state"] == protocol.STATE_REJECTED


def test_immediate_background_has_a_global_inflight_limit() -> None:
    adapter = _adapter()
    for index in range(adapter_module._MAX_BACKGROUND_TASKS):
        future = adapter._add_pending(
            f"task-{index}",
            f"context-{index}",
            bounded_background=True,
        )
        assert future is not None

    assert adapter._add_pending(
        "task-over-limit",
        "context-over-limit",
        bounded_background=True,
    ) is None


def test_orphan_sweep_preserves_a_live_nonblocking_task() -> None:
    adapter = _adapter()
    pending, _future = _pending(adapter)
    adapter._prepare_task = lambda params, peer, agent=None: (None, pending)  # type: ignore
    adapter._rpc_message_send(
        1,
        {"configuration": {"returnImmediately": True}, "message": {}},
        "doctor",
    )
    adapter.tasks._tasks["task-long"]["created_at"] = time.time() - 600

    adapter._sweep_orphans()

    record = adapter.tasks.get("task-long")
    assert record is not None
    assert record["state"] == protocol.STATE_WORKING


def test_orphan_sweep_preserves_a_reply_awaiting_finalization() -> None:
    adapter = _adapter()
    _pending_record, future = _pending(adapter)
    adapter.tasks._tasks["task-long"]["created_at"] = time.time() - 600
    future.set_result((protocol.STATE_COMPLETED, "finished"))

    adapter._sweep_orphans()

    record = adapter.tasks.get("task-long")
    assert record is not None
    assert record["state"] == protocol.STATE_WORKING
    assert "task-long" in adapter._pending


def test_orphan_sweep_bounds_a_hung_nonblocking_task(monkeypatch) -> None:
    adapter = _adapter()
    pending, _future = _pending(adapter)
    persisted: list[str] = []
    pushed: list[str] = []
    adapter._prepare_task = lambda params, peer, agent=None: (None, pending)  # type: ignore
    monkeypatch.setattr(
        protocol,
        "persist_message",
        lambda _context, _role, text, _task: persisted.append(text),
    )
    monkeypatch.setattr(security, "audit", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        adapter,
        "_send_push_notification",
        lambda *_args, **_kwargs: pushed.append("sent"),
    )
    adapter._rpc_message_send(
        1,
        {"configuration": {"returnImmediately": True}, "message": {}},
        "doctor",
    )
    monkeypatch.setattr(adapter_module, "_BACKGROUND_TASK_TIMEOUT", 30)
    adapter.tasks._tasks["task-long"]["created_at"] = time.time() - 301

    adapter._sweep_orphans()

    record = adapter.tasks.get("task-long")
    assert record is not None
    assert record["state"] == protocol.STATE_FAILED
    assert "task-long" not in adapter._pending
    assert persisted == ["[task orphaned — no reply produced]"]
    assert pushed == ["sent"]


def test_late_reply_cannot_refinalize_a_failed_task(monkeypatch) -> None:
    adapter = _adapter()
    pending, _future = _pending(adapter)
    monkeypatch.setattr(adapter_module, "_BACKGROUND_TASK_TIMEOUT", 30)
    adapter.tasks._tasks["task-long"]["created_at"] = time.time() - 301
    persisted: list[str] = []
    monkeypatch.setattr(
        protocol,
        "persist_message",
        lambda _context, _role, text, _task: persisted.append(text),
    )
    monkeypatch.setattr(security, "audit", lambda *_args, **_kwargs: None)

    adapter._sweep_orphans()
    completed_before = protocol.metrics.tasks_completed
    state, reply = adapter._finalize_task(
        pending,
        protocol.STATE_COMPLETED,
        "late success",
    )

    assert state == protocol.STATE_FAILED
    assert "orphaned" in reply
    assert persisted == ["[task orphaned — no reply produced]"]
    assert protocol.metrics.tasks_completed == completed_before


def test_losing_terminal_transition_has_no_completion_side_effects(monkeypatch) -> None:
    adapter = _adapter()
    pending, _future = _pending(adapter)
    persisted: list[str] = []
    pushed: list[str] = []
    original_complete = adapter.tasks.complete

    def lose_transition(task_id: str, _state: str, _reply: str = ""):
        original_complete(task_id, protocol.STATE_FAILED, "[task already failed]")
        return None

    monkeypatch.setattr(adapter.tasks, "complete", lose_transition)
    monkeypatch.setattr(
        protocol,
        "persist_message",
        lambda _context, _role, text, _task: persisted.append(text),
    )
    monkeypatch.setattr(security, "audit", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        adapter,
        "_send_push_notification",
        lambda *_args, **_kwargs: pushed.append("sent"),
    )
    completed_before = protocol.metrics.tasks_completed

    state, reply = adapter._finalize_task(
        pending,
        protocol.STATE_COMPLETED,
        "late success",
    )

    assert state == protocol.STATE_FAILED
    assert reply == "[task already failed]"
    assert persisted == []
    assert pushed == []
    assert protocol.metrics.tasks_completed == completed_before


def test_input_required_can_be_canceled_but_not_completed() -> None:
    store = protocol.TaskStore()
    store.create("task-question", "ctx-question", "doctor")

    first = store.complete(
        "task-question",
        protocol.STATE_INPUT_REQUIRED,
        "Which account?",
    )
    late_success = store.complete(
        "task-question",
        protocol.STATE_COMPLETED,
        "late answer",
    )
    canceled = store.complete(
        "task-question",
        protocol.STATE_CANCELED,
        "",
    )

    assert first is not None
    assert late_success is None
    assert canceled is not None
    record = store.get("task-question")
    assert record is not None
    assert record["state"] == protocol.STATE_CANCELED


def test_cancel_task_uses_the_terminal_finalization_path(monkeypatch) -> None:
    adapter = _adapter()
    _pending_record, future = _pending(adapter, task_id="task-cancel")
    persisted: list[str] = []
    pushed: list[str] = []
    audited: list[str] = []
    monkeypatch.setattr(
        protocol,
        "persist_message",
        lambda _context, _role, text, _task: persisted.append(text),
    )
    monkeypatch.setattr(
        security,
        "audit",
        lambda direction, *_args: audited.append(direction),
    )
    monkeypatch.setattr(
        adapter,
        "_send_push_notification",
        lambda *_args, **_kwargs: pushed.append("sent"),
    )

    response = adapter._rpc_tasks_cancel(1, {"id": "task-cancel"})

    assert response["result"]["status"]["state"] == protocol.STATE_CANCELED
    assert future.result(timeout=0) == (protocol.STATE_CANCELED, "")
    assert persisted == [""]
    assert audited == ["outbound"]
    assert pushed == ["sent"]


def test_completion_retention_uses_completion_order(monkeypatch) -> None:
    store = protocol.TaskStore()
    monkeypatch.setattr(store, "_MAX_TERMINAL", 2)
    store.create("old-long-task", "ctx-old", "doctor")
    for task_id in ("newer-1", "newer-2"):
        store.create(task_id, f"ctx-{task_id}", "doctor")
        store.complete(task_id, protocol.STATE_COMPLETED, task_id)

    completed = store.complete(
        "old-long-task",
        protocol.STATE_COMPLETED,
        "finished late",
    )

    assert completed is not None
    assert store.get("old-long-task") is not None
    assert store.get("newer-1") is None
    assert store.get("newer-2") is not None


def test_interrupted_tasks_share_the_bounded_completion_retention(monkeypatch) -> None:
    store = protocol.TaskStore()
    monkeypatch.setattr(store, "_MAX_TERMINAL", 2)
    for index in range(3):
        task_id = f"question-{index}"
        store.create(task_id, f"ctx-{index}", "doctor")
        store.complete(task_id, protocol.STATE_INPUT_REQUIRED, "Which account?")

    assert store.get("question-0") is None
    assert store.get("question-1") is not None
    assert store.get("question-2") is not None


def test_orphan_sweep_preserves_input_required_task() -> None:
    store = protocol.TaskStore()
    store.create("task-question", "ctx-question", "doctor")
    store.complete(
        "task-question",
        protocol.STATE_INPUT_REQUIRED,
        "Which account?",
    )
    store._tasks["task-question"]["created_at"] = time.time() - 600

    failed = store.fail_orphans(timeout_seconds=300)

    assert failed == []
    record = store.get("task-question")
    assert record is not None
    assert record["state"] == protocol.STATE_INPUT_REQUIRED


def test_client_can_request_immediate_acknowledgement(monkeypatch) -> None:
    monkeypatch.setattr(
        tools,
        "_load_config",
        lambda: {"a2a_agents": {"finance": {"url": "http://finance.local"}}},
    )
    monkeypatch.setattr(
        tools,
        "_fetch_card",
        lambda *_args, **_kwargs: {"url": "http://finance.local"},
    )
    monkeypatch.setattr(protocol, "persist_message", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(security, "audit", lambda *_args, **_kwargs: None)
    captured: dict = {}

    def fake_post(_url, body, _headers, _timeout):
        captured["body"] = body
        context_id = body["params"]["message"]["contextId"]
        task = protocol.build_task(
            "finance-task-1",
            context_id,
            protocol.STATE_WORKING,
        )
        return protocol.jsonrpc_result(body["id"], {"task": task})

    monkeypatch.setattr(tools, "_http_post_json", fake_post)

    result = tools.a2a_call({
        "agent": "finance",
        "message": "analyze this",
        "return_immediately": True,
    })

    assert captured["body"]["params"]["configuration"] == {
        "returnImmediately": True,
    }
    assert "finance-task-1" in result
    assert "working" in result
    assert "a2a_get_task" in result


def test_get_task_reports_the_latest_working_progress(monkeypatch) -> None:
    monkeypatch.setattr(
        tools,
        "_load_config",
        lambda: {"a2a_agents": {"finance": {"url": "http://finance.local"}}},
    )
    monkeypatch.setattr(
        tools,
        "_fetch_card",
        lambda *_args, **_kwargs: {"url": "http://finance.local"},
    )

    def fake_post(_url, body, _headers, _timeout):
        assert body["method"] == "GetTask"
        assert body["params"] == {"id": "finance-task-1"}
        task = protocol.build_task(
            "finance-task-1",
            "ctx-finance",
            protocol.STATE_WORKING,
            "reviewing statements",
        )
        return protocol.jsonrpc_result(body["id"], task)

    monkeypatch.setattr(tools, "_http_post_json", fake_post)

    result = tools.a2a_get_task({
        "agent": "finance",
        "task_id": "finance-task-1",
    })

    assert "working" in result
    assert "reviewing statements" in result
    assert "finance-task-1" in result


def test_repeated_terminal_poll_persists_the_reply_once(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(
        tools,
        "_load_config",
        lambda: {"a2a_agents": {"finance": {"url": "http://finance.local"}}},
    )
    monkeypatch.setattr(
        tools,
        "_fetch_card",
        lambda *_args, **_kwargs: {"url": "http://finance.local"},
    )
    monkeypatch.setattr(protocol, "_conv_dir", lambda: tmp_path)

    def fake_post(_url, body, _headers, _timeout):
        task = protocol.build_task(
            "finance-task-1",
            "ctx-finance",
            protocol.STATE_COMPLETED,
            "analysis complete",
        )
        return protocol.jsonrpc_result(body["id"], task)

    monkeypatch.setattr(tools, "_http_post_json", fake_post)
    inbound_before = protocol.metrics.inbound_total

    first = tools.a2a_get_task({"agent": "finance", "task_id": "finance-task-1"})
    second = tools.a2a_get_task({"agent": "finance", "task_id": "finance-task-1"})

    assert "analysis complete" in first
    assert "analysis complete" in second
    messages = protocol.load_conversation("ctx-finance")
    assert [(message["role"], message["text"]) for message in messages] == [
        ("agent", "analysis complete"),
    ]
    assert protocol.metrics.inbound_total == inbound_before + 1


def test_terminal_poll_persists_once_across_processes(tmp_path) -> None:
    context = multiprocessing.get_context("spawn")
    start = context.Event()
    processes = [
        context.Process(target=_persist_once_worker, args=(str(tmp_path), start))
        for _ in range(2)
    ]
    for process in processes:
        process.start()
    start.set()
    for process in processes:
        process.join(10)
        assert process.exitcode == 0

    records = [
        json.loads(line)
        for line in (tmp_path / "ctx-shared-process.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert [(record["role"], record["task_id"]) for record in records] == [
        ("agent", "task-shared-process"),
    ]


def test_repeated_terminal_task_render_is_stable() -> None:
    store = protocol.TaskStore()
    store.create("task-stable", "ctx-stable", "finance")
    store.complete("task-stable", protocol.STATE_COMPLETED, "final result")

    record = store.get("task-stable")
    assert record is not None
    first = store.to_task(record)
    time.sleep(0.01)
    second = store.to_task(record)

    assert first["status"]["timestamp"] == second["status"]["timestamp"]
    assert first["artifacts"][0]["artifactId"] == second["artifacts"][0][
        "artifactId"
    ]


def test_initial_terminal_response_matches_later_get_task(monkeypatch) -> None:
    adapter = _adapter()
    pending, _future = _pending(adapter)
    adapter._prepare_task = lambda params, peer, agent=None: (None, pending)  # type: ignore
    monkeypatch.setattr(
        adapter,
        "_await_reply",
        lambda _pending: (protocol.STATE_COMPLETED, "final result"),
    )
    monkeypatch.setattr(protocol, "persist_message", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(security, "audit", lambda *_args, **_kwargs: None)

    response = adapter._rpc_message_send(1, {"message": {}}, "finance")
    initial = response["result"]
    polled = adapter._rpc_tasks_get(2, {"id": "task-long"})["result"]

    assert initial["status"]["timestamp"] == polled["status"]["timestamp"]
    assert initial["artifacts"][0]["artifactId"] == polled["artifacts"][0][
        "artifactId"
    ]


def test_nonlocal_terminal_response_matches_later_get_task(monkeypatch) -> None:
    adapter = _adapter()
    agent = {
        "local": False,
        "slug": "finance",
        "profile": "finance",
        "tenant": "finance",
    }
    monkeypatch.setattr(
        adapter,
        "_forward_to_profile",
        lambda *_args, **_kwargs: ("final result", protocol.STATE_COMPLETED),
    )
    monkeypatch.setattr(protocol, "persist_message", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(security, "audit", lambda *_args, **_kwargs: None)

    response = adapter._rpc_message_send(
        1,
        {
            "message": protocol.text_message(
                protocol.ROLE_USER,
                "short task",
                context_id="ctx-forwarded",
            )
        },
        "doctor",
        agent=agent,
    )
    initial = response["result"]
    polled = adapter._rpc_tasks_get(
        2,
        {"id": initial["id"], "tenant": "finance"},
        agent=agent,
    )["result"]

    assert initial["status"]["timestamp"] == polled["status"]["timestamp"]
    assert initial["artifacts"][0]["artifactId"] == polled["artifacts"][0][
        "artifactId"
    ]


def test_nonlocal_late_completion_cannot_override_cancellation(monkeypatch) -> None:
    adapter = _adapter()
    agent = {
        "local": False,
        "slug": "finance",
        "profile": "finance",
        "tenant": "finance",
    }
    persisted: list[tuple] = []
    pushes: list[tuple] = []
    completed_before = protocol.metrics.tasks_completed
    outbound_before = protocol.metrics.outbound_total

    def cancel_then_reply(*_args, **_kwargs):
        task_id = next(iter(adapter.tasks._tasks))
        adapter.tasks.complete(task_id, protocol.STATE_CANCELED, "canceled")
        return "late final result", protocol.STATE_COMPLETED

    monkeypatch.setattr(adapter, "_forward_to_profile", cancel_then_reply)
    monkeypatch.setattr(
        protocol,
        "persist_message",
        lambda *args, **kwargs: persisted.append((args, kwargs)),
    )
    monkeypatch.setattr(security, "audit", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        adapter,
        "_send_push_notification",
        lambda *args, **kwargs: pushes.append((args, kwargs)),
    )

    response = adapter._rpc_message_send(
        1,
        {
            "message": protocol.text_message(
                protocol.ROLE_USER,
                "short task",
                context_id="ctx-forwarded",
            )
        },
        "doctor",
        agent=agent,
    )
    task = response["result"]

    assert task["status"]["state"] == protocol.STATE_CANCELED
    assert protocol.extract_text(task["status"]["message"]) == "canceled"
    assert not any(args[1] == "agent" for args, _kwargs in persisted)
    assert protocol.metrics.tasks_completed == completed_before
    assert protocol.metrics.outbound_total == outbound_before
    assert pushes == []


def test_immediate_rejection_matches_later_get_task() -> None:
    adapter = _adapter()

    response = adapter._rpc_message_send(
        1,
        {
            "message": protocol.text_message(
                protocol.ROLE_USER,
                "",
                context_id="ctx-empty",
            )
        },
        "doctor",
    )
    initial = response["result"]
    polled = adapter._rpc_tasks_get(2, {"id": initial["id"]})["result"]

    assert initial == polled
    assert initial["status"]["state"] == protocol.STATE_REJECTED
    assert protocol.extract_text(initial["status"]["message"]) == (
        "Empty task — nothing to do."
    )


def test_get_task_exposes_the_latest_nonfinal_progress() -> None:
    adapter = _adapter()
    _pending_record, future = _pending(adapter)

    asyncio.run(adapter.send("ctx-long", "reading statements", metadata={}))
    asyncio.run(adapter.send("ctx-long", "calculating totals", metadata={}))

    response = adapter._rpc_tasks_get(1, {"id": "task-long"})
    task = response["result"]
    assert task["status"]["state"] == protocol.STATE_WORKING
    assert protocol.extract_text(task["status"]["message"]) == "calculating totals"
    assert future.done() is False


def test_subscribe_streams_current_progress_and_terminal_state() -> None:
    adapter = _adapter()
    adapter.tasks.create("task-subscribe", "ctx-subscribe", "doctor")
    adapter.tasks.set_state("task-subscribe", protocol.STATE_WORKING)

    class Handler:
        def __init__(self) -> None:
            self.wfile = io.BytesIO()
            self.close_connection = False

        def send_response(self, _status) -> None:
            return None

        def send_header(self, _name, _value) -> None:
            return None

        def end_headers(self) -> None:
            return None

    handler = Handler()
    subscriber = threading.Thread(
        target=adapter._rpc_tasks_subscribe,
        args=(handler, 1, {"id": "task-subscribe"}),
    )
    subscriber.start()

    deadline = time.time() + 1
    while b'"task"' not in handler.wfile.getvalue() and time.time() < deadline:
        time.sleep(0.01)
    adapter.tasks.set_progress("task-subscribe", "reviewing inputs")
    while b"reviewing inputs" not in handler.wfile.getvalue() and time.time() < deadline:
        time.sleep(0.01)
    adapter.tasks.complete(
        "task-subscribe",
        protocol.STATE_COMPLETED,
        "finished",
    )
    subscriber.join(1)

    output = handler.wfile.getvalue()
    assert subscriber.is_alive() is False
    assert b'"task"' in output
    assert b"reviewing inputs" in output
    assert b"finished" in output
    assert output.endswith(b": done\n\n")


def test_get_task_preserves_an_input_required_question() -> None:
    store = protocol.TaskStore()
    store.create("task-question", "ctx-question", "doctor")
    store.complete(
        "task-question",
        protocol.STATE_INPUT_REQUIRED,
        "Which date range should I use?",
    )

    record = store.get("task-question")
    assert record is not None
    task = store.to_task(record)

    assert task["status"]["state"] == protocol.STATE_INPUT_REQUIRED
    assert protocol.extract_text(task["status"]["message"]) == (
        "Which date range should I use?"
    )


def test_progress_snapshot_uses_outbound_redaction(monkeypatch) -> None:
    adapter = _adapter()
    _pending_record, _future = _pending(adapter)
    monkeypatch.setattr(security, "redact_outbound", lambda _text: "[redacted]")

    asyncio.run(adapter.send("ctx-long", "sensitive progress", metadata={}))

    response = adapter._rpc_tasks_get(1, {"id": "task-long"})
    task = response["result"]
    assert protocol.extract_text(task["status"]["message"]) == "[redacted]"


def test_late_reply_for_finished_task_cannot_complete_newer_task() -> None:
    adapter = _adapter()
    _first_pending, _first_future = _pending(
        adapter,
        task_id="task-first",
        context_id="ctx-shared",
    )
    adapter.tasks.complete("task-first", protocol.STATE_CANCELED, "")
    adapter._pop_pending("task-first")
    _second_pending, second_future = _pending(
        adapter,
        task_id="task-second",
        context_id="ctx-shared",
    )

    asyncio.run(
        adapter.send(
            "ctx-shared",
            "late reply from first task",
            reply_to="task-first",
            metadata={"notify": True},
        )
    )

    assert second_future.done() is False


def test_late_progress_for_finished_task_cannot_update_newer_task() -> None:
    adapter = _adapter()
    _first_pending, _first_future = _pending(
        adapter,
        task_id="task-first",
        context_id="ctx-shared",
    )
    adapter.tasks.complete("task-first", protocol.STATE_CANCELED, "")
    adapter._pop_pending("task-first")
    _second_pending, _second_future = _pending(
        adapter,
        task_id="task-second",
        context_id="ctx-shared",
    )

    asyncio.run(
        adapter.send(
            "ctx-shared",
            "late progress from first task",
            reply_to="task-first",
            metadata={},
        )
    )

    record = adapter.tasks.get("task-second")
    assert record is not None
    assert record["progress"] == ""


def test_get_task_can_wait_without_model_side_busy_polling(monkeypatch) -> None:
    monkeypatch.setattr(
        tools,
        "_load_config",
        lambda: {"a2a_agents": {"finance": {"url": "http://finance.local"}}},
    )
    monkeypatch.setattr(
        tools,
        "_fetch_card",
        lambda *_args, **_kwargs: {"url": "http://finance.local"},
    )
    calls = 0

    def fake_post(_url, body, _headers, _timeout):
        nonlocal calls
        calls += 1
        state = protocol.STATE_WORKING if calls == 1 else protocol.STATE_COMPLETED
        text = "still calculating" if calls == 1 else "finished"
        return protocol.jsonrpc_result(
            body["id"],
            protocol.build_task("finance-task-1", "ctx-finance", state, text),
        )

    clock = [0.0]
    monkeypatch.setattr(tools, "_http_post_json", fake_post)
    monkeypatch.setattr(tools.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(tools.time, "sleep", lambda seconds: clock.__setitem__(0, clock[0] + seconds))
    monkeypatch.setattr(protocol, "persist_message_once", lambda *_args: True)

    result = tools.a2a_get_task({
        "agent": "finance",
        "task_id": "finance-task-1",
        "wait_seconds": 5,
    })

    assert calls == 2
    assert "completed" in result
    assert "finished" in result


def test_wait_seconds_performs_a_final_deadline_poll(monkeypatch) -> None:
    monkeypatch.setattr(
        tools,
        "_load_config",
        lambda: {"a2a_agents": {"finance": {"url": "http://finance.local"}}},
    )
    monkeypatch.setattr(
        tools,
        "_fetch_card",
        lambda *_args, **_kwargs: {"url": "http://finance.local"},
    )
    calls = 0

    def fake_post(_url, body, _headers, _timeout):
        nonlocal calls
        calls += 1
        state = protocol.STATE_WORKING if calls == 1 else protocol.STATE_COMPLETED
        return protocol.jsonrpc_result(
            body["id"],
            protocol.build_task(
                "finance-task-1",
                "ctx-finance",
                state,
                "finished" if calls > 1 else "still working",
            ),
        )

    clock = [0.0]
    monkeypatch.setattr(tools, "_http_post_json", fake_post)
    monkeypatch.setattr(tools.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(
        tools.time,
        "sleep",
        lambda seconds: clock.__setitem__(0, clock[0] + seconds),
    )
    monkeypatch.setattr(protocol, "persist_message_once", lambda *_args: True)

    result = tools.a2a_get_task({
        "agent": "finance",
        "task_id": "finance-task-1",
        "wait_seconds": 1,
    })

    assert calls == 2
    assert "completed" in result
    assert "finished" in result


def test_rate_limit_after_a_working_snapshot_returns_that_snapshot(monkeypatch) -> None:
    monkeypatch.setattr(
        tools,
        "_load_config",
        lambda: {"a2a_agents": {"finance": {"url": "http://finance.local"}}},
    )
    monkeypatch.setattr(
        tools,
        "_fetch_card",
        lambda *_args, **_kwargs: {"url": "http://finance.local"},
    )
    calls = 0

    def fake_post(_url, body, _headers, _timeout):
        nonlocal calls
        calls += 1
        if calls > 1:
            raise tools.urllib.error.HTTPError(
                "http://finance.local",
                429,
                "rate limited",
                Message(),
                None,
            )
        return protocol.jsonrpc_result(
            body["id"],
            protocol.build_task(
                "finance-task-1",
                "ctx-finance",
                protocol.STATE_WORKING,
                "latest progress",
            ),
        )

    clock = [0.0]
    monkeypatch.setattr(tools, "_http_post_json", fake_post)
    monkeypatch.setattr(tools.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(
        tools.time,
        "sleep",
        lambda seconds: clock.__setitem__(0, clock[0] + seconds),
    )

    result = tools.a2a_get_task({
        "agent": "finance",
        "task_id": "finance-task-1",
        "wait_seconds": 5,
    })

    assert calls == 2
    assert "working" in result
    assert "latest progress" in result
    assert "Error:" not in result


def test_wait_seconds_caps_each_get_task_request_timeout(monkeypatch) -> None:
    monkeypatch.setattr(
        tools,
        "_load_config",
        lambda: {
            "a2a_agents": {
                "finance": {"url": "http://finance.local", "timeout": 120}
            }
        },
    )
    monkeypatch.setattr(
        tools,
        "_fetch_card",
        lambda *_args, **_kwargs: {"url": "http://finance.local"},
    )
    request_timeouts: list[int] = []

    def fake_post(_url, body, _headers, timeout):
        request_timeouts.append(timeout)
        return protocol.jsonrpc_result(
            body["id"],
            protocol.build_task(
                "finance-task-1",
                "ctx-finance",
                protocol.STATE_WORKING,
                "still working",
            ),
        )

    clock = [0.0]
    monkeypatch.setattr(tools, "_http_post_json", fake_post)
    monkeypatch.setattr(tools.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(
        tools.time,
        "sleep",
        lambda seconds: clock.__setitem__(0, clock[0] + seconds),
    )

    tools.a2a_get_task(
        {
            "agent": "finance",
            "task_id": "finance-task-1",
            "wait_seconds": 5,
        }
    )

    assert request_timeouts
    assert max(request_timeouts) <= 5


def test_bounded_wait_stays_below_the_peer_rate_limit(monkeypatch) -> None:
    monkeypatch.setattr(
        tools,
        "_load_config",
        lambda: {"a2a_agents": {"finance": {"url": "http://finance.local"}}},
    )
    monkeypatch.setattr(
        tools,
        "_fetch_card",
        lambda *_args, **_kwargs: {"url": "http://finance.local"},
    )
    calls = 0

    def fake_post(_url, body, _headers, _timeout):
        nonlocal calls
        calls += 1
        return protocol.jsonrpc_result(
            body["id"],
            protocol.build_task(
                "finance-task-1",
                "ctx-finance",
                protocol.STATE_WORKING,
                "still working",
            ),
        )

    clock = [0.0]
    monkeypatch.setattr(tools, "_http_post_json", fake_post)
    monkeypatch.setattr(tools.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(tools.time, "sleep", lambda seconds: clock.__setitem__(0, clock[0] + seconds))

    result = tools.a2a_get_task({
        "agent": "finance",
        "task_id": "finance-task-1",
        "wait_seconds": 60,
    })

    assert "working" in result
    assert calls <= 2
