"""Lifecycle-scoped gateway delivery regressions for terminal completions.

The gateway contract here is deliberately narrower than exactly-once: one live
GatewayRunner suppresses concurrent/replayed copies after successful adapter
injection, failed injection remains retryable, and durable async-delegation
state (when available) is acknowledged through its authoritative SQLite API.
"""

import asyncio
import json
import queue
from collections import OrderedDict
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import Platform
from gateway.run import GatewayRunner
from gateway.session import SessionSource
from tools.process_registry import ProcessRegistry, ProcessSession


@pytest.fixture(autouse=True)
def isolated_registry(tmp_path, monkeypatch):
    """Any current/future durable compatibility path must stay in tmp state."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    import tools.process_registry as pr_module

    monkeypatch.setattr(pr_module, "CHECKPOINT_PATH", tmp_path / "processes.json")
    registry = pr_module.ProcessRegistry()
    monkeypatch.setattr(pr_module, "process_registry", registry)
    return registry


def _runner(adapter, *, origins=None):
    runner = object.__new__(GatewayRunner)
    runner._running = True
    runner.adapters = {Platform.TELEGRAM: adapter}
    runner.session_store = SimpleNamespace(
        _ensure_loaded=lambda: None,
        _entries=origins or {},
    )
    runner._session_source_cache = {}
    runner._completion_delivery_lock = __import__("threading").Lock()
    runner._completion_deliveries_inflight = set()
    runner._completion_deliveries_delivered = OrderedDict()
    runner._completion_delivery_retention = 2048
    return runner


def _async_event(delegation_id="deleg_duplicate"):
    return {
        "type": "async_delegation",
        "delegation_id": delegation_id,
        "session_key": "agent:main:telegram:dm:12345:678",
        "goal": "Investigate flaky test",
        "status": "completed",
        "summary": "Found it",
        "api_calls": 1,
        "duration_seconds": 12.0,
        "dispatched_at": 1000.0,
        "completed_at": 1012.0,
        # PR #62479 stamps these on gateway-owned events. They must not
        # change the producer identity used for queue replay.
        "origin_profile": "default",
        "origin_hermes_home": "/tmp/hermes-default",
    }


def _completion_event(*, started_at, session_id="proc_reused"):
    return {
        "type": "completion",
        "session_id": session_id,
        "session_key": "agent:main:telegram:dm:123",
        "platform": "telegram",
        "chat_type": "dm",
        "chat_id": "123",
        "started_at": started_at,
        "command": "echo done",
        "exit_code": 0,
        "completion_reason": "exited",
        "output": "done\n",
    }


def _stop_after_sleeps(monkeypatch, runner, count):
    sleep_calls = 0

    async def _bounded_sleep(_delay):
        nonlocal sleep_calls
        sleep_calls += 1
        if sleep_calls >= count:
            runner._running = False

    monkeypatch.setattr(asyncio, "sleep", _bounded_sleep)


def test_duplicate_async_queue_replay_injects_once(monkeypatch, isolated_registry):
    """Byte-identical queue replays produce one turn in one gateway lifecycle."""
    isolated = queue.Queue()
    monkeypatch.setattr(isolated_registry, "completion_queue", isolated)
    isolated.put(dict(_async_event()))
    isolated.put(dict(_async_event()))

    adapter = SimpleNamespace(handle_message=AsyncMock())
    runner = _runner(adapter)
    _stop_after_sleeps(monkeypatch, runner, count=2)

    asyncio.run(runner._async_delegation_watcher(interval=0))

    adapter.handle_message.assert_awaited_once()


def test_gateway_claim_conflict_defers_pending_durable_event(
    monkeypatch, isolated_registry,
):
    import tools.async_delegation as delegation_mod

    event = _async_event("deleg_gateway_live_lease")
    deferred = []
    monkeypatch.setattr(delegation_mod, "claim_event_delivery", lambda *_args: None)
    monkeypatch.setattr(
        isolated_registry,
        "defer_unclaimed_delivery",
        lambda evt: deferred.append(evt) or True,
    )

    runner = _runner(SimpleNamespace(handle_message=AsyncMock()))
    result = asyncio.run(runner._deliver_completion_notification("ready", event))

    assert result is None
    assert deferred == [event]


def test_gateway_busy_route_is_atomic_with_tool_boundary_carrier(
    monkeypatch, isolated_registry,
):
    """The watcher cannot hide a dequeued inject before active-turn requeue."""
    import threading

    import agent.delegation_inject as inject_mod
    import tools.async_delegation as delegation_mod
    import tools.process_registry as registry_mod

    isolated = queue.Queue()
    monkeypatch.setattr(isolated_registry, "completion_queue", isolated)
    monkeypatch.setattr(registry_mod, "_format_async_delegation", lambda _evt: "ready")
    monkeypatch.setattr(inject_mod, "ensure_pending_inject_heartbeat", lambda _agent: True)
    monkeypatch.setattr(
        delegation_mod,
        "claim_event_delivery",
        lambda _event, _owner: "gateway-race-claim",
    )

    route_paused = threading.Event()
    release_route = threading.Event()
    original_coalesce = delegation_mod.coalesce_ready_after_turn_events

    def _pause_after_dequeue(events):
        route_paused.set()
        if not release_route.wait(3):
            raise TimeoutError("test did not release paused gateway route")
        return original_coalesce(events)

    monkeypatch.setattr(
        delegation_mod, "coalesce_ready_after_turn_events", _pause_after_dequeue
    )

    turn_id = "turn-gateway-inject"
    session_key = "agent:main:telegram:dm:12345:678"
    active_agent = SimpleNamespace(
        _active_turn_id=turn_id,
        _iteration_calls_made=0,
        max_iterations=1,
        session_id="",
    )
    event = {
        **_async_event("deleg_gateway_inject_race"),
        "delivery_event_key": "task:0",
        "result_delivery": "inject",
        "parent_turn_id": turn_id,
        "session_key": session_key,
    }
    isolated.put(event)

    runner = _runner(SimpleNamespace(handle_message=AsyncMock()))
    runner._running_agents = {session_key: active_agent}
    _stop_after_sleeps(monkeypatch, runner, count=2)

    watcher_thread = threading.Thread(
        target=lambda: asyncio.run(runner._async_delegation_watcher(interval=0))
    )
    messages = [
        {
            "role": "assistant",
            "tool_calls": [{"id": "tc", "function": {"name": "terminal"}}],
        },
        {"role": "tool", "tool_call_id": "tc", "content": "working"},
    ]
    drained = {}
    drain_done = threading.Event()

    def _drain():
        drained["count"] = inject_mod.attach_ready_injects_to_tool_results(
            active_agent, messages, 1, turn_id=turn_id
        )
        drain_done.set()

    drain_thread = threading.Thread(target=_drain)
    try:
        watcher_thread.start()
        assert route_paused.wait(3), "gateway watcher did not reach post-dequeue route"
        drain_thread.start()

        # Dequeue, coalesce, active-parent classification, and requeue must be one
        # routing critical section. Returning here would degrade inject to late.
        assert not drain_done.wait(0.5)

        release_route.set()
        watcher_thread.join(3)
        drain_thread.join(3)

        assert not watcher_thread.is_alive()
        assert not drain_thread.is_alive()
        assert drained == {"count": 1}
        assert messages[-1]["role"] == "tool"
        assert "ready" in messages[-1]["content"]
        assert isolated.empty()
    finally:
        release_route.set()
        runner._running = False
        watcher_thread.join(3)
        drain_thread.join(3)
        while not isolated.empty():
            isolated.get_nowait()


def test_gateway_watcher_coalesces_ready_after_turn_batch_children(
    monkeypatch, isolated_registry,
):
    isolated = queue.Queue()
    monkeypatch.setattr(isolated_registry, "completion_queue", isolated)
    base = {
        **_async_event("deleg_after_turn_group"),
        "result_delivery": "after_turn",
        "is_batch": True,
        "batch_size": 3,
        "goals": ["zero", "one", "two"],
    }
    for index in (0, 1):
        isolated.put(
            {
                **base,
                "delivery_event_key": f"task:{index}",
                "task_index": index,
                "results": [
                    {
                        "task_index": index,
                        "status": "completed",
                        "summary": f"ready-{index}",
                    }
                ],
            }
        )

    adapter = SimpleNamespace(handle_message=AsyncMock())
    runner = _runner(adapter)
    deliver = AsyncMock(return_value=True)
    runner._deliver_completion_notification = deliver
    _stop_after_sleeps(monkeypatch, runner, count=2)

    asyncio.run(runner._async_delegation_watcher(interval=0))

    deliver.assert_awaited_once()
    text, grouped = deliver.await_args_list[0].args
    assert grouped["delivery_event_keys"] == ["task:0", "task:1"]
    assert [result["task_index"] for result in grouped["results"]] == [0, 1]
    assert "RESULTS READY" in text and "2/3" in text
    assert isolated.empty()


def test_gateway_after_turn_waits_for_idle_boundary_then_delivers_ready_group(
    monkeypatch, isolated_registry,
):
    isolated = queue.Queue()
    monkeypatch.setattr(isolated_registry, "completion_queue", isolated)
    base = {
        **_async_event("deleg_after_turn_busy"),
        "result_delivery": "after_turn",
        "is_batch": True,
        "batch_size": 3,
        "goals": ["zero", "one", "two"],
    }
    for index in (0, 1):
        isolated.put(
            {
                **base,
                "delivery_event_key": f"task:{index}",
                "task_index": index,
                "results": [
                    {
                        "task_index": index,
                        "status": "completed",
                        "summary": f"ready-{index}",
                    }
                ],
            }
        )

    adapter = SimpleNamespace(handle_message=AsyncMock())
    runner = _runner(adapter)
    deliver = AsyncMock(return_value=True)
    runner._deliver_completion_notification = deliver
    runner._running_agents = {base["session_key"]: SimpleNamespace()}
    _stop_after_sleeps(monkeypatch, runner, count=2)

    asyncio.run(runner._async_delegation_watcher(interval=0))

    deliver.assert_not_awaited()
    assert isolated.qsize() == 1
    queued_group = isolated.get_nowait()
    assert queued_group["delivery_event_keys"] == ["task:0", "task:1"]
    isolated.put(queued_group)

    runner._running = True
    runner._running_agents = {}
    _stop_after_sleeps(monkeypatch, runner, count=2)
    asyncio.run(runner._async_delegation_watcher(interval=0))

    deliver.assert_awaited_once()
    _text, delivered_group = deliver.await_args_list[0].args
    assert delivered_group["delivery_event_keys"] == ["task:0", "task:1"]
    assert isolated.empty()


def test_unroutable_async_event_is_not_requeued_forever(
    monkeypatch, isolated_registry,
):
    isolated = queue.Queue()
    monkeypatch.setattr(isolated_registry, "completion_queue", isolated)
    event = _async_event("deleg_desktop_or_cli")
    event["session_key"] = "20260711_unparseable_ui_session"
    isolated.put(event)

    adapter = SimpleNamespace(handle_message=AsyncMock())
    runner = _runner(adapter)
    _stop_after_sleeps(monkeypatch, runner, count=2)

    asyncio.run(runner._async_delegation_watcher(interval=0))

    adapter.handle_message.assert_not_awaited()
    assert isolated.empty()


def test_concurrent_claims_share_the_same_narrow_delivery_seam():
    """Concurrent consumers in one runner cannot both enter the adapter."""
    entered = asyncio.Event()
    release = asyncio.Event()

    async def _blocked_injection(_event):
        entered.set()
        await release.wait()

    adapter = SimpleNamespace(handle_message=AsyncMock(side_effect=_blocked_injection))
    runner = _runner(adapter)
    event = _async_event()
    text = "completion"

    async def _exercise():
        first = asyncio.create_task(runner._deliver_completion_notification(text, dict(event)))
        await entered.wait()
        second = asyncio.create_task(runner._deliver_completion_notification(text, dict(event)))
        await asyncio.sleep(0)
        release.set()
        return await asyncio.gather(first, second)

    assert sorted(asyncio.run(_exercise()), key=str) == [None, True]
    adapter.handle_message.assert_awaited_once()


def test_failed_async_injection_is_retried_and_only_success_is_acked(
    monkeypatch, isolated_registry,
):
    isolated = queue.Queue()
    monkeypatch.setattr(isolated_registry, "completion_queue", isolated)
    isolated.put(_async_event())

    adapter = SimpleNamespace(
        handle_message=AsyncMock(side_effect=[RuntimeError("temporary"), None])
    )
    runner = _runner(adapter)
    _stop_after_sleeps(monkeypatch, runner, count=3)

    from tools import async_delegation

    acknowledgements = []
    monkeypatch.setattr(
        async_delegation,
        "complete_completion_delivery",
        lambda delegation_id, _claim_id: acknowledgements.append(delegation_id) or True,
        raising=False,
    )

    asyncio.run(runner._async_delegation_watcher(interval=0))

    assert adapter.handle_message.await_count == 2
    assert acknowledgements == ["deleg_duplicate"]


def _persist_pending_completion(event):
    from tools import async_delegation

    async_delegation._persist_dispatch({
        "delegation_id": event["delegation_id"],
        "session_key": event["session_key"],
        "origin_ui_session_id": "",
        "parent_session_id": event.get("parent_session_id"),
        "dispatched_at": event["dispatched_at"],
    })
    async_delegation._persist_completion(event, {
        "status": "completed",
        "summary": event["summary"],
    })


def test_explicit_kill_returns_output_before_consuming_notification(monkeypatch):
    import tools.process_registry as pr_module

    registry = ProcessRegistry()
    session = ProcessSession(
        id="proc_kill_consumed",
        command="sleep 999",
        task_id="task",
        started_at=1.0,
        output_buffer="important terminal output\n",
        notify_on_complete=True,
    )
    session.process = MagicMock()
    session.process.pid = 4242
    registry._running[session.id] = session
    monkeypatch.setattr(registry, "_terminate_host_pid", lambda *_a, **_kw: None)
    monkeypatch.setattr(registry, "_write_checkpoint", lambda: None)
    monkeypatch.setattr(pr_module, "process_registry", registry)

    result = registry.kill_process(session.id)
    assert result["status"] == "killed"
    assert result["output"] == "important terminal output\n"
    assert registry.is_completion_consumed(session.id)

    adapter = SimpleNamespace(handle_message=AsyncMock())
    runner = _runner(adapter)

    async def _instant_sleep(*_a, **_kw):
        pass

    monkeypatch.setattr(asyncio, "sleep", _instant_sleep)
    asyncio.run(runner._run_process_watcher({
        "session_id": session.id,
        "check_interval": 0,
        "session_key": "agent:main:telegram:dm:123",
        "platform": "telegram",
        "chat_type": "dm",
        "chat_id": "123",
        "notify_on_complete": True,
    }))

    adapter.handle_message.assert_not_awaited()


def test_process_tool_redacts_explicit_kill_output(monkeypatch):
    from tools import process_registry as pr_module

    registry = ProcessRegistry()
    session = ProcessSession(
        id="proc_kill_redacted",
        command="printenv",
        task_id="task",
        started_at=1.0,
        output_buffer="PRIVATE_TOKEN=opaque-value\n",
        exited=True,
        exit_code=0,
    )
    registry._finished[session.id] = session
    monkeypatch.setattr(pr_module, "process_registry", registry)

    def _redact(result):
        assert result["output"] == "PRIVATE_TOKEN=opaque-value\n"
        result["output"] = "PRIVATE_TOKEN=<redacted>\n"
        return result

    monkeypatch.setattr(pr_module, "_redact_process_result", _redact)

    result = json.loads(pr_module._handle_process({
        "action": "kill",
        "session_id": session.id,
    }))
    assert result["output"] == "PRIVATE_TOKEN=<redacted>\n"


def test_autonomous_completion_redacts_real_command_and_output_secrets(monkeypatch):
    import agent.redact as redact_module
    import tools.process_registry as pr_module

    secret = "abc123randomopaquetokenvalue999"
    registry = ProcessRegistry()
    session = ProcessSession(
        id="proc_autonomous_redaction",
        command=f"printenv MY_SERVICE_TOKEN={secret}",
        task_id="task",
        started_at=1234.5,
        output_buffer=f"MY_SERVICE_TOKEN={secret}\nHOME=/home/user\n",
        exited=True,
        exit_code=0,
        notify_on_complete=True,
    )
    registry._finished[session.id] = session
    monkeypatch.setattr(pr_module, "process_registry", registry)
    monkeypatch.setattr(redact_module, "_REDACT_ENABLED", True)

    adapter = SimpleNamespace(handle_message=AsyncMock())
    runner = _runner(adapter)

    async def _instant_sleep(*_a, **_kw):
        pass

    monkeypatch.setattr(asyncio, "sleep", _instant_sleep)
    asyncio.run(runner._run_process_watcher({
        "session_id": session.id,
        "check_interval": 0,
        "session_key": "agent:main:telegram:dm:123",
        "platform": "telegram",
        "chat_type": "dm",
        "chat_id": "123",
        "notify_on_complete": True,
    }))

    delivered = adapter.handle_message.await_args.args[0]
    assert secret not in delivered.text
    assert "HOME=/home/user" in delivered.text
