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


def test_compression_parent_delivery_targets_tip_and_is_acked(
    monkeypatch, isolated_registry,
):
    """A compression-rotated parent with a live tip is deliverable + acked."""
    from tools import async_delegation

    event = _async_event("deleg_compression")
    event["parent_session_id"] = "sess_parent"
    _persist_pending_completion(event)

    adapter = SimpleNamespace(handle_message=AsyncMock())
    runner = _runner(adapter)
    runner._session_db = SimpleNamespace(
        get_session=AsyncMock(side_effect=lambda session_id: {
            "sess_parent": {
                "id": "sess_parent",
                "ended_at": "2026-07-16T12:00:00",
                "end_reason": "compression",
            },
            "sess_tip": {"id": "sess_tip", "ended_at": None, "end_reason": None},
        }.get(session_id)),
        get_compression_tip=AsyncMock(return_value="sess_tip"),
    )

    assert asyncio.run(
        runner._deliver_completion_notification("completion", event)
    ) is True

    adapter.handle_message.assert_awaited_once()
    durable = async_delegation.get_durable_delegation(event["delegation_id"])
    assert durable is not None
    assert durable["delivery_state"] == "delivered"


def test_explicit_reset_drop_is_terminal_not_falsely_delivered(
    monkeypatch, isolated_registry,
):
    """An explicit /new boundary drop gets a terminal 'dropped' disposition.

    Not 'delivered' (the ack must stay honest — nothing was injected) and not
    'pending' (restart recovery would replay a completion that is fail-closed
    dropped again on every boot).
    """
    from tools import async_delegation

    event = _async_event("deleg_explicit_new")
    event["parent_session_id"] = "sess_reset"
    _persist_pending_completion(event)

    adapter = SimpleNamespace(handle_message=AsyncMock())
    runner = _runner(adapter)
    runner._session_db = SimpleNamespace(
        get_session=AsyncMock(return_value={
            "id": "sess_reset",
            "ended_at": "2026-07-16T12:00:00",
            "end_reason": "session_reset",
        }),
        get_compression_tip=AsyncMock(),
    )

    assert asyncio.run(
        runner._deliver_completion_notification("completion", event)
    ) is None

    adapter.handle_message.assert_not_awaited()
    durable = async_delegation.get_durable_delegation(event["delegation_id"])
    assert durable is not None
    assert durable["delivery_state"] == "dropped"
    restored = queue.Queue()
    assert async_delegation.restore_undelivered_completions(restored) == 0


def test_midflight_compression_rotation_stays_pending_for_retry(
    monkeypatch, isolated_registry,
):
    """A rotation without a visible continuation yet is retryable, not dropped."""
    from tools import async_delegation

    event = _async_event("deleg_midflight")
    event["parent_session_id"] = "sess_rotating"
    _persist_pending_completion(event)

    adapter = SimpleNamespace(handle_message=AsyncMock())
    runner = _runner(adapter)
    runner._session_db = SimpleNamespace(
        get_session=AsyncMock(return_value={
            "id": "sess_rotating",
            "ended_at": "2026-07-16T12:00:00",
            "end_reason": "compression",
        }),
        get_compression_tip=AsyncMock(return_value=None),
    )

    assert asyncio.run(
        runner._deliver_completion_notification("completion", event)
    ) is False

    adapter.handle_message.assert_not_awaited()
    durable = async_delegation.get_durable_delegation(event["delegation_id"])
    assert durable is not None
    assert durable["delivery_state"] == "pending"
    restored = queue.Queue()
    assert async_delegation.restore_undelivered_completions(restored) == 1
    assert restored.get_nowait()["delegation_id"] == event["delegation_id"]


def test_retry_attempts_are_capped_to_a_terminal_drop(
    monkeypatch, isolated_registry,
):
    """Endless claim/release churn converges to a terminal 'dropped' state."""
    from tools import async_delegation

    event = _async_event("deleg_attempt_cap")
    event["parent_session_id"] = "sess_rotating"
    _persist_pending_completion(event)

    adapter = SimpleNamespace(handle_message=AsyncMock())
    runner = _runner(adapter)
    runner._session_db = SimpleNamespace(
        get_session=AsyncMock(return_value={
            "id": "sess_rotating",
            "ended_at": "2026-07-16T12:00:00",
            "end_reason": "compression",
        }),
        get_compression_tip=AsyncMock(return_value=None),
    )

    async def _churn():
        for _ in range(async_delegation._MAX_DELIVERY_ATTEMPTS + 2):
            await runner._deliver_completion_notification("completion", event)

    asyncio.run(_churn())

    adapter.handle_message.assert_not_awaited()
    durable = async_delegation.get_durable_delegation(event["delegation_id"])
    assert durable is not None
    assert durable["delivery_state"] == "dropped"
    assert durable["delivery_attempts"] <= async_delegation._MAX_DELIVERY_ATTEMPTS
    restored = queue.Queue()
    assert async_delegation.restore_undelivered_completions(restored) == 0


def test_distinct_process_incarnations_are_not_deduplicated():
    """Producer spawn time distinguishes a reused process session ID."""
    adapter = SimpleNamespace(handle_message=AsyncMock())
    runner = _runner(adapter)

    async def _exercise():
        first = await runner._deliver_completion_notification(
            "first", _completion_event(started_at=10.0)
        )
        second = await runner._deliver_completion_notification(
            "second", _completion_event(started_at=20.0)
        )
        return first, second

    assert asyncio.run(_exercise()) == (True, True)

    assert adapter.handle_message.await_count == 2


def test_delivered_identity_retention_is_bounded():
    """Lifecycle dedupe cannot grow without bound in a long-running gateway."""
    adapter = SimpleNamespace(handle_message=AsyncMock())
    runner = _runner(adapter)
    runner._completion_delivery_retention = 2
    runner._completion_deliveries_delivered = OrderedDict()

    async def _exercise():
        for index in range(3):
            await runner._deliver_completion_notification(
                f"completion {index}",
                _async_event(f"deleg_retention_{index}"),
            )

    asyncio.run(_exercise())

    assert len(runner._completion_deliveries_delivered) == 2
    assert ("async_delegation", "deleg_retention_0", "") not in (
        runner._completion_deliveries_delivered
    )
    assert ("async_delegation", "deleg_retention_2", "") in (
        runner._completion_deliveries_delivered
    )


def test_delivery_state_is_isolated_per_gateway_profile_lifecycle():
    """A process-local claim in one profile never suppresses another runner."""
    default_adapter = SimpleNamespace(handle_message=AsyncMock())
    profile_adapter = SimpleNamespace(handle_message=AsyncMock())
    default_runner = _runner(default_adapter)
    profile_runner = _runner(profile_adapter)
    event = _async_event("deleg_same_producer_id")

    async def _exercise():
        first = await default_runner._deliver_completion_notification(
            "default", dict(event),
        )
        second = await profile_runner._deliver_completion_notification(
            "profile", dict(event),
        )
        return first, second

    assert asyncio.run(_exercise()) == (True, True)
    default_adapter.handle_message.assert_awaited_once()
    profile_adapter.handle_message.assert_awaited_once()


def test_async_completion_uses_canonical_origin_routing(monkeypatch, isolated_registry):
    isolated = queue.Queue()
    monkeypatch.setattr(isolated_registry, "completion_queue", isolated)
    event = _async_event("deleg_routing")
    isolated.put(event)

    canonical = SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="canonical-chat",
        chat_type="group",
        thread_id="canonical-topic",
    )
    entry = SimpleNamespace(origin=canonical)
    adapter = SimpleNamespace(handle_message=AsyncMock())
    runner = _runner(adapter, origins={event["session_key"]: entry})
    _stop_after_sleeps(monkeypatch, runner, count=2)

    asyncio.run(runner._async_delegation_watcher(interval=0))

    delivered = adapter.handle_message.await_args.args[0]
    assert delivered.source == canonical


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


def test_kill_of_already_exited_process_returns_output_before_consuming():
    registry = ProcessRegistry()
    session = ProcessSession(
        id="proc_already_exited",
        command="echo complete",
        task_id="task",
        started_at=1.0,
        output_buffer="complete\n",
        exited=True,
        exit_code=0,
    )
    registry._finished[session.id] = session

    result = registry.kill_process(session.id)

    assert result["status"] == "already_exited"
    assert result["output"] == "complete\n"
    assert registry.is_completion_consumed(session.id)


def test_read_log_only_consumes_when_terminal_output_page_is_observed():
    registry = ProcessRegistry()
    session = ProcessSession(
        id="proc_paged_log",
        command="printf lines",
        task_id="task",
        started_at=1.0,
        output_buffer="first\nsecond\nfinal\n",
        exited=True,
        exit_code=0,
    )
    registry._finished[session.id] = session

    middle_page = registry.read_log(session.id, offset=1, limit=1)
    assert middle_page["output"] == "second"
    assert not registry.is_completion_consumed(session.id)

    final_page = registry.read_log(session.id, offset=2, limit=1)
    assert final_page["output"] == "final"
    assert registry.is_completion_consumed(session.id)


def test_bulk_kill_does_not_consume_discarded_completion_output(monkeypatch):
    registry = ProcessRegistry()
    session = ProcessSession(
        id="proc_bulk_kill",
        command="sleep 999",
        task_id="task",
        started_at=1.0,
        output_buffer="output bulk cleanup does not return\n",
        notify_on_complete=True,
    )
    session.process = MagicMock()
    session.process.pid = 4243
    registry._running[session.id] = session
    monkeypatch.setattr(registry, "_terminate_host_pid", lambda *_a, **_kw: None)
    monkeypatch.setattr(registry, "_write_checkpoint", lambda: None)

    assert registry.kill_all() == 1
    assert not registry.is_completion_consumed(session.id)
    queued = registry.completion_queue.get_nowait()
    assert queued["session_id"] == session.id
    assert queued["started_at"] == session.started_at
    assert queued["output"] == "output bulk cleanup does not return\n"


def test_unobserved_normal_completion_still_notifies(monkeypatch):
    import tools.process_registry as pr_module

    class _Registry:
        def get(self, _session_id):
            return SimpleNamespace(
                output_buffer="done\n",
                exited=True,
                exit_code=0,
                command="echo done",
                started_at=1234.5,
            )

        def is_completion_consumed(self, _session_id):
            return False

    monkeypatch.setattr(pr_module, "process_registry", _Registry())
    adapter = SimpleNamespace(handle_message=AsyncMock())
    runner = _runner(adapter)

    async def _instant_sleep(*_a, **_kw):
        pass

    monkeypatch.setattr(asyncio, "sleep", _instant_sleep)
    asyncio.run(runner._run_process_watcher({
        "session_id": "proc_unobserved",
        "check_interval": 0,
        "session_key": "agent:main:telegram:dm:123",
        "platform": "telegram",
        "chat_type": "dm",
        "chat_id": "123",
        "notify_on_complete": True,
    }))

    adapter.handle_message.assert_awaited_once()


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
