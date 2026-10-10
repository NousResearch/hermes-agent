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
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gateway.config import Platform
from gateway.run import GatewayRunner
from tools.process_registry import ProcessRegistry, ProcessSession


class AdmittingHandler(AsyncMock):
    """Fake transport whose successful insertion issues the production receipt."""

    async def _execute_mock_call(self, event, *args, **kwargs):
        result = await super()._execute_mock_call(event, *args, **kwargs)
        event._gateway_accepted = True
        return result


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
    runner._background_tasks = set()
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

    adapter = SimpleNamespace(handle_message=AdmittingHandler())
    runner = _runner(adapter)
    _stop_after_sleeps(monkeypatch, runner, count=2)

    asyncio.run(runner._async_delegation_watcher(interval=0))

    adapter.handle_message.assert_awaited_once()


def test_unroutable_async_event_remains_retryable(
    monkeypatch, isolated_registry,
):
    isolated = queue.Queue()
    monkeypatch.setattr(isolated_registry, "completion_queue", isolated)
    event = _async_event("deleg_desktop_or_cli")
    event["session_key"] = "20260711_unparseable_ui_session"
    isolated.put(event)

    adapter = SimpleNamespace(handle_message=AdmittingHandler())
    runner = _runner(adapter)
    _stop_after_sleeps(monkeypatch, runner, count=2)

    asyncio.run(runner._async_delegation_watcher(interval=0))

    adapter.handle_message.assert_not_awaited()
    assert not isolated.empty()


def test_concurrent_claims_share_the_same_narrow_delivery_seam():
    """Concurrent consumers in one runner cannot both enter the adapter."""
    entered = asyncio.Event()
    release = asyncio.Event()

    async def _blocked_injection(_event):
        entered.set()
        await release.wait()

    adapter = SimpleNamespace(handle_message=AdmittingHandler(side_effect=_blocked_injection))
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
        handle_message=AdmittingHandler(side_effect=[RuntimeError("temporary"), None])
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

    adapter = SimpleNamespace(handle_message=AdmittingHandler())
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

    adapter = SimpleNamespace(handle_message=AdmittingHandler())
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


def test_concurrent_process_watchers_coalesce_one_session_completion_turn(monkeypatch):
    """Concurrent terminal watchers for one session must re-enter the agent once."""
    import tools.process_registry as pr_module

    registry = ProcessRegistry()
    watchers = []
    for index in range(3):
        session = ProcessSession(
            id=f"proc_batch_{index}",
            command=f"printf batch-{index}",
            task_id=f"task-{index}",
            started_at=1000.0 + index,
            output_buffer=f"batch-{index}\n",
            exited=True,
            exit_code=0,
            notify_on_complete=True,
        )
        registry._finished[session.id] = session
        watchers.append({
            "session_id": session.id,
            "check_interval": 0,
            "session_key": "agent:main:telegram:dm:123",
            "platform": "telegram",
            "chat_type": "dm",
            "chat_id": "123",
            "notify_on_complete": True,
        })
    monkeypatch.setattr(pr_module, "process_registry", registry)

    adapter = SimpleNamespace(handle_message=AdmittingHandler())
    runner = _runner(adapter)

    async def _exercise():
        await asyncio.gather(*(
            runner._run_process_watcher(watcher)
            for watcher in watchers
        ))

    asyncio.run(_exercise())

    adapter.handle_message.assert_awaited_once()
    delivered = adapter.handle_message.await_args.args[0]
    assert "3 background processes completed" in delivered.text
    for index in range(3):
        assert f"proc_batch_{index}" in delivered.text


def test_completion_arriving_during_batch_delivery_schedules_next_flush():
    """A new event cannot be stranded behind an in-flight batch for its route."""
    first_delivery_entered = asyncio.Event()
    release_first_delivery = asyncio.Event()
    delivery_count = 0

    async def _deliver(_event):
        nonlocal delivery_count
        delivery_count += 1
        if delivery_count == 1:
            first_delivery_entered.set()
            await release_first_delivery.wait()

    adapter = SimpleNamespace(handle_message=AdmittingHandler(side_effect=_deliver))
    runner = _runner(adapter)

    async def _exercise():
        first = asyncio.create_task(runner._enqueue_process_completion_notification(
            "first completion",
            _completion_event(started_at=1.0, session_id="proc_first"),
        ))
        await first_delivery_entered.wait()
        second = asyncio.create_task(runner._enqueue_process_completion_notification(
            "second completion",
            _completion_event(started_at=2.0, session_id="proc_second"),
        ))
        release_first_delivery.set()
        assert await first is True
        assert await asyncio.wait_for(second, timeout=1.0) is True

    asyncio.run(_exercise())

    assert adapter.handle_message.await_count == 2


def test_completion_batches_do_not_cross_conversation_routes():
    adapter = SimpleNamespace(handle_message=AdmittingHandler())
    runner = _runner(adapter)

    first = _completion_event(started_at=1.0, session_id="proc_route_a")
    second = _completion_event(started_at=2.0, session_id="proc_route_b")
    second["session_key"] = "agent:main:telegram:dm:456"
    second["chat_id"] = "456"

    async def _exercise():
        return await asyncio.gather(
            runner._enqueue_process_completion_notification("first", first),
            runner._enqueue_process_completion_notification("second", second),
        )

    assert asyncio.run(_exercise()) == [True, True]
    assert adapter.handle_message.await_count == 2


def test_failed_coalesced_delivery_retries_all_entries():
    attempts = 0

    async def _deliver(_event):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise RuntimeError("temporary adapter failure")

    adapter = SimpleNamespace(handle_message=AdmittingHandler(side_effect=_deliver))
    runner = _runner(adapter)
    events = [
        _completion_event(started_at=float(index), session_id=f"proc_retry_{index}")
        for index in range(2)
    ]

    async def _enqueue_all():
        return await asyncio.gather(*(
            runner._enqueue_process_completion_notification(f"event-{index}", event)
            for index, event in enumerate(events)
        ))

    async def _exercise():
        assert await _enqueue_all() == [False, False]
        assert await _enqueue_all() == [True, True]

    asyncio.run(_exercise())
    assert adapter.handle_message.await_count == 2


def test_coalesced_success_records_every_completion_identity():
    adapter = SimpleNamespace(handle_message=AdmittingHandler())
    runner = _runner(adapter)
    events = [
        _completion_event(started_at=float(index), session_id=f"proc_ledger_{index}")
        for index in range(3)
    ]

    async def _exercise():
        return await asyncio.gather(*(
            runner._enqueue_process_completion_notification(f"event-{index}", event)
            for index, event in enumerate(events)
        ))

    assert asyncio.run(_exercise()) == [True, True, True]
    for event in events:
        identity = runner._completion_delivery_identity(event)
        assert identity in runner._completion_deliveries_delivered


def test_coalesced_format_bounds_details_and_reports_omitted_count():
    async def _format():
        loop = asyncio.get_running_loop()
        entries = [
            (
                f"event-{index}",
                _completion_event(
                    started_at=float(index), session_id=f"proc_bound_{index}"
                ),
                loop.create_future(),
            )
            for index in range(12)
        ]
        return GatewayRunner._format_coalesced_process_completions(entries)

    text = asyncio.run(_format())

    for index in range(10):
        assert f"proc_bound_{index}" in text
    assert "proc_bound_10" not in text
    assert "proc_bound_11" not in text
    assert "and 2 more completion(s)" in text


def test_coalesced_format_force_redacts_output_when_redaction_disabled(monkeypatch):
    """A user setting cannot disable the gateway's outbound secret floor."""
    import agent.redact as redact_module

    secret = "abc123randomopaquetokenvalue999"
    monkeypatch.setattr(redact_module, "_REDACT_ENABLED", False)

    async def _format():
        loop = asyncio.get_running_loop()
        first = _completion_event(started_at=1.0, session_id="proc_secret")
        first["output"] = (
            f"MY_SERVICE_TOKEN={secret}\n"
            "HOME=/home/user\n"
        )
        second = _completion_event(started_at=2.0, session_id="proc_control")
        return GatewayRunner._format_coalesced_process_completions([
            ("first", first, loop.create_future()),
            ("second", second, loop.create_future()),
        ])

    text = asyncio.run(_format())

    assert secret not in text
    assert "HOME=/home/user" in text


def test_coalesced_format_redacts_before_truncating_output(monkeypatch):
    """Truncation cannot remove the prefix needed to recognize a secret."""
    import agent.redact as redact_module

    marker = "SHOULD_NOT_SURVIVE"
    monkeypatch.setattr(redact_module, "_REDACT_ENABLED", False)

    async def _format():
        loop = asyncio.get_running_loop()
        first = _completion_event(started_at=1.0, session_id="proc_long_secret")
        first["output"] = f"MY_SERVICE_TOKEN={'x' * 900}{marker}\n"
        second = _completion_event(started_at=2.0, session_id="proc_control")
        return GatewayRunner._format_coalesced_process_completions([
            ("first", first, loop.create_future()),
            ("second", second, loop.create_future()),
        ])

    text = asyncio.run(_format())

    assert marker not in text


def test_duplicate_primary_does_not_discard_fresh_batch_sibling():
    adapter = SimpleNamespace(handle_message=AdmittingHandler())
    runner = _runner(adapter)
    duplicate = _completion_event(started_at=1.0, session_id="proc_duplicate")
    fresh = _completion_event(started_at=2.0, session_id="proc_fresh")
    duplicate_identity = runner._completion_delivery_identity(duplicate)
    runner._completion_deliveries_delivered[duplicate_identity] = None

    async def _exercise():
        return await asyncio.gather(
            runner._enqueue_process_completion_notification("duplicate", duplicate),
            runner._enqueue_process_completion_notification("fresh", fresh),
        )

    assert asyncio.run(_exercise()) == [True, True]
    adapter.handle_message.assert_awaited_once()
    fresh_identity = runner._completion_delivery_identity(fresh)
    assert fresh_identity in runner._completion_deliveries_delivered


def test_batch_format_failure_resolves_waiters_for_retry(monkeypatch):
    adapter = SimpleNamespace(handle_message=AdmittingHandler())
    runner = _runner(adapter)
    monkeypatch.setattr(
        runner,
        "_format_coalesced_process_completions",
        MagicMock(side_effect=ValueError("bad batch")),
    )
    events = [
        _completion_event(started_at=float(index), session_id=f"proc_format_{index}")
        for index in range(2)
    ]

    async def _exercise():
        pending = asyncio.gather(*(
            runner._enqueue_process_completion_notification(f"event-{index}", event)
            for index, event in enumerate(events)
        ))
        return await asyncio.wait_for(pending, timeout=1.0)

    assert asyncio.run(_exercise()) == [False, False]
    adapter.handle_message.assert_not_awaited()


def test_shutdown_cancels_batch_during_window_and_settles_waiter_for_retry():
    adapter = SimpleNamespace(handle_message=AdmittingHandler())
    runner = _runner(adapter)
    sleep_entered = asyncio.Event()
    release_sleep = asyncio.Event()
    real_sleep = asyncio.sleep
    event = _completion_event(started_at=1.0, session_id="proc_cancel_window")

    async def _controlled_sleep(delay):
        if delay == runner._completion_notification_batch_window:
            sleep_entered.set()
            await release_sleep.wait()
            return
        await real_sleep(delay)

    async def _exercise():
        pending = asyncio.create_task(
            runner._enqueue_process_completion_notification("completion", event)
        )
        await sleep_entered.wait()
        flush_task = next(iter(runner._completion_notification_batch_tasks.values()))
        assert flush_task in runner._background_tasks

        await runner._cancel_process_completion_batch_tasks()

        assert await asyncio.wait_for(pending, timeout=1.0) is False
        assert flush_task.cancelled()
        assert flush_task not in runner._background_tasks
        assert runner._completion_notification_batches == {}
        assert runner._completion_notification_batch_tasks == {}

    with patch("gateway.run.asyncio.sleep", new=_controlled_sleep):
        asyncio.run(_exercise())
    adapter.handle_message.assert_not_awaited()


def test_shutdown_cancels_blocked_batch_delivery_and_keeps_it_retryable():
    delivery_entered = asyncio.Event()

    async def _blocked_delivery(_event):
        delivery_entered.set()
        await asyncio.Event().wait()

    adapter = SimpleNamespace(handle_message=AdmittingHandler(side_effect=_blocked_delivery))
    runner = _runner(adapter)
    runner._completion_notification_batch_window = 0
    event = _completion_event(started_at=1.0, session_id="proc_cancel_delivery")

    async def _exercise():
        pending = asyncio.create_task(
            runner._enqueue_process_completion_notification("completion", event)
        )
        await delivery_entered.wait()
        flush_task = next(iter(runner._completion_notification_batch_flush_tasks))

        await runner._cancel_process_completion_batch_tasks()

        assert await asyncio.wait_for(pending, timeout=1.0) is False
        assert flush_task.cancelled()
        assert runner._completion_delivery_identity(event) not in runner._completion_deliveries_inflight
        assert runner._completion_delivery_identity(event) not in runner._completion_deliveries_delivered
        assert runner._completion_notification_batches == {}
        assert runner._completion_notification_batch_tasks == {}

    asyncio.run(_exercise())
    adapter.handle_message.assert_awaited_once()


def test_completion_enqueue_stays_retryable_after_shutdown_starts():
    adapter = SimpleNamespace(handle_message=AdmittingHandler())
    runner = _runner(adapter)

    async def _exercise():
        await runner._cancel_process_completion_batch_tasks()
        return await runner._enqueue_process_completion_notification(
            "completion",
            _completion_event(started_at=1.0, session_id="proc_after_shutdown"),
        )

    assert asyncio.run(_exercise()) is False
    assert runner._completion_notification_batches == {}
    assert runner._completion_notification_batch_tasks == {}
    adapter.handle_message.assert_not_awaited()


def test_successful_batch_releases_all_lifecycle_task_references():
    adapter = SimpleNamespace(handle_message=AdmittingHandler(return_value=None))
    runner = _runner(adapter)
    runner._completion_notification_batch_window = 0

    async def _exercise():
        result = await runner._enqueue_process_completion_notification(
            "completion",
            _completion_event(started_at=1.0, session_id="proc_success_cleanup"),
        )
        await asyncio.sleep(0)
        return result

    assert asyncio.run(_exercise()) is True
    assert runner._completion_notification_batch_tasks == {}
    assert runner._completion_notification_batch_flush_tasks == set()
    assert runner._background_tasks == set()


def test_shutdown_cancels_overlapping_flushes_for_same_route():
    delivery_entered = asyncio.Event()

    async def _blocked_delivery(_event):
        delivery_entered.set()
        await asyncio.Event().wait()

    adapter = SimpleNamespace(handle_message=AdmittingHandler(side_effect=_blocked_delivery))
    runner = _runner(adapter)
    runner._completion_notification_batch_window = 0
    first_event = _completion_event(started_at=1.0, session_id="proc_old_flush")
    second_event = _completion_event(started_at=2.0, session_id="proc_new_flush")

    async def _exercise():
        first = asyncio.create_task(
            runner._enqueue_process_completion_notification("first", first_event)
        )
        await delivery_entered.wait()

        # The first task has detached from the route index while blocked in
        # adapter delivery.  A new completion for the same route must create a
        # second flush, and shutdown must still own and cancel both tasks.
        assert runner._completion_notification_batch_tasks == {}
        runner._completion_notification_batch_window = 3600
        second = asyncio.create_task(
            runner._enqueue_process_completion_notification("second", second_event)
        )
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        flush_tasks = set(runner._completion_notification_batch_flush_tasks)
        assert len(flush_tasks) == 2

        await runner._cancel_process_completion_batch_tasks()

        assert await asyncio.gather(first, second) == [False, False]
        assert all(task.cancelled() for task in flush_tasks)
        assert runner._completion_notification_batches == {}
        assert runner._completion_notification_batch_tasks == {}
        assert runner._completion_notification_batch_flush_tasks == set()
        assert runner._background_tasks == set()

    asyncio.run(_exercise())
    adapter.handle_message.assert_awaited_once()


# ---------------------------------------------------------------------------
# Async-delegation same-tick coalescing (#70300)
# ---------------------------------------------------------------------------


def _distinct_async_event(delegation_id, session_key="agent:main:telegram:dm:12345:678"):
    event = _async_event(delegation_id)
    event["session_key"] = session_key
    event["summary"] = f"Result for {delegation_id}"
    return event


def test_same_tick_async_batch_coalesces_into_one_turn_and_acks_all_rows(
    monkeypatch, isolated_registry,
):
    """Three same-session async completions in one drain -> one synthetic turn.

    All three durable delegation rows must be honestly acknowledged only
    after the single consolidated injection was accepted by the adapter.
    """
    from tools import async_delegation

    isolated = queue.Queue()
    monkeypatch.setattr(isolated_registry, "completion_queue", isolated)
    events = [_distinct_async_event(f"deleg_batch_{i}") for i in range(3)]
    for event in events:
        _persist_pending_completion(event)
        isolated.put(dict(event))

    adapter = SimpleNamespace(handle_message=AdmittingHandler())
    runner = _runner(adapter)
    _stop_after_sleeps(monkeypatch, runner, count=2)

    asyncio.run(runner._async_delegation_watcher(interval=0))

    adapter.handle_message.assert_awaited_once()
    delivered = adapter.handle_message.await_args.args[0]
    assert "3 background subagent delegations" in delivered.text
    for i in range(3):
        assert f"Result for deleg_batch_{i}" in delivered.text
    for event in events:
        row = async_delegation.get_durable_delegation(event["delegation_id"])
        assert row is not None
        assert row["delivery_state"] == "delivered"
    assert isolated.empty()


def test_same_tick_async_events_for_different_sessions_do_not_coalesce(
    monkeypatch, isolated_registry,
):
    isolated = queue.Queue()
    monkeypatch.setattr(isolated_registry, "completion_queue", isolated)
    isolated.put(_distinct_async_event("deleg_route_a"))
    isolated.put(_distinct_async_event(
        "deleg_route_b", session_key="agent:main:telegram:dm:99999:678",
    ))

    adapter = SimpleNamespace(handle_message=AdmittingHandler())
    runner = _runner(adapter)
    _stop_after_sleeps(monkeypatch, runner, count=2)

    asyncio.run(runner._async_delegation_watcher(interval=0))

    assert adapter.handle_message.await_count == 2
    texts = [call.args[0].text for call in adapter.handle_message.await_args_list]
    assert not any("background subagent delegations" in text for text in texts)
    assert any("deleg_route_a" in text for text in texts)
    assert any("deleg_route_b" in text for text in texts)


def test_single_async_event_latency_and_text_are_unchanged(
    monkeypatch, isolated_registry,
):
    """A lone completion keeps the plain per-event formatter output."""
    isolated = queue.Queue()
    monkeypatch.setattr(isolated_registry, "completion_queue", isolated)
    isolated.put(_distinct_async_event("deleg_single"))

    adapter = SimpleNamespace(handle_message=AdmittingHandler())
    runner = _runner(adapter)
    _stop_after_sleeps(monkeypatch, runner, count=2)

    asyncio.run(runner._async_delegation_watcher(interval=0))

    adapter.handle_message.assert_awaited_once()
    delivered = adapter.handle_message.await_args.args[0]
    assert "background subagent delegations" not in delivered.text
    assert "deleg_single" in delivered.text


def test_failed_coalesced_async_batch_releases_claims_and_retries(
    monkeypatch, isolated_registry,
):
    """A rejected consolidated injection leaves every durable row pending."""
    from tools import async_delegation

    isolated = queue.Queue()
    monkeypatch.setattr(isolated_registry, "completion_queue", isolated)
    events = [_distinct_async_event(f"deleg_retry_{i}") for i in range(2)]
    for event in events:
        _persist_pending_completion(event)
        isolated.put(dict(event))

    adapter = SimpleNamespace(
        handle_message=AdmittingHandler(side_effect=[RuntimeError("temporary"), None])
    )
    runner = _runner(adapter)
    _stop_after_sleeps(monkeypatch, runner, count=3)

    asyncio.run(runner._async_delegation_watcher(interval=0))

    # First tick fails as one batch, second tick delivers the same batch.
    assert adapter.handle_message.await_count == 2
    for event in events:
        row = async_delegation.get_durable_delegation(event["delegation_id"])
        assert row is not None
        assert row["delivery_state"] == "delivered"
    assert isolated.empty()


def test_sibling_claimed_by_other_consumer_is_not_double_delivered(
    monkeypatch, isolated_registry,
):
    """A sibling owned elsewhere is excluded from the consolidated turn."""
    from tools import async_delegation

    isolated = queue.Queue()
    monkeypatch.setattr(isolated_registry, "completion_queue", isolated)
    events = [_distinct_async_event(f"deleg_owned_{i}") for i in range(2)]
    for event in events:
        _persist_pending_completion(event)
        isolated.put(dict(event))
    # Simulate another live consumer holding the second row's claim.
    assert async_delegation.claim_completion_delivery(
        events[1]["delegation_id"], "other-consumer:claim",
    )

    adapter = SimpleNamespace(handle_message=AdmittingHandler())
    runner = _runner(adapter)
    _stop_after_sleeps(monkeypatch, runner, count=2)

    asyncio.run(runner._async_delegation_watcher(interval=0))

    adapter.handle_message.assert_awaited_once()
    delivered = adapter.handle_message.await_args.args[0]
    assert "Result for deleg_owned_0" in delivered.text
    assert "Result for deleg_owned_1" not in delivered.text
    row = async_delegation.get_durable_delegation(events[1]["delegation_id"])
    assert row["delivery_state"] == "pending"


@pytest.mark.parametrize("unavailable", ["raw_adapter", "transport", "owner_db", "api_db"])
@pytest.mark.parametrize("batch_size", [1, 2])
def test_unavailable_delivery_preserves_budget_across_restarts(tmp_path, unavailable, batch_size):
    """Unavailable owners/transports cannot consume any sibling's durable attempts."""
    from hermes_state import SessionDB
    from tools import async_delegation

    events = [_async_event(f"deleg_unavailable_{i}") for i in range(batch_size)]
    raw = unavailable in {"raw_adapter", "api_db"}
    for event in events:
        if raw:
            event["session_key"] = "opaque-client-session"
        if unavailable == "owner_db":
            event["parent_session_id"] = "parent-session"
        _persist_pending_completion(event)

    adapter = SimpleNamespace(handle_message=AdmittingHandler())
    api = SimpleNamespace(supports_async_delivery=False, _ensure_session_db=lambda: None)
    for _restart in range(3):
        runner = _runner(adapter)
        runner.adapters = {Platform.API_SERVER: api} if unavailable == "api_db" else {}
        if unavailable == "owner_db":
            runner.adapters = {Platform.TELEGRAM: adapter}
        assert asyncio.run(runner._deliver_async_delegation_group(events)) is False
        for event in events:
            row = async_delegation.get_durable_delegation(event["delegation_id"])
            assert (row["delivery_state"], row["delivery_attempts"]) == ("pending", 0)

    runner = _runner(adapter)
    db = SessionDB(tmp_path / "owner.db")
    try:
        if raw:
            db.create_session("opaque-client-session", "api_server")
            api._ensure_session_db = lambda: db
            runner.adapters = {Platform.API_SERVER: api}
        if unavailable == "owner_db":
            runner._session_db = SimpleNamespace(get_session=AsyncMock(return_value={"ended_at": None}))
        assert asyncio.run(runner._deliver_async_delegation_group(events)) is True
        for event in events:
            row = async_delegation.get_durable_delegation(event["delegation_id"])
            assert (row["delivery_state"], row["delivery_attempts"]) == ("delivered", 1)
        if raw:
            rows = db.get_messages("opaque-client-session")
            assert len(rows) == 1
            assert rows[0]["display_kind"] == "async_delegation_complete"
            adapter.handle_message.assert_not_awaited()
        else:
            adapter.handle_message.assert_awaited_once()
    finally:
        db.close()


@pytest.mark.parametrize("available", [False, True])
def test_completion_profile_transport_never_falls_back(monkeypatch, isolated_registry, available):
    primary = SimpleNamespace(handle_message=AdmittingHandler(), send=AsyncMock())
    secondary = SimpleNamespace(handle_message=AdmittingHandler(), send=AsyncMock())
    runner = _runner(primary)
    runner._profile_adapters = {"research": {Platform.TELEGRAM: secondary}} if available else {}
    evt = dict(_completion_event(started_at=1), session_key="agent:research:telegram:dm:12345")
    result = asyncio.run(runner._inject_watch_notification("private result", evt))
    assert result is available
    asyncio.run(runner._send_watcher_message("telegram", "12345", None, "raw result", evt))
    primary.send.assert_not_awaited()
    assert secondary.send.await_count == int(available)
    primary.handle_message.assert_not_awaited()
    assert secondary.handle_message.await_count == int(available)
    if available:
        assert secondary.handle_message.await_args.args[0].source.profile == "research"


@pytest.mark.parametrize("event_type", ["watch_match", "watch_disabled"])
@pytest.mark.parametrize("mode", ["concise", "off"])
def test_idle_watch_drain_respects_notify_mode(monkeypatch, isolated_registry, event_type, mode):
    adapter = SimpleNamespace(handle_message=AdmittingHandler())
    runner = _runner(adapter)
    runner._load_background_notifications_mode = lambda: mode
    evt = dict(_completion_event(started_at=1), type=event_type,
               pattern="READY", output="READY", message="Watch patterns disabled")
    isolated_registry.completion_queue.put(evt)
    _stop_after_sleeps(monkeypatch, runner, count=2)
    asyncio.run(runner._async_delegation_watcher(interval=0))
    assert adapter.handle_message.await_count == (0 if mode == "off" else 1)
    assert isolated_registry.completion_queue.empty()


def test_watch_drain_retries_transport_failure(monkeypatch, isolated_registry):
    adapter = SimpleNamespace(handle_message=AdmittingHandler(side_effect=[RuntimeError("offline"), None]))
    runner = _runner(adapter)
    runner._load_background_notifications_mode = lambda: "concise"
    evt = dict(_completion_event(started_at=1), type="watch_match", pattern="READY", output="READY")
    isolated_registry.completion_queue.put(evt)
    _stop_after_sleeps(monkeypatch, runner, count=3)
    asyncio.run(runner._async_delegation_watcher(interval=0))
    assert adapter.handle_message.await_count == 2
    assert isolated_registry.completion_queue.empty()


def test_slow_ledger_transaction_does_not_block_the_event_loop(monkeypatch, isolated_registry):
    """A state.db commit/close that checkpoints a large WAL can block for tens of seconds. Every
    durable claim and settle on the delivery path must run off the loop, so the loop keeps
    answering liveness probes while the ledger is slow."""
    import contextlib
    import time
    from itertools import pairwise

    from hermes_cli import sqlite_util
    from tools import async_delegation

    events = [_distinct_async_event(f"deleg_slow_{i}") for i in range(2)]
    for event in events:
        _persist_pending_completion(event)

    stall_s = 1.2
    real_transaction = sqlite_util.transaction

    @contextlib.contextmanager
    def _slow_transaction(conn, **kwargs):
        with real_transaction(conn, **kwargs) as inner:
            yield inner
        time.sleep(stall_s)  # the blocking close/checkpoint seen in the gateway freeze

    monkeypatch.setattr(sqlite_util, "transaction", _slow_transaction)
    adapter = SimpleNamespace(handle_message=AdmittingHandler())
    runner = _runner(adapter)

    async def _exercise():
        ticks = [time.monotonic()]
        delivery_done = asyncio.Event()

        async def _heartbeat():
            while not delivery_done.is_set():
                await asyncio.sleep(0.05)
                ticks.append(time.monotonic())

        heartbeat = asyncio.create_task(_heartbeat())
        try:
            result = await runner._deliver_async_delegation_group([dict(e) for e in events])
        finally:
            delivery_done.set()
            await heartbeat
        return result, max(b - a for a, b in pairwise(ticks))

    result, worst_gap = asyncio.run(_exercise())

    assert result is True
    adapter.handle_message.assert_awaited_once()
    for event in events:
        assert async_delegation.get_durable_delegation(event["delegation_id"])["delivery_state"] == "delivered"
    assert worst_gap < 1.0, f"event loop stalled {worst_gap:.2f}s behind a slow ledger transaction"


@pytest.mark.parametrize("siblings", [False, True])
def test_cancelled_claim_refunds_during_executor_shutdown(monkeypatch, siblings):
    """asyncio.run teardown must refund claims without keeping the loop alive for release."""
    import concurrent.futures
    import threading

    from gateway.run_notifications_ledger import claim_off_loop, claim_siblings_off_loop
    from tools import async_delegation

    events = [_distinct_async_event(f"deleg_shutdown_{i}") for i in range(2 if siblings else 1)]
    for event in events:
        _persist_pending_completion(event)
    entered, shutting_down = threading.Event(), threading.Event()
    pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    real_shutdown = pool.shutdown
    real_claim = async_delegation.claim_event_delivery

    def _shutdown(*args, **kwargs):
        # asyncio has forbidden new executor submissions before it calls shutdown.
        shutting_down.set()
        return real_shutdown(*args, **kwargs)

    def _slow_claim(event, consumer):
        entered.set()
        assert shutting_down.wait(5)
        return real_claim(event, consumer)

    monkeypatch.setattr(pool, "shutdown", _shutdown)
    monkeypatch.setattr(async_delegation, "claim_event_delivery", _slow_claim)

    async def _exercise():
        asyncio.get_running_loop().set_default_executor(pool)
        if siblings:
            claiming = claim_siblings_off_loop([(event, "result") for event in events], "old")
        else:
            claiming = claim_off_loop(
                lambda: _slow_claim(events[0], "old"),
                lambda claim_id: async_delegation.defer_completion_delivery(
                    events[0]["delegation_id"], claim_id,
                ),
            )
        task = asyncio.create_task(claiming)
        while not entered.is_set():
            await asyncio.sleep(0)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        # Return immediately; only asyncio.run's executor teardown unblocks the claim.

    asyncio.run(_exercise())
    for event in events:
        row = async_delegation.get_durable_delegation(event["delegation_id"])
        assert (row["delivery_state"], row["delivery_attempts"]) == ("pending", 0)
        assert async_delegation.claim_event_delivery(event, "next-consumer")


def test_cancelled_finished_claim_refunds_after_executor_shutdown(monkeypatch):
    """A finished worker whose result is not consumed needs an executor-independent refund."""
    import concurrent.futures
    import contextvars
    import threading

    from gateway.run_notifications_ledger import claim_off_loop
    from tools import async_delegation

    event = _distinct_async_event("deleg_finished_shutdown")
    _persist_pending_completion(event)
    pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    real_submit = pool.submit
    submitted = []
    released = threading.Event()
    release_contexts = []
    scope = contextvars.ContextVar("claim_release_scope", default=None)

    def _submit(*args, **kwargs):
        future = real_submit(*args, **kwargs)
        submitted.append(future)
        return future

    def _release(claim_id):
        async_delegation.defer_completion_delivery(event["delegation_id"], claim_id)
        release_contexts.append((scope.get(), threading.current_thread().daemon))
        released.set()

    monkeypatch.setattr(pool, "submit", _submit)

    async def _exercise():
        asyncio.get_running_loop().set_default_executor(pool)
        scope.set("owning-profile")
        task = asyncio.create_task(claim_off_loop(
            lambda: async_delegation.claim_event_delivery(event, "old"), _release,
        ))
        await asyncio.sleep(0)  # The claimant submits and suspends at its await.
        # Block this test's loop so the worker finishes before its result can be consumed.
        assert submitted[0].result(timeout=5)
        pool.shutdown(wait=True)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(_exercise())
    assert released.wait(5), "finished claim was not refunded after executor shutdown"
    assert release_contexts == [("owning-profile", False)]
    row = async_delegation.get_durable_delegation(event["delegation_id"])
    assert (row["delivery_state"], row["delivery_attempts"]) == ("pending", 0)
    assert async_delegation.claim_event_delivery(event, "next-consumer")


def test_cancelled_sibling_claim_releases_its_lease(monkeypatch, isolated_registry):
    """A claim taken in a worker thread must not strand its lease when the awaiting task is
    cancelled (shutdown): the row stays immediately claimable instead of waiting out the lease."""
    import threading

    from tools import async_delegation

    events = [_distinct_async_event(f"deleg_cancel_{i}") for i in range(2)]
    for event in events:
        _persist_pending_completion(event)
    entered, proceed = threading.Event(), threading.Event()
    real_claim = async_delegation.claim_event_delivery

    def _gated_claim(evt, consumer):
        entered.set()
        proceed.wait(5)
        return real_claim(evt, consumer)

    monkeypatch.setattr(async_delegation, "claim_event_delivery", _gated_claim)
    runner = _runner(SimpleNamespace(handle_message=AdmittingHandler()))

    async def _exercise():
        task = asyncio.create_task(runner._deliver_async_delegation_group([dict(e) for e in events]))
        while not entered.is_set():
            await asyncio.sleep(0.01)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        proceed.set()
        for _ in range(300):
            await asyncio.sleep(0.01)
            if async_delegation.claim_completion_delivery(events[1]["delegation_id"], "next-consumer"):
                return True
        return False

    assert asyncio.run(_exercise()), "cancelled sibling claim stranded its lease"


def test_cancelled_primary_claim_is_refunded_not_released(monkeypatch, isolated_registry):
    """A shutdown that cancels delivery after the primary claim commits must refund the attempt
    exactly once; releasing it as a failed delivery would spend the budget on every restart."""
    import threading

    from tools import async_delegation

    event = _distinct_async_event("deleg_cancel_primary")
    _persist_pending_completion(event)
    committed, proceed = threading.Event(), threading.Event()
    real_claim = async_delegation.claim_completion_delivery

    def _claim_then_stall(delegation_id, claim_id):
        result = real_claim(delegation_id, claim_id)
        committed.set()
        proceed.wait(5)
        return result

    monkeypatch.setattr(async_delegation, "claim_completion_delivery", _claim_then_stall)
    runner = _runner(SimpleNamespace(handle_message=AdmittingHandler()))

    async def _exercise():
        task = asyncio.create_task(runner._deliver_async_delegation_group([dict(event)]))
        while not committed.is_set():
            await asyncio.sleep(0.01)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        proceed.set()
        for _ in range(300):
            await asyncio.sleep(0.01)
            if real_claim(event["delegation_id"], "next-consumer"):
                return True
        return False

    assert asyncio.run(_exercise()), "cancelled primary claim stranded its lease"
    row = async_delegation.get_durable_delegation(event["delegation_id"])
    assert row["delivery_attempts"] == 1, f"cancellation spent an attempt: {row['delivery_attempts']}"


def test_cancelled_primary_delivery_refunds_all_claimed_batch_rows(monkeypatch, isolated_registry):
    """Cancellation after sibling and primary claims are held refunds every row without spending attempts."""
    from tools import async_delegation

    events = [_distinct_async_event(f"deleg_cancel_batch_{i}") for i in range(2)]
    for event in events:
        _persist_pending_completion(event)

    entered = asyncio.Event()
    blocked = asyncio.Event()

    async def _blocked_injection(_event):
        entered.set()
        await blocked.wait()

    adapter = SimpleNamespace(handle_message=AdmittingHandler(side_effect=_blocked_injection))
    runner = _runner(adapter)

    async def _exercise():
        task = asyncio.create_task(
            runner._deliver_async_delegation_group([dict(event) for event in events])
        )
        await entered.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(_exercise())

    for event in events:
        row = async_delegation.get_durable_delegation(event["delegation_id"])
        assert row is not None
        assert (row["delivery_state"], row["delivery_attempts"]) == ("pending", 0)
        assert async_delegation.claim_completion_delivery(
            event["delegation_id"], f"next-consumer:{event['delegation_id']}",
        )


@pytest.mark.parametrize(("verdict", "expected_operation"), [("terminal", "drop"), ("retry", "release")])
def test_cancelled_preflight_settle_is_not_released_again(
    monkeypatch, isolated_registry, verdict, expected_operation,
):
    """Cancellation during a pre-flight settle must not race a second finally settle."""
    import threading

    from gateway import run_notifications

    entered = threading.Event()
    unblock = threading.Event()
    operations = []

    async def _settle(ops):
        def _settle_in_worker():
            for operation in ops:
                operations.append(operation)
                if operation[0] == expected_operation:
                    entered.set()
                    assert unblock.wait(5)

        await asyncio.to_thread(_settle_in_worker)

    runner = _runner(SimpleNamespace(handle_message=AdmittingHandler()))
    runner._completion_delivery_ready = AsyncMock(return_value=True)
    runner._classify_completion_target = AsyncMock(return_value=verdict)
    runner._settle_durable_claims = _settle
    monkeypatch.setattr(run_notifications, "claim_off_loop", AsyncMock(return_value=True))
    event = _async_event("deleg_terminal_drop_race")
    event["parent_session_id"] = "gone-session"

    async def _exercise():
        task = asyncio.create_task(
            runner._deliver_completion_notification_scoped("completion", event)
        )
        while not entered.is_set():
            await asyncio.sleep(0)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        unblock.set()
        for _ in range(100):
            await asyncio.sleep(0.01)
            if not any(operation[0] == expected_operation for operation in operations):
                continue
            if len(operations) == 1:
                return
        raise AssertionError(f"terminal claim was settled more than once: {operations!r}")

    asyncio.run(_exercise())
    assert [operation[0] for operation in operations] == [expected_operation]


def test_cancelled_settle_waits_for_queued_worker():
    """Cancellation cannot cancel a durable settle that is queued behind a busy worker."""
    import concurrent.futures
    import threading

    from gateway.run_notifications_ledger import settle_durable_claims

    pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    entered = threading.Event()
    unblock = threading.Event()
    submitted = threading.Event()
    settled = threading.Event()
    operations = []

    async def _exercise():
        loop = asyncio.get_running_loop()
        loop.set_default_executor(pool)
        blocker = loop.run_in_executor(None, lambda: (entered.set(), unblock.wait(5)))
        while not entered.is_set():
            await asyncio.sleep(0)

        real_run_in_executor = loop.run_in_executor
        calls = 0

        def _track_submission(executor, func, *args):
            nonlocal calls
            calls += 1
            if calls == 1:
                submitted.set()
            return real_run_in_executor(executor, func, *args)

        def _settle(*operation) -> None:
            operations.append(operation)
            settled.set()

        loop.run_in_executor = _track_submission
        task = asyncio.create_task(
            settle_durable_claims(
                [("drop", "delegation", "claim")],
                _settle,
            )
        )
        while not submitted.is_set():
            await asyncio.sleep(0)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

        unblock.set()
        await asyncio.shield(blocker)
        while not settled.is_set():
            await asyncio.sleep(0)

    try:
        asyncio.run(_exercise())
    finally:
        pool.shutdown(wait=True)
    assert operations == [("drop", "delegation", "claim")]
