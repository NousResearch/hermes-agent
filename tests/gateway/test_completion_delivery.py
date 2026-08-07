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


# ---------------------------------------------------------------------------
# Process completion batching / coalescing tests (#70300 / PR #71900)
# ---------------------------------------------------------------------------

def test_concurrent_process_watchers_coalesce_one_session_completion_turn(monkeypatch):
    """Three concurrent terminal watchers for one session must re-enter the
    agent once with a batched turn."""
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

    adapter = SimpleNamespace(handle_message=AsyncMock())
    runner = _runner(adapter)

    async def _exercise():
        await asyncio.gather(*(
            runner._run_process_watcher(w)
            for w in watchers
        ))

    asyncio.run(_exercise())

    # Three watchers → one batched turn (not 3 separate injections)
    adapter.handle_message.assert_awaited_once()
    delivered = adapter.handle_message.await_args.args[0]
    assert "3 background processes completed" in delivered.text
    for index in range(3):
        assert f"proc_batch_{index}" in delivered.text


def test_single_completion_uses_original_notification(monkeypatch):
    """A single completion should still use its rich original format text."""
    import tools.process_registry as pr_module

    registry = ProcessRegistry()
    session = ProcessSession(
        id="proc_single",
        command="echo hello",
        task_id="task-single",
        started_at=1000.0,
        output_buffer="hello\n",
        exited=True,
        exit_code=0,
        notify_on_complete=True,
    )
    registry._finished[session.id] = session
    monkeypatch.setattr(pr_module, "process_registry", registry)

    adapter = SimpleNamespace(handle_message=AsyncMock())
    runner = _runner(adapter)

    async def _exercise():
        await runner._run_process_watcher({
            "session_id": session.id,
            "check_interval": 0,
            "session_key": "agent:main:telegram:dm:123",
            "platform": "telegram",
            "chat_type": "dm",
            "chat_id": "123",
            "notify_on_complete": True,
        })

    asyncio.run(_exercise())

    adapter.handle_message.assert_awaited_once()
    delivered = adapter.handle_message.await_args.args[0]
    # Should NOT have batch boilerplate for a single completion
    assert "background processes completed for this session" not in delivered.text


def test_completion_batches_do_not_cross_conversation_routes():
    """Completions for different routes must produce separate deliveries."""
    adapter = SimpleNamespace(handle_message=AsyncMock())
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

    assert asyncio.run(_exercise()) == [
        runner.CompletionDisposition.DELIVERED,
        runner.CompletionDisposition.DELIVERED,
    ]
    assert adapter.handle_message.await_count == 2


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

    adapter = SimpleNamespace(handle_message=AsyncMock(side_effect=_deliver))
    runner = _runner(adapter)

    async def _exercise():
        first = asyncio.create_task(
            runner._enqueue_process_completion_notification(
                "first completion",
                _completion_event(started_at=1.0, session_id="proc_first"),
            )
        )
        await first_delivery_entered.wait()
        second = asyncio.create_task(
            runner._enqueue_process_completion_notification(
                "second completion",
                _completion_event(started_at=2.0, session_id="proc_second"),
            )
        )
        release_first_delivery.set()
        assert await first is runner.CompletionDisposition.DELIVERED
        assert await asyncio.wait_for(second, timeout=2.0) is (
            runner.CompletionDisposition.DELIVERED
        )

    asyncio.run(_exercise())

    assert adapter.handle_message.await_count == 2


def test_failed_coalesced_delivery_retries_all_entries():
    """When an adapter error occurs during batch delivery, every entry
    gets a False result so the watcher retry loop can re-enqueue."""
    attempts = 0

    async def _deliver(_event):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise RuntimeError("temporary adapter failure")

    adapter = SimpleNamespace(handle_message=AsyncMock(side_effect=_deliver))
    runner = _runner(adapter)
    events = [
        _completion_event(started_at=float(i), session_id=f"proc_retry_{i}")
        for i in range(2)
    ]

    async def _enqueue_all():
        return await asyncio.gather(*(
            runner._enqueue_process_completion_notification(
                f"event-{i}", evt,
            )
            for i, evt in enumerate(events)
        ))

    async def _exercise():
        # First batch fails → every waiter is resolved as retryable
        assert await _enqueue_all() == [
            runner.CompletionDisposition.RETRY,
            runner.CompletionDisposition.RETRY,
        ]
        # Second batch succeeds
        assert await _enqueue_all() == [
            runner.CompletionDisposition.DELIVERED,
            runner.CompletionDisposition.DELIVERED,
        ]

    asyncio.run(_exercise())
    assert adapter.handle_message.await_count == 2


def test_coalesced_success_records_every_completion_identity():
    """After a successful batched delivery, every completion identity must
    be present in the lifecycle ledger to prevent re-delivery."""
    adapter = SimpleNamespace(handle_message=AsyncMock())
    runner = _runner(adapter)
    events = [
        _completion_event(started_at=float(i), session_id=f"proc_ledger_{i}")
        for i in range(3)
    ]

    async def _exercise():
        return await asyncio.gather(*(
            runner._enqueue_process_completion_notification(
                f"event-{i}", evt,
            )
            for i, evt in enumerate(events)
        ))

    assert asyncio.run(_exercise()) == [
        runner.CompletionDisposition.DELIVERED,
        runner.CompletionDisposition.DELIVERED,
        runner.CompletionDisposition.DELIVERED,
    ]
    for evt in events:
        identity = runner._completion_delivery_identity(evt)
        assert identity in runner._completion_deliveries_delivered


def test_duplicate_primary_does_not_discard_fresh_batch_sibling():
    """If the first entry in a batch is a duplicate, the next fresh entry
    should be tried as primary so no sibling is discarded."""
    adapter = SimpleNamespace(handle_message=AsyncMock())
    runner = _runner(adapter)
    duplicate = _completion_event(started_at=1.0, session_id="proc_duplicate")
    fresh = _completion_event(started_at=2.0, session_id="proc_fresh")
    duplicate_identity = runner._completion_delivery_identity(duplicate)
    runner._completion_deliveries_delivered[duplicate_identity] = None

    async def _exercise():
        return await asyncio.gather(
            runner._enqueue_process_completion_notification("dup", duplicate),
            runner._enqueue_process_completion_notification("fresh", fresh),
        )

    assert asyncio.run(_exercise()) == [
        runner.CompletionDisposition.DELIVERED,
        runner.CompletionDisposition.DELIVERED,
    ]
    adapter.handle_message.assert_awaited_once()
    fresh_identity = runner._completion_delivery_identity(fresh)
    assert fresh_identity in runner._completion_deliveries_delivered


def test_batch_format_failure_resolves_waiters_for_retry(monkeypatch):
    """If batch formatting throws, all waiter futures should resolve with
    False so watchers retry rather than being stranded forever."""
    adapter = SimpleNamespace(handle_message=AsyncMock())
    runner = _runner(adapter)
    monkeypatch.setattr(
        runner,
        "_format_coalesced_process_completions",
        MagicMock(side_effect=ValueError("bad batch")),
    )
    events = [
        _completion_event(started_at=float(i), session_id=f"proc_format_{i}")
        for i in range(2)
    ]

    async def _exercise():
        pending = asyncio.gather(*(
            runner._enqueue_process_completion_notification(
                f"event-{i}", evt,
            )
            for i, evt in enumerate(events)
        ))
        return await asyncio.wait_for(pending, timeout=2.0)

    # Formatter failure resolves every waiter as retryable, not as a drop.
    assert asyncio.run(_exercise()) == [
        runner.CompletionDisposition.RETRY,
        runner.CompletionDisposition.RETRY,
    ]
    adapter.handle_message.assert_not_awaited()


def test_batch_message_format_includes_exit_codes_and_summary():
    """Verify the batched message format includes per-process status,
    exit codes, and an aggregate summary line."""
    adapter = SimpleNamespace(handle_message=AsyncMock())
    runner = _runner(adapter)

    events = [
        {**_completion_event(started_at=1.0, session_id="proc_ok"),
         "exit_code": 0, "output": "success\n"},
        {**_completion_event(started_at=2.0, session_id="proc_fail"),
         "exit_code": 1, "completion_reason": "error", "output": "fail\n"},
        {**_completion_event(started_at=3.0, session_id="proc_ok2"),
         "exit_code": 0, "output": "also ok\n"},
    ]

    async def _exercise():
        return await asyncio.gather(*(
            runner._enqueue_process_completion_notification(
                f"event-{i}", evt,
            )
            for i, evt in enumerate(events)
        ))

    asyncio.run(_exercise())

    adapter.handle_message.assert_awaited_once()
    text = adapter.handle_message.await_args.args[0].text

    # Must include per-process details
    assert "proc_ok" in text
    assert "proc_fail" in text
    assert "proc_ok2" in text

    # Must include exit codes
    assert "exit_code=0" in text
    assert "exit_code=1" in text

    # Must include aggregate summary
    assert "Summary:" in text
    assert "2 succeeded" in text
    assert "1 failed" in text


def test_batch_message_truncates_entries():
    """When >10 processes complete, show only first 10 in detail."""
    adapter = SimpleNamespace(handle_message=AsyncMock())
    runner = _runner(adapter)

    events = [
        _completion_event(started_at=float(i), session_id=f"proc_{i}")
        for i in range(12)
    ]

    async def _exercise():
        return await asyncio.gather(*(
            runner._enqueue_process_completion_notification(
                f"event-{i}", evt,
            )
            for i, evt in enumerate(events)
        ))

    asyncio.run(_exercise())

    adapter.handle_message.assert_awaited_once()
    text = adapter.handle_message.await_args.args[0].text

    # Must say 12 completed
    assert "12 background processes" in text

    # Must mention omitted entries
    assert "more" in text


def test_batch_message_output_truncates_long_tails():
    """Output >800 chars in batch entries should be truncated."""
    adapter = SimpleNamespace(handle_message=AsyncMock())
    runner = _runner(adapter)

    # Use distinguishable content to verify truncation direction
    prefix = "PREFIX_MARKER_"
    suffix = "_SUFFIX_MARKER"
    long_output = prefix + "x" * 2000 + suffix
    events = [
        _completion_event(started_at=1.0, session_id="proc_long"),
        _completion_event(started_at=2.0, session_id="proc_short"),
    ]
    events[0]["output"] = long_output

    async def _exercise():
        return await asyncio.gather(*(
            runner._enqueue_process_completion_notification(
                f"event-{i}", evt,
            )
            for i, evt in enumerate(events)
        ))

    asyncio.run(_exercise())

    adapter.handle_message.assert_awaited_once()
    text = adapter.handle_message.await_args.args[0].text

    # Must mention truncation
    assert "truncated" in text.lower()

    # The suffix (near end) should survive
    assert suffix in text

    # The prefix (near beginning) should not survive truncation
    assert prefix not in text


def test_single_completion_bounded_latency():
    """A single completion pays the batching window (up to 100 ms) so
    coalescing remains correct — true zero-latency singletons would
    require limiting coalescing to work already available in the same
    loop tick, which is incompatible with a time-window batch."""
    import time as _time

    adapter = SimpleNamespace(handle_message=AsyncMock())
    runner = _runner(adapter)

    evt = _completion_event(started_at=1.0)

    t0 = _time.monotonic()
    result = asyncio.run(
        runner._enqueue_process_completion_notification("single", evt)
    )
    elapsed = _time.monotonic() - t0

    assert result is runner.CompletionDisposition.DELIVERED
    adapter.handle_message.assert_awaited_once()
    # Bounded batching delay — well under 1 s even on slow CI.
    assert elapsed < 1.0, f"Single completion took {elapsed:.3f}s, expected < 1.0s"
def test_cancellation_during_batch_window_resolves_waiters_retryable():
    """Cancel during the batching sleep resolves pending waiters as False
    (retryable) — no stale batch or unresolved Future remains."""
    adapter = SimpleNamespace(handle_message=AsyncMock())
    runner = _runner(adapter)
    # Pre-initialise the batch state so the attribute path exists
    if not hasattr(runner, "_completion_notification_batches"):
        runner._completion_notification_batches = {}
        runner._completion_notification_batch_tasks = {}
        runner._completion_notification_batch_lock = __import__("threading").Lock()
        runner._completion_notification_batch_window = 100.0  # long window

    evt = _completion_event(started_at=1.0, session_id="proc_cancel_window")

    async def _cancel_during_window():
        # Create a task that we can cancel mid-flight
        enq_task = asyncio.create_task(
            runner._enqueue_process_completion_notification(
                "text", evt,
            )
        )
        # Let the flush task enter its sleep
        await asyncio.sleep(0.01)
        # Find and cancel the flush task
        key = runner._completion_notification_batch_key(evt)
        flush_task = runner._completion_notification_batch_tasks.get(key)
        assert flush_task is not None, "flush task should be scheduled"
        flush_task.cancel()
        # Wait for the enqueue to resolve
        result = await enq_task
        # After cleanup: no stale batch entry
        assert key not in runner._completion_notification_batches
        # After cleanup: no stale task key
        assert key not in runner._completion_notification_batch_tasks
        return result

    result = asyncio.run(_cancel_during_window())
    # Cancelled completion is retryable
    assert result is runner.CompletionDisposition.RETRY


def test_cancellation_during_delivery_resolves_waiters_retryable():
    """Cancelling the real flush mid-delivery resolves the waiter as RETRY.

    Exercises GatewayRunner._flush_process_completion_batch itself.  The
    previous version replaced that method with a local reimplementation via
    types.MethodType and asserted on the reimplementation, so it passed while
    the production path mapped CancelledError onto DROP_UNROUTABLE — the
    defect the sweeper review on #72675 reported as finding 1.  After the
    assertion was migrated to CompletionDisposition.RETRY the stand-in still
    resolved its futures with False, so the test failed for a reason
    unrelated to the code under test.

    The adapter blocks inside delivery so the flush task can be cancelled
    while it is awaiting the send, which is the case the finding describes.
    """
    delivery_entered = asyncio.Event()
    hold_delivery = asyncio.Event()

    async def _deliver(_event):
        delivery_entered.set()
        await hold_delivery.wait()

    adapter = SimpleNamespace(handle_message=AsyncMock(side_effect=_deliver))
    runner = _runner(adapter)
    runner._background_tasks = set()

    evt = _completion_event(started_at=1.0, session_id="proc_cancel_delivery")

    async def _exercise():
        enq_task = asyncio.create_task(
            runner._enqueue_process_completion_notification("text", evt)
        )
        await asyncio.wait_for(delivery_entered.wait(), timeout=2.0)

        # The real flush pops the batch and its task key before delivery
        # begins, so the task has to be located directly.
        flush_task = None
        for task in asyncio.all_tasks():
            coro = getattr(task, "get_coro", lambda: None)()
            if getattr(coro, "__name__", "") == "_flush_process_completion_batch":
                flush_task = task
                break
        assert flush_task is not None, "real flush task should be running"
        flush_task.cancel()

        result = await asyncio.wait_for(enq_task, timeout=2.0)
        assert result is runner.CompletionDisposition.RETRY, (
            f"Expected RETRY after cancellation during delivery, got {result!r}"
        )
        key = runner._completion_notification_batch_key(evt)
        assert key not in runner._completion_notification_batches
        assert key not in runner._completion_notification_batch_tasks

    asyncio.run(_exercise())


def test_batch_summary_counts_failures_in_omitted_tail():
    """12-entry batch with failures only outside entries[:10] must still
    reflect the correct aggregate success/failure counts in the summary."""
    adapter = SimpleNamespace(handle_message=AsyncMock())
    runner = _runner(adapter)

    events = [
        {**_completion_event(started_at=float(i), session_id=f"proc_{i}"),
         "exit_code": (1 if i >= 10 else 0),
         "output": f"out_{i}\n"}
        for i in range(12)
    ]

    async def _exercise():
        return await asyncio.gather(*(
            runner._enqueue_process_completion_notification(
                f"event-{i}", evt,
            )
            for i, evt in enumerate(events)
        ))

    asyncio.run(_exercise())

    adapter.handle_message.assert_awaited_once()
    text = adapter.handle_message.await_args.args[0].text

    # All entries counted, not just shown slice
    assert "Summary:" in text
    assert "10 succeeded" in text
    assert "2 failed" in text

    # Failures not shown in detail (they're in the omitted tail)
    assert "proc_10" not in text
    assert "proc_11" not in text

    # Omitted tail message present
    assert "more completion" in text
