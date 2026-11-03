"""A background delegation completion the gateway keeps refusing must not go silent.

A transient refusal (``admission_conflict``, ``session_busy``, ``queue_full``...) defers the durable row and
refunds its delivery attempt, so it can stay ``pending`` at attempts=0 with nothing shown to the user. After
repeated refusals (or ~5 minutes of deferral) the watcher posts ONE concise notice to the same chat; the row is
never dropped or acknowledged and keeps retrying the real delivery.
"""

import asyncio
import queue
from collections import OrderedDict
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from gateway import run_notifications
from gateway.config import Platform
from gateway.run import GatewayRunner
from gateway.wake import WakeNotAccepted
from tests.module_clock import install_module_clock
from tools import async_delegation as ad


@pytest.fixture(autouse=True)
def isolated_registry(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    import tools.process_registry as pr_module

    monkeypatch.setattr(pr_module, "CHECKPOINT_PATH", tmp_path / "processes.json")
    registry = pr_module.ProcessRegistry()
    monkeypatch.setattr(pr_module, "process_registry", registry)
    monkeypatch.setattr(registry, "completion_queue", queue.Queue())
    return registry


def _runner(adapter):
    runner = object.__new__(GatewayRunner)
    runner._running = True
    runner.adapters = {Platform.TELEGRAM: adapter}
    runner.session_store = SimpleNamespace(_ensure_loaded=lambda: None, _entries={})
    runner._session_source_cache = {}
    runner._completion_delivery_lock = __import__("threading").Lock()
    runner._completion_deliveries_inflight = set()
    runner._completion_deliveries_delivered = OrderedDict()
    runner._completion_delivery_retention = 2048
    runner._background_tasks = set()
    return runner


def _adapter():
    return SimpleNamespace(handle_message=AsyncMock(), send=AsyncMock())


def _event(delegation_id="deleg_stuck", goal="Investigate flaky test"):
    return {
        "type": "async_delegation", "delegation_id": delegation_id,
        "session_key": "agent:main:telegram:dm:12345:678", "goal": goal,
        "status": "completed", "summary": "Found it", "dispatched_at": 1000.0, "completed_at": 1012.0,
    }


def _persist(*events):
    for evt in events:
        ad._persist_dispatch(evt)
        ad._persist_completion(evt, {"status": evt["status"], "summary": evt["summary"]})


def _run(monkeypatch, runner, ticks, *, tick_s=60.0):
    """Run the real watcher for *ticks* passes, each *tick_s* later (so every refusal backoff is due)."""
    sleeps = 0
    clock = install_module_clock(monkeypatch, run_notifications)

    async def _sleep(_delay):
        nonlocal sleeps
        sleeps += 1
        clock.now += tick_s
        if sleeps >= ticks:
            runner._running = False

    monkeypatch.setattr(asyncio, "sleep", _sleep)
    asyncio.run(runner._async_delegation_watcher(interval=0))


def _refuse_admission(monkeypatch, reason="admission_conflict"):
    async def _refuse(*_args, **_kwargs):
        raise WakeNotAccepted(reason, reason=reason)

    monkeypatch.setattr("gateway.wake.admit_internal_event", _refuse)


def _notices(adapter):
    return [call.args[1] for call in adapter.send.await_args_list]


def test_seven_refusals_stay_quiet(monkeypatch, isolated_registry):
    monkeypatch.setattr(run_notifications, "_STUCK_DELEGATION_NOTICE_AFTER_S", 1e9)  # count trigger only
    event = _event()
    _persist(event)
    isolated_registry.completion_queue.put(dict(event))
    _refuse_admission(monkeypatch)
    adapter = _adapter()

    _run(monkeypatch, _runner(adapter), ticks=8)  # N ticks = N-1 passes (the last sleep stops the loop)

    assert adapter.send.await_count == 0
    assert isolated_registry.completion_queue.get_nowait()["_watch_retry_step"] == 7


def test_eight_refusals_post_exactly_one_notice_and_keep_the_row_pending(monkeypatch, isolated_registry):
    # Count trigger only: 60 s ticks would otherwise cross the 5 min trigger at refusal 6.
    monkeypatch.setattr(run_notifications, "_STUCK_DELEGATION_NOTICE_AFTER_S", 1e9)
    event = _event()
    _persist(event)
    isolated_registry.completion_queue.put(dict(event))
    _refuse_admission(monkeypatch)
    adapter = _adapter()
    runner = _runner(adapter)

    _run(monkeypatch, runner, ticks=16)  # 15 refusals: crosses 8 and keeps refusing; the notice is not repeated

    (notice,) = _notices(adapter)
    assert "deleg_stuck" in notice and "completed" in notice
    assert "admission_conflict" in notice
    assert "Investigate flaky test" in notice
    assert "async_delegations" in notice and "result_json" in notice and "state.db" in notice
    assert adapter.send.await_args.args[0] == "12345"
    row = ad.get_durable_delegation("deleg_stuck")
    assert (row["delivery_state"], row["delivery_attempts"]) == ("pending", 0)  # neither dropped, acked nor drained
    queued = isolated_registry.completion_queue.get_nowait()
    assert queued["delegation_id"] == "deleg_stuck" and queued["_watch_retry_step"] == 15
    assert queued["_refusal_reason"] == "admission_conflict"


def test_continuous_deferral_for_five_minutes_posts_the_notice_before_eight_refusals(monkeypatch, isolated_registry):
    event = _event()
    _persist(event)
    isolated_registry.completion_queue.put(dict(event))
    _refuse_admission(monkeypatch, "session_busy")
    adapter = _adapter()
    runner = _runner(adapter)

    _run(monkeypatch, runner, ticks=5, tick_s=100.0)  # 5 refusals spanning 400 s

    (notice,) = _notices(adapter)
    assert "session_busy" in notice
    queued = isolated_registry.completion_queue.get_nowait()
    assert queued["_watch_retry_step"] < 8


def test_each_stuck_delegation_gets_its_own_single_notice(monkeypatch, isolated_registry):
    first, second = _event("deleg_a", "goal alpha"), _event("deleg_b", "goal beta")
    first["session_key"] = "agent:main:telegram:dm:111:1"
    second["session_key"] = "agent:main:telegram:dm:222:2"
    _persist(first, second)
    isolated_registry.completion_queue.put(dict(first))
    isolated_registry.completion_queue.put(dict(second))
    _refuse_admission(monkeypatch, "queue_full")
    adapter = _adapter()
    runner = _runner(adapter)

    _run(monkeypatch, runner, ticks=12)

    notices = _notices(adapter)
    assert len(notices) == 2
    assert sum("deleg_a" in text and "goal alpha" in text for text in notices) == 1
    assert sum("deleg_b" in text and "goal beta" in text for text in notices) == 1
    assert {call.args[0] for call in adapter.send.await_args_list} == {"111", "222"}
    for delegation_id in ("deleg_a", "deleg_b"):
        row = ad.get_durable_delegation(delegation_id)
        assert (row["delivery_state"], row["delivery_attempts"]) == ("pending", 0)


def test_successful_delivery_posts_no_notice(monkeypatch, isolated_registry):
    event = _event()
    _persist(event)
    isolated_registry.completion_queue.put(dict(event))

    async def _accept(_adapter, synth_event):
        synth_event._gateway_accepted = True

    monkeypatch.setattr("gateway.wake.admit_internal_event", _accept)
    adapter = _adapter()
    runner = _runner(adapter)

    _run(monkeypatch, runner, ticks=12)

    assert adapter.send.await_count == 0
    assert ad.get_durable_delegation("deleg_stuck")["delivery_state"] == "delivered"


def test_recovery_before_the_threshold_posts_no_notice(monkeypatch, isolated_registry):
    event = _event()
    _persist(event)
    isolated_registry.completion_queue.put(dict(event))
    calls = 0

    async def _flaky(_adapter, synth_event):
        nonlocal calls
        calls += 1
        if calls <= 3:
            raise WakeNotAccepted("session_busy", reason="session_busy")
        synth_event._gateway_accepted = True

    monkeypatch.setattr("gateway.wake.admit_internal_event", _flaky)
    adapter = _adapter()
    runner = _runner(adapter)

    _run(monkeypatch, runner, ticks=12)

    assert calls == 4
    assert adapter.send.await_count == 0
    assert ad.get_durable_delegation("deleg_stuck")["delivery_state"] == "delivered"


def test_suppress_warning_notifications_silences_the_notice_but_not_the_retry(
    monkeypatch, isolated_registry, tmp_path,
):
    (tmp_path / "config.yaml").write_text("display: {suppress_warning_notifications: true}\n", encoding="utf-8")
    event = _event()
    _persist(event)
    isolated_registry.completion_queue.put(dict(event))
    _refuse_admission(monkeypatch)
    adapter = _adapter()
    runner = _runner(adapter)

    _run(monkeypatch, runner, ticks=12)

    assert adapter.send.await_count == 0
    row = ad.get_durable_delegation("deleg_stuck")
    assert (row["delivery_state"], row["delivery_attempts"]) == ("pending", 0)
    assert isolated_registry.completion_queue.get_nowait()["_watch_retry_step"] >= 8
    assert "deleg_stuck" in runner._stuck_delegations_notified  # decided once; not re-evaluated every pass


def test_long_goal_is_one_truncated_line(monkeypatch, isolated_registry):
    event = _event(goal="line one\n" + "x" * 500)
    _persist(event)
    isolated_registry.completion_queue.put(dict(event))
    _refuse_admission(monkeypatch)
    adapter = _adapter()
    runner = _runner(adapter)

    _run(monkeypatch, runner, ticks=10)

    (notice,) = _notices(adapter)
    goal = notice.split("Goal: ", 1)[1].split(". To read", 1)[0]
    assert "\n" not in goal and goal.startswith("line one xxx") and goal.endswith("…")
    assert len(goal) <= run_notifications._STUCK_DELEGATION_GOAL_CHARS


def test_realistic_event_without_routing_keys_still_gets_one_notice(monkeypatch, isolated_registry):
    """An async-delegation event carries only ``session_key`` (no platform/chat_id/thread_id); the notice must
    resolve its route from it, not from keys the event never has."""
    event = _event()
    assert not {"platform", "chat_id", "thread_id"} & set(event)
    adapter = _adapter()
    runner = _runner(adapter)

    sent = asyncio.run(runner._send_stuck_delegation_notice(event, 8, 300.0))

    assert sent is True
    (notice,) = _notices(adapter)
    assert "deleg_stuck" in notice and "Investigate flaky test" in notice
    assert adapter.send.await_args.args[0] == "12345"


def test_unroutable_event_returns_false_so_the_retry_tries_again(monkeypatch, isolated_registry):
    event = _event()
    event["session_key"] = ""
    adapter = _adapter()

    assert asyncio.run(_runner(adapter)._send_stuck_delegation_notice(event, 8, 300.0)) is False
    assert adapter.send.await_count == 0


def test_notice_uses_the_session_store_origin_thread(monkeypatch, isolated_registry):
    from gateway.config import Platform as P
    from gateway.session import SessionSource

    event = _event()
    adapter = _adapter()
    runner = _runner(adapter)
    origin = SessionSource(platform=P.TELEGRAM, chat_id="12345", chat_type="dm", thread_id="999")
    runner.session_store._entries[event["session_key"]] = SimpleNamespace(origin=origin, transport_profile=None)

    assert asyncio.run(runner._send_stuck_delegation_notice(event, 8, 300.0)) is True
    assert adapter.send.await_args.kwargs["metadata"]["thread_id"] == "999"
