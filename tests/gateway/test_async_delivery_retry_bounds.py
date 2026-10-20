"""Bounds on the in-process async-completion delivery loop (#141893).

An unadmitted durable completion used to retry on every watcher tick forever: the defer settlement
refunds the delivery attempt, so ``delivery_attempts`` stayed 0 and nothing converged — one
stranded row injected every 2s (~43k log lines/day). Refused and unready deliveries now back off
exponentially, and a refusal for a row past the replay-age cap settles it as dropped, the same
verdict startup replay already gives its twin.
"""

import asyncio
import queue
import time
from collections import OrderedDict
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

import tools.async_delegation as ad
from gateway.run import GatewayRunner
from gateway.wake import WakeNotAccepted
from tools.process_registry import ProcessRegistry


@pytest.fixture(autouse=True)
def isolated_registry(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    import tools.process_registry as pr_module

    monkeypatch.setattr(pr_module, "CHECKPOINT_PATH", tmp_path / "processes.json")
    monkeypatch.setattr(pr_module, "process_registry", ProcessRegistry())


def _runner(adapter):
    runner = object.__new__(GatewayRunner)
    runner._running = True
    runner.adapters = {__import__("gateway.config", fromlist=["Platform"]).Platform.TELEGRAM: adapter}
    runner.session_store = SimpleNamespace(_ensure_loaded=lambda: None, _entries={})
    runner._session_source_cache = {}
    runner._completion_delivery_lock = __import__("threading").Lock()
    runner._completion_deliveries_inflight = set()
    runner._completion_deliveries_delivered = OrderedDict()
    runner._completion_delivery_retention = 2048
    runner._background_tasks = set()
    return runner


def _async_event(delegation_id="deleg_bounds"):
    now = time.time()
    return {
        "type": "async_delegation",
        "delegation_id": delegation_id,
        "session_key": "agent:main:telegram:dm:12345:678",
        "status": "completed",
        "summary": "Found it",
        "dispatched_at": now - 60.0,
        "completed_at": now - 12.0,
    }


def _persist_pending(event, *, completed_at=None):
    ad._persist_dispatch({
        "delegation_id": event["delegation_id"],
        "session_key": event["session_key"],
        "origin_ui_session_id": "",
        "parent_session_id": event.get("parent_session_id"),
        "dispatched_at": event["dispatched_at"],
    })
    ad._persist_completion(event, {"status": "completed", "summary": event["summary"]})
    if completed_at is not None:
        with ad._DB_LOCK, ad._transaction() as conn:
            conn.execute("UPDATE async_delegations SET completed_at=? WHERE delegation_id=?",
                         (completed_at, event["delegation_id"]))


def _stop_after_sleeps(monkeypatch, runner, count):
    sleep_calls = 0

    async def _bounded_sleep(_delay):
        nonlocal sleep_calls
        sleep_calls += 1
        if sleep_calls >= count:
            runner._running = False

    monkeypatch.setattr(asyncio, "sleep", _bounded_sleep)


def _refusing_admission(monkeypatch):
    """Make every injection raise WakeNotAccepted: the session exists but will not admit."""
    refusals = []
    raise_or_admit = {"raise": True}

    async def _refuse(adapter, event):
        refusals.append(event)
        if raise_or_admit["raise"]:
            raise WakeNotAccepted("session busy")

    import gateway.wake as wake_module
    monkeypatch.setattr(wake_module, "admit_internal_event", _refuse)
    return refusals, raise_or_admit


def test_refused_delivery_backs_off_instead_of_retrying_every_tick(monkeypatch, isolated_registry):
    """After one refused injection the event waits out a growing gap instead of re-attempting
    on every 2s watcher tick; the durable row stays pending with its budget unspent."""
    event = _async_event()
    _persist_pending(event)
    refusals, _flags = _refusing_admission(monkeypatch)
    isolated = queue.Queue()
    import tools.process_registry as pr_module
    monkeypatch.setattr(pr_module.process_registry, "completion_queue", isolated)
    isolated.put(event)
    runner = _runner(SimpleNamespace(handle_message=MagicMock()))
    _stop_after_sleeps(monkeypatch, runner, count=5)  # 4 work ticks

    asyncio.run(runner._async_delegation_watcher(interval=0))

    assert len(refusals) == 1, "ticks inside the backoff window must not attempt delivery"
    assert not isolated.empty(), "a refused delivery stays retryable"
    requeued = isolated.get_nowait()
    assert requeued["_delivery_defers"] == 1
    assert requeued["_delivery_retry_at"] > time.monotonic()
    row = ad.get_durable_delegation(event["delegation_id"])
    assert (row["delivery_state"], row["delivery_attempts"]) == ("pending", 0)


def test_backoff_gap_grows_with_each_deferral_and_caps_out():
    """The stamp's series is bounded and monotonic: 2s first gap (today's cadence), then
    exponentially wider, never past the cap."""
    from gateway.run_notifications import _ASYNC_DELIVERY_BACKOFF_S, _stamp_async_delivery_backoff

    evt = _async_event()
    gaps = []
    for _ in range(len(_ASYNC_DELIVERY_BACKOFF_S) + 4):
        _stamp_async_delivery_backoff(evt)
        gaps.append(evt["_delivery_retry_at"] - time.monotonic())
    assert gaps[0] <= _ASYNC_DELIVERY_BACKOFF_S[0] + 0.5
    assert gaps[-1] <= max(_ASYNC_DELIVERY_BACKOFF_S) + 0.5
    assert evt["_delivery_defers"] == len(_ASYNC_DELIVERY_BACKOFF_S) + 4

    process_event = {"type": "completion", "session_id": "proc"}
    _stamp_async_delivery_backoff(process_event)
    assert "_delivery_retry_at" not in process_event, "non-durable events keep their cadence"


def test_refusal_within_replay_age_defers_without_spending_attempts(monkeypatch, isolated_registry):
    """A fresh completion refused admission keeps today's contract: pending, budget unspent."""
    event = _async_event()
    _persist_pending(event)
    _refusing_admission(monkeypatch)
    runner = _runner(SimpleNamespace(handle_message=MagicMock()))

    result = asyncio.run(runner._deliver_completion_notification("done", event))

    assert result is False
    row = ad.get_durable_delegation(event["delegation_id"])
    assert (row["delivery_state"], row["delivery_attempts"]) == ("pending", 0)


def test_refusal_past_replay_age_terminally_drops(monkeypatch, isolated_registry):
    """A completion that has been pending longer than the replay-age cap settles as dropped on
    its next refusal — the same verdict startup replay gives it — and leaves the queue."""
    event = _async_event()
    _persist_pending(event, completed_at=time.time() - (ad._MAX_COMPLETION_REPLAY_AGE_S + 3600))
    _refusing_admission(monkeypatch)
    isolated = queue.Queue()
    import tools.process_registry as pr_module
    monkeypatch.setattr(pr_module.process_registry, "completion_queue", isolated)
    isolated.put(event)
    runner = _runner(SimpleNamespace(handle_message=MagicMock()))
    _stop_after_sleeps(monkeypatch, runner, count=3)  # 2 work ticks

    asyncio.run(runner._async_delegation_watcher(interval=0))

    row = ad.get_durable_delegation(event["delegation_id"])
    assert row["delivery_state"] == "dropped"
    assert isolated.empty(), "a terminal drop must not requeue the event"


def test_unready_delivery_is_stamped_and_claims_nothing(monkeypatch, isolated_registry):
    """The not-ready preflight (no routable adapter) spends no claim; its retry also backs off."""
    event = _async_event("deleg_unready")
    event["session_key"] = "20260711_unparseable_ui_session"  # no routing metadata -> not ready
    isolated = queue.Queue()
    import tools.process_registry as pr_module
    monkeypatch.setattr(pr_module.process_registry, "completion_queue", isolated)
    isolated.put(event)
    runner = _runner(SimpleNamespace(handle_message=MagicMock()))
    _stop_after_sleeps(monkeypatch, runner, count=4)  # 3 work ticks

    asyncio.run(runner._async_delegation_watcher(interval=0))

    runner.adapters[__import__("gateway.config", fromlist=["Platform"]).Platform.TELEGRAM]\
        .handle_message.assert_not_called()
    assert not isolated.empty()
    requeued = isolated.get_nowait()
    assert requeued.get("_delivery_defers") == 1
