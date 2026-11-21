"""A completion whose platform nothing in the gateway can resolve must leave the drain loop.

``_build_process_event_source`` answers ``None`` for a platform name that is neither builtin nor
registered, so the delivery gate vetoes the completion on every pass. The drain paces those retries
(#141153) but never stops them: the event is re-offered every tick, its durable row stays ``pending``
(so restart recovery replays it), and each attempt logs the same warning — the unbounded loop and
log flood in #152734. The bounded lane here is only for that proved-unroutable shape; every
refusal that can still clear (a transport that is down, a parked profile, a restart wait) keeps
waiting as long as it takes, which the last two tests pin.
"""

import asyncio
import logging
import queue
from collections import OrderedDict
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from gateway import run_notifications
from gateway.config import GatewayConfig, Platform
from gateway.run import GatewayRunner
from gateway.synthetic_event_platforms import (
    UNREGISTERED_PLATFORM_ATTEMPTS,
    resolve_platform,
    unroutable_completion_exhausted,
)
from tests.module_clock import install_module_clock
from tools.process_registry import ProcessRegistry


class AdmittingHandler(AsyncMock):
    """Fake transport whose successful insertion issues the production receipt."""

    async def _execute_mock_call(self, event, *args, **kwargs):
        result = await super()._execute_mock_call(event, *args, **kwargs)
        event._gateway_accepted = True
        return result


@pytest.fixture(autouse=True)
def isolated_registry(tmp_path, monkeypatch):
    """The durable delegation ledger and the completion queue stay in tmp state."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    import tools.process_registry as pr_module

    monkeypatch.setattr(pr_module, "CHECKPOINT_PATH", tmp_path / "processes.json")
    registry = ProcessRegistry()
    monkeypatch.setattr(pr_module, "process_registry", registry)
    return registry


@pytest.fixture
def plugin_platform():
    """A platform a plugin registered, so the gateway can resolve it."""
    from gateway.platform_registry import PlatformEntry, platform_registry

    name = "customprobe"
    platform_registry.register(PlatformEntry(
        name=name, label="Custom Probe", adapter_factory=lambda cfg: None, check_fn=lambda: True))
    try:
        yield name
    finally:
        platform_registry.unregister(name)


def _runner(adapter=None):
    runner = object.__new__(GatewayRunner)
    runner._running = True
    runner.config = GatewayConfig()
    runner.adapters = {Platform.TELEGRAM: adapter} if adapter is not None else {}
    runner.session_store = SimpleNamespace(_ensure_loaded=lambda: None, _entries={})
    runner._session_source_cache = {}
    runner._completion_delivery_lock = __import__("threading").Lock()
    runner._completion_deliveries_inflight = set()
    runner._completion_deliveries_delivered = OrderedDict()
    runner._completion_delivery_retention = 2048
    runner._background_tasks = set()
    return runner


def _unregistered_event(delegation_id="deleg_custom_platform"):
    """A background subagent spawned from a plugin platform this gateway never registered."""
    return {
        "type": "async_delegation",
        "delegation_id": delegation_id,
        "session_key": "customprobe-chat:some-id",
        "platform": "customprobe-chat",
        "chat_type": "dm",
        "chat_id": "12345",
        "goal": "Investigate flaky test",
        "status": "completed",
        "summary": "Found it",
        "api_calls": 1,
        "duration_seconds": 12.0,
        "dispatched_at": 1000.0,
        "completed_at": 1012.0,
    }


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
    return async_delegation


def _stop_after_sleeps(monkeypatch, runner, count):
    """End the watcher after *count* sleeps, each one a minute later so backoff is due."""
    sleep_calls = 0
    clock = install_module_clock(monkeypatch, run_notifications)

    async def _bounded_sleep(_delay):
        nonlocal sleep_calls
        sleep_calls += 1
        clock.now += 60
        if sleep_calls >= count:
            runner._running = False

    monkeypatch.setattr(asyncio, "sleep", _bounded_sleep)


def test_unregistered_platform_completion_leaves_the_drain_loop(
    monkeypatch, isolated_registry, caplog,
):
    """The flood: an event whose platform never resolves is dropped once, not re-offered forever."""
    isolated = queue.Queue()
    monkeypatch.setattr(isolated_registry, "completion_queue", isolated)
    event = _unregistered_event()
    delegation = _persist_pending_completion(event)
    isolated.put(event)

    adapter = SimpleNamespace(handle_message=AdmittingHandler())
    runner = _runner(adapter)
    _stop_after_sleeps(monkeypatch, runner, count=UNREGISTERED_PLATFORM_ATTEMPTS + 4)

    with caplog.at_level(logging.WARNING, logger="gateway.run"):
        asyncio.run(runner._async_delegation_watcher(interval=0))

    assert isolated.empty(), "the refused event was re-offered after its attempts ran out"
    adapter.handle_message.assert_not_awaited()
    assert delegation.get_durable_delegation("deleg_custom_platform")["delivery_state"] == "dropped"
    dropped = [rec for rec in caplog.records if "was undeliverable in" in rec.getMessage()]
    assert len(dropped) == 1


def test_a_registered_plugin_platform_completion_is_never_bounded(
    monkeypatch, isolated_registry, plugin_platform,
):
    """A transport that is merely down can reconnect: those attempts must never drop a completion."""
    assert resolve_platform(plugin_platform) is not None
    runner = _runner()
    event = _unregistered_event("deleg_registered_plugin")
    event["platform"] = plugin_platform
    event["session_key"] = f"agent:main:{plugin_platform}:dm:12345"
    delegation = _persist_pending_completion(event)

    for _ in range(UNREGISTERED_PLATFORM_ATTEMPTS + 3):
        assert asyncio.run(unroutable_completion_exhausted(runner, event)) is False

    assert "dropped" != delegation.get_durable_delegation("deleg_registered_plugin")["delivery_state"]


def test_a_builtin_platform_with_no_transport_is_never_bounded(isolated_registry):
    runner = _runner()
    event = _unregistered_event("deleg_telegram_down")
    event["platform"] = "telegram"
    event["session_key"] = "agent:main:telegram:dm:12345"
    delegation = _persist_pending_completion(event)

    for _ in range(UNREGISTERED_PLATFORM_ATTEMPTS + 3):
        assert asyncio.run(unroutable_completion_exhausted(runner, event)) is False

    assert delegation.get_durable_delegation("deleg_telegram_down")["delivery_state"] == "pending"


def test_an_interim_task_notice_never_settles_the_batch_row(isolated_registry):
    """The interim per-task notice shares the batch's delegation_id; only the final result may settle it."""
    runner = _runner()
    event = {**_unregistered_event("deleg_batch"), "task_failure_notice": True}
    delegation = _persist_pending_completion(event)

    verdicts = [asyncio.run(unroutable_completion_exhausted(runner, event))
                for _ in range(UNREGISTERED_PLATFORM_ATTEMPTS)]

    assert verdicts[-1] is True, "the notice itself must still leave the loop"
    assert delegation.get_durable_delegation("deleg_batch")["delivery_state"] == "pending"


def test_the_unregistered_platform_warning_is_logged_once(caplog):
    """One line per name, not one per delivery attempt (#152734): 10k lines/hour came from the repeat."""
    runner = _runner(SimpleNamespace(handle_message=AdmittingHandler()))
    event = {**_unregistered_event(), "type": "completion", "session_id": "proc_watch"}

    with caplog.at_level(logging.WARNING, logger="gateway.run"):
        for _ in range(3):
            assert runner._build_process_event_source(event) is None

    mentions = [rec for rec in caplog.records if "customprobe-chat" in rec.getMessage()]
    assert len(mentions) == 1, f"the per-attempt warning repeated: {[r.getMessage() for r in mentions]}"
    assert "unregistered" in mentions[0].getMessage()


def test_unregistered_secondary_completion_settles_its_own_ledger(
    tmp_path, monkeypatch, isolated_registry,
):
    """An invalid plugin platform must not strand a secondary profile's pending ledger row."""
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override

    secondary_home = tmp_path / "profiles" / "research"
    secondary_home.mkdir(parents=True)
    event = _unregistered_event("deleg_secondary_unregistered")
    event["session_key"] = "agent:research:customprobe-chat:dm:12345"

    # Dispatch/completion were persisted in the named profile, not in the host's state.db.
    token = set_hermes_home_override(str(secondary_home))
    try:
        delegation = _persist_pending_completion(event)
        assert delegation.get_durable_delegation(event["delegation_id"])["delivery_state"] == "pending"
    finally:
        reset_hermes_home_override(token)

    runner = _runner()

    def _profile_home(source):
        assert source.profile == "research"
        return secondary_home

    monkeypatch.setattr(runner, "_resolve_profile_home_for_source", _profile_home)
    for attempt in range(UNREGISTERED_PLATFORM_ATTEMPTS):
        assert asyncio.run(unroutable_completion_exhausted(runner, event)) is (
            attempt == UNREGISTERED_PLATFORM_ATTEMPTS - 1
        )

    assert delegation.get_durable_delegation(event["delegation_id"]) is None
    token = set_hermes_home_override(str(secondary_home))
    try:
        assert delegation.get_durable_delegation(event["delegation_id"])["delivery_state"] == "dropped"
    finally:
        reset_hermes_home_override(token)
