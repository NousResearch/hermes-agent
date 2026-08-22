"""Watcher-level tests for scale-to-zero: the idle watcher's dormant sequence and
the arm-gate wiring, exercised against the real GatewayRunner methods bound onto
a lightweight stand-in (booting a full gateway is unnecessary for this logic and
would be slow/flaky).

These cover the parts gateway/test_scale_to_zero.py (pure helpers) can't: that
the watcher calls the relay adapter's go_dormant() exactly when idle+armed,
respects the cooldown, and skips when busy — the F7/D3 + D12 behaviour.
"""

from __future__ import annotations

import asyncio
import time

import pytest

from gateway.run import GatewayRunner


class _FakeRelayAdapter:
    def __init__(self):
        self.go_dormant_calls = 0

    async def go_dormant(self):
        self.go_dormant_calls += 1
        return True


def _runner_with(monkeypatch, *, idle, armed_adapter=True):
    """Build a GatewayRunner without booting it, stubbing just what the watcher
    touches. Real methods (_scale_to_zero_is_idle composition, the watcher body)
    run; only their dependencies are stubbed."""
    r = GatewayRunner.__new__(GatewayRunner)
    r._running = True
    r._scale_to_zero_cooldown_until = 0.0
    r._last_inbound_at = time.time()
    r._running_agents = {}
    r._background_tasks = set()
    adapter = _FakeRelayAdapter() if armed_adapter else None

    monkeypatch.setattr(r, "_scale_to_zero_is_idle", lambda: idle, raising=False)
    monkeypatch.setattr(r, "_relay_adapter_for_dormancy", lambda: adapter, raising=False)
    monkeypatch.setattr(r, "_scale_to_zero_idle_timeout_seconds", lambda: 300.0, raising=False)
    monkeypatch.setattr(r, "_update_runtime_status", lambda *a, **k: None, raising=False)
    return r, adapter


@pytest.mark.asyncio
async def test_watcher_goes_dormant_when_idle(monkeypatch):
    r, adapter = _runner_with(monkeypatch, idle=True)
    # Run one iteration: stop after the first sleep so the loop exits cleanly.
    task = asyncio.create_task(r._scale_to_zero_watcher(interval=0.01))
    await asyncio.sleep(0.1)
    r._running = False
    await asyncio.wait_for(task, timeout=2)
    assert adapter.go_dormant_calls >= 1
    # After driving dormant, a re-arm cooldown is set (0.F).
    assert r._scale_to_zero_cooldown_until > time.time()


    # No exception, loop exits cleanly — nothing to assert beyond survival.


def test_bg_work_blocks_idle_via_background_tasks(monkeypatch):
    """_scale_to_zero_has_live_background_work() reports True when a tracked
    background task is still live (D3/F7) — the guard that keeps a gateway with
    an in-flight backgrounded subagent/terminal awake."""
    r = GatewayRunner.__new__(GatewayRunner)

    async def _never():
        await asyncio.sleep(0.2)

    loop = asyncio.new_event_loop()
    try:
        t = loop.create_task(_never())
        r._background_tasks = {t}
        # process_registry has nothing active in this fresh process.
        assert r._scale_to_zero_has_live_background_work() is True
        t.cancel()
    finally:
        loop.run_until_complete(asyncio.gather(t, return_exceptions=True))
        loop.close()


def test_real_inbound_after_dormancy_restores_running_status(monkeypatch):
    """Once a dormant gateway receives real inbound after wake, the runtime
    lifecycle must not remain stuck in the watcher-written `draining` state."""
    r = GatewayRunner.__new__(GatewayRunner)
    r._last_inbound_at = 0.0
    r._scale_to_zero_cooldown_until = time.time() + 60.0
    status_updates = []
    monkeypatch.setattr(
        r,
        "_update_runtime_status",
        lambda state=None, *a, **k: status_updates.append(state),
        raising=False,
    )

    r._scale_to_zero_note_real_inbound()

    assert r._last_inbound_at > 0.0
    assert status_updates == ["running"]


# ── _scale_to_zero_should_arm: the CALL SITE feeds config.platforms (the F25 bug) ──
#
# config.platforms is pre-seeded with a DISABLED placeholder PlatformConfig for every
# known platform, so list(config.platforms.keys()) is always the full ~20-entry catalog
# regardless of what the instance runs. The arm check must filter to ENABLED platforms
# (mirroring the connect loop) before asking messaging_is_relay_only_or_absent — passing
# the bare placeholder keys made it see disabled `discord`/`telegram`/… as live direct
# platforms and refuse to arm on a real relay-only instance. The pure-helper tests in
# test_scale_to_zero.py pass bare names so they never exercised this call site.


def _arm_runner(monkeypatch, platform_states, *, enabled=True, wake_url="https://wake.example"):
    """Build a GatewayRunner stand-in whose config.platforms mirrors a real load:
    `platform_states` is {Platform: enabled_bool}; everything runs the REAL
    _scale_to_zero_should_arm. Only the env flag + wake_url resolution are stubbed."""
    from types import SimpleNamespace

    from gateway.config import PlatformConfig

    r = GatewayRunner.__new__(GatewayRunner)
    platforms = {p: PlatformConfig(enabled=en) for p, en in platform_states.items()}
    r.config = SimpleNamespace(platforms=platforms)

    monkeypatch.setattr("gateway.scale_to_zero.scale_to_zero_enabled", lambda *a, **k: enabled)
    monkeypatch.setattr("gateway.relay.relay_wake_url", lambda: wake_url)
    return r


def test_arm_true_for_relay_only_with_disabled_placeholders(monkeypatch):
    """The F25 regression test: relay ENABLED, every other platform present but
    DISABLED (the real load_gateway_config() shape). Must arm — the disabled
    placeholders must NOT count as live direct-socket platforms."""
    from gateway.platforms.base import Platform

    r = _arm_runner(
        monkeypatch,
        {
            Platform.TELEGRAM: False,
            Platform.DISCORD: False,
            Platform.SLACK: False,
            Platform.MATRIX: False,
            Platform.RELAY: True,
        },
    )
    assert r._scale_to_zero_should_arm() is True


def test_no_arm_when_a_direct_platform_is_actually_enabled(monkeypatch):
    """A genuinely-enabled direct-socket platform (real Discord token) DOES disarm —
    the filter must not over-broaden to 'ignore everything but relay'."""
    from gateway.platforms.base import Platform

    r = _arm_runner(
        monkeypatch,
        {Platform.DISCORD: True, Platform.RELAY: True},
    )
    assert r._scale_to_zero_should_arm() is False



# ── the self-suspend step: fires only after a clean quiesce, in order ─────────
#
# The gateway owns the suspend (Fly Proxy autostop is inbound-only/job-blind and
# no longer held open by outbound sockets), so the watcher must (a) suspend only
# AFTER go_dormant succeeded — the relay flip precedes the freeze, closing the
# buffered-event black hole — and (b) never suspend when the quiesce failed or
# inbound landed mid-quiesce.


@pytest.mark.asyncio
async def test_watcher_self_suspends_after_dormant(monkeypatch):
    r, adapter = _runner_with(monkeypatch, idle=True)
    calls = []

    async def fake_suspend():
        calls.append(("suspend", adapter.go_dormant_calls))
        r._running = False  # stop the loop after the first full sequence

    monkeypatch.setattr(r, "_scale_to_zero_self_suspend", fake_suspend, raising=False)
    task = asyncio.create_task(r._scale_to_zero_watcher(interval=0.01))
    await asyncio.wait_for(task, timeout=2)
    # Suspend fired exactly once, and only AFTER go_dormant ran (flip-before-freeze).
    assert calls == [("suspend", 1)]


@pytest.mark.asyncio
async def test_watcher_skips_suspend_when_dormant_fails(monkeypatch):
    r, adapter = _runner_with(monkeypatch, idle=True)

    async def broken_dormant():
        raise RuntimeError("quiesce failed")

    adapter.go_dormant = broken_dormant
    suspend_calls = []

    async def fake_suspend():
        suspend_calls.append(1)

    monkeypatch.setattr(r, "_scale_to_zero_self_suspend", fake_suspend, raising=False)
    task = asyncio.create_task(r._scale_to_zero_watcher(interval=0.01))
    await asyncio.sleep(0.1)
    r._running = False
    await asyncio.wait_for(task, timeout=2)
    # A failed quiesce means an UNFLIPPED relay — suspending would black-hole
    # inbound events. Must stay awake.
    assert suspend_calls == []


@pytest.mark.asyncio
async def test_watcher_skips_suspend_when_inbound_lands_mid_quiesce(monkeypatch):
    r, adapter = _runner_with(monkeypatch, idle=True)
    # First idle check (loop gate) True, second (post-quiesce re-check) False.
    reads = iter([True, False, False, False, False, False])
    monkeypatch.setattr(
        r, "_scale_to_zero_is_idle", lambda: next(reads, False), raising=False
    )
    suspend_calls = []

    async def fake_suspend():
        suspend_calls.append(1)

    monkeypatch.setattr(r, "_scale_to_zero_self_suspend", fake_suspend, raising=False)
    task = asyncio.create_task(r._scale_to_zero_watcher(interval=0.01))
    await asyncio.sleep(0.15)
    r._running = False
    await asyncio.wait_for(task, timeout=2)
    assert adapter.go_dormant_calls == 1
    assert suspend_calls == []


@pytest.mark.asyncio
async def test_self_suspend_noop_off_fly(monkeypatch):
    """Off-Fly (no flaps socket/identity) the helper is a silent no-op —
    dormancy without platform suspend, never an error."""
    r = GatewayRunner.__new__(GatewayRunner)
    monkeypatch.setattr(
        "gateway.scale_to_zero.self_suspend_available", lambda *a, **k: False
    )
    called = []
    monkeypatch.setattr(
        "gateway.scale_to_zero.suspend_self",
        lambda *a, **k: called.append(1) or True,
    )
    await r._scale_to_zero_self_suspend()
    assert called == []


# ── non-messaging platforms must not disarm (the api_server-key regression) ──
#
# The Docker stage2 hook now generates API_SERVER_KEY for every container, and
# key presence force-enables the api_server platform (gateway/config.py). The
# arm gate counted every enabled platform, so `api_server` (a loopback
# listener, not a messaging socket) made messaging_is_relay_only_or_absent
# False on EVERY hosted instance — silently disarming scale-to-zero. The gate
# must only count messaging platforms (excluding LOCAL/API_SERVER/WEBHOOK,
# mirroring _connect_platforms' messaging_platforms exclusion set).


def test_arm_true_with_api_server_enabled(monkeypatch):
    from gateway.platforms.base import Platform

    r = _arm_runner(
        monkeypatch,
        {
            Platform.RELAY: True,
            Platform.API_SERVER: True,
            Platform.TELEGRAM: False,
        },
    )
    assert r._scale_to_zero_should_arm() is True


def test_arm_true_with_all_non_messaging_surfaces_enabled(monkeypatch):
    from gateway.platforms.base import Platform

    r = _arm_runner(
        monkeypatch,
        {
            Platform.RELAY: True,
            Platform.API_SERVER: True,
            Platform.WEBHOOK: True,
            Platform.LOCAL: True,
        },
    )
    assert r._scale_to_zero_should_arm() is True


def test_direct_platform_still_disarms_alongside_api_server(monkeypatch):
    """The messaging-only filter must not over-broaden: a genuinely enabled
    direct-socket platform still disarms even with api_server also enabled."""
    from gateway.platforms.base import Platform

    r = _arm_runner(
        monkeypatch,
        {
            Platform.RELAY: True,
            Platform.API_SERVER: True,
            Platform.DISCORD: True,
        },
    )
    assert r._scale_to_zero_should_arm() is False


# ── supervised watchers must NOT count as live background work (staging bug) ──
#
# _spawn_supervised parks every permanent watcher task (session-expiry, kanban,
# reconnect, the scale-to-zero watcher ITSELF, ...) in _background_tasks. The
# bg-work check counted them, so an armed gateway considered itself busy
# forever and never went dormant — verified live on staging 2026-08-12 (armed
# at 05:25, fully idle 25+ min, zero "going dormant" lines). Fly's coarse
# autostop masked this until the gateway took ownership of the suspend.
# These tests exercise the REAL _spawn_supervised path — the earlier tests
# stubbed _background_tasks and missed the call site (same trap as F25).


@pytest.mark.asyncio
async def test_supervised_watchers_do_not_block_idle():
    r = GatewayRunner.__new__(GatewayRunner)
    r._running = True
    r._background_tasks = set()

    async def _forever():
        await asyncio.sleep(3600)

    # Spawn like production does — through _spawn_supervised.
    for name in ("session_expiry", "kanban", "scale_to_zero_watcher"):
        r._spawn_supervised(lambda: _forever(), name)
    await asyncio.sleep(0)  # let tasks start
    try:
        assert r._scale_to_zero_has_live_background_work() is False
    finally:
        for t in r._background_tasks:
            t.cancel()
        await asyncio.gather(*r._background_tasks, return_exceptions=True)


@pytest.mark.asyncio
async def test_transient_background_task_still_blocks_idle():
    """A plain (untagged) task in _background_tasks — startup-resume events,
    ad-hoc work — must still count as live background work."""
    r = GatewayRunner.__new__(GatewayRunner)
    r._running = True

    async def _work():
        await asyncio.sleep(3600)

    t = asyncio.create_task(_work())
    r._background_tasks = {t}
    try:
        assert r._scale_to_zero_has_live_background_work() is True
    finally:
        t.cancel()
        await asyncio.gather(t, return_exceptions=True)


@pytest.mark.asyncio
async def test_done_supervised_watcher_is_ignored_either_way():
    r = GatewayRunner.__new__(GatewayRunner)
    r._running = True

    async def _quick():
        return None

    t = asyncio.create_task(_quick())
    await t
    r._background_tasks = {t}
    assert r._scale_to_zero_has_live_background_work() is False


# ── permanent tasks spawned OUTSIDE _spawn_supervised must also be tagged ──
#
# _loop_heartbeat_task and _heartbeat_poll_task are both infinite while-True
# loops added to _background_tasks via plain asyncio.create_task() + manual
# add(), NOT through _spawn_supervised — so they were untagged and defeated
# the fix above: _loop_heartbeat_task starts unconditionally on every
# gateway boot (start()), which would make the busy check return True
# forever regardless of the _spawn_supervised fix, on every armed instance.


@pytest.mark.asyncio
async def test_loop_heartbeat_task_does_not_block_idle():
    r = GatewayRunner.__new__(GatewayRunner)
    r._running = True
    r._background_tasks = set()
    r._loop_heartbeat_task = None
    r._gateway_started_at = time.time()

    r._start_loop_heartbeat_task()
    await asyncio.sleep(0)  # let the task start
    try:
        assert r._scale_to_zero_has_live_background_work() is False
    finally:
        r._loop_heartbeat_task.cancel()
        await asyncio.gather(r._loop_heartbeat_task, return_exceptions=True)


@pytest.mark.asyncio
async def test_heartbeat_poll_task_does_not_block_idle():
    r = GatewayRunner.__new__(GatewayRunner)
    r._running = True
    r._background_tasks = set()
    r._heartbeat_poll_task = None
    r._heartbeat_watch = {}
    r._running_agents = {}

    r._start_heartbeat_poller()
    await asyncio.sleep(0)  # let the task start
    try:
        assert r._scale_to_zero_has_live_background_work() is False
    finally:
        r._heartbeat_poll_task.cancel()
        await asyncio.gather(r._heartbeat_poll_task, return_exceptions=True)


# ── in-flight cron / API-server work must block suspend (the 10:45 near-miss) ──
#
# Cron jobs run on the scheduler's thread pool and API-server runs live on the
# adapter — both outside _running_agents (the #60432 blind spot). The idle
# predicate must consume _active_work_count() (agents + cron + api runs), or a
# suspend can freeze a cron job mid-run: observed on staging 2026-08-20, where
# is_idle held True throughout a live cron run and only tick timing saved it.


def _work_count_runner(monkeypatch, *, agents=0, cron_ids=(), api_runs=0):
    from types import SimpleNamespace

    r = GatewayRunner.__new__(GatewayRunner)
    r._running = True
    r._running_agents = {f"a{i}": object() for i in range(agents)}
    r._background_tasks = set()
    r._last_inbound_at = 0.0  # inbound-quiet for hours
    monkeypatch.setattr(
        r, "_scale_to_zero_idle_timeout_seconds", lambda: 300.0, raising=False
    )
    monkeypatch.setattr(
        "cron.scheduler.get_running_job_ids", lambda: set(cron_ids)
    )
    api_adapter = SimpleNamespace(active_agent_work_count=lambda: api_runs)
    from gateway.platforms.base import Platform

    r.adapters = {Platform.API_SERVER: api_adapter}
    return r


def test_running_cron_job_blocks_idle(monkeypatch):
    r = _work_count_runner(monkeypatch, cron_ids={"job1"})
    assert r._scale_to_zero_is_idle() is False


def test_active_api_run_blocks_idle(monkeypatch):
    r = _work_count_runner(monkeypatch, api_runs=1)
    assert r._scale_to_zero_is_idle() is False


def test_idle_true_when_all_work_sources_quiet(monkeypatch):
    r = _work_count_runner(monkeypatch)
    assert r._scale_to_zero_is_idle() is True


def test_unreadable_cron_source_fails_awake(monkeypatch):
    """A transient failure reading the cron work source must count as WORK
    (stay awake), not as idle — fail-open accounting would reopen the
    mid-job-freeze hole exactly when bookkeeping is broken."""
    r = _work_count_runner(monkeypatch)

    def _boom():
        raise RuntimeError("registry unavailable")

    monkeypatch.setattr("cron.scheduler.get_running_job_ids", _boom)
    assert r._scale_to_zero_is_idle() is False


def test_unreadable_api_source_fails_awake(monkeypatch):
    from types import SimpleNamespace

    def _boom():
        raise RuntimeError("adapter wedged")

    r = _work_count_runner(monkeypatch)
    from gateway.platforms.base import Platform

    r.adapters = {Platform.API_SERVER: SimpleNamespace(active_agent_work_count=_boom)}
    assert r._scale_to_zero_is_idle() is False


def test_missing_api_adapter_is_not_work(monkeypatch):
    """No api_server adapter at all (common: relay-only instance before the
    key existed) is a NORMAL state, not an unreadable source — must not hold
    the machine awake."""
    r = _work_count_runner(monkeypatch)
    r.adapters = {}
    assert r._scale_to_zero_is_idle() is True
