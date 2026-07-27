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


@pytest.mark.asyncio
async def test_watcher_does_not_go_dormant_when_busy(monkeypatch):
    r, adapter = _runner_with(monkeypatch, idle=False)
    task = asyncio.create_task(r._scale_to_zero_watcher(interval=0.01))
    await asyncio.sleep(0.1)
    r._running = False
    await asyncio.wait_for(task, timeout=2)
    assert adapter.go_dormant_calls == 0


@pytest.mark.asyncio
async def test_watcher_respects_cooldown(monkeypatch):
    r, adapter = _runner_with(monkeypatch, idle=True)
    # Cooldown active far in the future: even though idle, no dormancy fires.
    r._scale_to_zero_cooldown_until = time.time() + 3600
    task = asyncio.create_task(r._scale_to_zero_watcher(interval=0.01))
    await asyncio.sleep(0.1)
    r._running = False
    await asyncio.wait_for(task, timeout=2)
    assert adapter.go_dormant_calls == 0


@pytest.mark.asyncio
async def test_watcher_noop_when_no_relay_adapter(monkeypatch):
    # Armed-but-no-relay-adapter (e.g. relay not yet connected): must not crash.
    r, _ = _runner_with(monkeypatch, idle=True, armed_adapter=False)
    task = asyncio.create_task(r._scale_to_zero_watcher(interval=0.01))
    await asyncio.sleep(0.1)
    r._running = False
    await asyncio.wait_for(task, timeout=2)
    # No exception, loop exits cleanly — nothing to assert beyond survival.


def test_bg_work_blocks_idle_via_background_tasks(monkeypatch):
    """_scale_to_zero_has_live_background_work() reports True when a tracked
    background task is still live (D3/F7) — the guard that keeps a gateway with
    an in-flight backgrounded subagent/terminal awake."""
    r = GatewayRunner.__new__(GatewayRunner)

    async def _never():
        await asyncio.sleep(3600)

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


def test_bg_work_blocks_idle_via_async_delegation(monkeypatch):
    """delegate_task(background=true) lives in tools.async_delegation, not the
    process registry. An active background delegation must block suspend too."""
    r = GatewayRunner.__new__(GatewayRunner)
    r._background_tasks = set()

    monkeypatch.setattr("tools.async_delegation.active_count", lambda: 1)

    assert r._scale_to_zero_has_live_background_work() is True


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


def test_bg_work_false_when_quiet():
    r = GatewayRunner.__new__(GatewayRunner)
    r._background_tasks = set()
    # No background tasks, no active processes in this fresh process.
    assert r._scale_to_zero_has_live_background_work() is False


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


def test_arm_when_no_platform_enabled_at_all(monkeypatch):
    """Chronos-only / no-messaging agent (all placeholders disabled) can scale to zero."""
    from gateway.platforms.base import Platform

    r = _arm_runner(
        monkeypatch,
        {Platform.TELEGRAM: False, Platform.DISCORD: False},
    )
    assert r._scale_to_zero_should_arm() is True


def test_no_arm_when_not_opted_in(monkeypatch):
    """Relay-only but the Labs stamp is off ⇒ never arm (fail-safe default)."""
    from gateway.platforms.base import Platform

    r = _arm_runner(monkeypatch, {Platform.RELAY: True}, enabled=False)
    assert r._scale_to_zero_should_arm() is False


def test_no_arm_without_wake_url(monkeypatch):
    """Relay-only + opted in but no registered wake URL ⇒ no arm (§3.4(1))."""
    from gateway.platforms.base import Platform

    r = _arm_runner(monkeypatch, {Platform.RELAY: True}, wake_url=None)
    assert r._scale_to_zero_should_arm() is False
