"""The embedded dispatcher re-evaluates lock ownership on every tick.

A gateway that starts with ``kanban.dispatch_in_gateway: false``, or one that
loses the boot race, must never strand the machine-global singleton lock: the
gateway whose config enables dispatch either dispatches or says loudly that it
cannot. A boot-once lock behind a boot-once config gate produced a 16h40m
fleet-wide dispatch outage — a gateway configured OUT of dispatching held the
lock for its whole process lifetime while the one configured to dispatch had
already lost the race and never retried.

Two harness choices make these tests able to fail on the pre-fix code:

- ``_kanban_dispatcher_boot`` is NOT stubbed. The defect lives in that method's
  early return, so a test that replaces it can only ever prove the loop works.
  Config and ``kanban_home`` are patched at their real seams instead.
- Each gateway's watcher stays alive as a real task across ticks, so "gateway B
  keeps running" is never quietly simulated as "gateway B restarts". A
  boot-once dispatcher RETURNS from the coroutine when it loses the race and can
  then never dispatch again — ``_ExhaustedWatcher`` preserves that so the
  assertion fires instead of a fresh boot papering over it.
"""

from __future__ import annotations

import asyncio
import logging

import pytest

import gateway.kanban_watchers as kw
from gateway.kanban_watchers_common import _acquire_singleton_lock, _release_singleton_lock


class _ExhaustedWatcher:
    """Stands in for a watcher coroutine that returned — it can never tick again."""

    def cancel(self):
        return None


class _Gateway:
    """One gateway process: a live watcher task stepped one tick at a time."""

    def __init__(self, harness, *, enabled: bool):
        self._harness = harness
        self.runner = object.__new__(kw.GatewayKanbanWatchersMixin)
        self.runner._running = True
        self.config = {"kanban": {"dispatch_in_gateway": enabled}}
        self.spawns: list[str] = []
        self._resume = asyncio.Event()
        self._tick_done = asyncio.Event()
        self._task = None
        self.runner._sleep_between_ticks = self._park

    async def _park(self, _interval):
        self._tick_done.set()
        await self._resume.wait()
        self._resume.clear()

    @property
    def enabled(self) -> bool:
        return self.config["kanban"]["dispatch_in_gateway"]

    @enabled.setter
    def enabled(self, value: bool) -> None:
        self.config["kanban"]["dispatch_in_gateway"] = value

    @property
    def exhausted(self) -> bool:
        return isinstance(self._task, _ExhaustedWatcher)

    def owns_lock(self) -> bool:
        return self.runner._owns_kanban_dispatcher_lock()

    async def tick(self) -> None:
        """Advance this gateway by exactly one dispatcher tick."""
        self._harness.active = self
        if self.exhausted:
            return
        if self._task is None:
            self._task = asyncio.create_task(self.runner._kanban_dispatcher_watcher())
        else:
            self._resume.set()
        self._tick_done.clear()
        waiter = asyncio.create_task(self._tick_done.wait())
        done, _ = await asyncio.wait([self._task, waiter],
                                     return_when=asyncio.FIRST_COMPLETED, timeout=5.0)
        waiter.cancel()
        if self._task in done:
            self._task.result()
            self._task = _ExhaustedWatcher()

    async def restart(self) -> None:
        """Process death and relaunch: every fd closes, the loop re-enters from boot."""
        await self.shutdown()
        self.runner = object.__new__(kw.GatewayKanbanWatchersMixin)
        self.runner._running = True
        self._resume = asyncio.Event()
        self._tick_done = asyncio.Event()
        self.runner._sleep_between_ticks = self._park
        self._task = None

    async def shutdown(self) -> None:
        self.runner._running = False
        self._resume.set()
        if isinstance(self._task, asyncio.Task):
            self._task.cancel()
            try:
                await self._task
            except BaseException:
                pass
        self.runner._release_kanban_dispatcher_lock()
        self._task = None


class _Harness:
    """Several gateway processes sharing one machine-global lock path."""

    def __init__(self):
        self.active = None
        self.built: list[_Gateway] = []

    def make(self, *, enabled: bool) -> _Gateway:
        gw = _Gateway(self, enabled=enabled)
        self.built.append(gw)
        return gw

    def load_config(self) -> dict:
        return self.active.config


@pytest.fixture
def gateways(tmp_path, monkeypatch):
    """Run a scenario against gateways whose real boot path reads patched seams."""
    (tmp_path / "kanban").mkdir(parents=True, exist_ok=True)
    harness = _Harness()

    from hermes_cli import config as hermes_config
    from hermes_cli import kanban_db as kb
    from hermes_cli import kanban_db_dispatch as kbd

    monkeypatch.setattr(hermes_config, "load_config", harness.load_config)
    monkeypatch.setattr(kb, "kanban_home", lambda: tmp_path)
    monkeypatch.setattr(kbd, "reap_worker_zombies", lambda: [])

    class _Dispatcher:
        def __init__(self, kb_module, settings):
            self.spawns = harness.active.spawns

        def tick_once(self):
            self.spawns.append("tick")
            return []

        def ready_nonempty(self):
            return False

    async def _direct(fn, *args):
        return fn(*args)

    async def _no_sleep(_delay):
        return None

    monkeypatch.setattr(kw, "_KanbanDispatcher", _Dispatcher)
    monkeypatch.setattr(kw, "_resolve_dispatcher_settings",
                        lambda cfg, kb_module: type("S", (), {"interval": 1.0})())
    monkeypatch.setattr(kw, "_to_thread_process_service", _direct)
    monkeypatch.setattr(kw, "_kanban_dispatch_allowed", lambda: True)
    monkeypatch.setattr(kw, "_resolve_auto_decompose_settings", lambda load_config: (False, 0))
    monkeypatch.setattr(kw.asyncio, "sleep", _no_sleep)
    monkeypatch.delenv("HERMES_KANBAN_DISPATCH_IN_GATEWAY", raising=False)

    def _run(scenario):
        async def _outer():
            try:
                await scenario(harness)
            finally:
                for gw in harness.built:
                    await gw.shutdown()

        asyncio.run(asyncio.wait_for(_outer(), timeout=30.0))

    return _run


def test_disabled_gateway_never_holds_the_lock(gateways):
    """"Disabled" must mean "not holding the lock", not "returned before reaching it"."""

    async def scenario(h):
        disabled = h.make(enabled=False)
        enabled = h.make(enabled=True)

        await disabled.tick()
        await disabled.tick()
        assert disabled.spawns == []
        assert not disabled.owns_lock()

        await enabled.tick()
        assert enabled.owns_lock()
        assert enabled.spawns, "the gateway configured to dispatch never dispatched"

    gateways(scenario)


def test_enabled_gateway_booting_second_takes_over_from_a_disabled_squatter(gateways):
    """Acceptance 1: A boots disabled first, B boots enabled second — B must dispatch."""

    async def scenario(h):
        a = h.make(enabled=False)
        await a.tick()

        b = h.make(enabled=True)
        await b.tick()

        assert b.owns_lock()
        assert b.spawns, "B was locked out by a gateway that does not dispatch"
        assert a.spawns == []

    gateways(scenario)


def test_lock_frees_to_a_running_gateway_when_the_holder_is_reconfigured(gateways):
    """Acceptance 2, the 16h40m outage reproduced.

    A wins the lock while enabled. B loses the race — pre-fix it gave up for its
    whole process lifetime. A is then reconfigured to dispatch=false and
    restarted; pre-fix it returned before the lock code, orphaning the lock with
    nobody dispatching. B must take over on its own next tick, without B
    restarting.
    """

    async def scenario(h):
        a = h.make(enabled=True)
        b = h.make(enabled=True)

        await a.tick()
        assert a.owns_lock()

        await b.tick()
        assert b.spawns == [], "two gateways dispatched concurrently"
        assert not b.owns_lock()

        a.enabled = False
        await a.restart()
        await a.tick()
        assert not a.owns_lock(), "a gateway configured out of dispatch still holds the lock"

        await b.tick()
        assert not b.exhausted, (
            "B gave up for its process lifetime after losing the boot race — this is the outage")
        assert b.owns_lock(), "the enabled gateway never re-acquired the freed lock"
        assert b.spawns

    gateways(scenario)


def test_holder_releases_on_the_tick_its_config_flips(gateways):
    """No restart needed on the holder either: the flip lands on the next tick."""

    async def scenario(h):
        a = h.make(enabled=True)
        b = h.make(enabled=True)

        await a.tick()
        assert a.owns_lock()
        await b.tick()
        assert not b.owns_lock()

        a.enabled = False
        await a.tick()
        assert not a.owns_lock()
        spawns_at_flip = len(a.spawns)

        await b.tick()
        assert b.owns_lock()
        assert len(a.spawns) == spawns_at_flip

    gateways(scenario)


def test_locked_out_gateway_warns_every_time_it_cannot_dispatch(gateways, caplog):
    """A gateway that wants to dispatch and cannot must say so at WARNING.

    Pre-fix this was a one-shot INFO at boot, so the fleet ran 16h40m with a full
    ready queue and nothing in the logs — the "dispatcher stuck" warning lives
    INSIDE the dispatch loop and never runs when no loop is dispatching.
    """

    async def scenario(h):
        holder = h.make(enabled=True)
        await holder.tick()
        assert holder.owns_lock()

        locked_out = h.make(enabled=True)
        with caplog.at_level(logging.WARNING, logger=kw.logger.name):
            await locked_out.tick()

        warnings = [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING]
        assert any("dispatch_in_gateway is true but another" in m for m in warnings), warnings

    gateways(scenario)


def test_unreadable_config_keeps_the_current_owner(gateways, monkeypatch):
    """A transient config read failure must not hand the lock around mid-flight."""

    async def scenario(h):
        a = h.make(enabled=True)
        await a.tick()
        assert a.owns_lock()

        def _explode():
            raise RuntimeError("config unreadable")

        from hermes_cli import config as hermes_config
        monkeypatch.setattr(hermes_config, "load_config", _explode)
        await a.tick()
        assert a.owns_lock()

    gateways(scenario)


def test_env_override_never_touches_the_lock(gateways, monkeypatch):
    """The env escape hatch is a permanent opt-out; it must not squat the lock."""

    async def scenario(h):
        monkeypatch.setenv("HERMES_KANBAN_DISPATCH_IN_GATEWAY", "false")
        off = h.make(enabled=True)
        await off.tick()
        assert off.spawns == []
        assert not off.owns_lock()

        monkeypatch.delenv("HERMES_KANBAN_DISPATCH_IN_GATEWAY")
        on = h.make(enabled=True)
        await on.tick()
        assert on.owns_lock()

    gateways(scenario)


def test_singleton_lock_still_serialises_two_enabled_gateways(tmp_path):
    """Regression fence: the per-tick retry must not weaken the single-owner backstop."""
    lock_path = tmp_path / ".dispatcher.lock"
    winner, winner_state = _acquire_singleton_lock(lock_path)
    loser, loser_state = _acquire_singleton_lock(lock_path)
    try:
        assert winner_state == "held"
        assert loser_state == "contended"
        assert loser is None
    finally:
        _release_singleton_lock(loser)
        _release_singleton_lock(winner)
