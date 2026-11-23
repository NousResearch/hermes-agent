"""Plugin idle tasks under gateway housekeeping: a slow task never holds the thread up, and a task
still running when housekeeping stops never records afterwards.

The curator tick also polls the plugins' idle tasks (``run_idle_jobs``). That thread stamps the
runtime heartbeat, which ``hermes gateway status`` and ``/api/status`` read to tell a live gateway
from a wedged one, and it runs every other chore (the cron delivery drain, misfire catch-up, the
plugin update check, state.db maintenance). An idle task that makes model calls for minutes must
not stall any of them, so each due run goes to its own thread and the tick moves on.

Plugin code cannot be killed, so when housekeeping stops (gateway shutdown, or an in-process
restart of the multiplexer) it waits a bounded time for its runs, then returns; a run that ends
later writes no state, and no run starts after the stop.
"""

from __future__ import annotations

import threading
import time

import pytest

import gateway.run as gateway_run
from hermes_cli import plugins_idle_tasks
from hermes_cli.plugins import PluginContext, PluginManager, PluginManifest
from hermes_cli.plugins_idle_tasks import load_state, run_plugin_idle_tasks
from tests.waiting import wait_until

BUDGET = 1.0  # the shutdown join budget, shortened from SHUTDOWN_JOIN_SECONDS for the test


class _Ticks:
    """Stop event that lets the housekeeping loop run exactly ``n`` ticks with no sleeping."""

    def __init__(self, n: int):
        self.n, self.left = 0, n

    def is_set(self):
        return self.n >= self.left

    def wait(self, timeout=None):
        self.n += 1
        return True


class _Stop:
    """Stop event that lets the loop run ``n`` ticks with no sleeping, then holds it until ``set()``."""

    def __init__(self, n: int):
        self.n, self.left = 0, n
        self.held, self._set = threading.Event(), threading.Event()

    def is_set(self):
        return self._set.is_set()

    def set(self):
        self._set.set()

    def wait(self, timeout=None):
        self.n += 1
        if self.n >= self.left:
            self.held.set()
            self._set.wait(30)
        return self._set.is_set()


class _Gateway:
    def __init__(self, manager, ctx):
        self.manager, self.ctx, self.heartbeats, self.after_curator, self.fences = manager, ctx, [], [], []
        self.state_path = ctx.state.data_dir / "idle-tasks" / "promote.json"
        self.lock = ctx.state.data_dir / "idle-tasks" / ".promote.lock"


@pytest.fixture
def gw(tmp_path, monkeypatch):
    """A manager with the ``wisdom`` plugin, the gateway's other chores stubbed, fences captured."""
    import agent.curator
    import hermes_cli.plugins

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    manager = PluginManager()
    gateway = _Gateway(manager, PluginContext(PluginManifest(name="wisdom", source="user"), manager))

    class Fence(plugins_idle_tasks.IdleTaskFence):
        def __init__(self, stop=None):
            super().__init__(stop)
            gateway.fences.append(self)

    monkeypatch.setattr(plugins_idle_tasks, "IdleTaskFence", Fence)
    monkeypatch.setattr(plugins_idle_tasks, "SHUTDOWN_JOIN_SECONDS", BUDGET)
    monkeypatch.setattr(hermes_cli.plugins, "get_plugin_manager", lambda: manager)
    monkeypatch.setattr(agent.curator, "maybe_run_curator", lambda **kw: None)
    monkeypatch.setattr(gateway_run, "_write_runtime_status_quiet", lambda **kw: gateway.heartbeats.append(1))
    monkeypatch.setattr(gateway_run, "_housekeeping_state_db_maintenance",
                        lambda *a: gateway.after_curator.append(1))
    return gateway


def _serve(stop) -> threading.Thread:
    loop = threading.Thread(target=gateway_run._start_gateway_housekeeping, args=(stop,),
                            kwargs={"interval": 0}, daemon=True)
    loop.start()
    return loop


def _blocking_first_run(entered, release, calls):
    def promote(run):  # the first run is a promotion pass that makes model calls for minutes
        calls.append(run.run_count)
        if len(calls) == 1:
            entered.set()
            release.wait(30)
            return "old run"
        return "promoted 2 skills"

    return promote


def _register_blocking(gw):
    entered, release, calls = threading.Event(), threading.Event(), []
    gw.ctx.register_idle_task("promote", _blocking_first_run(entered, release, calls), interval_hours=1,
                              first_run="immediate")
    return entered, release, calls


def test_a_slow_idle_task_does_not_delay_the_next_chore_or_the_heartbeat(gw):
    entered, release, _calls = _register_blocking(gw)

    # Tick 60 runs the curator tick, then the state.db tick; tick 61 stamps the heartbeat again.
    loop = _serve(_Ticks(61))
    try:
        loop.join(10)
        assert entered.is_set()
        assert not loop.is_alive(), "the housekeeping loop waited for the idle task"
        assert gw.after_curator == [1] and len(gw.heartbeats) == 61
    finally:
        release.set()
    wait_until(lambda: not gw.lock.exists(), what="the run's claim released")
    # The run ended after housekeeping stopped: it is not recorded, so it runs again later.
    state = load_state(gw.state_path)
    assert (state["run_count"], state["last_status"], state["last_run_at"]) == (0, None, None)


def test_stop_with_a_run_going_returns_within_its_budget_and_nothing_runs_or_records_after(gw):
    entered, release, calls = _register_blocking(gw)
    stop = _Stop(61)
    loop = _serve(stop)
    try:
        assert stop.held.wait(10) and entered.wait(5)
        began = time.monotonic()
        stop.set()
        loop.join(BUDGET + 10)
        assert not loop.is_alive()
        assert time.monotonic() - began < BUDGET + 2.0  # bounded: the plugin's run is still going
        assert gw.lock.exists()  # held by the run still going in this process
        (fence,) = gw.fences
        assert run_plugin_idle_tasks(idle_for_seconds=float("inf"), manager=gw.manager, fence=fence) == {}
    finally:
        release.set()
    wait_until(lambda: not gw.lock.exists(), what="the run's claim released")
    assert calls == [0]
    assert load_state(gw.state_path)["run_count"] == 0


def test_stop_waits_for_a_run_that_ends_within_its_budget(gw, monkeypatch):
    monkeypatch.setattr(plugins_idle_tasks, "SHUTDOWN_JOIN_SECONDS", 10.0)
    entered, release, tail = threading.Event(), threading.Event(), threading.Event()

    def promote(run):
        entered.set()
        release.wait(30)
        tail.wait(0.3)  # the run's last batch, finishing after the stop
        return "promoted"

    gw.ctx.register_idle_task("promote", promote, interval_hours=1, first_run="immediate")
    stop = _Stop(61)
    loop = _serve(stop)
    try:
        assert stop.held.wait(10) and entered.wait(5)
        stop.set()
    finally:
        release.set()
    loop.join(20)
    assert not loop.is_alive()
    assert not gw.lock.exists()  # housekeeping returned only once the run had settled
    assert load_state(gw.state_path)["run_count"] == 0


def test_a_run_that_ends_before_the_stop_is_recorded(gw):
    gw.ctx.register_idle_task("promote", lambda run: "promoted 2 skills", interval_hours=1, first_run="immediate")
    stop = _Stop(61)
    loop = _serve(stop)
    try:
        assert stop.held.wait(10)
        state = wait_until(lambda: (s := load_state(gw.state_path))["run_count"] == 1 and s, what="the run's record")
        assert (state["last_status"], state["last_summary"]) == ("ok", "promoted 2 skills")
    finally:
        stop.set()
        loop.join(BUDGET + 10)
    assert not loop.is_alive() and not gw.lock.exists()


def test_after_an_in_process_restart_only_the_new_housekeeping_records(gw):
    entered, release, calls = _register_blocking(gw)
    old_stop = _Stop(61)
    old = _serve(old_stop)
    assert old_stop.held.wait(10) and entered.wait(5)
    old_stop.set()
    old.join(BUDGET + 10)
    assert not old.is_alive()

    new_stop = _Stop(61)
    new = _serve(new_stop)  # the multiplexer restarted in this process; the old run is still going
    try:
        assert new_stop.held.wait(10)
        assert calls == [0]  # its curator tick did not start a second run beside the old one
        release.set()
        wait_until(lambda: not gw.lock.exists(), what="the old run's claim released")
        assert load_state(gw.state_path)["run_count"] == 0  # the old run's write never landed

        _old_fence, new_fence = gw.fences
        started = run_plugin_idle_tasks(idle_for_seconds=float("inf"), manager=gw.manager, fence=new_fence)
        started["wisdom:promote"].join(10)
        state = load_state(gw.state_path)
        assert (state["run_count"], state["last_summary"]) == (1, "promoted 2 skills")
    finally:
        release.set()
        new_stop.set()
        new.join(BUDGET + 10)
