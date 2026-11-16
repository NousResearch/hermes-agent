"""Plugin idle tasks run on their own thread: the host's poll (gateway housekeeping, CLI startup,
``hermes serve``'s ticker) starts each due run and returns at once, the claim still keeps one run
at a time, and the worker records the run and releases the claim whatever ``fn`` does."""

from __future__ import annotations

import os
import threading
import time
from datetime import UTC, datetime, timedelta

import pytest

from agent.idle_schedule import CLAIM_STALE_SECONDS
from hermes_cli.plugins import PluginContext, PluginManager, PluginManifest
from hermes_cli.plugins_idle_tasks import load_state, run_plugin_idle_tasks

T0 = datetime(2026, 10, 1, 12, 0, tzinfo=UTC)
IDLE = 3 * 3600.0


@pytest.fixture
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    return tmp_path


def _ctx(manager=None):
    manager = manager or PluginManager()
    return manager, PluginContext(PluginManifest(name="wisdom", source="user"), manager)


def _poll(manager, now):
    return run_plugin_idle_tasks(idle_for_seconds=IDLE, manager=manager, now=now)


def _blocking(entered, release, calls):
    def fn(run):
        calls.append(threading.current_thread())
        entered.set()
        release.wait(5)
        return "promoted 2 skills"
    return fn


def test_a_slow_task_runs_on_its_own_thread_and_the_poll_returns_at_once(home):
    manager, ctx = _ctx()
    entered, release, calls = threading.Event(), threading.Event(), []
    ctx.register_idle_task("promote", _blocking(entered, release, calls), interval_hours=1, first_run="immediate")
    state_path = ctx.state.data_dir / "idle-tasks" / "promote.json"

    began = time.monotonic()
    try:
        started = _poll(manager, T0)
        returned_after = time.monotonic() - began
        assert entered.wait(5)
        assert returned_after < 2.0  # the poll did not wait for the run
        assert calls[0] is not threading.current_thread()
        assert load_state(state_path)["run_count"] == 0  # still running
    finally:
        release.set()

    worker = started["wisdom:promote"]
    worker.join(10)
    assert not worker.is_alive()
    state = load_state(state_path)
    assert (state["run_count"], state["last_status"], state["last_summary"]) == (1, "ok", "promoted 2 skills")
    assert not (ctx.state.data_dir / "idle-tasks" / ".promote.lock").exists()


def test_no_second_run_starts_while_one_is_in_flight(home):
    manager, ctx = _ctx()
    entered, release, calls = threading.Event(), threading.Event(), []
    ctx.register_idle_task("promote", _blocking(entered, release, calls), interval_hours=1, first_run="immediate")
    lock = ctx.state.data_dir / "idle-tasks" / ".promote.lock"

    try:
        first = _poll(manager, T0)
        assert entered.wait(5)
        assert _poll(manager, T0 + timedelta(hours=5)) == {}
        # The run outlasts the claim's stale age: this process still knows it is running.
        old = time.time() - CLAIM_STALE_SECONDS - 60
        os.utime(lock, (old, old))
        assert _poll(manager, T0 + timedelta(hours=6)) == {}
        assert len(calls) == 1
    finally:
        release.set()
    first["wisdom:promote"].join(10)

    entered.clear()
    again = _poll(manager, T0 + timedelta(hours=7))
    again["wisdom:promote"].join(10)
    assert len(calls) == 2


def test_the_claim_is_released_and_the_run_recorded_after_an_exception(home):
    manager, ctx = _ctx()

    def boom(run):
        raise RuntimeError("model unreachable")

    ctx.register_idle_task("boom", boom, interval_hours=1, first_run="immediate")
    _poll(manager, T0)["wisdom:boom"].join(10)

    state = load_state(ctx.state.data_dir / "idle-tasks" / "boom.json")
    assert (state["run_count"], state["last_status"]) == (1, "error") and "model unreachable" in state["last_error"]
    assert not (ctx.state.data_dir / "idle-tasks" / ".boom.lock").exists()
    _poll(manager, T0 + timedelta(hours=1))["wisdom:boom"].join(10)  # the next interval runs again
    assert load_state(ctx.state.data_dir / "idle-tasks" / "boom.json")["run_count"] == 2


def test_the_worker_keeps_the_profile_scope_of_the_poll(home, tmp_path):
    from hermes_constants import get_hermes_home, reset_hermes_home_override, set_hermes_home_override

    coder = tmp_path / "profiles" / "coder"
    coder.mkdir(parents=True)
    release, seen = threading.Event(), []

    def fn(run):
        release.wait(5)  # read the home only after the poller has left the profile's scope
        seen.append(get_hermes_home())

    token = set_hermes_home_override(coder)
    try:
        manager, ctx = _ctx(PluginManager(scope_key=str(coder)))
        ctx.register_idle_task("suggest", fn, interval_hours=1, first_run="immediate")
        state_path = ctx.state.data_dir / "idle-tasks" / "suggest.json"
        started = _poll(manager, T0)
    finally:
        reset_hermes_home_override(token)
        release.set()
    started["wisdom:suggest"].join(10)

    assert seen == [coder]
    assert state_path.is_relative_to(coder) and load_state(state_path)["run_count"] == 1
