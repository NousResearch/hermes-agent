"""The idle-task stop fence: once a host stops, none of its runs starts or records.

A stopping host (gateway housekeeping) polls its plugins' idle tasks under an ``IdleTaskFence``.
Plugin code cannot be killed, so closing the fence waits a bounded time for the runs and returns;
a run that ends after that leaves the task's state as it was before the run and releases its claim.
"""

from __future__ import annotations

import threading
from datetime import UTC, datetime

import pytest

from hermes_cli import plugins_idle_tasks
from hermes_cli.plugins import PluginContext, PluginManager, PluginManifest
from hermes_cli.plugins_idle_tasks import IdleTaskFence, load_state, run_plugin_idle_tasks
from tests.waiting import wait_until

T0 = datetime(2026, 1, 1, tzinfo=UTC)
IDLE = 10 * 3600.0


@pytest.fixture
def ctx(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    return PluginContext(PluginManifest(name="wisdom", source="user"), PluginManager())


def _poll(ctx, fence):
    return run_plugin_idle_tasks(idle_for_seconds=IDLE, manager=ctx._manager, now=T0, fence=fence)


def _paths(ctx, name="promote"):
    base = ctx.state.data_dir / "idle-tasks"
    return base / f"{name}.json", base / f".{name}.lock"


def test_a_closed_fence_starts_no_run_and_takes_no_claim(ctx):
    calls = []
    ctx.register_idle_task("promote", calls.append, interval_hours=1, first_run="immediate")
    fence = IdleTaskFence()
    assert fence.close(0) is True
    assert _poll(ctx, fence) == {}
    assert calls == [] and not _paths(ctx)[1].exists()


def test_a_set_stop_event_closes_the_fence(ctx):
    calls, stop = [], threading.Event()
    ctx.register_idle_task("promote", calls.append, interval_hours=1, first_run="immediate")
    stop.set()
    assert _poll(ctx, IdleTaskFence(stop)) == {} and calls == []


def test_a_run_outlasting_the_close_records_nothing_and_releases_its_claim(ctx):
    entered, release = threading.Event(), threading.Event()

    def promote(run):
        entered.set()
        release.wait(30)
        return "promoted"

    ctx.register_idle_task("promote", promote, interval_hours=1, first_run="immediate")
    state_path, lock = _paths(ctx)
    fence = IdleTaskFence()
    worker = _poll(ctx, fence)["wisdom:promote"]
    try:
        assert entered.wait(5)
        assert fence.close(0.05) is False  # bounded: the run is still going
    finally:
        release.set()
    worker.join(10)
    assert not worker.is_alive() and not lock.exists()
    assert load_state(state_path)["run_count"] == 0


def test_a_run_ending_within_the_close_budget_is_recorded(ctx):
    ctx.register_idle_task("promote", lambda run: "promoted", interval_hours=1, first_run="immediate")
    fence = IdleTaskFence()
    worker = _poll(ctx, fence)["wisdom:promote"]
    worker.join(10)
    assert fence.close(5) is True
    state = load_state(_paths(ctx)[0])
    assert (state["run_count"], state["last_summary"]) == (1, "promoted")


def test_close_waits_for_a_record_in_progress(ctx, monkeypatch):
    in_save, finish_save, order = threading.Event(), threading.Event(), []
    real_save = plugins_idle_tasks.save_state

    def slow_save(path, state):
        in_save.set()
        finish_save.wait(10)
        real_save(path, state)
        order.append("recorded")

    monkeypatch.setattr(plugins_idle_tasks, "save_state", slow_save)
    ctx.register_idle_task("promote", lambda run: "promoted", interval_hours=1, first_run="immediate")
    fence = IdleTaskFence()
    worker = _poll(ctx, fence)["wisdom:promote"]
    assert in_save.wait(5)
    closer = threading.Thread(target=lambda: (fence.close(5), order.append("closed")), daemon=True)
    closer.start()
    closer.join(0.3)  # gives a close that skipped the record's lock time to finish first
    finish_save.set()
    closer.join(10)
    worker.join(10)
    assert order == ["recorded", "closed"]


def test_a_closed_fence_refuses_to_start_a_worker():
    fence, ran = IdleTaskFence(), []
    fence.close(0)
    worker = threading.Thread(target=lambda: ran.append(1))
    assert fence.start(worker) is False
    assert worker.ident is None and ran == []


def test_close_waits_for_a_run_that_ends_within_its_budget(ctx):
    entered, release, seen = threading.Event(), threading.Event(), {}
    ctx.register_idle_task("promote", lambda run: (entered.set(), release.wait(30)) and "promoted",
                           interval_hours=1, first_run="immediate")
    state_path, lock = _paths(ctx)
    fence = IdleTaskFence()
    worker = _poll(ctx, fence)["wisdom:promote"]
    assert entered.wait(5)

    def close():
        seen["settled"] = fence.close(10)
        seen["claim_left"] = lock.exists()

    closer = threading.Thread(target=close, daemon=True)
    closer.start()
    try:
        wait_until(lambda: fence.closed, what="the fence closed")
    finally:
        release.set()
    closer.join(15)
    worker.join(10)
    assert seen == {"settled": True, "claim_left": False}  # close returned once the run had settled
    assert load_state(state_path)["run_count"] == 0  # it ended after the stop: not recorded


def test_without_a_fence_a_run_is_recorded_as_before(ctx):
    ctx.register_idle_task("promote", lambda run: "promoted", interval_hours=1, first_run="immediate")
    run_plugin_idle_tasks(idle_for_seconds=IDLE, manager=ctx._manager, now=T0)["wisdom:promote"].join(10)
    wait_until(lambda: load_state(_paths(ctx)[0])["run_count"] == 1, what="the run's record")
