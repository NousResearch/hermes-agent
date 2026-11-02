"""A slow plugin idle task never holds up the gateway's housekeeping thread.

The curator tick also polls the plugins' idle tasks (``run_idle_jobs``). That thread stamps the
runtime heartbeat, which ``hermes gateway status`` and ``/api/status`` read to tell a live gateway
from a wedged one, and it runs every other chore (the cron delivery drain, misfire catch-up, the
plugin update check, state.db maintenance). An idle task that makes model calls for minutes must
not stall any of them, so each due run goes to its own thread and the tick moves on.
"""

from __future__ import annotations

import threading

import pytest

import gateway.run as gateway_run
from hermes_cli.plugins import PluginContext, PluginManager, PluginManifest
from hermes_cli.plugins_idle_tasks import load_state
from tests.waiting import wait_until


class _Ticks:
    """Stop event that lets the housekeeping loop run exactly ``n`` ticks with no sleeping."""

    def __init__(self, n: int):
        self.n, self.left = 0, n

    def is_set(self):
        return self.n >= self.left

    def wait(self, timeout=None):
        self.n += 1
        return True


@pytest.fixture
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    return tmp_path


def test_a_slow_idle_task_does_not_delay_the_next_chore_or_the_heartbeat(home, monkeypatch):
    import agent.curator
    import hermes_cli.plugins

    manager = PluginManager()
    ctx = PluginContext(PluginManifest(name="wisdom", source="user"), manager)
    entered, release = threading.Event(), threading.Event()

    def promote(run):  # a promotion pass that makes model calls for minutes
        entered.set()
        release.wait(20)
        return "promoted 2 skills"

    ctx.register_idle_task("promote", promote, interval_hours=1, first_run="immediate")
    state_path = ctx.state.data_dir / "idle-tasks" / "promote.json"

    heartbeats, after_curator = [], []
    monkeypatch.setattr(hermes_cli.plugins, "get_plugin_manager", lambda: manager)
    monkeypatch.setattr(agent.curator, "maybe_run_curator", lambda **kw: None)
    monkeypatch.setattr(gateway_run, "_write_runtime_status_quiet", lambda **kw: heartbeats.append(1))
    monkeypatch.setattr(gateway_run, "_housekeeping_state_db_maintenance", lambda *a: after_curator.append(1))

    # Tick 60 runs the curator tick, then the state.db tick; tick 61 stamps the heartbeat again.
    loop = threading.Thread(target=gateway_run._start_gateway_housekeeping, args=(_Ticks(61),),
                            kwargs={"interval": 0}, daemon=True)
    loop.start()
    try:
        loop.join(10)
        assert entered.is_set()
        assert not loop.is_alive(), "the housekeeping loop waited for the idle task"
        assert after_curator == [1] and len(heartbeats) == 61
        assert load_state(state_path)["run_count"] == 0  # the task is still running
    finally:
        release.set()
        loop.join(30)

    state = wait_until(lambda: (s := load_state(state_path))["run_count"] == 1 and s, what="the run's record")
    assert (state["last_status"], state["last_summary"]) == ("ok", "promoted 2 skills")
