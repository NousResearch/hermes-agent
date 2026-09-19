"""A parked /goal resumes from the idle hook once its barrier lifts, without waiting for another turn."""
import queue
import time
from unittest.mock import patch

import pytest

from hermes_cli import goals
from hermes_cli.cli_loops_mixin import CLILoopsMixin


@pytest.fixture
def hermes_home(tmp_path, monkeypatch):
    from pathlib import Path
    home = tmp_path / ".hermes"; home.mkdir()
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(home))
    goals._DB_CACHE.clear()
    yield home
    goals._DB_CACHE.clear()


class _Cli(CLILoopsMixin):
    def __init__(self, mgr):
        self._pending_input = queue.Queue()
        self._mgr = mgr

    def _get_goal_manager(self):
        return self._mgr


def test_idle_hook_queues_the_continuation_when_a_timed_barrier_has_elapsed(hermes_home):
    mgr = goals.GoalManager(session_id="resume-idle")
    mgr.set("finish the thing")
    mgr.wait_for_seconds(1, reason="cooldown")
    cli = _Cli(mgr)
    with patch("cli._cprint"), patch("cli._DIM", ""), patch("cli._RST", ""):
        cli._maybe_resume_parked_goal()
        assert cli._pending_input.empty()            # still parked
        mgr.state.waiting_until = time.time() - 1
        mgr._save()
        cli._last_goal_barrier_check = 0.0
        cli._maybe_resume_parked_goal()
    assert not cli._pending_input.empty()            # continuation queued
    assert "finish the thing" in cli._pending_input.get()
    assert mgr.state.waiting_until == 0.0            # barrier cleared


def test_idle_hook_is_a_no_op_for_an_unparked_or_inactive_goal(hermes_home):
    mgr = goals.GoalManager(session_id="resume-noop")
    mgr.set("g")
    cli = _Cli(mgr)
    cli._maybe_resume_parked_goal()
    assert cli._pending_input.empty()
    mgr.clear()
    cli._last_goal_barrier_check = 0.0
    cli._maybe_resume_parked_goal()
    assert cli._pending_input.empty()
