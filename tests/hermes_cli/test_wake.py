"""One-shot session self-wake: an armed wake deadline re-enters the loop with no user input (#122444).

The orchestrator turn arms a wake via ``schedule_wake``; the classic-TUI idle hook fires it
into ``_pending_input`` exactly once, so the next process-loop dequeue is a chat turn and the
chain never dies waiting for a human.
"""

import json
import queue
import time

import pytest

from hermes_cli import goals
from hermes_cli.cli_loops_mixin import CLILoopsMixin


@pytest.fixture
def hermes_home(tmp_path, monkeypatch):
    from pathlib import Path

    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(home))
    goals._DB_CACHE.clear()
    yield home
    goals._DB_CACHE.clear()


class _Cli(CLILoopsMixin):
    def __init__(self, session_id):
        self._pending_input = queue.Queue()
        self.session_id = session_id


def test_due_wake_deadline_queues_prompt_without_user_input(hermes_home):
    """A due wake fires exactly once into ``_pending_input`` — the loop's next dequeue is a
    chat turn with no user input in between (the broken chain reported in #122444)."""
    from hermes_cli.wake import schedule_wake

    schedule_wake("wake-sid", "check the fleet and re-arm", time.time() - 1)
    cli = _Cli("wake-sid")

    cli._maybe_fire_wake()
    assert not cli._pending_input.empty()
    assert "check the fleet" in cli._pending_input.get()

    # One-shot: the wake is consumed on fire and never re-fires.
    cli._last_wake_check = 0.0
    cli._maybe_fire_wake()
    assert cli._pending_input.empty()


def test_not_yet_due_wake_stays_armed_and_the_tool_arms_it(hermes_home):
    """``schedule_wake`` (the orchestrator-facing tool) persists the deadline for the current
    session; before ``fires_at`` the idle hook stays silent."""
    from tools.schedule_wake_tool import schedule_wake_tool

    out = schedule_wake_tool(
        {"prompt": "resume the migration gate", "delay_secs": 60}, session_id="wake-tool-sid"
    )
    assert json.loads(out)["success"] is True

    from hermes_cli.wake import due_wake_prompt, load_wake

    state = load_wake("wake-tool-sid")
    assert state is not None
    assert state.prompt == "resume the migration gate"
    assert state.fires_at > time.time()
    assert due_wake_prompt("wake-tool-sid") is None  # not due yet → stays armed
