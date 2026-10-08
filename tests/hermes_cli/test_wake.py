"""One-shot session self-wake: an armed wake deadline re-enters the loop with no user input (#122444).

The orchestrator turn arms a wake via ``schedule_wake``; the owning driver (classic-CLI idle hook,
TUI/Desktop session-owner poller, messaging-gateway watcher) fires it exactly once, so the next
turn is a chat turn and the chain never dies waiting for a human.
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


def test_due_wake_fires_once_as_a_chat_turn_even_when_the_prompt_looks_like_a_command(hermes_home):
    """A due wake fires exactly once into ``_pending_input`` and the injected text is the rendered
    ``[Wake …]`` message, never the raw prompt: a model-authored ``/exit`` or ``!rm`` can't reach
    the CLI's slash/bang router (the reviewer's blocker on the first cut). Gateway-routed wakes are
    skipped by the CLI driver — they belong to the gateway watcher."""
    from hermes_cli.wake import load_wake, schedule_wake

    schedule_wake("wake-sid", "/exit", time.time() - 1)
    cli = _Cli("wake-sid")

    cli._maybe_fire_wake()
    injected = cli._pending_input.get_nowait()
    assert injected.startswith("[Wake") and "/exit" in injected and not injected.startswith("/")
    assert load_wake("wake-sid").fire_count == 1 and not load_wake("wake-sid").armed

    cli._last_wake_check = 0.0
    cli._maybe_fire_wake()
    assert cli._pending_input.empty()  # one-shot: consumed on fire

    schedule_wake("wake-sid", "routed", time.time() - 1, route={"platform": "telegram", "chat_id": "42"})
    cli._last_wake_check = 0.0
    cli._maybe_fire_wake()
    assert cli._pending_input.empty() and load_wake("wake-sid").armed


def test_tool_arms_with_budget_and_refund_keeps_an_unstarted_fire_armed(hermes_home, monkeypatch):
    """``schedule_wake`` persists the deadline (not due before ``fires_at``); the per-session fire
    budget refuses the (N+1)th arm instead of letting a self-re-arming model loop forever; a driver
    whose dispatch never started a turn rewinds the fire so the wake stays armed."""
    from hermes_cli import wake
    from tools.schedule_wake_tool import schedule_wake_tool

    out = schedule_wake_tool({"prompt": "resume the migration gate", "delay_secs": 60}, session_id="s")
    assert json.loads(out)["success"] is True
    state = wake.load_wake("s")
    assert state.prompt == "resume the migration gate" and state.fires_at > time.time()
    assert wake.due_wake_prompt("s") is None and wake.load_wake("s").armed  # not due yet → stays armed

    monkeypatch.setattr(wake, "max_fires", lambda: 2)
    for _ in range(2):
        wake.schedule_wake("s", "again", time.time() - 1)
        assert wake.due_wake_prompt("s")
    refused = json.loads(schedule_wake_tool({"prompt": "again", "delay_secs": 60}, session_id="s"))
    assert "budget" in refused["error"] and not wake.load_wake("s").armed

    monkeypatch.setattr(wake, "max_fires", lambda: 0)
    wake.schedule_wake("s", "refund me", time.time() - 1)
    assert wake.due_wake_prompt("s") and not wake.load_wake("s").armed
    assert wake.abandon_wake_fire("s") is True
    assert wake.load_wake("s").armed and wake.load_wake("s").fire_count == 2
    assert wake.abandon_wake_fire("s") is False  # nothing to refund once armed again
