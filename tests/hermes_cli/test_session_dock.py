"""Active /goal and queued /queue prompts share the classic CLI live-work dock."""
import queue
from types import SimpleNamespace

import pytest
from prompt_toolkit.utils import get_cwidth


@pytest.fixture
def hermes_home(tmp_path, monkeypatch):
    from pathlib import Path

    from hermes_cli import goals

    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(home))
    goals._DB_CACHE.clear()
    yield home
    goals._DB_CACHE.clear()


def test_dock_paints_goal_on_top_and_queue_last_and_follows_their_lifecycle(hermes_home):
    from cli import _VoiceInputMessage
    from hermes_cli.cli_subagent_monitor import SubagentMonitor
    from hermes_cli.goals import GoalManager
    from tools.process_registry_notifications import TimelineNotification

    mgr = GoalManager(session_id="sid-a", default_max_turns=20)
    mgr.set("ship the dock feature")
    pending = queue.Queue()
    for item in ("first follow-up\nwith a newline", ("look at this", ["shot.png"]),
                 _VoiceInputMessage("spoken ask"),
                 TimelineNotification("raw wall", "Process abc exited 0", "notice")):
        pending.put(item)
    cli = SimpleNamespace(agent=None, session_id="sid-a", _goal_manager=mgr, _pending_input=pending)
    dock = SubagentMonitor(cli)

    assert dock.refresh() and dock.has_rows
    lines = dock.dock_text(columns=80, rows=40).splitlines()
    assert "⊙ Goal (active, 0/20 turns): ship the dock feature" in lines[0]
    queue_at = next(i for i, line in enumerate(lines) if "Queue · 4 queued" in line)
    assert [line.strip() for line in lines[queue_at + 1:]] == [
        "1. first follow-up with a newline", "2. look at this", "3. spoken ask", "+1 more"]
    assert all(get_cwidth(line) <= 80 for line in lines)
    dock.collapsed = True
    collapsed = dock.dock_text(columns=80, rows=40)
    assert "\n" not in collapsed and "goal active · 4 queued · F7 restore · ⊙ Goal" in collapsed
    assert "Ctrl+T" not in collapsed  # the monitor only lists subagents/processes
    assert get_cwidth(collapsed) <= 80

    # A paused goal stays visible; draining the queue and clearing the goal empties the dock.
    dock.collapsed = False
    mgr.pause("user-paused")
    pending.queue.clear()
    assert dock.refresh() and dock.dock_text(columns=80, rows=40).startswith(" ⏸ Goal (paused")
    mgr.clear()
    assert dock.refresh() and not dock.has_rows and dock.dock_text(columns=80, rows=40) == ""


def test_dock_ignores_a_goal_manager_left_over_from_another_session(hermes_home):
    """After /new the cached manager still names the old session until it is rebuilt; the dock
    must not paint the previous session's goal (and never opens state.db itself)."""
    from hermes_cli.cli_session_dock import goal_line
    from hermes_cli.goals import GoalManager

    mgr = GoalManager(session_id="old-sid")
    mgr.set("previous session objective")
    assert goal_line(SimpleNamespace(session_id="old-sid", _goal_manager=mgr)).startswith("⊙ Goal")
    assert goal_line(SimpleNamespace(session_id="new-sid", _goal_manager=mgr)) == ""
    assert goal_line(SimpleNamespace(session_id="new-sid")) == ""
