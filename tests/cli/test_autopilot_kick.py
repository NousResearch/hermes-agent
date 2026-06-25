"""Tests for the classic-CLI autopilot enable-kick.

Autopilot is reactive (it engages at the end of a running turn via the engine's
continuation seam). When the user flips /autopilot ON while the conversation is
idle, ``_maybe_kick_autopilot`` injects a single resume turn onto
``_pending_input`` so the engine actually starts driving. These tests exercise
that method in isolation against a stub instance.
"""

import queue

import cli as cli_mod


def _stub(**overrides):
    s = object.__new__(cli_mod.HermesCLI)
    s.agent = object()
    s._agent_running = False
    s._pending_input = queue.Queue()
    s.conversation_history = [
        {"role": "user", "content": "do X"},
        {"role": "assistant", "content": "partial work"},
    ]
    s._autopilot_goal = ""
    for k, v in overrides.items():
        setattr(s, k, v)
    return s


def test_kick_enqueues_resume_when_idle_with_history():
    s = _stub()
    s._maybe_kick_autopilot()
    item = s._pending_input.get_nowait()
    assert item.startswith("[Autopilot]")
    assert "Resume" in item


def test_kick_includes_goal_when_set():
    s = _stub(_autopilot_goal="ship the feature")
    s._maybe_kick_autopilot()
    item = s._pending_input.get_nowait()
    assert "ship the feature" in item


def test_kick_noop_when_agent_running():
    s = _stub(_agent_running=True)
    s._maybe_kick_autopilot()
    assert s._pending_input.empty()


def test_kick_noop_when_no_agent():
    s = _stub(agent=None)
    s._maybe_kick_autopilot()
    assert s._pending_input.empty()


def test_kick_noop_when_input_already_queued():
    s = _stub()
    s._pending_input.put("real user message")
    s._maybe_kick_autopilot()
    # only the pre-existing message remains; no kick added
    assert s._pending_input.get_nowait() == "real user message"
    assert s._pending_input.empty()


def test_kick_noop_when_no_history():
    s = _stub(conversation_history=[])
    s._maybe_kick_autopilot()
    assert s._pending_input.empty()


def test_kick_starts_on_goal_when_no_history_but_goal_set():
    # Regression: `/autopilot goal <text>` on a COLD session (no history) must
    # START by enqueuing the goal as the opening task — previously it bailed
    # ("send a task") so autopilot was ON but nothing ran.
    s = _stub(conversation_history=[], _autopilot_goal="ship the feature")
    s._maybe_kick_autopilot()
    item = s._pending_input.get_nowait()
    assert item == "ship the feature"
    assert s._pending_input.empty()


# ── /autopilot argument parsing (explicit-subcommand; footgun fix) ──────────
def _toggle_stub(monkeypatch, **overrides):
    """A HermesCLI stub wired for _toggle_autopilot in isolation."""
    s = object.__new__(cli_mod.HermesCLI)
    s.agent = type("A", (), {"autopilot_mode": False, "_autopilot_goal": ""})()
    s._agent_running = False
    s._pending_input = queue.Queue()
    s.conversation_history = []
    s._autopilot_on = False
    s._autopilot_goal = ""
    # neutralize side-effecting helpers/printers
    s._maybe_kick_autopilot = lambda: None
    for k, v in overrides.items():
        setattr(s, k, v)
    monkeypatch.setattr(cli_mod, "_cprint", lambda *a, **k: None)
    return s


def test_toggle_on(monkeypatch):
    s = _toggle_stub(monkeypatch)
    s._toggle_autopilot("/autopilot on")
    assert s._autopilot_on is True
    assert s.agent.autopilot_mode is True


def test_toggle_off_with_trailing_word_still_disables(monkeypatch):
    # Regression: the old parser treated "off now" as a goal and ENABLED.
    s = _toggle_stub(monkeypatch, _autopilot_on=True)
    s.agent.autopilot_mode = True
    s._toggle_autopilot("/autopilot off now")
    assert s._autopilot_on is False
    assert s.agent.autopilot_mode is False


def test_bare_positional_is_not_a_goal(monkeypatch):
    # Unknown argument must NOT enable or set a goal — just show usage.
    s = _toggle_stub(monkeypatch)
    s._toggle_autopilot("/autopilot ship the feature")
    assert s._autopilot_on is False
    assert s._autopilot_goal == ""


def test_goal_subcommand_sets_goal_and_enables(monkeypatch):
    s = _toggle_stub(monkeypatch)
    s._toggle_autopilot("/autopilot goal ship the feature")
    assert s._autopilot_goal == "ship the feature"
    assert s.agent._autopilot_goal == "ship the feature"
    assert s._autopilot_on is True


def test_clear_resets_goal(monkeypatch):
    s = _toggle_stub(monkeypatch, _autopilot_goal="old goal")
    s.agent._autopilot_goal = "old goal"
    s._toggle_autopilot("/autopilot clear")
    assert s._autopilot_goal == ""
    assert s.agent._autopilot_goal == ""


def test_goal_clear_resets_goal(monkeypatch):
    s = _toggle_stub(monkeypatch, _autopilot_goal="old goal")
    s.agent._autopilot_goal = "old goal"
    s._toggle_autopilot("/autopilot goal clear")
    assert s._autopilot_goal == ""
