"""Contract tests for the embedder goal-turn protocol (``hermes_cli.goal_session``).

These pin the properties an EMBEDDER depends on. Each one corresponds to a
failure class measured in a real embedded deployment, where the same logic was
reimplemented outside the process and drifted.
"""

import sys
import types

import pytest


def _install_fake_goals(monkeypatch, manager_cls, contract_cls=None):
    module = types.SimpleNamespace(
        GoalManager=manager_cls,
        GoalContract=contract_cls or (lambda **kw: dict(kw)),
        gather_background_processes=lambda: [],
    )
    monkeypatch.setitem(sys.modules, "hermes_cli.goals", module)
    return module


class _State:
    def __init__(self, status="active", turns_used=1, max_turns=10, paused_reason=""):
        self.status = status
        self.turns_used = turns_used
        self.max_turns = max_turns
        self.paused_reason = paused_reason


def test_unknown_action_is_an_envelope_not_an_exception():
    from hermes_cli.goal_session import run_goal_turn

    result = run_goal_turn("s1", "teleport")
    assert result["error"]["code"] == "invalid_action"
    assert result["state"] is None


def test_missing_session_id_is_rejected():
    from hermes_cli.goal_session import run_goal_turn

    assert run_goal_turn("  ", "status")["error"]["code"] == "invalid_session"


def test_evaluate_restarts_a_vanished_goal_when_the_objective_is_supplied(monkeypatch):
    """The single commonest embedded failure: `goal_not_found` on evaluate.

    An embedder that knows the objective should never have to recognize the
    error, issue a `start`, and retry — that round-trip is the bug.
    """
    started = {}

    class Manager:
        def __init__(self, session_id, *, default_max_turns):
            self.state = None

        def set(self, goal, *, max_turns=None, contract=None):
            started["goal"] = goal
            self.state = _State()
            return self.state

        def add_gate(self, command):
            started.setdefault("gates", []).append(command)

        def evaluate_after_turn(self, response, background_processes=None):
            return {"verdict": "continue", "reason": "keep going",
                    "should_continue": True, "continuation_prompt": "next"}

    _install_fake_goals(monkeypatch, Manager)
    from hermes_cli.goal_session import run_goal_turn

    result = run_goal_turn("s1", "evaluate", response="did a thing", goal="Ship it")

    assert started["goal"] == "Ship it"
    assert result["restarted"] is True
    assert result["decision"]["verdict"] == "continue"


def test_evaluate_without_an_objective_still_reports_goal_not_found(monkeypatch):
    """Self-healing must not invent a goal out of nothing."""

    class Manager:
        def __init__(self, session_id, *, default_max_turns):
            self.state = None

    _install_fake_goals(monkeypatch, Manager)
    from hermes_cli.goal_session import run_goal_turn

    assert run_goal_turn("s1", "evaluate", response="x")["error"]["code"] == "goal_not_found"


@pytest.mark.parametrize("where", ["state", "decision"])
def test_a_judge_outage_is_tagged_rather_than_read_as_a_failed_verdict(monkeypatch, where):
    """An outage and a real `failed` both arrive as `failed`; only one may self-heal.

    The reason can surface on the state OR on the decision. Checking one
    spelling silently returns the loop, so both are pinned.
    """

    class Manager:
        def __init__(self, session_id, *, default_max_turns):
            self.state = _State(paused_reason="judge error: BadRequestError" if where == "state" else "")

        def evaluate_after_turn(self, response, background_processes=None):
            reason = "judge error: BadRequestError" if where == "decision" else "the work is wrong"
            return {"verdict": "failed", "reason": reason,
                    "should_continue": False, "continuation_prompt": None}

    _install_fake_goals(monkeypatch, Manager)
    from hermes_cli.goal_session import run_goal_turn

    decision = run_goal_turn("s1", "evaluate", response="x")["decision"]
    assert decision["verdict"] == "failed"
    assert decision["failure_kind"] == "judge_unavailable"
    assert "BadRequestError" in decision["reason"]


def test_a_genuine_failed_verdict_is_not_tagged_as_an_outage(monkeypatch):
    """A real judgment about the work must reach the operator immediately."""

    class Manager:
        def __init__(self, session_id, *, default_max_turns):
            self.state = _State()

        def evaluate_after_turn(self, response, background_processes=None):
            return {"verdict": "failed", "reason": "the tests do not pass",
                    "should_continue": False, "continuation_prompt": None}

    _install_fake_goals(monkeypatch, Manager)
    from hermes_cli.goal_session import run_goal_turn

    assert run_goal_turn("s1", "evaluate", response="x")["decision"]["failure_kind"] == ""


def test_an_unrecognized_verdict_is_normalized_to_failed(monkeypatch):
    """A verdict nobody defined must never drive another agent turn."""

    class Manager:
        def __init__(self, session_id, *, default_max_turns):
            self.state = _State()

        def evaluate_after_turn(self, response, background_processes=None):
            return {"verdict": "vibes", "reason": "?", "should_continue": True,
                    "continuation_prompt": "go"}

    _install_fake_goals(monkeypatch, Manager)
    from hermes_cli.goal_session import run_goal_turn

    decision = run_goal_turn("s1", "evaluate", response="x")["decision"]
    assert decision["verdict"] == "failed"
    assert decision["should_continue"] is False
    assert decision["continuation_prompt"] is None
    assert "vibes" in decision["reason"]


def test_an_inactive_goal_resolves_rather_than_failing(monkeypatch):
    """`inactive` is an ordinary condition, not a malformed verdict.

    ``evaluate_after_turn`` returns it when the goal is no longer active — done,
    cleared, or paused between turns. Normalizing it to `failed` cost the
    operator a Resume for a mission that had simply finished: production showed
    three such resumes within minutes of the first deployment.
    """

    class Manager:
        def __init__(self, session_id, *, default_max_turns):
            self.state = _State(status="done")

        def evaluate_after_turn(self, response, background_processes=None):
            return {"verdict": "inactive", "reason": "no active goal",
                    "should_continue": False, "continuation_prompt": None}

    _install_fake_goals(monkeypatch, Manager)
    from hermes_cli.goal_session import run_goal_turn

    result = run_goal_turn("s1", "evaluate", response="x")
    assert result["decision"]["verdict"] == "done"
    assert result["decision"]["failure_kind"] == ""
    assert result["decision"]["should_continue"] is False


def test_an_inactive_goal_that_was_paused_reports_a_pause(monkeypatch):
    class Manager:
        def __init__(self, session_id, *, default_max_turns):
            self.state = _State(status="paused")

        def evaluate_after_turn(self, response, background_processes=None):
            return {"verdict": "inactive", "reason": "no active goal",
                    "should_continue": False, "continuation_prompt": None}

    _install_fake_goals(monkeypatch, Manager)
    from hermes_cli.goal_session import run_goal_turn

    result = run_goal_turn("s1", "evaluate", response="x")
    assert result["decision"]["verdict"] == "hard_pause"
    assert result["state"]["status"] == "hard_paused"


def test_a_publication_blocker_returns_a_continuation_without_re_judging(monkeypatch):
    """The verdict already exists; re-grading it would spend an aux call for nothing."""
    judged = []

    class Manager:
        def __init__(self, session_id, *, default_max_turns):
            self.state = _State()

        def evaluate_after_turn(self, response, background_processes=None):
            judged.append(response)
            return {"verdict": "continue", "reason": "x", "should_continue": True,
                    "continuation_prompt": "y"}

    _install_fake_goals(monkeypatch, Manager)
    from hermes_cli.goal_session import run_goal_turn

    result = run_goal_turn(
        "s1", "evaluate", response="done", blocked_reason="branch-not-pushed",
        blocked_remedy="Push it.", blocked_paths=["a.py"], blocked_attempts_left=2,
    )

    assert judged == [], "the judge must not run again for a publication blocker"
    decision = result["decision"]
    assert decision["verdict"] == "continue"
    assert "branch-not-pushed" in decision["continuation_prompt"]
    assert "Push it." in decision["continuation_prompt"]
    assert "a.py" in decision["continuation_prompt"]
    assert "2 more time" in decision["continuation_prompt"]


def test_a_publication_blocker_hard_pauses_once_the_budget_is_spent(monkeypatch):
    """If naming the blocker did not clear it, a human must see it."""

    class Manager:
        def __init__(self, session_id, *, default_max_turns):
            self.state = _State()

        def evaluate_after_turn(self, response, background_processes=None):
            raise AssertionError("the judge must not be called")

    _install_fake_goals(monkeypatch, Manager)
    from hermes_cli.goal_session import run_goal_turn

    result = run_goal_turn("s1", "evaluate", response="done",
                           blocked_reason="worktree-dirty", blocked_attempts_left=0)

    assert result["decision"]["verdict"] == "hard_pause"
    assert result["decision"]["should_continue"] is False
    assert result["state"]["status"] == "hard_paused"


def test_the_publication_continuation_forbids_new_work():
    """The failure mode is an agent that reads 'blocked' as 'keep going'."""
    from hermes_cli.goal_session import publication_continuation

    text = publication_continuation("branch-not-pushed", "Push the branch.", 1)
    assert "already verified this work as DONE" in text
    assert "Do NOT start new work" in text
    # A bulleted path list must not run into the sentence's period.
    listed = publication_continuation("worktree-dirty", "Commit it.", 1, ["x.py", "  "])
    assert listed.count("- x.py") == 1
    assert "  - x.py." not in listed


def test_a_paused_state_becomes_a_hard_pause(monkeypatch):
    class Manager:
        def __init__(self, session_id, *, default_max_turns):
            self.state = _State(status="paused")

        def evaluate_after_turn(self, response, background_processes=None):
            return {"verdict": "continue", "status": "paused", "reason": "budget",
                    "should_continue": True, "continuation_prompt": "go"}

    _install_fake_goals(monkeypatch, Manager)
    from hermes_cli.goal_session import run_goal_turn

    result = run_goal_turn("s1", "evaluate", response="x")
    assert result["decision"]["verdict"] == "hard_pause"
    assert result["decision"]["should_continue"] is False
    assert result["state"]["status"] == "hard_paused"


def test_a_broken_background_process_list_never_fails_the_turn(monkeypatch):
    """The live process list is an optimization, not a precondition."""

    class Manager:
        def __init__(self, session_id, *, default_max_turns):
            self.state = _State()

        def evaluate_after_turn(self, response, background_processes=None):
            assert background_processes is None
            return {"verdict": "continue", "reason": "ok", "should_continue": True,
                    "continuation_prompt": "next"}

    def explode():
        raise RuntimeError("process registry is down")

    module = _install_fake_goals(monkeypatch, Manager)
    module.gather_background_processes = explode

    from hermes_cli.goal_session import run_goal_turn

    assert run_goal_turn("s1", "evaluate", response="x")["decision"]["verdict"] == "continue"


def test_start_requires_a_goal(monkeypatch):
    class Manager:
        def __init__(self, session_id, *, default_max_turns):
            self.state = None

    _install_fake_goals(monkeypatch, Manager)
    from hermes_cli.goal_session import run_goal_turn

    assert run_goal_turn("s1", "start", goal="   ")["error"]["code"] == "invalid_goal"
