"""Regression tests for `/goal draft` word-boundary matching in the classic CLI.

`/goal` is dispatched separately in the classic CLI
(``CLICommandsMixin._handle_goal_command``) and in the gateway
(``GatewayRunner._handle_goal_command``); the gateway side is covered by
``tests/gateway/test_goal_draft_prefix.py``. Both carry the same guard, so
both need the same regression.

A bare ``lower.startswith("draft")`` misroutes any goal whose first word
merely *begins* with "draft" (``drafting``, ``drafts``, ``draftsman``) into
the aux-LLM contract-draft path AND slices the first 5 characters off the
objective via ``arg[len("draft"):]`` — so ``/goal drafting the roadmap``
silently became the goal ``ing the roadmap``.
"""

import queue

import pytest

from hermes_cli import goals
from hermes_cli.cli_commands_mixin import CLICommandsMixin


class _Stub(CLICommandsMixin):
    """Minimal host for the goal command: a real GoalManager + input queue.

    ``_get_goal_manager`` lives on the concrete CLI class, not the mixin, so
    the stub supplies it.
    """

    def __init__(self, session_key: str):
        self._mgr = goals.GoalManager(session_key)
        self._pending_input = queue.Queue()

    def _get_goal_manager(self):
        return self._mgr


@pytest.fixture()
def hermes_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    goals._DB_CACHE.clear()
    yield home
    goals._DB_CACHE.clear()


@pytest.fixture()
def draft_spy(monkeypatch):
    """Record calls to the aux-LLM contract drafter without invoking it."""
    calls = []

    def _spy(objective, *args, **kwargs):
        calls.append(objective)
        return None

    monkeypatch.setattr(goals, "draft_contract", _spy)
    return calls


def test_cli_goal_starting_with_draft_word_stays_free_form(hermes_home, draft_spy):
    """`/goal drafting …` is a free-form goal, not a draft-contract request.

    It must not invoke the draft helper and must be stored verbatim — no
    ``draft`` prefix sliced off the front.
    """
    sid = "sid-cli-goal-draft-prefix-freeform"
    stub = _Stub(sid)

    stub._handle_goal_command("/goal drafting the roadmap")

    assert draft_spy == [], "free-form 'drafting…' goal must not hit draft_contract"

    state = goals.GoalManager(sid).state
    assert state is not None
    assert state.goal == "drafting the roadmap"
    assert stub._pending_input.get_nowait() == "drafting the roadmap"


def test_cli_goal_draft_still_routes_to_contract_helper(hermes_home, draft_spy):
    """Positive control: a genuine `/goal draft <objective>` still routes to
    the draft helper with the ``draft`` keyword stripped."""
    sid = "sid-cli-goal-draft-prefix-contract"
    stub = _Stub(sid)

    stub._handle_goal_command("/goal draft the quarterly report")

    assert draft_spy == ["the quarterly report"]

    state = goals.GoalManager(sid).state
    assert state is not None
    assert state.goal == "the quarterly report"


def test_cli_goal_bare_draft_returns_usage(hermes_home, draft_spy):
    """Bare `/goal draft` is still the usage path — no goal set, no helper."""
    sid = "sid-cli-goal-draft-prefix-empty"
    stub = _Stub(sid)

    stub._handle_goal_command("/goal draft")

    assert draft_spy == []
    assert goals.GoalManager(sid).state is None
