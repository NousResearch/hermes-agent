"""`/goal <text>` kicks the loop with a framed goal prompt, never with a bare slash-shaped message.

Two behaviours are pinned here:

* A goal the user just pasted kicks with a short pointer, not a 2,000-char replay (the original
  reason this module exists: one run spent 11 API calls / 6 min deciding a re-paste was a replay).
* Every other goal kicks with the framed prompt. The kickoff used to be the raw goal text, so a
  goal starting with "/" (`/goal /work fix the leak`) was handed to the surface as a message that
  begins with a slash — every surface parses that as a slash command, the kickoff turn never
  starts, and the goal sits active at 0 turns. Framing makes that impossible.
"""
import queue

import pytest

from hermes_cli.cli_commands_mixin import CLICommandsMixin
from hermes_cli.goals import GOAL_ALREADY_SEEN_KICK, GOAL_KICK_TEMPLATE


@pytest.fixture(autouse=True)
def _stub_cli_print_seams(monkeypatch):
    """``/goal`` prints through the mixin's ``_cp``/``_dim_line``. ``_cp`` does ``from cli import
    _cprint``, and importing ``cli`` activates the payload environment against the REAL Hermes home
    (the home-I/O guard refuses that). These tests pin the queued prompt, not the printing."""
    import hermes_cli.cli_commands_mixin as mixin

    monkeypatch.setattr(mixin, "_cp", lambda *lines: None, raising=False)
    monkeypatch.setattr(mixin, "_dim_line", lambda text: text, raising=False)


def _cli(history):
    cli = CLICommandsMixin.__new__(CLICommandsMixin)
    cli.conversation_history = history
    cli._pending_input = queue.Queue()
    cli._get_goal_manager = lambda: _StubGoalManager()
    return cli


class _StubState:
    def __init__(self):
        self.goal = ""
        self.max_turns = 20

    def has_contract(self):
        return False


class _StubGoalManager:
    """These tests pin the prompt the CLI queues, not persistence — a real GoalManager here
    opened state.db under the live Hermes home and tripped the home-I/O guard."""

    def __init__(self):
        self.state = _StubState()

    def has_goal(self):
        return bool(self.state.goal)

    def set(self, goal, contract=None):
        self.state.goal = goal
        return self.state


def _kick(cli, text):
    cli._handle_goal_command("/goal " + text)
    return cli._pending_input.get_nowait()


HANDOFF = "HANDOFF: resume round 3 integration.\n" + "\n".join(f"  - step {i}: merge r3-{i} and run its targeted suite, then the full suite" for i in range(8))


def test_goal_that_the_user_just_pasted_kicks_with_a_pointer_not_the_text():
    cli = _cli([{"role": "user", "content": HANDOFF + "\n\nGo."},
                {"role": "assistant", "content": "ok"}])
    assert _kick(cli, HANDOFF) == GOAL_ALREADY_SEEN_KICK
    # block-style content is handled too
    cli = _cli([{"role": "user", "content": [{"type": "text", "text": HANDOFF}]}])
    assert _kick(cli, HANDOFF) == GOAL_ALREADY_SEEN_KICK


def test_a_new_goal_or_a_goal_from_an_older_turn_is_kicked_framed():
    assert _kick(_cli([]), "Ship the release") == GOAL_KICK_TEMPLATE.format(goal="Ship the release")
    cli = _cli([{"role": "user", "content": "Something unrelated"}])
    assert _kick(cli, HANDOFF) == GOAL_KICK_TEMPLATE.format(goal=" ".join(HANDOFF.split()))
    # only the LAST user message counts: the agent has moved on since an older paste
    cli = _cli([{"role": "user", "content": HANDOFF}, {"role": "assistant", "content": "done"},
                {"role": "user", "content": "now something else"}])
    assert _kick(cli, HANDOFF) == GOAL_KICK_TEMPLATE.format(goal=" ".join(HANDOFF.split()))


def test_a_short_goal_that_selects_one_option_from_the_last_message_keeps_the_selection():
    """Independent-review witness: after a message offering API or UI work, `/goal ship the API` and
    `/goal ship the UI` produced identical kickoffs. A goal that is a fragment of the last message
    carries the selection; only a near-whole re-paste is replaced by the pointer."""
    offer = "I can either ship the API or ship the UI next; which do you want? " * 8
    cli = _cli([{"role": "user", "content": offer}])
    api, ui = _kick(cli, "ship the API"), _kick(cli, "ship the UI")
    assert "ship the API" in api and "ship the UI" not in api
    assert "ship the UI" in ui and "ship the API" not in ui
    # a long goal that is only a minority of a much longer message is also kept
    long_goal = "x" * 500
    cli = _cli([{"role": "user", "content": long_goal + " " + "y" * 2000}])
    assert long_goal in _kick(cli, long_goal)


def test_a_goal_that_starts_with_a_slash_never_produces_a_slash_shaped_prompt():
    """The regression: the raw goal text was the kickoff, so `/goal /token-signal …` handed the
    surface a message beginning with "/" — it was parsed as a command and no turn ever ran."""
    for goal in ("/token-signal Perform an adversarial audit", "/work fix the leak"):
        prompt = _kick(_cli([]), goal)
        assert not prompt.startswith("/")
        assert goal in prompt
        assert prompt.startswith("[Standing goal]")


def test_gateway_and_tui_surfaces_use_the_same_rule(tmp_path, monkeypatch):
    """Independent review: other surfaces still duplicated the full goal. One shared function now."""
    from hermes_cli import goals
    long_goal = "HANDOFF " + "step; " * 120
    assert goals.goal_kick_prompt(long_goal, long_goal) == goals.GOAL_ALREADY_SEEN_KICK
    short = goals.goal_kick_prompt("ship the API", "ship the API or ship the UI? " * 8)
    assert short == goals.GOAL_KICK_TEMPLATE.format(goal="ship the API")
    # DB-backed lookup fails safe to "" (goal kicked framed) when no session/db
    assert goals.last_user_message_from_db(None) == ""
