"""`/goal <text>` kicks the loop with a short pointer when the user's last message already carries the goal.

In one run `/goal <2,000-char handoff note>` was issued 16 minutes after the same note had been pasted as
a user message; the kickoff re-sent it and the agent spent 11 API calls / 6 min deciding it was a replay.
"""
import queue

from hermes_cli.cli_commands_mixin import CLICommandsMixin
from hermes_cli.goals import GOAL_ALREADY_SEEN_KICK


def _cli(history):
    cli = CLICommandsMixin.__new__(CLICommandsMixin)
    cli.conversation_history = history
    cli._pending_input = queue.Queue()
    return cli


HANDOFF = "HANDOFF: resume round 3 integration.\n" + "\n".join(f"  - step {i}: merge r3-{i} and run its targeted suite, then the full suite" for i in range(8))


def test_goal_that_the_user_just_pasted_kicks_with_a_pointer_not_the_text():
    cli = _cli([{"role": "user", "content": HANDOFF + "\n\nGo."},
                {"role": "assistant", "content": "ok"}])
    assert cli._goal_kick_prompt(HANDOFF) == GOAL_ALREADY_SEEN_KICK
    # block-style content is handled too
    cli = _cli([{"role": "user", "content": [{"type": "text", "text": HANDOFF}]}])
    assert cli._goal_kick_prompt(HANDOFF) == GOAL_ALREADY_SEEN_KICK


def test_a_new_goal_or_a_goal_from_an_older_turn_is_kicked_verbatim():
    assert _cli([])._goal_kick_prompt("Ship the release")  == "Ship the release"
    cli = _cli([{"role": "user", "content": "Something unrelated"}])
    assert cli._goal_kick_prompt(HANDOFF) == HANDOFF
    # only the LAST user message counts: the agent has moved on since an older paste
    cli = _cli([{"role": "user", "content": HANDOFF}, {"role": "assistant", "content": "done"},
                {"role": "user", "content": "now something else"}])
    assert cli._goal_kick_prompt(HANDOFF) == HANDOFF


def test_a_short_goal_that_selects_one_option_from_the_last_message_is_kicked_verbatim():
    """Independent-review witness: after a message offering API or UI work, `/goal ship the API` and
    `/goal ship the UI` produced identical kickoffs. A goal that is a fragment of the last message
    carries the selection; only a near-whole re-paste is replaced by the pointer."""
    offer = "I can either ship the API or ship the UI next; which do you want? " * 8
    cli = _cli([{"role": "user", "content": offer}])
    assert cli._goal_kick_prompt("ship the API") == "ship the API"
    assert cli._goal_kick_prompt("ship the UI") == "ship the UI"
    # a long goal that is only a minority of a much longer message is also kept verbatim
    long_goal = "x" * 500
    cli = _cli([{"role": "user", "content": long_goal + " " + "y" * 2000}])
    assert cli._goal_kick_prompt(long_goal) == long_goal


def test_gateway_and_tui_surfaces_use_the_same_rule(tmp_path, monkeypatch):
    """Independent review: other surfaces still duplicated the full goal. One shared function now."""
    from hermes_cli import goals
    long_goal = "HANDOFF " + "step; " * 120
    assert goals.goal_kick_prompt(long_goal, long_goal) == goals.GOAL_ALREADY_SEEN_KICK
    assert goals.goal_kick_prompt("ship the API", "ship the API or ship the UI? " * 8) == "ship the API"
    # DB-backed lookup fails safe to "" (goal kicked verbatim) when no session/db
    assert goals.last_user_message_from_db(None) == ""
    import gateway.slash_commands_goals as g, tui_gateway.methods_tools as m  # noqa: E401
    import inspect
    assert "goal_kick_prompt" in inspect.getsource(g) and "goal_kick_prompt" in inspect.getsource(m)
