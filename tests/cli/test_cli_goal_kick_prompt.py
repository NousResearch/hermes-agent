"""`/goal <text>` kicks the loop with a short pointer when the user's last message already carries the goal.

In one run `/goal <2,000-char handoff note>` was issued 16 minutes after the same note had been pasted as
a user message; the kickoff re-sent it and the agent spent 11 API calls / 6 min deciding it was a replay.
"""
import queue

from hermes_cli.cli_commands_mixin import CLICommandsMixin


def _cli(history):
    cli = CLICommandsMixin.__new__(CLICommandsMixin)
    cli.conversation_history = history
    cli._pending_input = queue.Queue()
    return cli


HANDOFF = "HANDOFF: resume round 3 integration.\n  - merge r3-16 first\n  - then run the full suite"


def test_goal_that_the_user_just_pasted_kicks_with_a_pointer_not_the_text():
    cli = _cli([{"role": "user", "content": "Here is the plan.\n\n" + HANDOFF + "\n\nGo."},
                {"role": "assistant", "content": "ok"}])
    assert cli._goal_kick_prompt(HANDOFF) == CLICommandsMixin._GOAL_ALREADY_SEEN_KICK
    # block-style content is handled too
    cli = _cli([{"role": "user", "content": [{"type": "text", "text": HANDOFF}]}])
    assert cli._goal_kick_prompt(HANDOFF) == CLICommandsMixin._GOAL_ALREADY_SEEN_KICK


def test_a_new_goal_or_a_goal_from_an_older_turn_is_kicked_verbatim():
    assert _cli([])._goal_kick_prompt("Ship the release")  == "Ship the release"
    cli = _cli([{"role": "user", "content": "Something unrelated"}])
    assert cli._goal_kick_prompt(HANDOFF) == HANDOFF
    # only the LAST user message counts: the agent has moved on since an older paste
    cli = _cli([{"role": "user", "content": HANDOFF}, {"role": "assistant", "content": "done"},
                {"role": "user", "content": "now something else"}])
    assert cli._goal_kick_prompt(HANDOFF) == HANDOFF
