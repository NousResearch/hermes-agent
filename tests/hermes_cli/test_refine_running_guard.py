"""/refine must refuse mid-turn on the CLI, the way the gateway already does.

``cli.py`` assigns ``conversation_history`` only once a turn returns ("Update
history with full conversation"), so a snapshot taken while the agent is
working holds the PREVIOUS turn. Without a guard the CLI spawned the review
against that stale history and still printed the success line — the review
silently missed the work the user was watching, which is usually the very
thing that prompted the /refine.
"""

from __future__ import annotations

import types

from hermes_cli.cli_commands_mixin import CLICommandsMixin


class _StubAgent:
    valid_tool_names = {"skill_manage"}

    def __init__(self):
        self.spawned = False
        self.snapshot = None
        self.focus = None

    def _spawn_background_review(
        self, messages_snapshot, review_memory, review_skills, focus=None,
    ):
        self.spawned = True
        self.snapshot = list(messages_snapshot)
        self.focus = focus


def _cli(*, running: bool, history: list[dict]):
    cli = types.SimpleNamespace()
    cli.agent = _StubAgent()
    cli.conversation_history = list(history)
    cli._agent_running = running
    cli._handle_refine_command = types.MethodType(
        CLICommandsMixin._handle_refine_command, cli,
    )
    return cli


FINISHED_TURN = [
    {"role": "user", "content": "hello"},
    {"role": "assistant", "content": "hi"},
]


def test_refine_is_refused_while_a_turn_is_running():
    """The in-flight turn is not in conversation_history yet, so don't review."""
    cli = _cli(running=True, history=FINISHED_TURN)

    cli._handle_refine_command("/refine save the deploy workflow as a skill")

    assert cli.agent.spawned is False, (
        "spawned a review against a stale pre-turn snapshot while the agent "
        f"was running (snapshot={cli.agent.snapshot!r})"
    )


def test_refine_runs_when_the_session_is_idle():
    """The guard must not swallow the ordinary case, and focus still rides along."""
    cli = _cli(running=False, history=FINISHED_TURN)

    cli._handle_refine_command("/refine save the deploy workflow as a skill")

    assert cli.agent.spawned is True
    assert cli.agent.snapshot == FINISHED_TURN
    assert cli.agent.focus == "save the deploy workflow as a skill"


def test_refine_without_focus_still_runs_when_idle():
    """A bare /refine passes focus=None, as the automatic post-turn review does."""
    cli = _cli(running=False, history=FINISHED_TURN)

    cli._handle_refine_command("/refine")

    assert cli.agent.spawned is True
    assert cli.agent.focus is None
