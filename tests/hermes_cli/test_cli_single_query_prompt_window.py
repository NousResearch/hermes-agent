"""`chat -q` has nobody at a prompt: its turn must not bind the wait-until-answered policy.

The interactive CLI binds ``set_prompts_wait_for_answer()`` around each turn so approval panels stay
up until answered. A single-query turn has no prompt_toolkit app (and often no stdin), so binding it
there parks every approval for ``MAX_SAFE_TIMEOUT_S`` instead of ``approvals.timeout``.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from cli import HermesCLI, _ChatTurn
from tools.approval_context import prompts_wait_for_answer


def _run_turn(single_query: bool) -> bool:
    cli = HermesCLI.__new__(HermesCLI)
    cli.session_id = "prompt-window"
    cli.conversation_history = [{"role": "user", "content": "go"}]
    if single_query:
        setattr(cli, "_single_query_mode", True)
    seen = {}

    def _run_conversation(**_kwargs):
        seen["wait_for_answer"] = prompts_wait_for_answer()
        return {}

    cli.agent = SimpleNamespace(run_conversation=Mock(side_effect=_run_conversation))
    cli._chat_run_agent(_ChatTurn(), "go")
    assert prompts_wait_for_answer() is False, "turn leaked the wait-for-answer binding"
    return seen["wait_for_answer"]


@pytest.mark.parametrize(("single_query", "expected"), [(False, True), (True, False)])
def test_only_an_attended_turn_waits_for_an_answer(single_query, expected):
    assert _run_turn(single_query) is expected
