"""A gateway turn that COMPLETED with empty text still reaches the /goal hook (the goal manager
counts the empty streak and re-prompts / pauses visibly); interrupted or failed empty turns stay
out. Before, any empty text skipped the hook and the goal stalled in ``active`` with nothing queued.
Inspired by Codex CLI 0.155 (openai/codex#44320)."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.run import GatewayRunner


def _runner() -> tuple[GatewayRunner, AsyncMock, AsyncMock]:
    runner = object.__new__(GatewayRunner)
    runner.session_store = MagicMock()
    runner.session_store.get_or_create_session.return_value = MagicMock(session_id="s1")
    goal_hook, loop_hook = AsyncMock(), AsyncMock()
    runner._post_turn_goal_continuation = goal_hook
    runner._post_turn_loop_completion = loop_hook
    return runner, goal_hook, loop_hook


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "agent_result, goal_hook_runs",
    [
        ({"final_response": "", "completed": True}, True),      # completed, empty → streak accounting
        ({"final_response": "", "interrupted": True}, False),   # soft interrupt → never drives /goal
        ({"final_response": "", "failed": True}, False),        # failed turn → never drives /goal
        (None, False),                                          # nothing known about the turn
    ],
)
async def test_empty_text_drives_goal_hook_only_for_completed_turns(agent_result, goal_hook_runs):
    runner, goal_hook, loop_hook = _runner()
    await runner._run_post_turn_hooks(agent_result=agent_result, source=object(), is_internal=False)
    assert goal_hook.await_count == (1 if goal_hook_runs else 0)
    loop_hook.assert_awaited_once()   # an in-flight /loop tick is always released
    if goal_hook_runs:
        assert goal_hook.await_args.kwargs["final_response"] == ""
