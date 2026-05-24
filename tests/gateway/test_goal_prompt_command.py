import types
from pathlib import Path

import pytest

from gateway.platforms.base import MessageEvent, MessageType
from gateway.run import GatewayRunner


@pytest.mark.asyncio
async def test_gateway_goal_prompt_loads_prompt_and_dispatches_goal(tmp_path: Path):
    prompt = tmp_path / "docs" / "runbooks" / "GOAL_PROMPT.md"
    prompt.parent.mkdir(parents=True)
    prompt.write_text("```text\nContinue from NEXT_ACTIONS.md\n```", encoding="utf-8")

    runner = object.__new__(GatewayRunner)
    seen = {}

    async def fake_goal_handler(self, event):
        seen["text"] = event.text
        return "Goal set."

    runner._handle_goal_command = types.MethodType(fake_goal_handler, runner)
    event = MessageEvent(
        text=f"/goal_prompt {tmp_path}",
        message_type=MessageType.TEXT,
    )

    result = await runner._handle_goal_prompt_command(event)

    assert seen["text"] == "/goal Continue from NEXT_ACTIONS.md"
    assert "Loading goal prompt" in result
    assert "Goal set." in result
    # The handler restores the original event text after dispatch.
    assert event.text == f"/goal_prompt {tmp_path}"


@pytest.mark.asyncio
async def test_gateway_goal_prompt_oneshot_wraps_prompt_and_dispatches_goal(tmp_path: Path):
    prompt = tmp_path / "docs" / "runbooks" / "GOAL_PROMPT.md"
    prompt.parent.mkdir(parents=True)
    prompt.write_text("```text\n/goal Continue from NEXT_ACTIONS.md\n```", encoding="utf-8")

    runner = object.__new__(GatewayRunner)
    seen = {}

    async def fake_goal_handler(self, event):
        seen["text"] = event.text
        seen["metadata"] = dict(getattr(self, "_pending_goal_metadata", {}) or {})
        return "Goal set."

    runner._handle_goal_command = types.MethodType(fake_goal_handler, runner)
    event = MessageEvent(
        text=f"/goal_prompt_oneshot {tmp_path}",
        message_type=MessageType.TEXT,
    )

    result = await runner._handle_goal_prompt_command(event, oneshot=True)

    assert seen["text"].startswith("/goal Continue from NEXT_ACTIONS.md")
    assert not seen["text"].startswith("/goal /goal")
    assert "/goal_prompt_oneshot mode" in seen["text"]
    assert seen["metadata"]["goal_mode"] == "goal_prompt_oneshot"
    assert seen["metadata"]["goal_prompt_path"] == str(prompt)
    assert seen["metadata"]["max_turns"] == 200
    assert seen["metadata"]["compaction_refresh_interval"] == 5
    assert "Loading one-shot goal prompt" in result
    assert "Goal set." in result
    assert event.text == f"/goal_prompt_oneshot {tmp_path}"


@pytest.mark.asyncio
async def test_gateway_goal_prompt_reports_missing_file(tmp_path: Path):
    runner = object.__new__(GatewayRunner)
    event = MessageEvent(
        text=f"/goal_prompt {tmp_path}",
        message_type=MessageType.TEXT,
    )

    result = await runner._handle_goal_prompt_command(event)

    assert "No GOAL_PROMPT.md found" in result
    assert "Usage: `/goal_prompt [project-root-or-prompt-file]`" in result


def test_gateway_session_split_carries_persisted_oneshot_goal_state(tmp_path: Path):
    from hermes_cli.goals import GoalManager, save_goal

    runner = object.__new__(GatewayRunner)
    old_session_id = f"gateway_parent_{tmp_path.name}"
    new_session_id = f"gateway_child_{tmp_path.name}"
    state = GoalManager(old_session_id, default_max_turns=321).set(
        "Continue gateway oneshot goal",
        goal_mode="goal_prompt_oneshot",
        goal_prompt_path=str(tmp_path / "docs" / "runbooks" / "GOAL_PROMPT.md"),
        compaction_refresh_interval=5,
    )
    state.turns_used = 9
    save_goal(old_session_id, state)

    assert runner._carry_goal_state_between_sessions(
        old_session_id,
        new_session_id,
        reason="compression",
    ) is True

    carried = GoalManager(new_session_id).state
    assert carried is not None
    assert carried.goal == "Continue gateway oneshot goal"
    assert carried.goal_mode == "goal_prompt_oneshot"
    assert carried.goal_prompt_path.endswith("GOAL_PROMPT.md")
    assert carried.compaction_refresh_interval == 5
    assert carried.turns_used == 9
    assert carried.max_turns == 321
