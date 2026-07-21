"""Discord rich Kanban approval gate tests."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

# Trigger shared discord mock before importing adapter module.
from plugins.platforms.discord.adapter import KanbanApprovalView  # noqa: E402


def _interaction(user_id="458519787346198530", display_name="capt.america"):
    embed = MagicMock()
    embed.set_footer = MagicMock()
    return SimpleNamespace(
        user=SimpleNamespace(id=user_id, display_name=display_name, roles=[]),
        response=SimpleNamespace(edit_message=AsyncMock(), send_message=AsyncMock()),
        message=SimpleNamespace(embeds=[embed]),
    )


def test_kanban_approval_approve_comments_and_unblocks_not_completes():
    async def _run():
        view = KanbanApprovalView(
            task_id="t_gate",
            board="default",
            channel_id="1523615301965447298",
            allowed_user_ids={"458519787346198530"},
        )
        interaction = _interaction()

        with patch("hermes_cli.kanban.run_slash", return_value="OK") as run_slash:
            await view.approve(interaction, MagicMock())

        commands = [call.args[0] for call in run_slash.call_args_list]
        assert len(commands) == 2
        assert commands[0].startswith("--board default comment t_gate ")
        assert "HUMAN_DECISION: APPROVED" in commands[0]
        assert "capt.america" in commands[0]
        assert commands[1] == "--board default unblock --reason 'HUMAN_DECISION: APPROVED by capt.america' t_gate"
        assert not any("kanban complete" in cmd for cmd in commands)
        interaction.response.edit_message.assert_awaited_once()

    asyncio.run(_run())


def test_kanban_approval_approve_continues_when_discord_ack_expired():
    async def _run():
        view = KanbanApprovalView(
            task_id="t_gate",
            board="default",
            channel_id="1523615301965447298",
            allowed_user_ids={"458519787346198530"},
        )
        interaction = _interaction()
        interaction.response.edit_message.side_effect = RuntimeError("Unknown interaction")

        with patch("hermes_cli.kanban.run_slash", return_value="OK") as run_slash:
            await view.approve(interaction, MagicMock())

        commands = [call.args[0] for call in run_slash.call_args_list]
        assert any(cmd.startswith("--board default comment t_gate ") for cmd in commands)
        assert any(cmd.startswith("--board default unblock ") for cmd in commands)

    asyncio.run(_run())


def test_kanban_approval_reject_comments_and_keeps_task_blocked():
    async def _run():
        view = KanbanApprovalView(
            task_id="t_gate",
            board="default",
            channel_id="1523615301965447298",
            allowed_user_ids={"458519787346198530"},
        )
        interaction = _interaction()

        with patch("hermes_cli.kanban.run_slash", return_value="OK") as run_slash:
            await view.reject(interaction, MagicMock())

        commands = [call.args[0] for call in run_slash.call_args_list]
        assert len(commands) == 1
        assert commands[0].startswith("--board default comment t_gate ")
        assert "HUMAN_DECISION: REJECTED" in commands[0]
        assert "capt.america" in commands[0]
        assert not any(" unblock " in cmd for cmd in commands)
        assert not any("kanban complete" in cmd for cmd in commands)
        interaction.response.edit_message.assert_awaited_once()

    asyncio.run(_run())
