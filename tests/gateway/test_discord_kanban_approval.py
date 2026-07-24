"""Discord rich Kanban approval gate tests."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

# Trigger shared discord mock before importing adapter module.
from plugins.platforms.discord.adapter import (  # noqa: E402
    KanbanApprovalView,
    KanbanProtocolViolationView,
)


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
        assert len(commands) == 2
        assert commands[0].startswith("--board default comment t_gate ")
        assert "HUMAN_DECISION: REJECTED" in commands[0]
        assert "capt.america" in commands[0]
        assert commands[1].startswith("--board default create ")
        assert "Address rejection for t_gate" in commands[1]
        assert "--parent t_gate" in commands[1]
        assert "--triage" in commands[1]
        assert "--idempotency-key kanban-approval-rejection:t_gate" in commands[1]
        assert "HUMAN_DECISION: REJECTED by capt.america" in commands[1]
        assert not any(" unblock " in cmd for cmd in commands)
        assert not any("kanban complete" in cmd for cmd in commands)
        interaction.response.edit_message.assert_awaited_once()

    asyncio.run(_run())


def test_kanban_approval_comment_reason_is_recorded():
    async def _run():
        view = KanbanApprovalView(
            task_id="t_gate",
            board="default",
            channel_id="1523615301965447298",
            allowed_user_ids={"458519787346198530"},
        )
        interaction = _interaction()

        with patch("hermes_cli.kanban.run_slash", return_value="OK") as run_slash:
            await view._resolve(
                interaction,
                "reject",
                MagicMock(),
                "❌ Rejected",
                note="Need safer rollback plan first.",
            )

        commands = [call.args[0] for call in run_slash.call_args_list]
        assert "Reason: Need safer rollback plan first." in commands[0]
        assert "Reason: Need safer rollback plan first." in commands[1]

    asyncio.run(_run())


def test_protocol_violation_retry_comments_and_promotes():
    async def _run():
        view = KanbanProtocolViolationView(
            task_id="t_proto",
            board="default",
            channel_id="1523615301965447298",
            allowed_user_ids={"458519787346198530"},
        )
        interaction = _interaction()

        with patch("hermes_cli.kanban.run_slash", return_value="OK") as run_slash:
            await view.retry(interaction, MagicMock())

        commands = [call.args[0] for call in run_slash.call_args_list]
        assert len(commands) == 2
        assert commands[0].startswith("--board default comment t_proto ")
        assert "PROTOCOL_VIOLATION_DECISION: RETRY" in commands[0]
        assert "capt.america" in commands[0]
        assert commands[1] == "--board default promote --force t_proto 'PROTOCOL_VIOLATION_DECISION: RETRY by capt.america'"
        interaction.response.edit_message.assert_awaited_once()

    asyncio.run(_run())


def test_protocol_violation_remediation_creates_idempotent_child():
    async def _run():
        view = KanbanProtocolViolationView(
            task_id="t_proto",
            board="default",
            channel_id="1523615301965447298",
            allowed_user_ids={"458519787346198530"},
            error="worker exited cleanly without terminal call",
        )
        interaction = _interaction()

        with patch("hermes_cli.kanban.run_slash", return_value="OK") as run_slash:
            await view.create_remediation(interaction, MagicMock())

        commands = [call.args[0] for call in run_slash.call_args_list]
        assert len(commands) == 2
        assert "PROTOCOL_VIOLATION_DECISION: REMEDIATE" in commands[0]
        assert commands[1].startswith("--board default create ")
        assert "Remediate protocol violation for t_proto" in commands[1]
        assert "--parent t_proto" in commands[1]
        assert "--triage" in commands[1]
        assert "--idempotency-key kanban-protocol-violation:t_proto" in commands[1]
        assert "worker exited cleanly without terminal call" in commands[1]
        assert not any(" promote " in cmd for cmd in commands)

    asyncio.run(_run())


def test_protocol_violation_mark_verified_done_completes_with_human_summary():
    async def _run():
        view = KanbanProtocolViolationView(
            task_id="t_proto",
            board="default",
            channel_id="1523615301965447298",
            allowed_user_ids={"458519787346198530"},
        )
        interaction = _interaction()

        with patch("hermes_cli.kanban.run_slash", return_value="OK") as run_slash:
            await view.mark_verified_done(interaction, MagicMock())

        commands = [call.args[0] for call in run_slash.call_args_list]
        assert len(commands) == 2
        assert "PROTOCOL_VIOLATION_DECISION: HUMAN_VERIFIED_DONE" in commands[0]
        assert commands[1].startswith("--board default complete t_proto ")
        assert "Human verified completion after protocol violation" in commands[1]

    asyncio.run(_run())
