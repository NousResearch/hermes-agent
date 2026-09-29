import types

import pytest
from unittest.mock import AsyncMock, patch

from gateway.config import PlatformConfig

from types import SimpleNamespace
import asyncio
from gateway.platforms.base import SendResult
from plugins.platforms.matrix.adapter import MatrixAdapter
from tools import approval
from tools.approval_gateway_wait import _ApprovalEntry


class TestMatrixExecApprovalReactions:
    @pytest.mark.asyncio
    async def test_reaction_resolves_pending_approval(self, monkeypatch):
        monkeypatch.setenv("MATRIX_ALLOWED_USERS", "@liizfq:liizfq.top")
        from plugins.platforms.matrix.adapter import MatrixAdapter
        from plugins.platforms.matrix.approval_lifecycle import _MatrixApprovalPrompt

        adapter = MatrixAdapter(
            PlatformConfig(
                enabled=True,
                token="tok",
                extra={"homeserver": "https://matrix.example.org"},
            )
        )
        # Resolve user_id so _is_self_sender doesn't defensively drop all traffic (#15763).
        adapter._user_id = "@bot:example.org"
        adapter._approval_prompts_by_event["$target"] = _MatrixApprovalPrompt(
            session_key="sess-1",
            chat_id="!room:example.org",
            message_id="$target",
            approval_id="req-1",
        )

        content = {"m.relates_to": {"event_id": "$target", "key": "✅"}}
        event = types.SimpleNamespace(
            sender="@liizfq:liizfq.top",
            event_id="$react1",
            room_id="!room:example.org",
            content=content,
        )

        adapter.edit_message = AsyncMock(
            return_value=types.SimpleNamespace(success=True, message_id="$edit")
        )
        with patch(
            "tools.approval.resolve_gateway_approval", return_value=1
        ) as mock_resolve:
            await adapter._on_reaction(event)

        mock_resolve.assert_called_once_with("sess-1", "once", approval_id="req-1")
        assert "$target" not in adapter._approval_prompts_by_event


@pytest.mark.asyncio
@pytest.mark.parametrize("identity_source", ["request_id", "metadata"])
@pytest.mark.parametrize("card_state", ["current", "stale", "unbound"])
async def test_native_matrix_card_targets_the_displayed_request(
    monkeypatch, identity_source, card_state
):
    monkeypatch.setenv("MATRIX_ALLOWED_USERS", "@owner:example.org")
    session = "agent:main:matrix:dm:!room:example.org"
    old = _ApprovalEntry({"command": "old command", "request_id": "old"})
    new = _ApprovalEntry({"command": "new sensitive command", "request_id": "new"})
    approval._gateway_queues[session] = [old, new] if card_state == "current" else [new]
    adapter = MatrixAdapter(
        PlatformConfig(
            enabled=True,
            token="test",
            extra={"homeserver": "https://matrix.example.org"},
        )
    )
    adapter._client = SimpleNamespace()
    adapter.send = AsyncMock(return_value=SendResult(success=True, message_id="$card"))
    adapter.edit_message = AsyncMock(
        return_value=SendResult(success=True, message_id="$card")
    )
    adapter._send_reaction = AsyncMock(return_value="$seed")
    adapter._redact_bot_approval_reactions = AsyncMock()
    kwargs = {}
    if card_state != "unbound":
        identity = "new" if card_state == "current" else "old"
        kwargs = (
            {"request_id": identity}
            if identity_source == "request_id"
            else {"metadata": {"approval_id": identity}}
        )
    try:
        sent = await adapter.send_exec_approval(
            "!room:example.org", "displayed command", session, **kwargs
        )
        if sent.success:
            await adapter._handle_approval_reaction(
                "!room:example.org", "$card", "✅", "@owner:example.org"
            )
        assert (sent.success, old.result, new.result) == (
            card_state != "unbound",
            None,
            "once" if card_state == "current" else None,
        )
    finally:
        approval.clear_session(session)
        tasks = tuple(getattr(adapter, "_approval_tasks", set()))
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
