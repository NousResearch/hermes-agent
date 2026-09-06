"""Bare-word approvals resolve only outside multi-user chats.

A pending dangerous-command approval plus any participant's conversational
"ok"/"no" in a group/channel/forum must not resolve someone else's prompt.
Slash /approve keeps working everywhere; DMs (and DM-only channels) keep
the bare-word convenience.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from gateway.run_busy import GatewayBusySessionMixin


def _event(text, chat_type):
    return SimpleNamespace(
        text=text,
        allow_gateway_control=True,
        source=SimpleNamespace(chat_type=chat_type),
    )


def _runner():
    return SimpleNamespace(
        _PLAINTEXT_APPROVAL_WORDS=GatewayBusySessionMixin._PLAINTEXT_APPROVAL_WORDS,
        _handle_approve_command=AsyncMock(return_value="approved"),
        _handle_deny_command=AsyncMock(return_value="denied"),
        _adapter_for_source=lambda source: None,
    )


async def _route(runner, event):
    return await GatewayBusySessionMixin._route_plaintext_approval_while_busy(
        runner, event, "sess-1")


class TestPlaintextApprovalScope:
    @pytest.mark.asyncio
    async def test_group_ok_ignored(self):
        with patch("tools.approval.has_blocking_approval", return_value=True):
            assert await _route(_runner(), _event("ok", "group")) is False

    @pytest.mark.asyncio
    async def test_group_deny_ignored(self):
        with patch("tools.approval.has_blocking_approval", return_value=True):
            assert await _route(_runner(), _event("no", "channel")) is False

    @pytest.mark.asyncio
    @pytest.mark.parametrize("chat_type", ["dm", "private", None])
    async def test_dm_ok_resolves(self, chat_type):
        runner = _runner()
        with patch("tools.approval.has_blocking_approval", return_value=True):
            assert await _route(runner, _event("ok", chat_type)) is True
        runner._handle_approve_command.assert_awaited_once()
