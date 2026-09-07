"""Plain-text approval replies ("yes"/"no" while an approval blocks) must honour the same
slash-command gating as a typed ``/approve`` / ``/deny``.

``_route_plaintext_approval_while_busy`` synthesizes ``/approve`` or ``/deny`` from a bare
word and calls the handler directly.  Before this fix it skipped ``_check_slash_access``,
so an operator who made ``/approve`` admin-only via ``group_allow_admin_from`` was bypassed
by any allowed non-admin typing "yes" (unverified-approver hole).
"""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.base import MessageEvent, MessageType
from gateway.session import SessionSource

ADMIN = "111"
NON_ADMIN = "222"


def _make_event(text: str, user_id: str) -> MessageEvent:
    return MessageEvent(
        text=text,
        message_type=MessageType.TEXT,
        source=SessionSource(
            platform=Platform.DISCORD,
            user_id=user_id,
            chat_id="chan-1",
            user_name=f"user-{user_id}",
            chat_type="channel",
        ),
        message_id="m1",
    )


def _make_runner(monkeypatch, extra: dict):
    from gateway.run import GatewayRunner

    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(
        platforms={Platform.DISCORD: PlatformConfig(enabled=True, token="***", extra=extra)}
    )
    adapter = MagicMock()
    adapter._send_with_retry = AsyncMock(
        return_value=SimpleNamespace(success=True, message_id="reply1")
    )
    adapter._unwrap_ephemeral = lambda r: (r, 0) if isinstance(r, str) else (None, 0)
    runner.adapters = {Platform.DISCORD: adapter}
    runner._running_agents = {}
    runner._pending_messages = {}

    recorder = AsyncMock(return_value="✅ approved")
    runner._handle_approve_command = recorder
    runner._handle_deny_command = AsyncMock(return_value="denied")
    monkeypatch.setattr("tools.approval.has_blocking_approval", lambda _key: True)
    return runner, adapter, recorder


def _route(runner, text: str, user_id: str) -> bool:
    return asyncio.run(
        runner._route_plaintext_approval_while_busy(_make_event(text, user_id), "agent:main:discord:channel:chan-1")
    )


def test_non_admin_plaintext_yes_is_denied_and_consumed(monkeypatch):
    runner, adapter, recorder = _make_runner(monkeypatch, {"group_allow_admin_from": [ADMIN]})

    assert _route(runner, "yes", NON_ADMIN) is True

    recorder.assert_not_awaited()
    adapter._send_with_retry.assert_awaited_once()
    sent = adapter._send_with_retry.await_args.kwargs["content"]
    assert "/approve" in sent and "admin" in sent.lower()


def test_admin_plaintext_yes_reaches_approve_handler(monkeypatch):
    runner, _adapter, recorder = _make_runner(monkeypatch, {"group_allow_admin_from": [ADMIN]})

    assert _route(runner, "yes", ADMIN) is True

    recorder.assert_awaited_once()


def test_no_admin_list_keeps_plaintext_yes_ungated(monkeypatch):
    """No ``allow_admin_from`` configured -> gating disabled -> behaviour unchanged."""
    runner, _adapter, recorder = _make_runner(monkeypatch, {})

    assert _route(runner, "yes", NON_ADMIN) is True

    recorder.assert_awaited_once()
