"""Tests for approvals.escalate_to gateway approval routing."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from gateway.config import Platform
from gateway.run import (
    _APPROVAL_ESCALATED_ORIGIN_NOTICE,
    _resolve_approval_prompt_route,
    _send_approval_escalated_origin_notice_sync,
)
from gateway.session import SessionSource


class _OriginAdapter:
    typed_command_prefix = "/"

    def __init__(self) -> None:
        self.sent = []

    async def send(self, chat_id, content, metadata=None):
        self.sent.append(
            {"chat_id": chat_id, "content": content, "metadata": metadata}
        )
        return SimpleNamespace(success=True)

    def pause_typing_for_chat(self, chat_id):
        return None


class _ButtonApprovalAdapter(_OriginAdapter):
    async def send_exec_approval(
        self, chat_id, command, session_key, description="", metadata=None
    ):
        self.sent.append(
            {
                "chat_id": chat_id,
                "command": command,
                "session_key": session_key,
                "description": description,
                "metadata": metadata,
            }
        )
        return SimpleNamespace(success=True)


def _source(
    *,
    platform: Platform = Platform.BLUEBUBBLES,
    chat_id: str = "origin-chat",
    thread_id: str | None = None,
) -> SessionSource:
    return SessionSource(
        platform=platform,
        user_id="origin-user",
        chat_id=chat_id,
        user_name="Origin",
        chat_type="dm",
        thread_id=thread_id,
    )


def test_approval_prompt_route_defaults_to_origin_when_key_missing():
    origin = _OriginAdapter()
    metadata = {"thread_id": "origin-thread"}

    route = _resolve_approval_prompt_route(
        {},
        {Platform.BLUEBUBBLES: origin},
        source=_source(platform=Platform.BLUEBUBBLES),
        origin_adapter=origin,
        origin_chat_id="origin-chat",
        origin_metadata=metadata,
    )

    assert route.adapter is origin
    assert route.chat_id == "origin-chat"
    assert route.metadata is metadata
    assert route.escalated is False


def test_approval_prompt_route_uses_configured_interactive_target():
    origin = _OriginAdapter()
    operator = _ButtonApprovalAdapter()

    route = _resolve_approval_prompt_route(
        {"approvals": {"escalate_to": "telegram:operator-chat"}},
        {Platform.BLUEBUBBLES: origin, Platform.TELEGRAM: operator},
        source=_source(platform=Platform.BLUEBUBBLES, chat_id="origin-chat"),
        origin_adapter=origin,
        origin_chat_id="origin-chat",
        origin_metadata={"thread_id": "origin-thread"},
    )

    assert route.adapter is operator
    assert route.chat_id == "operator-chat"
    assert route.metadata is None
    assert route.escalated is True


def test_approval_prompt_route_rejects_cross_session_text_only_target():
    origin = _OriginAdapter()
    text_only_operator = _OriginAdapter()

    with pytest.raises(RuntimeError, match="interactive exec approvals"):
        _resolve_approval_prompt_route(
            {"approvals": {"escalate_to": "sms:operator-chat"}},
            {Platform.BLUEBUBBLES: origin, Platform.SMS: text_only_operator},
            source=_source(platform=Platform.BLUEBUBBLES, chat_id="origin-chat"),
            origin_adapter=origin,
            origin_chat_id="origin-chat",
            origin_metadata=None,
        )


def test_approval_prompt_route_rejects_malformed_configured_target():
    with pytest.raises(ValueError, match="platform:chat_id"):
        _resolve_approval_prompt_route(
            {"approvals": {"escalate_to": "telegram"}},
            {},
            source=_source(),
            origin_adapter=_OriginAdapter(),
            origin_chat_id="origin-chat",
            origin_metadata=None,
        )


def test_escalated_origin_notice_is_neutral_and_bounded(monkeypatch):
    origin = _OriginAdapter()
    scheduled = []

    def fake_schedule(coro, loop, **kwargs):
        import asyncio

        scheduled.append(kwargs)
        result = asyncio.run(coro)

        class _Future:
            def result(self, timeout=None):
                return result

        return _Future()

    monkeypatch.setattr("gateway.run.safe_schedule_threadsafe", fake_schedule)

    _send_approval_escalated_origin_notice_sync(
        origin,
        "origin-chat",
        {"thread_id": "origin-thread"},
        loop=object(),
    )

    assert origin.sent == [
        {
            "chat_id": "origin-chat",
            "content": _APPROVAL_ESCALATED_ORIGIN_NOTICE,
            "metadata": {"thread_id": "origin-thread"},
        }
    ]
    notice = origin.sent[0]["content"]
    assert "rm -rf" not in notice
    assert "/approve" not in notice
    assert "flagged" in notice
    assert scheduled
