from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import MessageEvent, MessageType
from gateway.session import SessionSource


def _event(text: str = "Patch the confirmed failure") -> MessageEvent:
    return MessageEvent(
        text=text,
        message_type=MessageType.TEXT,
        message_id="message-1",
        source=SessionSource(
            platform=Platform.DISCORD,
            chat_id="project-updates",
            chat_type="group",
            user_id="operator-1",
        ),
    )


def _adapter(monkeypatch):
    import plugins.platforms.discord.adapter as discord_platform
    from plugins.platforms.discord.adapter import DiscordAdapter

    monkeypatch.setattr(
        discord_platform.discord,
        "DMChannel",
        type("DMChannel", (), {}),
        raising=False,
    )
    value = DiscordAdapter(
        PlatformConfig(
            enabled=True,
            token="fake-token",
            extra={
                "specialist_routing": {
                    "enabled": True,
                    "board": "project-maintenance",
                    "capabilities": {
                        "burndown-patch-steward": {
                            "domain": "repository-evidence",
                            "actions": ["audit", "inspect", "read", "review", "validate"],
                            "evidence_class": "diagnostic-only",
                            "requested_permissions": ["repository-evidence:read"],
                        }
                    },
                }
            },
        )
    )
    value._client = SimpleNamespace(user=SimpleNamespace(id=999))
    value.send = AsyncMock()
    return value


def test_specialist_route_creates_one_handoff_and_acknowledges(monkeypatch):
    from gateway.specialist_handoff import HandoffResult
    from gateway.specialist_routing import RouteKind, SpecialistRouteDecision

    adapter = _adapter(monkeypatch)
    adapter._classify_specialist_event = AsyncMock(
        return_value=SpecialistRouteDecision(
            kind=RouteKind.SPECIALIST,
            profile="burndown-patch-steward",
            confidence=0.95,
            reason="bounded patch",
            title="Patch confirmed failure",
        )
    )
    create = AsyncMock(return_value=HandoffResult(True, task_id="t_abc", created=True))

    async def fake_to_thread(func, **kwargs):
        return await create(**kwargs)

    monkeypatch.setattr(asyncio, "to_thread", fake_to_thread)

    handled = asyncio.run(adapter._maybe_route_specialist_event(_event()))

    assert handled is True
    create.assert_awaited_once()
    adapter.send.assert_awaited_once_with(
        "project-updates",
        content="Planning `t_abc` with the task orchestrator. I’ll post the worker plan and progress here.",
        reply_to="message-1",
    )


def test_general_route_preserves_normal_chat_path(monkeypatch):
    from gateway.specialist_routing import RouteKind, SpecialistRouteDecision

    adapter = _adapter(monkeypatch)
    adapter._classify_specialist_event = AsyncMock(
        return_value=SpecialistRouteDecision(
            kind=RouteKind.GENERAL,
            reason="ordinary conversation",
            confidence=0.0,
            audit_reason="general",
        )
    )

    assert asyncio.run(adapter._maybe_route_specialist_event(_event("Hello"))) is False
    adapter.send.assert_not_awaited()
