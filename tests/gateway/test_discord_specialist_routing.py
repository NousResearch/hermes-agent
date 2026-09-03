from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import MessageEvent, MessageType
from gateway.session import SessionSource


@pytest.fixture
def adapter(monkeypatch):
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
                    "profiles": {
                        "task-orchestrator": "broad coordinated work",
                        "patch-steward": "narrow corrective patches",
                    },
                }
            },
        )
    )
    value._client = SimpleNamespace(user=SimpleNamespace(id=999))
    value.send = AsyncMock()
    return value


def _event(text="Patch the confirmed failure"):
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


def test_settings_keep_only_bounded_explicit_profile_descriptions(adapter):
    settings = adapter._specialist_routing_settings()

    assert settings["enabled"] is True
    assert settings["profiles"] == {
        "task-orchestrator": "broad coordinated work",
        "patch-steward": "narrow corrective patches",
    }


def test_empty_profile_map_disables_routing(adapter):
    adapter.config.extra["specialist_routing"]["profiles"] = {}

    assert adapter._specialist_routing_settings() == {"enabled": False}


def test_config_bridges_discord_specialist_routing(monkeypatch, tmp_path):
    from gateway.config import load_gateway_config

    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir()
    (hermes_home / "config.yaml").write_text(
        "discord:\n"
        "  specialist_routing:\n"
        "    enabled: true\n"
        "    board: project-maintenance\n"
        "    profiles:\n"
        "      patch-steward: narrow corrective patches\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    config = load_gateway_config()

    assert config.platforms[Platform.DISCORD].extra["specialist_routing"][
        "profiles"
    ] == {"patch-steward": "narrow corrective patches"}


def test_specialist_route_creates_one_handoff_and_acknowledges(adapter, monkeypatch):
    from gateway.specialist_handoff import HandoffResult
    from gateway.specialist_routing import RouteKind, SpecialistRouteDecision

    adapter._classify_specialist_event = AsyncMock(
        return_value=SpecialistRouteDecision(
            kind=RouteKind.SPECIALIST,
            profile="patch-steward",
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
        content="Planning `t_abc` with `patch-steward`.",
        reply_to="message-1",
    )


def test_general_route_preserves_normal_chat_path(adapter):
    from gateway.specialist_routing import RouteKind, SpecialistRouteDecision

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
