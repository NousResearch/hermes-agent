from __future__ import annotations

from unittest.mock import patch

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.base import MessageEvent
from gateway.run import GatewayRunner
from gateway.session import SessionSource
from agent.skill_commands import scan_skill_commands


def _make_runner(config: GatewayConfig) -> GatewayRunner:
    runner = object.__new__(GatewayRunner)
    runner.config = config
    runner.adapters = {}
    runner._model = "openai/gpt-4.1-mini"
    runner._base_url = None
    return runner


def _make_skill(skills_dir, name, body="Do the thing."):
    skill_dir = skills_dir / name
    skill_dir.mkdir(parents=True, exist_ok=True)
    (skill_dir / "SKILL.md").write_text(
        f"""---
name: {name}
description: Description for {name}.
---
# {name}

{body}
""",
        encoding="utf-8",
    )
    return skill_dir


class _FakeAdapter:
    def __init__(self):
        self.sent = []

    async def send(self, chat_id, message, metadata=None):
        self.sent.append((chat_id, message, metadata))


def _discord_thread_source(user_name="크리스", user_id="chris-discord-id"):
    return SessionSource(
        platform=Platform.DISCORD,
        chat_id="1486771654809096396",
        chat_name="일반",
        chat_type="thread",
        thread_id="1511031451796373514",
        user_id=user_id,
        user_name=user_name,
    )


@pytest.mark.asyncio
async def test_prepare_inbound_routes_ultragoal_phrase_before_sender_prefix(tmp_path):
    _make_skill(
        tmp_path,
        "kanban-ultragoal-ingress",
        body="Ultragoal ingress skill body.",
    )
    runner = _make_runner(
        GatewayConfig(
            platforms={Platform.DISCORD: PlatformConfig(enabled=True, token="fake", extra={"ultragoal_authorized_user_ids": ["chris-discord-id"]})},
            thread_sessions_per_user=False,
        )
    )
    source = _discord_thread_source()
    event = MessageEvent(text="ULTRAGOAL로 진행해", source=source)

    with patch("tools.skills_tool.SKILLS_DIR", tmp_path):
        scan_skill_commands()
        result = await runner._prepare_inbound_message_text(
            event=event,
            source=source,
            history=[],
        )

    assert result is not None
    assert "kanban-ultragoal-ingress" in result
    assert "ULTRAGOAL로 진행해" in result
    assert "[크리스]" not in result
    assert "Kanban Ultragoal" in result
    assert "direct-kanban" in result
    assert "fail closed" in result.lower()


@pytest.mark.asyncio
async def test_prepare_inbound_ultragoal_phrase_fails_closed_when_ingress_missing(tmp_path):
    runner = _make_runner(
        GatewayConfig(
            platforms={Platform.DISCORD: PlatformConfig(enabled=True, token="fake", extra={"ultragoal_authorized_user_ids": ["chris-discord-id"]})},
            thread_sessions_per_user=False,
        )
    )
    source = _discord_thread_source()
    event = MessageEvent(text="BO-203 ultragoal로 진행", source=source)

    fake_adapter = _FakeAdapter()
    runner.adapters = {Platform.DISCORD: fake_adapter}

    with patch("tools.skills_tool.SKILLS_DIR", tmp_path):
        scan_skill_commands()
        result = await runner._prepare_inbound_message_text(
            event=event,
            source=source,
            history=[],
        )

    assert result is None
    assert len(fake_adapter.sent) == 1
    sent_chat_id, sent_message, _ = fake_adapter.sent[0]
    assert sent_chat_id == source.chat_id
    assert "blocked" in sent_message.lower()
    assert "kanban-ultragoal-ingress" in sent_message
    assert "BO-203" in sent_message
    assert "Autopilot" in sent_message


@pytest.mark.asyncio
async def test_prepare_inbound_does_not_route_ultragoal_discussion(tmp_path):
    runner = _make_runner(
        GatewayConfig(
            platforms={Platform.DISCORD: PlatformConfig(enabled=True, token="fake", extra={"ultragoal_authorized_user_ids": ["chris-discord-id"]})},
            thread_sessions_per_user=False,
        )
    )
    source = _discord_thread_source()
    event = MessageEvent(text="ultragoal 얘기 다시 설명해봐", source=source)

    with patch("tools.skills_tool.SKILLS_DIR", tmp_path):
        result = await runner._prepare_inbound_message_text(
            event=event,
            source=source,
            history=[],
        )

    assert result == "[크리스] ultragoal 얘기 다시 설명해봐"


@pytest.mark.asyncio
async def test_prepare_inbound_does_not_route_negated_ultragoal_operator_text(tmp_path):
    runner = _make_runner(
        GatewayConfig(
            platforms={Platform.DISCORD: PlatformConfig(enabled=True, token="fake", extra={"ultragoal_authorized_user_ids": ["chris-discord-id"]})},
            thread_sessions_per_user=False,
        )
    )
    source = _discord_thread_source()
    event = MessageEvent(text="BO-203 ultragoal로 진행 말고 설명해줘", source=source)

    with patch("tools.skills_tool.SKILLS_DIR", tmp_path):
        result = await runner._prepare_inbound_message_text(
            event=event,
            source=source,
            history=[],
        )

    assert result == "[크리스] BO-203 ultragoal로 진행 말고 설명해줘"


@pytest.mark.asyncio
async def test_prepare_inbound_non_chris_sender_does_not_route_ultragoal(tmp_path):
    runner = _make_runner(
        GatewayConfig(
            platforms={Platform.DISCORD: PlatformConfig(enabled=True, token="fake", extra={"ultragoal_authorized_user_ids": ["chris-discord-id"]})},
            thread_sessions_per_user=False,
        )
    )
    source = _discord_thread_source(user_name="다른사람", user_id="other-discord-id")
    event = MessageEvent(text="ULTRAGOAL로 진행해", source=source)
    fake_adapter = _FakeAdapter()
    runner.adapters = {Platform.DISCORD: fake_adapter}

    with patch("tools.skills_tool.SKILLS_DIR", tmp_path):
        result = await runner._prepare_inbound_message_text(
            event=event,
            source=source,
            history=[],
        )

    assert result is None
    assert len(fake_adapter.sent) == 1
    assert "immutable sender ID" in fake_adapter.sent[0][1]
    assert "Autopilot" in fake_adapter.sent[0][1]
