"""Regression tests for visible Discord /model picker current-state content."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from gateway.config import PlatformConfig
from plugins.platforms.discord.adapter import DiscordAdapter


def _capture_channel(adapter):
    sent = {}

    async def fake_send(**kwargs):
        sent.update(kwargs)
        return SimpleNamespace(id=1234)

    channel = SimpleNamespace(send=AsyncMock(side_effect=fake_send))
    adapter._client = SimpleNamespace(
        get_channel=lambda _chat_id: channel,
        fetch_channel=AsyncMock(),
    )
    return sent


@pytest.mark.asyncio
async def test_model_picker_sends_current_model_in_plain_content():
    adapter = DiscordAdapter(PlatformConfig(enabled=True, token="***"))
    sent = _capture_channel(adapter)

    result = await adapter.send_model_picker(
        chat_id="555",
        providers=[
            {
                "slug": "openrouter",
                "name": "OpenRouter",
                "models": ["openai/gpt-5.5"],
                "total_models": 1,
                "is_current": True,
            }
        ],
        current_model="openai/gpt-5.5",
        current_provider="openrouter",
        session_key="discord:555",
        on_model_selected=AsyncMock(return_value="switched"),
    )

    assert result.success is True
    assert sent["view"] is not None
    assert sent["embed"] is not None

    content = sent["content"]
    assert "Model Configuration" in content
    assert "Current model: `openai/gpt-5.5`" in content
    assert "Provider: OpenRouter" in content
    assert "Select a provider" in content
