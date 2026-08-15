"""Tests for config-driven Feishu application quick launchers."""

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import MessageEvent, MessageType
from gateway.session import SessionSource
from plugins.platforms.feishu.adapter import FeishuAdapter


def _launcher_config() -> PlatformConfig:
    return PlatformConfig(
        enabled=True,
        extra={
            "quick_launchers": [
                {
                    "name": "open reports",
                    "title": "Reports",
                    "description": "Open the reporting application.",
                    "button_text": "Open reports",
                    "template": "green",
                    "url": "https://example.com/reports",
                    "keywords": ["reports"],
                }
            ]
        },
    )


def _event(text: str) -> MessageEvent:
    return MessageEvent(
        text=text,
        message_type=MessageType.TEXT,
        source=SessionSource(
            platform=Platform.FEISHU,
            chat_id="oc_test",
            chat_name="Test chat",
            chat_type="dm",
            user_id="ou_user",
            user_name="User",
        ),
        message_id="om_trigger",
    )


def test_quick_launcher_matches_only_exact_configured_commands():
    adapter = FeishuAdapter(_launcher_config())

    assert adapter._match_quick_launcher("open reports") is not None
    assert adapter._match_quick_launcher(" /reports. ") is not None
    assert adapter._match_quick_launcher("please analyze the reports") is None


def test_quick_launcher_rejects_non_https_url():
    config = _launcher_config()
    config.extra["quick_launchers"][0]["url"] = "javascript:alert(1)"
    adapter = FeishuAdapter(config)

    assert adapter._match_quick_launcher("reports") is None


@pytest.mark.asyncio
async def test_quick_launcher_sends_native_url_card():
    adapter = FeishuAdapter(_launcher_config())
    adapter._client = MagicMock()
    response = SimpleNamespace(
        success=lambda: True,
        data=SimpleNamespace(message_id="om_launcher"),
    )

    with patch.object(
        adapter,
        "_feishu_send_with_retry",
        new_callable=AsyncMock,
        return_value=response,
    ) as send:
        result = await adapter._send_quick_launcher_card(
            "oc_test", adapter._match_quick_launcher("reports")
        )

    assert result.success is True
    kwargs = send.call_args.kwargs
    assert kwargs["msg_type"] == "interactive"
    card = json.loads(kwargs["payload"])
    button = card["elements"][1]["actions"][0]
    assert button["url"] == "https://example.com/reports"
    assert button["text"]["content"] == "Open reports"


@pytest.mark.asyncio
async def test_matching_launcher_bypasses_agent_but_other_text_does_not():
    adapter = FeishuAdapter(_launcher_config())
    adapter.handle_message = AsyncMock()
    adapter._send_quick_launcher_card = AsyncMock(
        return_value=SimpleNamespace(success=True, error=None)
    )

    await adapter._handle_message_with_guards(_event("reports"))
    adapter._send_quick_launcher_card.assert_awaited_once()
    adapter.handle_message.assert_not_awaited()

    adapter._send_quick_launcher_card.reset_mock()
    await adapter._handle_message_with_guards(_event("analyze expenses"))
    adapter._send_quick_launcher_card.assert_not_awaited()
    adapter.handle_message.assert_awaited_once()
