"""Tests for webhook-triggered approvals delivered to interactive platforms."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import SendResult
from gateway.platforms.webhook import WebhookAdapter


class _InteractiveTarget:
    def __init__(self):
        self.approval_mock = AsyncMock(return_value=SendResult(success=True))

    async def send_exec_approval(self, **kwargs):
        return await self.approval_mock(**kwargs)

    async def send(self, chat_id, content, metadata=None):
        return SendResult(success=True)


def _make_adapter() -> WebhookAdapter:
    return WebhookAdapter(
        PlatformConfig(
            enabled=True,
            extra={"host": "127.0.0.1", "port": 0, "routes": {}},
        )
    )


@pytest.mark.asyncio
async def test_forwards_approval_buttons_with_webhook_session_key():
    adapter = _make_adapter()
    target = _InteractiveTarget()
    runner = MagicMock()
    runner.adapters = {Platform.DISCORD: target}
    runner.config.get_home_channel.return_value = None
    adapter.gateway_runner = runner
    webhook_chat_id = "webhook:pr-review:delivery-1"
    adapter._delivery_info[webhook_chat_id] = {
        "deliver": "discord",
        "deliver_extra": {
            "chat_id": "channel-1",
            "thread_id": "thread-9",
        },
    }

    result = await adapter.send_exec_approval(
        chat_id=webhook_chat_id,
        command="git status",
        session_key="agent:main:webhook:pr-review:delivery-1",
        description="test approval",
        allow_permanent=False,
        smart_denied=True,
    )

    assert result.success is True
    target.approval_mock.assert_awaited_once_with(
        chat_id="channel-1",
        command="git status",
        session_key="agent:main:webhook:pr-review:delivery-1",
        description="test approval",
        metadata={"thread_id": "thread-9"},
        allow_permanent=False,
        smart_denied=True,
    )


@pytest.mark.asyncio
async def test_non_interactive_destination_returns_failure_for_text_fallback():
    adapter = _make_adapter()
    adapter._delivery_info["webhook:r:d"] = {"deliver": "log"}

    result = await adapter.send_exec_approval(
        chat_id="webhook:r:d",
        command="git status",
        session_key="agent:main:webhook:r:d",
    )

    assert result.success is False
    assert result.error == "No gateway runner for cross-platform delivery"


@pytest.mark.asyncio
async def test_regular_cross_platform_delivery_still_uses_same_target_resolution():
    adapter = _make_adapter()
    target = _InteractiveTarget()
    target.send = AsyncMock(return_value=SendResult(success=True))
    runner = MagicMock()
    runner.adapters = {Platform.DISCORD: target}
    runner.config.get_home_channel.return_value = None
    adapter.gateway_runner = runner

    result = await adapter._deliver_cross_platform(
        "discord",
        "hello",
        {
            "deliver_extra": {
                "chat_id": "channel-1",
                "thread_id": "thread-9",
            }
        },
    )

    assert result.success is True
    target.send.assert_awaited_once_with(
        "channel-1", "hello", metadata={"thread_id": "thread-9"}
    )
