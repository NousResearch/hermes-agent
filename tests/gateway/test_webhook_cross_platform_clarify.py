from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import Platform
from gateway.platforms.base import SendResult
from gateway.platforms.webhook import WebhookAdapter


@pytest.mark.asyncio
async def test_webhook_clarify_delegates_to_telegram_native_buttons():
    target = MagicMock()
    target.send_clarify = AsyncMock(return_value=SendResult(success=True, message_id="99"))
    runner = MagicMock()
    runner._authorization_adapter.return_value = target

    adapter = object.__new__(WebhookAdapter)
    adapter.gateway_runner = runner
    adapter._delivery_info = {
        "opaque-delivery": {
            "deliver": "telegram",
            "deliver_extra": {"chat_id": "327791380", "message_thread_id": "42"},
        }
    }

    result = await adapter.send_clarify(
        chat_id="opaque-delivery",
        question="Approve?",
        choices=["Approve", "Do not approve"],
        clarify_id="clarify-1",
        session_key="webhook-session",
        metadata={"github_approval": True},
    )

    assert result.success is True
    assert result.message_id == "99"
    runner._authorization_adapter.assert_called_once_with(Platform.TELEGRAM, None)
    target.send_clarify.assert_awaited_once_with(
        chat_id="327791380",
        question="Approve?",
        choices=["Approve", "Do not approve"],
        clarify_id="clarify-1",
        session_key="webhook-session",
        metadata={
            "github_approval": True,
            "thread_id": "42",
            "is_approval_prompt": True,
        },
    )


@pytest.mark.asyncio
async def test_webhook_clarify_without_interactive_destination_fails():
    adapter = object.__new__(WebhookAdapter)
    adapter.gateway_runner = MagicMock()
    adapter._delivery_info = {"opaque-delivery": {"deliver": "log"}}

    result = await adapter.send_clarify(
        chat_id="opaque-delivery", question="Approve?",
        choices=["Approve", "Do not approve"], clarify_id="clarify-2",
        session_key="webhook-session",
    )

    assert result.success is False
    assert "interactive destination" in (result.error or "")
