"""Runtime seam tests for the extracted webhook delivery mixin."""

import asyncio
from unittest.mock import AsyncMock

import pytest

from gateway.config import PlatformConfig
from gateway.platforms.base import SendResult
from gateway.platforms.webhook import WebhookAdapter
from gateway.platforms.webhook_delivery import WebhookDeliveryMixin


def _adapter() -> WebhookAdapter:
    return WebhookAdapter(
        PlatformConfig(
            enabled=True,
            extra={"host": "127.0.0.1", "port": 0, "routes": {}},
        )
    )


def test_delivery_methods_resolve_through_adapter_mro():
    assert WebhookAdapter.__mro__[1] is WebhookDeliveryMixin
    assert WebhookAdapter._direct_deliver is WebhookDeliveryMixin._direct_deliver
    assert (
        WebhookAdapter._deliver_github_comment
        is WebhookDeliveryMixin._deliver_github_comment
    )
    assert (
        WebhookAdapter._deliver_cross_platform
        is WebhookDeliveryMixin._deliver_cross_platform
    )


@pytest.mark.asyncio
async def test_direct_log_delivery_succeeds_without_invoking_agent():
    adapter = _adapter()
    adapter.handle_message = AsyncMock()

    result = await adapter._direct_deliver("hello", {"deliver": "log"})

    assert isinstance(result, SendResult)
    assert result.success is True
    adapter.handle_message.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("delivery", "error"),
    [
        ({"deliver_extra": {}}, "Missing repo or pr_number"),
        (
            {"deliver_extra": {"repo": "owner/repo", "pr_number": "abc"}},
            "Invalid pr_number",
        ),
        (
            {"deliver_extra": {"repo": "bad repo", "pr_number": "1"}},
            "Invalid repo format",
        ),
    ],
)
async def test_github_comment_validation_errors(delivery, error):
    result = await _adapter()._deliver_github_comment("hello", delivery)

    assert result.success is False
    assert result.error == error


@pytest.mark.asyncio
async def test_cross_platform_without_gateway_runner_fails():
    result = await _adapter()._deliver_cross_platform(
        "telegram", "hello", {"deliver": "telegram"}
    )

    assert result.success is False
    assert result.error == "No gateway runner for cross-platform delivery"


@pytest.mark.asyncio
async def test_original_adapter_monkeypatch_affects_instances(monkeypatch):
    sentinel = object()

    async def replacement(self, content, delivery):
        return sentinel

    monkeypatch.setattr(WebhookAdapter, "_direct_deliver", replacement)

    assert WebhookAdapter._direct_deliver is replacement
    assert await _adapter()._direct_deliver("hello", {}) is sentinel
