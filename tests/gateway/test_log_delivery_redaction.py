"""An explicitly selected log destination keeps its response and masks secrets before clipping."""

import logging

import pytest

from gateway.config import PlatformConfig


@pytest.mark.asyncio
@pytest.mark.parametrize("platform", ["webhook", "msgraph_webhook"])
async def test_selected_log_delivery_keeps_response_and_masks_before_clipping(caplog, monkeypatch, platform):
    from gateway.platforms.msgraph_webhook import MSGraphWebhookAdapter
    from gateway.platforms.webhook import WebhookAdapter

    monkeypatch.setenv("HERMES_REDACT_SECRETS", "false")
    secret = "sk-proj-" + "syntheticcredential" * 8
    content = "Useful response. " * 10 + " api_key=" + secret
    adapter = (WebhookAdapter if platform == "webhook" else MSGraphWebhookAdapter)(PlatformConfig(enabled=True))
    with caplog.at_level(logging.INFO, logger=f"gateway.platforms.{platform}"):
        result = await adapter.send("private-chat-identity", content)

    assert result.success
    assert "Useful response." in caplog.text
    assert "syntheticcred" not in caplog.text
    assert "private-chat-identity" not in caplog.text
    assert content.endswith(secret)
