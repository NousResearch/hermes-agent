"""Model-authored outbound text must reach the adapter without classification."""

from __future__ import annotations

import pytest

from gateway.config import GatewayConfig, Platform
from gateway.delivery import DeliveryRouter, DeliveryTarget


class RecordingAdapter:
    def __init__(self):
        self.calls = []

    async def send(self, chat_id, content, metadata=None):
        self.calls.append(
            {"chat_id": chat_id, "content": content, "metadata": metadata}
        )
        return {"success": True, "message_id": "receipt-1"}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "content",
    [
        "*(silent)*",
        "🔇",
        ".",
        "…",
        "no response",
        "No Reply.",
        "Silence is golden — here is the plan...",
    ],
)
async def test_authored_text_is_delivered_without_keyword_filter(
    content, tmp_path, monkeypatch
):
    monkeypatch.setattr("gateway.delivery.get_hermes_home", lambda: tmp_path)
    adapter = RecordingAdapter()
    router = DeliveryRouter(GatewayConfig(), adapters={Platform.DISCORD: adapter})
    target = DeliveryTarget.parse("discord:99887766")

    result = await router._deliver_to_platform(target, content, metadata=None)

    assert adapter.calls == [
        {"chat_id": "99887766", "content": content, "metadata": None}
    ]
    assert result == {"success": True, "message_id": "receipt-1"}


def test_gateway_config_has_no_semantic_delivery_filter():
    config = GatewayConfig.from_dict({"filter_silence_narration": True})

    assert not hasattr(config, "filter_silence_narration")
    assert "filter_silence_narration" not in config.to_dict()
