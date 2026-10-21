"""Explicit webhook cards reach Slack without changing ordinary delivery or retries."""

from unittest.mock import AsyncMock, MagicMock

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from gateway.config import PlatformConfig
from gateway.platforms.webhook import WebhookAdapter, _INSECURE_NO_AUTH
from plugins.platforms.slack.adapter import SlackAdapter


def _slack():
    adapter = SlackAdapter(PlatformConfig(enabled=True, token="xoxb-fake"))
    client = AsyncMock()
    client.chat_postMessage.return_value = {"ts": "111.222"}
    adapter._app = MagicMock()
    adapter._get_client = MagicMock(return_value=client)
    adapter.stop_typing = AsyncMock()
    return adapter, client


@pytest.mark.asyncio
@pytest.mark.parametrize("source", ["template", "literal", "ordinary", "missing", "malformed", "long"])
async def test_http_delivery_preserves_explicit_cards_and_plain_text_fallback(source):
    slack, transport = _slack()
    # More than the prompt renderer's 2000-character structured-value limit.
    blocks = [{"type": "section", "text": {"type": "plain_text", "text": "card " * 500}}]
    content = "Notification body" if source != "long" else "x" * (slack.MAX_MESSAGE_LENGTH + 100)
    extra = {"chat_id": "C_CARD", "thread_id": "123.456"}
    if source != "ordinary":
        extra["slack_blocks"] = blocks if source == "literal" else "{card.blocks}"
    payload = {"text": content, "card": {"blocks": blocks}, "slack_blocks": blocks}
    if source == "missing":
        payload.pop("card")
    elif source == "malformed":
        payload["card"]["blocks"] = {"type": "section"}
    webhook = WebhookAdapter(PlatformConfig(enabled=True, extra={
        "host": "127.0.0.1", "port": 0, "routes": {"card": {
            "secret": _INSECURE_NO_AUTH, "deliver": "slack", "deliver_only": True,
            "prompt": "{text}", "deliver_extra": extra}}}))
    webhook.handle_message = AsyncMock()
    runner = MagicMock()
    runner._authorization_adapter.return_value = slack
    webhook.gateway_runner = runner
    app = web.Application()
    app.router.add_post("/webhooks/{route_name}", webhook._handle_webhook)

    async with TestClient(TestServer(app)) as client:
        response = await client.post("/webhooks/card", json=payload)
        assert response.status == 200
        assert (await response.json())["status"] == "delivered"

    webhook.handle_message.assert_not_awaited()
    calls = [call.kwargs for call in transport.chat_postMessage.await_args_list]
    assert [call["text"] for call in calls] == slack.truncate_message(
        slack.format_message(content), slack.MAX_MESSAGE_LENGTH)
    assert all(call["channel"] == "C_CARD" and call["thread_ts"] == "123.456" for call in calls)
    if source in {"template", "literal"}:
        assert len(calls) == 1
        assert calls[0]["blocks"] == blocks
    else:
        assert all("blocks" not in call for call in calls)


class _RejectedBlocks(Exception):
    response = {"error": "invalid_blocks"}


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["invalid_blocks", "timeout"])
async def test_explicit_cards_only_retry_rejections_without_changing_text_or_thread(failure):
    slack, transport = _slack()
    blocks = [{"type": "section", "text": {"type": "plain_text", "text": "Card"}}]
    error = _RejectedBlocks("invalid_blocks") if failure == "invalid_blocks" else TimeoutError("timed out")
    transport.chat_postMessage.side_effect = [error, {"ts": "111.222"}]

    result = await slack.send("C_CARD", "Notification body", metadata={
        "thread_id": "123.456", "team_id": "T_CARD", "blocks": blocks})

    calls = [call.kwargs for call in transport.chat_postMessage.await_args_list]
    assert calls[0]["blocks"] == blocks
    assert calls[0]["text"] == "Notification body"
    assert calls[0]["thread_ts"] == "123.456"
    assert slack._get_client.call_args.kwargs == {"team_id": "T_CARD"}
    if failure == "invalid_blocks":
        assert result.success
        assert len(calls) == 2
        assert calls[1] == {key: value for key, value in calls[0].items() if key != "blocks"}
    else:
        assert not result.success
        assert len(calls) == 1
