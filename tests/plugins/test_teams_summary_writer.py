"""Tests for Teams meeting-summary webhook fan-out."""

from types import SimpleNamespace

import httpx
import pytest

from plugins.platforms.teams.summary_writer import TeamsSummaryWriter


@pytest.mark.anyio
async def test_incoming_webhook_urls_deliver_to_each_endpoint():
    requests = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(str(request.url))
        return httpx.Response(202)

    writer = TeamsSummaryWriter(transport=httpx.MockTransport(handler))
    payload = SimpleNamespace(title="Weekly sync", summary="Summary", key_decisions=[], action_items=[], risks=[])

    result = await writer.write_summary(
        payload,
        {"mode": "incoming_webhook", "incoming_webhook_urls": ["https://one.example/hook", "https://two.example/hook"]},
    )

    assert requests == ["https://one.example/hook", "https://two.example/hook"]
    assert [item["status_code"] for item in result["deliveries"]] == [202, 202]


@pytest.mark.anyio
async def test_singular_incoming_webhook_url_remains_supported():
    requests = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(str(request.url))
        return httpx.Response(200)

    writer = TeamsSummaryWriter(transport=httpx.MockTransport(handler))
    payload = SimpleNamespace(title="Weekly sync", summary="Summary", key_decisions=[], action_items=[], risks=[])

    await writer.write_summary(payload, {"mode": "incoming_webhook", "incoming_webhook_url": "https://one.example/hook"})

    assert requests == ["https://one.example/hook"]
