"""Proposed pytest coverage for Discord multipart cron delivery.

These tests target the API-compatible behavior proposed in
/tmp/cron-multipart-bug-fix.patch. They intentionally use small fakes instead
of a live Discord connection.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import pytest

HERMES_ROOT = Path("/usr/local/lib/hermes-agent")
if str(HERMES_ROOT) not in sys.path:
    sys.path.insert(0, str(HERMES_ROOT))

from gateway.config import PlatformConfig  # noqa: E402
from gateway.platforms.discord import DiscordAdapter  # noqa: E402


class FakeMessage:
    def __init__(self, message_id: int):
        self.id = message_id


class FakeChannel:
    def __init__(self, *, fail_on_call: int | None = None):
        self.fail_on_call = fail_on_call
        self.sent: list[dict] = []
        self.type = 0  # not a forum channel

    async def send(self, **kwargs):
        call_number = len(self.sent) + 1
        if self.fail_on_call == call_number:
            raise RuntimeError(f"simulated Discord send failure on call {call_number}")
        self.sent.append(kwargs)
        return FakeMessage(10_000 + call_number)


class FakeClient:
    def __init__(self, channel: FakeChannel):
        self.channel = channel

    def get_channel(self, channel_id: int):
        return self.channel

    async def fetch_channel(self, channel_id: int):  # pragma: no cover - fallback only
        return self.channel


def make_adapter(channel: FakeChannel) -> DiscordAdapter:
    adapter = DiscordAdapter(PlatformConfig(enabled=True, token="fake-token"))
    adapter._client = FakeClient(channel)
    return adapter


@pytest.mark.asyncio
async def test_short_message_sends_once_and_succeeds():
    channel = FakeChannel()
    adapter = make_adapter(channel)

    result = await adapter.send("123", "short cron response")

    assert result.success is True
    assert len(channel.sent) == 1
    assert result.error is None
    assert result.raw_response["message_ids"] == ["10001"]
    assert result.raw_response["total_parts"] == 1
    assert result.raw_response["succeeded_parts"] == 1
    assert result.raw_response["failed_parts"] == 0


@pytest.mark.asyncio
async def test_long_message_sends_multiple_parts_and_reports_all_success():
    channel = FakeChannel()
    adapter = make_adapter(channel)
    long_message = "A" * 4300

    result = await adapter.send("123", long_message)

    assert result.success is True
    assert len(channel.sent) > 1
    assert result.error is None
    assert result.raw_response["total_parts"] == len(channel.sent)
    assert result.raw_response["succeeded_parts"] == len(channel.sent)
    assert result.raw_response["failed_parts"] == 0
    assert len(result.raw_response["message_ids"]) == len(channel.sent)


@pytest.mark.asyncio
async def test_long_message_partial_failure_is_reported_as_failed():
    channel = FakeChannel(fail_on_call=2)
    adapter = make_adapter(channel)
    long_message = "B" * 4300

    result = await adapter.send("123", long_message)

    assert result.success is False
    assert result.raw_response["total_parts"] > 1
    assert result.raw_response["succeeded_parts"] < result.raw_response["total_parts"]
    assert result.raw_response["failed_parts"] >= 1
    assert "PARTIAL delivery" in result.error
    assert "2/" in result.raw_response["warnings"][0]
    assert "simulated Discord send failure" in result.raw_response["warnings"][0]
