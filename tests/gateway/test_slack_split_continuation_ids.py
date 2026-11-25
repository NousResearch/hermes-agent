"""A split Slack reply reports every chunk, like the other split-sending adapters."""
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import PlatformConfig
from plugins.platforms.slack.adapter import SlackAdapter


@pytest.fixture()
def adapter():
    a = SlackAdapter(PlatformConfig(enabled=True, token="***"))
    a._app = MagicMock()
    a._app.client = AsyncMock()
    a._bot_user_id = "U_BOT"
    a._running = True
    a.handle_message = AsyncMock()
    return a


def _post_returning(*timestamps):
    return AsyncMock(side_effect=[{"ok": True, "ts": ts} for ts in timestamps])


@pytest.mark.asyncio
async def test_split_send_reports_earlier_chunks_as_continuations(adapter):
    adapter._app.client.chat_postMessage = _post_returning("1.001", "1.002", "1.003")

    result = await adapter.send("C123", "x" * (adapter.MAX_MESSAGE_LENGTH * 2 + 500))

    assert adapter._app.client.chat_postMessage.call_count == 3
    assert result.success
    assert result.message_id == "1.003"
    assert result.continuation_message_ids == ("1.001", "1.002")


@pytest.mark.asyncio
async def test_single_chunk_send_has_no_continuations(adapter):
    adapter._app.client.chat_postMessage = _post_returning("2.001")

    result = await adapter.send("C123", "short reply")

    assert result.message_id == "2.001"
    assert result.continuation_message_ids == ()


@pytest.mark.asyncio
async def test_every_chunk_is_tracked_as_a_bot_message(adapter):
    adapter._app.client.chat_postMessage = _post_returning("3.001", "3.002")

    await adapter.send("C123", "y" * (adapter.MAX_MESSAGE_LENGTH + 500))

    for ts in ("3.001", "3.002"):
        assert adapter._workspace_message_marker(None, ts) in adapter._bot_message_ts


@pytest.mark.asyncio
async def test_chunk_without_ts_is_skipped(adapter):
    adapter._app.client.chat_postMessage = AsyncMock(
        side_effect=[{"ok": True}, {"ok": True, "ts": "4.002"}])

    result = await adapter.send("C123", "z" * (adapter.MAX_MESSAGE_LENGTH + 500))

    assert result.message_id == "4.002"
    assert result.continuation_message_ids == ()
