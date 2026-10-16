"""RPC result variants must all feed the existing timestamp-based echo guard."""
from unittest.mock import AsyncMock

import pytest

from gateway.config import PlatformConfig
from gateway.platforms.signal import SignalAdapter


@pytest.fixture
def adapter(monkeypatch):
    monkeypatch.setenv("SIGNAL_GROUP_ALLOWED_USERS", "abc")
    return SignalAdapter(PlatformConfig(enabled=True, extra={
        "http_url": "http://localhost:8080", "account": "+15550000000",
    }))


@pytest.mark.asyncio
@pytest.mark.parametrize("response", [
    {"timestamp": 1234},
    [{"timestamp": 1234}],
    {"results": [{"timestamp": 1234}]},
    {"result": {"results": [{"timestamp": 1234}]}},
])
async def test_send_then_sync_echo_does_not_dispatch(adapter, response):
    adapter._rpc = AsyncMock(return_value=response)
    adapter._stop_typing_indicator = AsyncMock()
    adapter.handle_message = AsyncMock()
    assert (await adapter.send("group:abc", "hello")).success
    await adapter._handle_envelope({
        "sourceNumber": adapter.account,
        "syncMessage": {"sentMessage": {
            "timestamp": 1234, "message": "hello", "groupInfo": {"groupId": "abc"},
        }},
    })
    adapter.handle_message.assert_not_awaited()
    assert "1234" in adapter._sent_message_timestamps  # quote/reply detection remains available


@pytest.mark.asyncio
async def test_identical_human_reply_in_same_chat_is_not_an_echo(adapter):
    adapter._rpc = AsyncMock(return_value={"results": [{"timestamp": 1234}]})
    adapter._stop_typing_indicator = AsyncMock()
    adapter.handle_message = AsyncMock()
    assert (await adapter.send("group:abc", "yes")).success
    await adapter._handle_envelope({
        "sourceNumber": "+15550000001", "timestamp": 1235,
        "dataMessage": {"message": "yes", "groupInfo": {"groupId": "abc"}},
    })
    adapter.handle_message.assert_awaited_once()


def test_nested_results_record_all_timestamps_and_ignore_empty_values(adapter):
    adapter._track_sent_timestamp({"timestamp": 1, "results": [
        None, {"timestamp": None}, {"timestamp": 2}, {"result": [{"timestamp": 3}]},
    ]})
    assert all(adapter._consume_sent_timestamp(ts) for ts in (1, 2, 3))
    assert not adapter._consume_sent_timestamp(None)
