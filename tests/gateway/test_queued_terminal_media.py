"""Queued text projection preserves explicit native attachment paths."""
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from gateway.config import Platform
from gateway.platforms.base import BasePlatformAdapter, SendResult
from gateway.run import GatewayRunner
from gateway.session import SessionSource


@pytest.mark.asyncio
@pytest.mark.parametrize("name", ["report.png", "sk-abc123def456ghi789jkl012mno345pqr678stu.png"])
async def test_queued_terminal_projection_preserves_attachment_path(tmp_path, monkeypatch, name):
    image = tmp_path / name
    image.write_bytes(b"synthetic")
    monkeypatch.setattr("gateway.platforms.base.MEDIA_DELIVERY_SAFE_ROOTS", (tmp_path,))
    source = SessionSource(platform=Platform.TELEGRAM, chat_id="synthetic-review", chat_type="dm")
    runner = object.__new__(GatewayRunner)
    runner._pop_post_delivery_callback = lambda *args: None
    adapter = SimpleNamespace(
        name="recording", extract_media=BasePlatformAdapter.extract_media,
        extract_images=BasePlatformAdapter.extract_images,
        send=AsyncMock(return_value=SendResult(success=True, message_id="text")),
        send_multiple_images=AsyncMock(),
    )
    turn_ctx = SimpleNamespace(
        source=source, session_key="synthetic-review", stream_consumer_holder=[None],
        _status_thread_metadata={"thread_id": "synthetic"}, event_message_id="input", run_generation=1,
        inbound_message_id=None,
    )
    raw = f"Report attached\nMEDIA:{image}"
    result = {"final_response": raw, "messages": [{"role": "assistant", "content": raw}]}
    await runner._run_agent_deliver_first_response(turn_ctx, adapter, result, result, None)
    adapter.send_multiple_images.assert_awaited_once_with(
        chat_id=source.chat_id, images=[(image.as_uri(), "")], metadata=turn_ctx._status_thread_metadata,
    )
    assert result["final_response"] == raw
    assert result["messages"][0]["content"] == raw
