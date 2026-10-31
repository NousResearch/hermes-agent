from unittest.mock import AsyncMock

import pytest

from gateway.platforms.base import SendResult
from plugins.platforms.photon.adapter import PhotonAdapter


@pytest.mark.asyncio
async def test_send_splits_long_messages_without_losing_content():
    adapter = object.__new__(PhotonAdapter)
    adapter.format_message = lambda content: content
    sent = 0

    async def send_chunk(_chat_id, _chunk):
        nonlocal sent
        sent += 1
        return SendResult(success=True, message_id=f"message-{sent}")

    adapter._sidecar_send = AsyncMock(side_effect=send_chunk)
    content = ("x" * 7000) + "\n" + ("y" * 7000)

    result = await adapter.send("space", content)

    assert result.success is True
    assert result.message_id == "message-2"
    assert result.continuation_message_ids == ("message-1",)
    assert adapter._sidecar_send.await_count == 2
    chunks = [call.args[1] for call in adapter._sidecar_send.await_args_list]
    assert all(len(chunk) <= PhotonAdapter.MAX_MESSAGE_LENGTH for chunk in chunks)
    assert chunks[0].startswith("x" * 100)
    assert chunks[1].removesuffix(" (2/2)").endswith("y" * 100)


def test_photon_declares_native_long_message_splitting():
    assert PhotonAdapter.splits_long_messages is True
