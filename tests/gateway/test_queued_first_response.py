from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from gateway.config import Platform
from gateway.run import GatewayRunner
from gateway.session import SessionSource


class _Adapter:
    name = "fake"

    def __init__(self) -> None:
        self.send = AsyncMock(return_value=SimpleNamespace(success=True))

    def extract_media(self, response: str):
        lines = response.splitlines()
        cleaned = "\n".join(line for line in lines if not line.startswith("MEDIA:"))
        return [], cleaned


@pytest.mark.asyncio
async def test_queued_first_response_strips_think_blocks_before_direct_send():
    adapter = _Adapter()
    source = SessionSource(platform=Platform.TELEGRAM, chat_id="chat-1")

    await GatewayRunner._deliver_queued_first_response(
        object(),
        "<think>private reasoning</think>\nVisible answer.\nMEDIA:/tmp/result.png",
        source,
        adapter,
        deliver_media=False,
    )

    adapter.send.assert_awaited_once()
    assert adapter.send.await_args.args[:2] == ("chat-1", "Visible answer.")
