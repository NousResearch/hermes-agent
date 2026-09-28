import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from gateway.config import Platform
from gateway.delivery import DeliveryTransport
from gateway.delivery_guard import HandoffDeliveryBlocked, guard_pre_delivery
from gateway.platforms.base import BasePlatformAdapter


def _handoff(body: str) -> str:
    return f"```text\n{body}\n```"


def _guard(content: str) -> None:
    guard_pre_delivery(
        platform="telegram",
        content=content,
        target={"chat_id": "fabricated"},
        session_id="handoff-test",
        turn_id="turn-test",
        metadata={"handoff": True},
    )


def test_telegram_handoff_exactly_2000_characters_is_allowed():
    content = _handoff("x" * (2_000 - len(_handoff(""))))

    assert len(content) == 2_000
    _guard(content)


def test_telegram_handoff_over_2000_characters_is_blocked_before_send():
    content = _handoff("x" * (2_001 - len(_handoff(""))))

    with pytest.raises(HandoffDeliveryBlocked, match=r"HANDOFF_DELIVERY_BLOCKED: 2001 > 2000"):
        _guard(content)


@pytest.mark.parametrize("content", [
    "handoff\n```text\nbody\n```",
    "```text\nbody\n```\nmore",
    "```python\nbody\n```",
    "```text\nbody\n```\n```text\nsecond\n```",
])
def test_telegram_handoff_requires_one_text_fence_only(content):
    with pytest.raises(HandoffDeliveryBlocked, match="HANDOFF_DELIVERY_BLOCKED: invalid fenced text block"):
        _guard(content)


def test_non_handoff_delivery_is_unchanged():
    guard_pre_delivery(
        platform="telegram", content="ordinary Telegram response that is not fenced",
        target={"chat_id": "fabricated"}, session_id="ordinary", turn_id="ordinary", metadata={},
    )


def test_handoff_transport_blocks_before_the_adapter_can_send_or_chunk():
    adapter = SimpleNamespace(send=AsyncMock())
    transport = DeliveryTransport(adapter, None, Platform.TELEGRAM)

    with pytest.raises(HandoffDeliveryBlocked):
        asyncio.run(transport.send(Platform.TELEGRAM, "fabricated", _handoff("x" * 2_000), {"handoff": True}))

    adapter.send.assert_not_awaited()


def test_final_delivery_retry_path_blocks_before_adapter_send_or_fallback():
    class Adapter(BasePlatformAdapter):
        async def connect(self, is_reconnect=False):
            raise NotImplementedError

        async def disconnect(self):
            raise NotImplementedError

        async def get_chat_info(self, _chat_id):
            raise NotImplementedError

        async def send(self, *_args, **_kwargs):
            raise NotImplementedError

    adapter = object.__new__(Adapter)
    adapter.platform = Platform.TELEGRAM
    adapter.send = AsyncMock()

    with pytest.raises(HandoffDeliveryBlocked):
        asyncio.run(BasePlatformAdapter._send_with_retry(
            adapter, "fabricated", _handoff("x" * 2_000), metadata={"handoff": True},
        ))

    adapter.send.assert_not_awaited()
