"""Gateway status/progress sends should not trigger WhatsApp human cascade."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from gateway.run import _send_or_update_status_coro


class _SendOnlyAdapter:
    def __init__(self):
        self.send = AsyncMock(return_value=SimpleNamespace(success=True, message_id="m1"))


class _StatusAdapter(_SendOnlyAdapter):
    def __init__(self):
        super().__init__()
        self.send_or_update_status = AsyncMock(return_value=SimpleNamespace(success=True, message_id="m1"))


@pytest.mark.asyncio
async def test_status_fallback_send_forces_single_delivery_style():
    adapter = _SendOnlyAdapter()

    await _send_or_update_status_coro(
        adapter,
        "chat-1",
        "thinking",
        "thinking\n\nstill checking",
        {"thread_id": "thread-1"},
    )

    metadata = adapter.send.call_args.kwargs["metadata"]
    assert metadata["thread_id"] == "thread-1"
    assert metadata["delivery_style"] == "single"


@pytest.mark.asyncio
async def test_status_update_adapter_forces_single_delivery_style():
    adapter = _StatusAdapter()

    await _send_or_update_status_coro(
        adapter,
        "chat-1",
        "thinking",
        "thinking\n\nstill checking",
        None,
    )

    metadata = adapter.send_or_update_status.call_args.kwargs["metadata"]
    assert metadata["delivery_style"] == "single"
