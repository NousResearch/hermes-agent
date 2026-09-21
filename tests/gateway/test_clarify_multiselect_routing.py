"""Gateway render-path contracts for clarify multi-select prompts."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from gateway.platforms.base import BasePlatformAdapter, SendResult
from gateway.run import _send_gateway_clarify
from tools import clarify_gateway


class _Adapter:
    def __init__(self) -> None:
        self.calls: list[str] = []

    async def send_clarify(self, **kwargs):
        self.calls.append("native")
        return SimpleNamespace(success=True, message_id="native")

    async def send_clarify_text_fallback(self, **kwargs):
        self.calls.append("text")
        return SimpleNamespace(success=True, message_id="text")


class _PortableAdapter(BasePlatformAdapter):
    def __init__(self) -> None:
        self.calls: list[str] = []
        self.content = ""

    async def connect(self, *, is_reconnect: bool = False) -> bool:
        return True

    async def disconnect(self) -> None:
        return None

    async def send(
        self,
        chat_id: str,
        content: str,
        reply_to: str | None = None,
        metadata: dict | None = None,
    ) -> SendResult:
        self.content = content
        return SendResult(success=True, message_id="text")

    async def get_chat_info(self, chat_id: str) -> dict:
        return {}


async def _send(
    adapter: _Adapter | BasePlatformAdapter,
    *,
    multi_select: bool,
):
    return await _send_gateway_clarify(
        adapter,
        chat_id="chat",
        question="Pick",
        choices=["A", "B"],
        clarify_id="clarify",
        session_key="session",
        metadata={},
        multi_select=multi_select,
    )


@pytest.mark.asyncio
async def test_single_select_keeps_native_adapter_controls() -> None:
    adapter = _Adapter()

    result = await _send(adapter, multi_select=False)

    assert result.message_id == "native"
    assert adapter.calls == ["native"]


@pytest.mark.asyncio
async def test_multi_select_uses_portable_text_fallback() -> None:
    """Native one-shot buttons cannot represent a checkbox answer set."""
    adapter = _Adapter()

    result = await _send(adapter, multi_select=True)

    assert result.message_id == "text"
    assert adapter.calls == ["text"]


@pytest.mark.asyncio
async def test_portable_fallback_explains_multi_select_reply_format() -> None:
    adapter = _PortableAdapter()
    clarify_gateway.register(
        clarify_id="clarify",
        session_key="session",
        question="Pick",
        choices=["A", "B"],
        multi_select=True,
    )

    try:
        await _send(adapter, multi_select=True)
    finally:
        clarify_gateway.clear_session("session")

    assert "Multiple selections allowed" in adapter.content
    assert 'e.g. "1, 3"' in adapter.content
