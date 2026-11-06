"""Regression: the Telegram adapter seals an orphaned draft when a turn dies.

``GatewayStreamConsumerTransport._abandon_native_stream`` guards the seal with
``getattr(type(adapter), "abandon_open_draft", None)`` and returns silently when the
adapter does not implement it. ``RelayAdapter`` did; the bundled Telegram adapter did
not, so a turn that died mid-stream (stale-generation exit, cancellation, or an
interrupt before any final send) left the ``sendMessageDraft`` preview armed: the
streaming bubble kept looking live and the user got no final text and no error.

Telegram exposes no "cancel draft" API, so the only way to stop the indicator is to
persist the last delivered frame as a real message — which is what the method does,
with ``notify`` stamped so the normal post-send typing re-trigger is skipped (#48678).
"""

from typing import Any, Optional

import pytest

from gateway.config import Platform
from gateway.platforms.base import SendResult
from gateway.stream_consumer import GatewayStreamConsumer, StreamConsumerConfig
from plugins.platforms.telegram.adapter import TelegramAdapter


class _StubTelegram(TelegramAdapter):
    """The real adapter class with its connect machinery bypassed and its send recorded."""

    def __init__(self):
        self.platform = Platform.TELEGRAM
        # not connected: send_draft / supports_draft_streaming stay inert
        self._bot = None
        self.sent: list[dict[str, Any]] = []

    async def send(
        self,
        chat_id: str,
        content: str,
        reply_to: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> SendResult:
        self.sent.append({"chat_id": chat_id, "content": content, "metadata": metadata})
        return SendResult(success=True, message_id="seal-1")


def _consumer(adapter: _StubTelegram) -> GatewayStreamConsumer:
    sc = GatewayStreamConsumer(
        adapter,
        "chat-1",
        StreamConsumerConfig(transport="auto", chat_type="dm", cursor=" ▉"),
    )
    # The state a dying draft-path turn leaves behind (a draft has no message_id).
    sc._use_draft_streaming = True
    sc._last_sent_text = "partial answer on screen"
    sc._accumulated = "partial answer on screen"
    return sc


def test_adapter_satisfies_the_transport_guard():
    """The exact expression at stream_consumer_transport.py must no longer short-circuit."""
    assert getattr(TelegramAdapter, "abandon_open_draft", None) is not None


@pytest.mark.asyncio
async def test_dying_turn_seals_the_draft_as_a_real_message():
    adapter = _StubTelegram()
    sc = _consumer(adapter)

    await sc._abandon_native_stream()

    assert len(adapter.sent) == 1, "an orphaned draft must be sealed exactly once"
    seal = adapter.sent[0]
    assert seal["chat_id"] == "chat-1"
    assert seal["content"] == "partial answer on screen"
    assert (seal.get("metadata") or {}).get("notify") is True, (
        "seal must not re-trigger typing"
    )


@pytest.mark.asyncio
async def test_sealing_claims_no_delivery():
    """Sealing is not delivery: the final-response flag must stay clear (same as relay)."""
    adapter = _StubTelegram()
    sc = _consumer(adapter)

    await sc._abandon_native_stream()

    assert sc.final_response_sent is False


@pytest.mark.asyncio
async def test_empty_frame_is_a_noop():
    """Nothing on screen to persist — the server-side preview expires on its own."""
    adapter = _StubTelegram()
    sc = _consumer(adapter)
    sc._last_sent_text = ""
    sc._accumulated = ""

    await sc._abandon_native_stream()

    assert adapter.sent == []


@pytest.mark.asyncio
async def test_seal_never_raises_when_the_transport_fails():
    """Best-effort contract: a dead socket while sealing must not fail the dying turn."""

    class _Broken(_StubTelegram):
        async def send(
            self,
            chat_id: str,
            content: str,
            reply_to: Optional[str] = None,
            metadata: Optional[dict[str, Any]] = None,
        ) -> SendResult:
            raise RuntimeError("network down")

    sc = _consumer(_Broken())

    await sc._abandon_native_stream()  # must not raise
