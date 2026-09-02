"""Regression coverage for #98552 — a truncated preview must not ACK as delivered.

Telegram refuses to split an oversized *progressive* edit: splitting mid-stream
moves the editable message id and the next accumulated-token edit re-splits,
producing the #48648 duplication loop.  Instead the adapter clips the payload
to one message and edits with the clipped text.  That call succeeded, so it
returned a bare ``SendResult(success=True)`` — and the stream consumer advanced
``_last_sent_text`` to the text it *sent*, not the shorter text Telegram
*stored*.

Everything the consumer knows about "what the user has already seen" is derived
from that field:

* ``_visible_prefix()`` — the on-screen text;
* ``_continuation_text()`` — the tail a fallback send still owes the user;
* ``_mark_skip_redundant_finalize()`` / the failed-final-edit guard — the
  turn-final payload the gateway reconciles against ``final_response``.

With an over-long ``_last_sent_text`` all three describe text no API call ever
carried: the fallback skips the un-stored middle, and
``delivered_final_matches`` reports a match, so ``gateway/run.py`` suppresses
its corrective send ("final delivery already confirmed ... content_delivered=
True") over a frozen, clipped preview.

These tests pin the contract at both layers: the adapter must report what it
actually stored, and the consumer must believe the report over its own request.
"""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import BasePlatformAdapter, SendResult
from gateway.stream_consumer import GatewayStreamConsumer, StreamConsumerConfig
from plugins.platforms.telegram.adapter import TelegramAdapter

CURSOR = " ▉"
CAP = 4096


# ---------------------------------------------------------------------------
# Adapter layer — the truncation must be reported, not hidden behind success
# ---------------------------------------------------------------------------


def _telegram_adapter():
    adapter = TelegramAdapter(PlatformConfig(enabled=True, token="fake-token"))
    adapter._bot = MagicMock()
    adapter._bot.edit_message_text = AsyncMock()
    adapter._bot.send_message = AsyncMock()
    return adapter


@pytest.mark.asyncio
async def test_saturated_preview_edit_reports_what_telegram_stored():
    """The first oversized progressive edit clips; the result must say so."""
    adapter = _telegram_adapter()

    result = await adapter.edit_message("123", "456", "x" * 6000, finalize=False)

    assert result.success is True
    assert result.message_id == "456"
    stored = adapter._bot.edit_message_text.call_args.kwargs["text"]
    assert len(stored) <= CAP
    assert isinstance(result.raw_response, dict)
    assert result.raw_response.get("stream_preview_truncated") is True
    assert result.raw_response.get("delivered_prefix") == stored


@pytest.mark.asyncio
async def test_deduped_saturated_preview_still_reports_the_stored_text():
    """The no-API-call dedup path must not look like a full delivery either."""
    adapter = _telegram_adapter()

    first = await adapter.edit_message("123", "456", "x" * 6000, finalize=False)
    second = await adapter.edit_message("123", "456", "x" * 7000, finalize=False)

    # Dedup held: the growing stream truncates to the same preview.
    assert adapter._bot.edit_message_text.await_count == 1
    assert second.success is True
    assert second.raw_response.get("delivered_prefix") == first.raw_response.get(
        "delivered_prefix"
    )


@pytest.mark.asyncio
async def test_under_cap_edit_reports_no_truncation():
    """Ordinary edits keep the bare success contract (no false positives)."""
    adapter = _telegram_adapter()

    result = await adapter.edit_message("123", "456", "short reply", finalize=False)

    assert result.success is True
    assert not (
        isinstance(result.raw_response, dict)
        and result.raw_response.get("delivered_prefix")
    )


# ---------------------------------------------------------------------------
# Consumer layer — believe the adapter's report, not our own request
# ---------------------------------------------------------------------------


class ClippingAdapter(BasePlatformAdapter):
    """Telegram-shaped double: progressive edits are clipped to ``CAP``.

    Records only what a user could actually see, so the assertions below
    compare the consumer's claims against the rendered screen rather than
    against the calls it made.
    """

    REQUIRES_EDIT_FINALIZE = True
    FALLBACK_ON_FINAL_EDIT_FLOOD = True
    MAX_MESSAGE_LENGTH = CAP

    def __init__(self, *, fail_finalize_edit: bool = False):
        super().__init__(PlatformConfig(enabled=True, token="***"), Platform.TELEGRAM)
        self._fail_finalize_edit = fail_finalize_edit
        self.screen = {}
        self.order = []
        self._next_id = 0

    def prefers_fresh_final_streaming(self, content, metadata=None) -> bool:
        return False

    async def connect(self, *, is_reconnect: bool = False) -> bool:
        return True

    async def disconnect(self) -> None:
        return None

    async def get_chat_info(self, chat_id):
        return {}

    async def send_typing(self, chat_id, metadata=None) -> None:
        return None

    async def send(self, chat_id, content, reply_to=None, metadata=None) -> SendResult:
        self._next_id += 1
        mid = f"m{self._next_id}"
        # Telegram's send() chunks oversized content rather than clipping it,
        # so a fresh send always lands whole; only progressive EDITS clip.
        self.screen[mid] = content
        self.order.append(mid)
        return SendResult(success=True, message_id=mid)

    async def edit_message(
        self, chat_id, message_id, content, *, finalize: bool = False, metadata=None
    ) -> SendResult:
        if finalize:
            if self._fail_finalize_edit:
                return SendResult(
                    success=False, error="Flood control exceeded. Retry in 30 seconds"
                )
            self.screen[message_id] = content
            return SendResult(success=True, message_id=message_id)
        if len(content) <= CAP:
            self.screen[message_id] = content
            return SendResult(success=True, message_id=message_id)
        clipped = content[:CAP]
        self.screen[message_id] = clipped
        return SendResult(
            success=True,
            message_id=message_id,
            raw_response={
                "stream_preview_truncated": True,
                "delivered_prefix": clipped,
            },
        )

    async def delete_message(self, chat_id, message_id) -> bool:
        self.screen.pop(message_id, None)
        if message_id in self.order:
            self.order.remove(message_id)
        return True

    def rendered(self) -> str:
        return "".join(self.screen[m] for m in self.order)


def _consumer_claims_final_delivery(consumer, final_text: str) -> bool:
    """Mirror of the suppression decision in ``gateway/run.py``."""
    if consumer.delivered_final_matches(final_text) is False:
        return False
    return bool(consumer.final_response_sent or consumer.final_content_delivered)


async def _drive(adapter, deltas, final_text):
    consumer = GatewayStreamConsumer(
        adapter,
        "chat-1",
        StreamConsumerConfig(cursor=CURSOR, edit_interval=0.0),
    )
    # Keep the whole reply in one preview so the adapter (not the consumer)
    # owns the overflow, which is what Telegram's rich-message cap does via
    # ``streaming_overflow_limit``.
    consumer._raw_message_limit = lambda: 32768
    task = asyncio.create_task(consumer.run())
    for delta in deltas:
        consumer.on_delta(delta)
        await asyncio.sleep(0.02)
    consumer.finish(final_text)
    try:
        await asyncio.wait_for(task, timeout=3.0)
    except (asyncio.TimeoutError, asyncio.CancelledError):
        task.cancel()
    return consumer


def _oversized_deltas():
    # First delta creates the preview message (well under the cap); the second
    # grows it past the cap so the clip happens on a progressive EDIT, which is
    # the path Telegram takes for a rich-capable bot (streaming_overflow_limit
    # raises the consumer's own split threshold to 32,768).
    opening = "报告开头。"
    body = "内容" * 2600
    tail = "最后一句：过去 4 天又发了 135 条）"
    return [opening, body, tail], opening + body + tail


@pytest.mark.asyncio
async def test_clipped_preview_never_claims_delivery_it_cannot_back_up():
    """The #98552 shape: clipped preview + a turn-final edit that fails.

    Whatever recovery path the consumer takes, it may only claim the turn
    final was delivered if the complete answer is actually on screen.
    """
    deltas, final_text = _oversized_deltas()
    adapter = ClippingAdapter(fail_finalize_edit=True)

    consumer = await _drive(adapter, deltas, final_text)

    if _consumer_claims_final_delivery(consumer, final_text):
        rendered = adapter.rendered()
        assert final_text.strip() in rendered, (
            "consumer claims the turn final was delivered, but the platform "
            "only stored a clipped preview — the gateway would suppress its "
            "corrective send and the answer would be lost. "
            f"flags=(response_sent={consumer.final_response_sent}, "
            f"content_delivered={consumer.final_content_delivered}) "
            f"verdict={consumer.delivered_final_matches(final_text)!r} "
            f"rendered_len={len(rendered)} final_len={len(final_text)}"
        )


@pytest.mark.asyncio
async def test_clipped_preview_leaves_a_reconcilable_record():
    """A clipped preview must never reconcile as the complete final answer."""
    deltas, final_text = _oversized_deltas()
    adapter = ClippingAdapter(fail_finalize_edit=True)

    consumer = await _drive(adapter, deltas, final_text)

    if final_text.strip() not in adapter.rendered():
        assert consumer.delivered_final_matches(final_text) is not True, (
            "a clipped preview reconciled as the completed response"
        )


@pytest.mark.asyncio
async def test_delivered_text_for_prefers_the_adapter_report():
    """Unit contract for the helper the success path now goes through."""
    plain = SimpleNamespace(raw_response=None)
    reported = SimpleNamespace(
        raw_response={"stream_preview_truncated": True, "delivered_prefix": "abc"}
    )
    empty = SimpleNamespace(raw_response={"delivered_prefix": ""})

    assert GatewayStreamConsumer._delivered_text_for(plain, "abcdef") == "abcdef"
    assert GatewayStreamConsumer._delivered_text_for(reported, "abcdef") == "abc"
    assert GatewayStreamConsumer._delivered_text_for(empty, "abcdef") == "abcdef"
