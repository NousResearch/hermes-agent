"""Held-outbound behaviour for the Weixin (iLink) adapter.

iLink refuses bot-initiated sends once the peer's reply window has closed (``ret=-2 prepare
failed``). The adapter must hold the undelivered chunks per peer and flush them on that peer's
next inbound instead of discarding the reply (#135921), never duplicating a chunk the peer
already saw, and never letting the hold queue outlive ``held_outbound_ttl_seconds``.
"""

import asyncio
import time
from typing import Optional
from unittest.mock import AsyncMock, patch

import pytest

from gateway.config import PlatformConfig
from plugins.platforms.weixin import adapter as weixin
from plugins.platforms.weixin.adapter import WeixinAdapter
from tests.waiting import async_wait_until


def _make_adapter() -> WeixinAdapter:
    return WeixinAdapter(
        PlatformConfig(
            enabled=True,
            token="test-token",
            extra={"account_id": "test-account"},
        )
    )


def _connected_adapter(context_token="ctx-token") -> WeixinAdapter:
    adapter = _make_adapter()
    adapter._session = object()
    adapter._send_session = adapter._session
    adapter._poll_session = adapter._session
    adapter._token = "test-token"
    adapter._base_url = "https://weixin.example.com"
    adapter._token_store.get = lambda account_id, chat_id: context_token
    adapter._token_store.set = AsyncMock()
    adapter._fetch_typing_ticket = AsyncMock()
    return adapter


def _inbound_message(sender_id: str, text: str, context_token: str = "fresh-token") -> dict:
    return {
        "from_user_id": sender_id,
        "message_id": "msg-1",
        "context_token": context_token,
        "item_list": [{"type": 1, "text_item": {"text": text}}],
    }


class _FakeILinkBridge:
    """Stateful stand-in for iLink's ``sendmessage`` endpoint. While the peer's reply window is
    closed every bot-initiated send is refused with the real wire response (``ret=-2 prepare
    failed``); once the peer messages the bot the window re-opens and sends are accepted.
    Delivered texts are recorded in order, which is what the assertions below check: ordering,
    no loss, and no duplicates."""

    def __init__(self, *, window_open: bool = True, close_after: Optional[int] = None):
        self.window_open = window_open
        self.close_after = close_after
        self.delivered: list[tuple[str, Optional[str]]] = []
        self.attempts = 0

    async def __call__(self, session, *, base_url, token, to, text, context_token, client_id) -> dict:
        self.attempts += 1
        if not self.window_open:
            return {"ret": -2, "errcode": None, "errmsg": "prepare failed"}
        self.delivered.append((text, context_token))
        if self.close_after is not None and len(self.delivered) >= self.close_after:
            self.window_open = False
        return {"ret": 0}

    @property
    def texts(self) -> list[str]:
        return [text for text, _ in self.delivered]


class TestWeixinHeldOutbound:
    """The hold itself: what a closed window keeps, and what the peer's next inbound delivers."""

    @patch("plugins.platforms.weixin.adapter.asyncio.sleep", new_callable=AsyncMock)
    @patch("plugins.platforms.weixin.adapter._send_message", new_callable=AsyncMock)
    def test_closed_window_holds_reply_and_flushes_it_on_next_inbound(self, send_message_mock, sleep_mock):
        adapter = _connected_adapter()
        send_message_mock.return_value = {"ret": weixin.RATE_LIMIT_ERRCODE, "errmsg": "prepare failed"}

        result = asyncio.run(adapter.send("wxid_test123", "the report"))

        # The send is *accepted*: the hold queue owns the text and will deliver it on the peer's
        # next inbound. Reporting failure here would make the delivery ledger arm its timer-based
        # redelivery lane after the same text (duplicating the reply once a flush delivers it) and
        # make callers retry a send iLink keeps refusing until the peer speaks.
        assert result.success is True
        assert result.raw_response["queued"] is True
        assert [chunks for _, chunks, _ in adapter._held_outbound["wxid_test123"]] == [["the report"]]

        # The peer messages again: the fresh context_token reopens the window, so the held reply goes out.
        send_message_mock.reset_mock()
        send_message_mock.return_value = {"ret": 0}
        adapter.handle_message = AsyncMock()
        adapter._text_batch_delay_seconds = 0.05
        adapter._text_batch_split_delay_seconds = 0.05

        async def _drive():
            await adapter._process_message(_inbound_message("wxid_test123", "any update?"))
            await async_wait_until(lambda: "wxid_test123" not in adapter._held_outbound)

        asyncio.run(_drive())

        assert "wxid_test123" not in adapter._held_outbound
        assert [call.kwargs["text"] for call in send_message_mock.await_args_list] == ["the report"]

    def test_held_queue_is_bounded_and_expires(self):
        adapter = _make_adapter()
        adapter._held_outbound_max = 2
        adapter._held_outbound_ttl_seconds = 30

        adapter._hold_outbound_text("wxid_test123", ["one"])
        adapter._hold_outbound_text("wxid_test123", ["two"])
        adapter._hold_outbound_text("wxid_test123", ["three"])

        assert [chunks for _, chunks, _ in adapter._held_outbound["wxid_test123"]] == [["two"], ["three"]]

        adapter._held_outbound["wxid_test123"][0] = (time.time() - 31, ["stale"], 30)
        adapter._hold_outbound_text("wxid_test123", ["four"])

        assert [chunks for _, chunks, _ in adapter._held_outbound["wxid_test123"]] == [["three"], ["four"]]

    def test_same_text_is_not_held_twice(self):
        """A refused reply gets retried many times over; one queue copy per attempt would flush
        the same text to the peer many times at once."""
        adapter = _connected_adapter()

        first = adapter._hold_outbound_text("wxid_test123", ["same text"])
        second = adapter._hold_outbound_text("wxid_test123", ["same text"])
        third = adapter._hold_outbound_text("wxid_test123", ["other text"])

        assert (first, second, third) == (1, 1, 2)
        assert [chunks for _, chunks, _ in adapter._held_outbound["wxid_test123"]] == [["same text"], ["other text"]]


class TestWeixinHeldOutboundTimeline:
    """End-to-end replay of the production failure: a long turn's chunks are refused once the
    peer's reply window closes, and the peer only receives them after messaging the bot again."""

    def _connected_adapter(self, context_token="ctx-token") -> WeixinAdapter:
        adapter = _connected_adapter(context_token)
        adapter._send_chunk_retries = 0
        adapter._send_chunk_delay_seconds = 0
        return adapter

    @patch("plugins.platforms.weixin.adapter.asyncio.sleep", new_callable=AsyncMock)
    def test_window_closing_mid_reply_holds_the_rest_and_flushes_it_in_order(self, sleep_mock):
        bridge = _FakeILinkBridge(close_after=1)  # accepts the first chunk, then the window closes
        adapter = self._connected_adapter()
        adapter.MAX_MESSAGE_LENGTH = 5  # one chunk per block, so the window closes between chunk 1 and 2

        with patch("plugins.platforms.weixin.adapter._send_message", new=bridge):
            result = asyncio.run(adapter.send("wxid_test123", "one\n\ntwo\n\nthree"))

            # The peer saw exactly the one chunk that made it out; the rest is held, nothing is duplicated.
            assert result.success is True
            assert result.raw_response["queued"] is True
            assert bridge.texts == ["one"]
            assert [chunks for _, chunks, _ in adapter._held_outbound["wxid_test123"]] == [["two", "three"]]

            # The peer messages again: the fresh context_token reopens the window (and it stays open).
            bridge.window_open = True
            bridge.close_after = None
            adapter.handle_message = AsyncMock()
            adapter._text_batch_delay_seconds = 0.05
            adapter._text_batch_split_delay_seconds = 0.05

            async def _inbound():
                await adapter._process_message(_inbound_message("wxid_test123", "any update?"))
                await async_wait_until(lambda: "wxid_test123" not in adapter._held_outbound)

            asyncio.run(_inbound())

            assert "wxid_test123" not in adapter._held_outbound
            assert bridge.texts == ["one", "two", "three"]

    @patch("plugins.platforms.weixin.adapter.asyncio.sleep", new_callable=AsyncMock)
    def test_flush_refused_midway_reholds_only_what_the_peer_has_not_seen(self, sleep_mock):
        """A flush refused halfway must re-queue from the first undelivered chunk: re-queueing
        the whole entry would re-send what the peer already received."""
        adapter = self._connected_adapter()
        adapter._hold_outbound_text("wxid_test123", ["a", "b", "c"])
        bridge = _FakeILinkBridge(close_after=1)  # accepts "a", then the window closes again

        with patch("plugins.platforms.weixin.adapter._send_message", new=bridge):
            asyncio.run(adapter._flush_held_outbound("wxid_test123"))

        assert bridge.texts == ["a"]
        assert [chunks for _, chunks, _ in adapter._held_outbound["wxid_test123"]] == [["b", "c"]]

    @patch("plugins.platforms.weixin.adapter.asyncio.sleep", new_callable=AsyncMock)
    def test_flush_preserves_order_across_several_held_replies(self, sleep_mock):
        adapter = self._connected_adapter()
        adapter._hold_outbound_text("wxid_test123", ["first reply"])
        adapter._hold_outbound_text("wxid_test123", ["second reply"])
        bridge = _FakeILinkBridge(window_open=True)

        with patch("plugins.platforms.weixin.adapter._send_message", new=bridge):
            asyncio.run(adapter._flush_held_outbound("wxid_test123"))

        assert bridge.texts == ["first reply", "second reply"]
        assert "wxid_test123" not in adapter._held_outbound

    @pytest.mark.parametrize("close_after", [1, 2, 3, 4])
    @patch("plugins.platforms.weixin.adapter.asyncio.sleep", new_callable=AsyncMock)
    def test_every_chunk_arrives_exactly_once_wherever_the_window_closes(self, sleep_mock, close_after):
        """Property: whatever chunk index the reply window closes at, the peer ends up with
        every chunk exactly once and in order — never a duplicate, never a gap."""
        bridge = _FakeILinkBridge(close_after=close_after)
        adapter = self._connected_adapter()
        adapter.MAX_MESSAGE_LENGTH = 5

        with patch("plugins.platforms.weixin.adapter._send_message", new=bridge):
            asyncio.run(adapter.send("wxid_test123", "one\n\ntwo\n\nthree\n\nfour"))

            bridge.window_open = True
            bridge.close_after = None
            asyncio.run(adapter._flush_held_outbound("wxid_test123"))

        assert bridge.texts == ["one", "two", "three", "four"]
        assert "wxid_test123" not in adapter._held_outbound

    @patch("plugins.platforms.weixin.adapter.asyncio.sleep", new_callable=AsyncMock)
    def test_flush_drops_an_expired_intermediate_update(self, sleep_mock):
        """TTL bounds intermediate updates only: a progress line the peer never saw inside the
        window is dropped, not delivered much later as if the conversation were still live."""
        adapter = self._connected_adapter()
        adapter._held_outbound_ttl_seconds = 60
        adapter._held_outbound["wxid_test123"] = [(time.time() - 61.0, ["stale progress"], 60)]
        bridge = _FakeILinkBridge(window_open=True)

        with patch("plugins.platforms.weixin.adapter._send_message", new=bridge):
            asyncio.run(adapter._flush_held_outbound("wxid_test123"))

        assert bridge.texts == []
        assert "wxid_test123" not in adapter._held_outbound

    @patch("plugins.platforms.weixin.adapter.asyncio.sleep", new_callable=AsyncMock)
    def test_a_final_reply_never_expires(self, sleep_mock):
        """A turn's final reply (``metadata["ledgered_final"]``) is the result the user is owed;
        held overnight — or over a weekend — it still flushes on the peer's next inbound."""
        adapter = self._connected_adapter()
        bridge = _FakeILinkBridge(window_open=False)  # the window is already closed

        with patch("plugins.platforms.weixin.adapter._send_message", new=bridge):
            result = asyncio.run(adapter._send_with_retry(chat_id="wxid_test123", content="overnight result"))

        assert result.success is True
        enqueued_at, chunks, ttl = adapter._held_outbound["wxid_test123"][0]
        assert chunks == ["overnight result"]
        assert ttl == float("inf")

        # 25 hours later the peer returns: the final reply still goes out.
        adapter._held_outbound["wxid_test123"] = [(enqueued_at - 90000.0, chunks, ttl)]
        bridge.window_open = True
        with patch("plugins.platforms.weixin.adapter._send_message", new=bridge):
            asyncio.run(adapter._flush_held_outbound("wxid_test123"))

        assert bridge.texts == ["overnight result"]
        assert "wxid_test123" not in adapter._held_outbound

    def test_the_gateway_final_path_marks_the_send_and_the_direct_path_does_not(self):
        """The gateway sends ledgered finals through ``_send_with_retry`` (marked → held forever)
        while typing notices / direct / cron pushes call ``send()`` directly (unmarked → bounded
        TTL). Both behaviours verified end-to-end here so the two lanes cannot drift apart."""
        async def _scenario():
            adapter = self._connected_adapter()
            bridge = _FakeILinkBridge(window_open=False)

            with patch("plugins.platforms.weixin.adapter._send_message", new=bridge):
                # Intermediate lane: a direct send is held with the bounded TTL...
                await adapter.send("wxid_test123", "progress note")
                # Final lane: the same text through the gateway's retry bracket is held forever.
                await adapter._send_with_retry(chat_id="wxid_test123", content="turn result")
            return adapter

        adapter = asyncio.run(_scenario())
        held = {chunks[0]: ttl for _, chunks, ttl in adapter._held_outbound["wxid_test123"]}
        assert held["progress note"] == 3600.0
        assert held["turn result"] == float("inf")

    @patch("plugins.platforms.weixin.adapter.asyncio.sleep", new_callable=AsyncMock)
    def test_flush_still_delivers_a_reply_inside_its_ttl(self, sleep_mock):
        """Positive control for the expiry rule: one second of margin left is still a live reply."""
        adapter = self._connected_adapter()
        adapter._held_outbound_ttl_seconds = 60
        adapter._held_outbound["wxid_test123"] = [(time.time() - 59.0, ["still warm"], 60)]
        bridge = _FakeILinkBridge(window_open=True)

        with patch("plugins.platforms.weixin.adapter._send_message", new=bridge):
            asyncio.run(adapter._flush_held_outbound("wxid_test123"))

        assert bridge.texts == ["still warm"]
        assert "wxid_test123" not in adapter._held_outbound

    @patch("plugins.platforms.weixin.adapter.asyncio.sleep", new_callable=AsyncMock)
    def test_refused_flush_reholds_under_the_original_deadline(self, sleep_mock):
        """A refused flush must not refresh the timestamp — otherwise a permanently closed
        window keeps an expired reply alive forever."""
        adapter = self._connected_adapter()
        adapter._hold_outbound_text("wxid_test123", ["a", "b"])
        held_at = adapter._held_outbound["wxid_test123"][0][0]
        bridge = _FakeILinkBridge(close_after=1)  # accepts "a", then the window closes again

        with patch("plugins.platforms.weixin.adapter._send_message", new=bridge):
            asyncio.run(adapter._flush_held_outbound("wxid_test123"))

        assert bridge.texts == ["a"]
        assert [chunks for _, chunks, _ in adapter._held_outbound["wxid_test123"]] == [["b"]]
        assert adapter._held_outbound["wxid_test123"][0][0] == held_at

    @patch("plugins.platforms.weixin.adapter.asyncio.sleep", new_callable=AsyncMock)
    def test_flush_paces_chunks_like_send_does(self, sleep_mock):
        """Up to 20 queued replies flushed back-to-back is what walks into iLink's rate limiter,
        so the flush keeps the same inter-chunk spacing ``send()`` uses."""
        adapter = self._connected_adapter()
        adapter._send_chunk_delay_seconds = 1.5
        adapter._hold_outbound_text("wxid_test123", ["a", "b"])
        bridge = _FakeILinkBridge(window_open=True)

        with patch("plugins.platforms.weixin.adapter._send_message", new=bridge):
            asyncio.run(adapter._flush_held_outbound("wxid_test123"))

        assert bridge.texts == ["a", "b"]
        assert [call.args[0] for call in sleep_mock.await_args_list] == [1.5]


class TestWeixinHeldOutboundHandoff:
    """The cron ``send_message`` path: a throwaway adapter built off-loop must hand its hold to
    the live adapter, and the caller must see "accepted, queued" — not a failure it would retry
    blindly while the window stays shut."""

    def test_throwaway_adapter_hands_the_hold_to_the_live_adapter(self):
        async def _scenario():
            live = _connected_adapter()
            live._loop = asyncio.get_running_loop()
            weixin._LIVE_ADAPTERS[live._token] = live
            try:
                throwaway = _connected_adapter()
                bridge = _FakeILinkBridge(window_open=False)  # the peer's reply window is closed
                with patch("plugins.platforms.weixin.adapter._send_message", new=bridge):
                    result = await throwaway.send("wxid_test123", "queued for later")
                await asyncio.sleep(0)  # let the loop run the hand-off callback
                return result, live, throwaway
            finally:
                weixin._LIVE_ADAPTERS.pop(live._token, None)

        result, live, throwaway = asyncio.run(_scenario())

        assert result.success is True
        assert result.raw_response["queued"] is True
        assert throwaway._held_outbound == {}
        assert [chunks for _, chunks, _ in live._held_outbound["wxid_test123"]] == [["queued for later"]]

    def test_direct_send_reports_a_queued_reply_as_accepted(self):
        async def _scenario():
            live = _connected_adapter()
            live._loop = asyncio.get_running_loop()
            weixin._LIVE_ADAPTERS[live._token] = live
            try:
                throwaway = _connected_adapter()
                bridge = _FakeILinkBridge(window_open=False)
                with patch("plugins.platforms.weixin.adapter._send_message", new=bridge):
                    result = await weixin._deliver_direct(throwaway, "wxid_test123", "night report", None, None)
                await asyncio.sleep(0)  # let the loop run the hand-off callback
                return result
            finally:
                weixin._LIVE_ADAPTERS.pop(live._token, None)

        result = asyncio.run(_scenario())

        assert result["success"] is True
        assert result["queued"] is True

    def test_live_adapter_on_the_same_loop_keeps_its_own_hold(self):
        """A live adapter holding for itself must not "hand off" to itself — the hold stays in
        its own queue and flushes on the peer's next inbound."""

        async def _scenario():
            live = _connected_adapter()
            live._loop = asyncio.get_running_loop()
            weixin._LIVE_ADAPTERS[live._token] = live
            try:
                bridge = _FakeILinkBridge(window_open=False)
                with patch("plugins.platforms.weixin.adapter._send_message", new=bridge):
                    result = await live.send("wxid_test123", "held by myself")
                return result, live
            finally:
                weixin._LIVE_ADAPTERS.pop(live._token, None)

        result, live = asyncio.run(_scenario())

        assert result.success is True
        assert result.raw_response["queued"] is True
        assert [chunks for _, chunks, _ in live._held_outbound["wxid_test123"]] == [["held by myself"]]


class TestWeixinHeldOutboundLedgerCoordination:
    """The hold queue and the delivery ledger must stay on disjoint lanes: the ledger owns
    final-response redelivery for timer-recoverable refusals; the hold queue owns text refused
    because the peer's window is closed, whose recovery event is the peer's next inbound — a
    timer can never fire at the right moment for that, and a ledger retry landing after a flush
    already delivered the text would duplicate it."""

    def test_a_held_send_closes_the_ledger_row_as_delivered(self):
        """Because ``send()`` reports success for held text, the gateway's
        ``_finalize_delivery_obligation`` marks the obligation delivered and never arms its
        timer-based redelivery after text the hold queue already owns."""
        from types import SimpleNamespace

        adapter = _connected_adapter()

        async def _scenario():
            bridge = _FakeILinkBridge(window_open=False)
            with patch("plugins.platforms.weixin.adapter._send_message", new=bridge):
                return await adapter.send("wxid_test123", "held final response")

        result = asyncio.run(_scenario())
        assert result.success is True
        assert result.raw_response["queued"] is True

        event = SimpleNamespace(
            source=SimpleNamespace(platform="weixin", chat_id="wxid_test123", thread_id=None),
            text="held final response",
        )

        with patch("gateway.delivery_ledger.mark_delivered") as delivered_mock, \
                patch("gateway.delivery_ledger.mark_failed") as failed_mock:
            asyncio.run(
                weixin.BasePlatformAdapter._finalize_delivery_obligation(
                    adapter, "obligation-1", result, event, adapter))
            assert delivered_mock.call_count == 1
            assert failed_mock.call_count == 0

    def test_session_not_ready_error_text_is_not_classified_retryable(self):
        """The stale-session error must stay out of the rate-limited redelivery lane: the window
        reopens on the peer's next message, not on a timer, and a timer retry would race the
        hold queue's flush."""
        from gateway.platforms.base import classify_send_error

        exc = weixin._session_not_ready_error(-2, None, "prepare failed")
        assert classify_send_error(exc) not in ("rate_limited", "transient")
