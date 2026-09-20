"""Tests for the Weixin outbound 24h quota guard.

iLink enforces an undocumented 24h send quota (community-reported ~10 messages); once it is
spent every send fails and the adapter's rate-limit circuit parks outbound for minutes — the
turn's final reply included. Three behaviours prevent that, all covered here:

1. ``_OutboundWindowCounter`` — the persisted rolling window every *real* send is recorded in,
   fail-open by construction (a corrupt file must degrade to "unlimited", never block a send).
2. The priority budget — heartbeat/status/media sends are shed as the window fills, so the final
   reply always keeps a slot.
3. The interact prompt — after 9 chunks in one turn the adapter posts the prompt (slot 10) and
   parks until the user replies, whose passive reply reopens iLink's window.
"""

import asyncio
import time
from unittest.mock import AsyncMock, patch

import pytest

from gateway.config import PlatformConfig
from gateway.platforms.weixin import WeixinAdapter, _OutboundWindowCounter


def _make_adapter(extra=None) -> WeixinAdapter:
    return WeixinAdapter(
        PlatformConfig(
            enabled=True,
            token="test-token",
            extra={"account_id": "test-account", **(extra or {})},
        )
    )


def _connected_adapter(**extra) -> WeixinAdapter:
    adapter = _make_adapter(extra or None)
    adapter._send_session = object()
    adapter._token = "test-token"
    adapter._token_store.get = lambda account_id, chat_id: "ctx-token"
    return adapter


class TestOutboundWindowCounter:
    def test_remaining_tracks_sends_inside_the_window(self, tmp_path):
        counter = _OutboundWindowCounter(str(tmp_path), "acct", quota=3, window_seconds=86400)

        assert counter.remaining() == 3
        asyncio.run(counter.record())
        assert counter.remaining() == 2

    def test_timestamps_older_than_the_window_are_pruned(self, tmp_path):
        counter = _OutboundWindowCounter(str(tmp_path), "acct", quota=3, window_seconds=3600)
        now = 1_000_000.0
        counter._timestamps = [now - 7200, now - 10, now - 5]

        with patch("gateway.platforms.weixin.time.time", return_value=now):
            assert counter.remaining() == 1  # the 7200s-old entry no longer occupies a slot

    def test_count_survives_a_restart(self, tmp_path):
        """The whole point of persisting: a gateway restart must not forget the spent quota."""
        counter = _OutboundWindowCounter(str(tmp_path), "acct", quota=2, window_seconds=86400)
        asyncio.run(counter.record())

        revived = _OutboundWindowCounter(str(tmp_path), "acct", quota=2, window_seconds=86400)
        assert revived.remaining() == 1

    def test_corrupt_counter_file_fails_open(self, tmp_path):
        counter = _OutboundWindowCounter(str(tmp_path), "acct", quota=2, window_seconds=86400)
        counter._path.write_text("{ this is not json", encoding="utf-8")

        revived = _OutboundWindowCounter(str(tmp_path), "acct", quota=2, window_seconds=86400)
        assert revived.remaining() == 2  # degrades to "unlimited", never raises

    def test_record_does_not_raise_when_persistence_fails(self, tmp_path):
        counter = _OutboundWindowCounter(str(tmp_path), "acct", quota=2, window_seconds=86400)

        with patch("gateway.platforms.weixin.atomic_json_write", side_effect=OSError("disk full")):
            assert asyncio.run(counter.record()) is False  # non-fatal: in-memory count still held
        assert counter.remaining() == 1


class TestOutboundPriorityBudget:
    def test_priority_of_reads_the_gateway_markers(self):
        assert WeixinAdapter._outbound_priority_of({"_heartbeat": True}) == "heartbeat"
        assert WeixinAdapter._outbound_priority_of({"_interim_send": True}) == "status"
        assert WeixinAdapter._outbound_priority_of({}) == "final"
        assert WeixinAdapter._outbound_priority_of(None) == "final"

    def test_final_reply_sends_even_with_the_window_full(self):
        adapter = _connected_adapter()
        adapter._outbound_counter._timestamps = [time.time()] * adapter._outbound_daily_quota

        with patch.object(adapter, "_send_text_chunk", new=AsyncMock()) as chunk_mock:
            result = asyncio.run(adapter.send("wxid_peer", "the answer"))

        assert result.success is True
        assert chunk_mock.await_count == 1  # the reply is never shed

    def test_heartbeat_is_skipped_while_slots_remain_for_the_reply(self):
        adapter = _connected_adapter()
        quota = adapter._outbound_daily_quota
        adapter._outbound_counter._timestamps = [time.time()] * (quota - 2)  # 2 slots left (heartbeat floor is 2)

        with patch.object(adapter, "_send_text_chunk", new=AsyncMock()) as chunk_mock:
            result = asyncio.run(adapter.send("wxid_peer", "⏳ Working — 9 min", metadata={"_heartbeat": True}))

        assert result.success is True  # a skip, not a failure: the caller must not retry
        assert result.message_id is None
        assert chunk_mock.await_count == 0

    def test_status_text_is_skipped_before_the_final_slot(self):
        adapter = _connected_adapter()
        quota = adapter._outbound_daily_quota
        adapter._outbound_counter._timestamps = [time.time()] * (quota - 1)  # 1 slot left (status floor is 1)

        with patch.object(adapter, "_send_text_chunk", new=AsyncMock()) as chunk_mock:
            result = asyncio.run(adapter.send("wxid_peer", "working on it…", metadata={"_interim_send": True}))

        assert result.success is True
        assert chunk_mock.await_count == 0

    def test_media_is_refused_when_only_the_final_slot_remains(self):
        adapter = _connected_adapter()
        quota = adapter._outbound_daily_quota
        adapter._outbound_counter._timestamps = [time.time()] * (quota - 1)

        with patch.object(adapter, "_send_file", new=AsyncMock()) as file_mock:
            result = asyncio.run(adapter.send_document("wxid_peer", "report.pdf"))

        assert result.success is False  # caller treats this as a skip
        assert file_mock.await_count == 0

    def test_heartbeat_still_sends_while_the_window_is_roomy(self):
        adapter = _connected_adapter()

        with patch.object(adapter, "_send_text_chunk", new=AsyncMock()) as chunk_mock:
            result = asyncio.run(adapter.send("wxid_peer", "⏳ Working — 3 min", metadata={"_heartbeat": True}))

        assert result.success is True
        assert chunk_mock.await_count == 1


class TestInteractPrompt:
    def test_prompt_fires_after_nine_chunks_and_parks_for_the_reply(self):
        adapter = _connected_adapter()
        chunks = [f"chunk-{i}" for i in range(11)]

        with patch.object(adapter, "_split_text", return_value=chunks), \
             patch.object(adapter, "_send_text_chunk", new=AsyncMock()) as chunk_mock, \
             patch.object(adapter, "_wait_for_interact_resume", new=AsyncMock()) as wait_mock:
            result = asyncio.run(adapter.send("wxid_peer", "long answer"))

        assert result.success is True
        # 11 business chunks + 1 prompt, and exactly one park.
        assert chunk_mock.await_count == 12
        assert wait_mock.await_count == 1
        # The prompt is slot 10 and must not itself inflate the batch count.
        assert adapter._sent_since_user_msg == 2

    def test_no_prompt_when_the_turn_stays_under_the_threshold(self):
        adapter = _connected_adapter()
        chunks = [f"chunk-{i}" for i in range(9)]

        with patch.object(adapter, "_split_text", return_value=chunks), \
             patch.object(adapter, "_send_text_chunk", new=AsyncMock()) as chunk_mock, \
             patch.object(adapter, "_wait_for_interact_resume", new=AsyncMock()) as wait_mock:
            asyncio.run(adapter.send("wxid_peer", "answer"))

        assert chunk_mock.await_count == 9
        assert wait_mock.await_count == 0

    def test_prompt_text_is_localizable_through_config_extra(self):
        adapter = _connected_adapter(outbound_interact_prompt="已发送 {sent} 条，回复任意消息接着发。")
        adapter._sent_since_user_msg = 9  # 9 sent → this prompt takes slot 10
        sent = []

        async def _capture(**kwargs):
            sent.append(kwargs["chunk"])

        with patch.object(adapter, "_send_text_chunk", new=AsyncMock(side_effect=_capture)):
            asyncio.run(adapter._send_interact_prompt("wxid_peer", "ctx-token"))

        assert sent == ["已发送 10 条，回复任意消息接着发。"]

    def test_prompt_falls_back_when_the_template_is_malformed(self):
        adapter = _connected_adapter(outbound_interact_prompt="{sent {} oops")
        adapter._sent_since_user_msg = 9
        sent = []

        async def _capture(**kwargs):
            sent.append(kwargs["chunk"])

        with patch.object(adapter, "_send_text_chunk", new=AsyncMock(side_effect=_capture)):
            asyncio.run(adapter._send_interact_prompt("wxid_peer", "ctx-token"))

        assert sent and "10" in sent[0]  # falls back to the built-in wording, still slot 10

    def test_short_reply_resumes_the_parked_turn_without_reaching_the_agent(self):
        adapter = _connected_adapter()
        adapter._waiting_interact_reply = True

        async def scenario():
            waiter = asyncio.create_task(adapter._wait_for_interact_resume())
            await asyncio.sleep(0.01)
            adapter._interact_resume_event.set()  # what _process_message does on a short reply
            await waiter

        asyncio.run(scenario())
        assert adapter._interact_resume_event.is_set() is False  # cleared for the next round

    def test_wait_times_out_instead_of_parking_forever(self):
        adapter = _connected_adapter()
        adapter._interact_reply_timeout = 0.05

        async def scenario():
            await adapter._wait_for_interact_resume()

        asyncio.run(scenario())  # returns rather than hanging the turn
