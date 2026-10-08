"""Tests for the pure-media wait-for-text cache pool (qqbot / weixin / feishu).

A pure-media message (no text of the user's own) is cached and acked with
"📷 收到" instead of being dispatched; the chat's next text message flushes the
cached media back in as REAL attachments (images — gateway image routing then
decides native vs text per the model) plus "[Media: …]" refs (documents).
"""
import asyncio
import time as _time
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from gateway.platforms.event import MessageEvent, MessageType


def _run(coro):
    return asyncio.run(coro)


class TestFeishuMediaCachePool:
    def _adapter(self):
        from plugins.platforms.feishu.adapter import FeishuAdapter

        adapter = object.__new__(FeishuAdapter)
        adapter._pending_media_cache = {}
        adapter._pending_media_ts = {}
        adapter.send = AsyncMock()
        adapter._handle_message_with_guards = AsyncMock()
        adapter._enqueue_text_event = AsyncMock()
        adapter._enqueue_media_event = AsyncMock()
        return adapter

    @staticmethod
    def _event(text="", media=(), mtype=MessageType.PHOTO, user="ou_user"):
        return MessageEvent(
            text=text, message_type=mtype,
            source=SimpleNamespace(chat_id="oc_1", thread_id="", user_id=user),
            media_urls=[p for p, _m in media], media_types=[m for _p, m in media],
        )

    def test_pure_media_is_cached_and_acked(self):
        adapter = self._adapter()
        _run(adapter._dispatch_inbound_event(self._event(media=[("/tmp/a.jpg", "image/jpeg")])))

        assert adapter._pending_media_cache["feishu:oc_1::ou_user"] == {
            "images": [("/tmp/a.jpg", "image/jpeg")], "files": []}
        adapter.send.assert_awaited_once()
        adapter._handle_message_with_guards.assert_not_awaited()
        # The upstream media-batch timer must not pick the event up either.
        adapter._enqueue_media_event.assert_not_awaited()

    def test_follow_up_text_restores_real_attachments(self):
        adapter = self._adapter()
        _run(adapter._dispatch_inbound_event(self._event(media=[("/tmp/a.jpg", "image/jpeg")])))
        event = self._event(text="看看这个", mtype=MessageType.TEXT)
        _run(adapter._dispatch_inbound_event(event))

        assert event.media_urls == ["/tmp/a.jpg"]
        assert event.media_types == ["image/jpeg"]
        adapter._enqueue_text_event.assert_awaited_once_with(event)
        assert "feishu:oc_1::ou_user" not in adapter._pending_media_cache

    def test_documents_are_flushed_as_text_refs(self):
        adapter = self._adapter()
        _run(adapter._dispatch_inbound_event(self._event(media=[("/tmp/doc.pdf", "application/pdf")])))
        event = self._event(text="处理下", mtype=MessageType.TEXT)
        _run(adapter._dispatch_inbound_event(event))

        assert event.media_urls == []
        assert "[Media: /tmp/doc.pdf]" in event.text

    def test_group_media_does_not_cross_senders(self):
        adapter = self._adapter()
        _run(adapter._dispatch_inbound_event(
            self._event(media=[("/tmp/a.jpg", "image/jpeg")], user="ou_a")))
        event = self._event(text="看这个", mtype=MessageType.TEXT, user="ou_b")
        _run(adapter._dispatch_inbound_event(event))

        assert event.media_urls == []  # B never receives A's parked image
        assert "feishu:oc_1::ou_a" in adapter._pending_media_cache  # still waiting for A's own text

    def test_stale_buckets_are_pruned(self):
        adapter = self._adapter()
        _run(adapter._dispatch_inbound_event(self._event(media=[("/tmp/a.jpg", "image/jpeg")])))
        adapter._pending_media_ts["feishu:oc_1::ou_user"] = _time.time() - adapter._MEDIA_CACHE_TTL_SECONDS - 1
        _run(adapter._dispatch_inbound_event(self._event(media=[("/tmp/b.jpg", "image/jpeg")])))

        assert list(adapter._pending_media_cache) == ["feishu:oc_1::ou_user"]
        assert adapter._pending_media_cache["feishu:oc_1::ou_user"]["images"] == [("/tmp/b.jpg", "image/jpeg")]


class TestQqbotMediaCachePool:
    def _adapter(self):
        from gateway.platforms.qqbot.adapter import QQAdapter

        adapter = object.__new__(QQAdapter)
        adapter._pending_attachments = {}
        adapter._pending_file_texts = {}
        adapter._pending_media_ts = {}
        adapter._chat_type_map = {}
        adapter.send = AsyncMock()
        adapter.handle_message = AsyncMock()
        adapter._process_quoted_context = AsyncMock(
            return_value={"quote_block": "", "image_urls": [], "image_media_types": []})
        adapter._parse_qq_timestamp = lambda ts: None
        adapter.build_source = lambda **kw: SimpleNamespace(**kw)
        return adapter

    @staticmethod
    def _att(images=(), info="", voice=()):
        return {
            "image_urls": [p for p, _m in images], "image_media_types": [m for _p, m in images],
            "voice_transcripts": list(voice), "attachment_info": info,
        }

    def test_pure_image_is_cached_and_acked(self):
        adapter = self._adapter()
        adapter._process_attachments = AsyncMock(return_value=self._att([("/tmp/a.jpg", "image/jpeg")]))
        _run(adapter._ingest({}, "m1", "", None, "t", chat_id="c1", qq_chat_type="c2c", user_id="u1"))

        assert adapter._pending_attachments["qqbot:c1:u1"] == [("/tmp/a.jpg", "image/jpeg")]
        adapter.send.assert_awaited_once()
        adapter.handle_message.assert_not_awaited()

    def test_pure_file_is_cached_and_acked(self):
        adapter = self._adapter()
        adapter._process_attachments = AsyncMock(return_value=self._att(info="[File: x.docx]"))
        _run(adapter._ingest({}, "m1", "", None, "t", chat_id="c1", qq_chat_type="c2c", user_id="u1"))

        assert adapter._pending_file_texts["qqbot:c1:u1"] == [("[File: x.docx]", "file")]
        adapter.send.assert_awaited_once()
        adapter.handle_message.assert_not_awaited()

    def test_follow_up_text_flushes_real_attachments(self):
        adapter = self._adapter()
        adapter._process_attachments = AsyncMock(return_value=self._att([("/tmp/a.jpg", "image/jpeg")]))
        _run(adapter._ingest({}, "m1", "", None, "t", chat_id="c1", qq_chat_type="c2c", user_id="u1"))

        adapter._process_attachments = AsyncMock(return_value=self._att())
        _run(adapter._ingest({}, "m2", "看这个", None, "t", chat_id="c1", qq_chat_type="c2c", user_id="u1"))

        event = adapter.handle_message.await_args.args[0]
        assert event.media_urls == ["/tmp/a.jpg"]
        assert event.media_types == ["image/jpeg"]

    def test_group_media_does_not_cross_senders(self):
        adapter = self._adapter()
        adapter._process_attachments = AsyncMock(return_value=self._att([("/tmp/a.jpg", "image/jpeg")]))
        _run(adapter._ingest({}, "m1", "", None, "t", chat_id="g1", qq_chat_type="group", user_id="u_a"))

        adapter._process_attachments = AsyncMock(return_value=self._att())
        _run(adapter._ingest({}, "m2", "看这个", None, "t", chat_id="g1", qq_chat_type="group", user_id="u_b"))

        event = adapter.handle_message.await_args.args[0]
        assert event.media_urls == []  # B never receives A's parked image
        assert "qqbot:g1:u_a" in adapter._pending_attachments  # still waiting for A's own text

    def test_voice_transcript_does_not_park(self):
        adapter = self._adapter()
        adapter._process_attachments = AsyncMock(return_value=self._att(voice=["hello"]))
        _run(adapter._ingest({}, "m1", "", None, "t", chat_id="c1", qq_chat_type="c2c", user_id="u1"))

        assert adapter._pending_attachments == {}
        adapter.handle_message.assert_awaited_once()


class TestWeixinMediaCachePool:
    def _adapter(self):
        from gateway.platforms.weixin import WeixinAdapter

        adapter = object.__new__(WeixinAdapter)
        adapter._pending_media_cache = {}
        adapter._pending_media_ts = {}
        adapter.platform = SimpleNamespace(value="weixin")  # backs the `name` property
        adapter._poll_session = object()
        adapter._account_id = "bot1"
        adapter._token = ""
        adapter._dedup = SimpleNamespace(is_duplicate=lambda *a, **k: False)
        adapter._is_dm_intake_allowed = lambda *a, **k: True
        adapter._is_group_allowed = lambda *a, **k: True
        adapter._typing_cache = {}
        adapter.send = AsyncMock()
        adapter.handle_message = AsyncMock()
        adapter._enqueue_text_event = lambda ev: None
        adapter.build_source = lambda **kw: SimpleNamespace(**kw)
        return adapter

    @staticmethod
    def _wire_media(adapter, path="/tmp/wx.jpg", mime="image/jpeg"):
        async def _collect(item, media_paths, media_types):
            media_paths.append(path)
            media_types.append(mime)

        adapter._collect_media = _collect

    @staticmethod
    def _msg(mtype, mid="m1", sender="u1"):
        return {"from_user_id": sender, "message_id": mid, "item_list": [{"type": mtype}]}

    def test_pure_image_is_cached_and_acked(self):
        from gateway.platforms.weixin import ITEM_IMAGE

        adapter = self._adapter()
        self._wire_media(adapter)
        with patch("gateway.platforms.weixin._extract_text", return_value=""), \
             patch("gateway.platforms.weixin._guess_chat_type", return_value=("dm", "u1")):
            _run(adapter._process_message(self._msg(ITEM_IMAGE)))

        assert adapter._pending_media_cache["weixin:u1:u1"] == {
            "images": [("/tmp/wx.jpg", "image/jpeg")], "files": []}
        adapter.send.assert_awaited_once()
        adapter.handle_message.assert_not_awaited()

    def test_follow_up_text_flushes_real_attachments(self):
        from gateway.platforms.weixin import ITEM_IMAGE

        adapter = self._adapter()
        self._wire_media(adapter)
        with patch("gateway.platforms.weixin._extract_text", return_value=""), \
             patch("gateway.platforms.weixin._guess_chat_type", return_value=("dm", "u1")):
            _run(adapter._process_message(self._msg(ITEM_IMAGE)))

        adapter._collect_media = AsyncMock()  # the text message carries no media
        with patch("gateway.platforms.weixin._extract_text", return_value="看这个"), \
             patch("gateway.platforms.weixin._guess_chat_type", return_value=("dm", "u1")):
            _run(adapter._process_message(self._msg(ITEM_IMAGE, mid="m2")))

        event = adapter.handle_message.await_args.args[0]
        assert event.media_urls == ["/tmp/wx.jpg"]
        assert event.media_types == ["image/jpeg"]
        assert "看这个" in event.text

    def test_group_media_does_not_cross_senders(self):
        from gateway.platforms.weixin import ITEM_IMAGE

        adapter = self._adapter()
        self._wire_media(adapter)
        with patch("gateway.platforms.weixin._extract_text", return_value=""), \
             patch("gateway.platforms.weixin._guess_chat_type", return_value=("group", "g1")):
            _run(adapter._process_message(self._msg(ITEM_IMAGE, sender="ua")))

        captured = {}
        adapter._enqueue_text_event = lambda ev: captured.setdefault("ev", ev)
        adapter._collect_media = AsyncMock()
        with patch("gateway.platforms.weixin._extract_text", return_value="看这个"), \
             patch("gateway.platforms.weixin._guess_chat_type", return_value=("group", "g1")):
            _run(adapter._process_message(self._msg(ITEM_IMAGE, mid="m2", sender="ub")))

        event = captured["ev"]  # the text turn goes through the text-batch path
        assert event.media_urls == []  # B never receives A's parked image
        assert "weixin:g1:ua" in adapter._pending_media_cache  # still waiting for A's own text

    def test_voice_is_exempt_from_the_pool(self):
        from gateway.platforms.weixin import ITEM_VOICE

        adapter = self._adapter()
        self._wire_media(adapter, path="/tmp/wx.silk", mime="audio/silk")
        with patch("gateway.platforms.weixin._extract_text", return_value=""), \
             patch("gateway.platforms.weixin._guess_chat_type", return_value=("dm", "u1")):
            _run(adapter._process_message(self._msg(ITEM_VOICE)))

        assert adapter._pending_media_cache == {}
        adapter.handle_message.assert_awaited_once()
