"""LINE: a text + images turn rides ONE Messaging API call.

LINE's reply token is single-use. Stock delivery sends the final text first (spending
the token on the free Reply API) and each image afterwards over the metered Push API —
in an N-member group that is N billed pushes per picture. When the turn is plain text
plus local image files and everything fits in one 5-object batch, the adapter folds the
images into the text's reply call instead. Every other shape (media-only, voice/video/
document attachments, oversized text, ephemeral notices, a pending slow-LLM button)
keeps the stock split path, and a failed combined send falls back to it.
"""

from __future__ import annotations

import asyncio
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from tests.gateway._plugin_adapter_loader import load_plugin_adapter

_line = load_plugin_adapter("line")
LineAdapter = _line.LineAdapter

CHAT = "C123"
IMAGE = "/tmp/line_fold_img.jpg"


def _make_adapter() -> LineAdapter:
    from gateway.config import PlatformConfig

    cfg = PlatformConfig(enabled=True, extra={"channel_access_token": "tok", "channel_secret": "sec"})
    ad = LineAdapter(cfg)
    ad._client = MagicMock()
    ad._client.reply = AsyncMock(return_value={"status": 200})
    ad._client.push = AsyncMock(return_value={"status": 200})
    ad._reply_tokens[CHAT] = ("reply-token", time.time() + 60)
    return ad


def _event():
    return SimpleNamespace(
        source=SimpleNamespace(chat_id=CHAT, platform="line", user_id="U1"),
        message_id="m1", text="hi", media_urls=[], channel_context=None)


def _extracted(text="", media=None, local=None, force_document=False):
    from gateway.platforms.base import _ExtractedResponse

    return _ExtractedResponse(
        text_content=text, images=[], media_files=media or [], local_files=local or [],
        force_document_attachments=force_document, pre_extract=text)


@pytest.fixture(autouse=True)
def _image_file(tmp_path):
    p = tmp_path / "gen.png"
    p.write_bytes(b"\x89PNG\r\n\x1a\n" + b"0" * 32)
    return p


async def _stage(adapter, text, image_path):
    """Run the staging half of the turn (extraction) like the gateway would."""
    event = _event()
    monkey_extract = getattr(adapter, "_extract_response_content")
    # Call the real override; the base extraction is mocked so only staging logic runs.
    adapter._extract_media = lambda response: ([str(image_path)], response.replace(f"MEDIA:{image_path}", "").strip())
    adapter.extract_images = lambda response: ([], response)
    adapter.filter_media_delivery_paths = lambda paths, session_key=None: paths
    adapter.extract_local_files = lambda text: ([], text)
    adapter.filter_local_delivery_paths = lambda paths, session_key=None: paths
    await monkey_extract(f"here you go MEDIA:{image_path}", event, "sk", is_ephemeral_response=False)


class TestFoldReply:
    def test_text_plus_image_rides_one_reply_call(self, _image_file):
        ad = _make_adapter()
        asyncio.run(_stage(ad, "here you go", _image_file))
        result = asyncio.run(ad.send(CHAT, "here you go"))
        assert result.success
        ad._client.reply.assert_awaited_once()
        ad._client.push.assert_not_awaited()
        messages = ad._client.reply.await_args.args[1]
        kinds = [m["type"] for m in messages]
        assert kinds == ["text", "image"]
        assert messages[0]["text"] == "here you go"

    def test_combined_send_marks_images_delivered_so_attachment_pass_skips_them(self, _image_file):
        ad = _make_adapter()
        asyncio.run(_stage(ad, "here you go", _image_file))
        asyncio.run(ad.send(CHAT, "here you go"))
        delivered = ad._fold_delivered[CHAT]
        assert delivered == {str(_image_file)}

    def test_expired_reply_token_folds_into_one_push(self, _image_file):
        ad = _make_adapter()
        ad._reply_tokens.clear()  # token already spent/expired
        asyncio.run(_stage(ad, "here you go", _image_file))
        result = asyncio.run(ad.send(CHAT, "here you go"))
        assert result.success
        ad._client.reply.assert_not_awaited()
        ad._client.push.assert_awaited_once()
        kinds = [m["type"] for m in ad._client.push.await_args.args[1]]
        assert kinds == ["text", "image"]

    def test_oversized_text_keeps_stock_split(self, _image_file):
        ad = _make_adapter()
        big = "\n\n".join(["x" * 4500] * 5) + f"\n\nMEDIA:{_image_file}"
        event = _event()
        ad._extract_media = lambda response: ([str(_image_file)], response.replace(f"MEDIA:{_image_file}", ""))
        ad.extract_images = lambda response: ([], response)
        ad.filter_media_delivery_paths = lambda paths, session_key=None: paths
        ad.extract_local_files = lambda text: ([], text)
        ad.filter_local_delivery_paths = lambda paths, session_key=None: paths
        asyncio.run(ad._extract_response_content(big, event, "sk", is_ephemeral_response=False))
        assert CHAT not in ad._fold_images  # too many bubbles to fit one batch

    def test_media_only_turn_is_not_staged(self, _image_file):
        ad = _make_adapter()
        event = _event()
        ad._extract_media = lambda response: ([str(_image_file)], "")
        ad.extract_images = lambda response: ([], response)
        ad.filter_media_delivery_paths = lambda paths, session_key=None: paths
        ad.extract_local_files = lambda text: ([], text)
        ad.filter_local_delivery_paths = lambda paths, session_key=None: paths
        extracted = asyncio.run(ad._extract_response_content(f"MEDIA:{_image_file}", event, "sk", is_ephemeral_response=False))
        assert extracted.text_content == ""
        assert CHAT not in ad._fold_images

    def test_ephemeral_notice_is_not_staged(self, _image_file):
        ad = _make_adapter()
        event = _event()
        ad._extract_media = lambda response: ([str(_image_file)], response)
        ad.extract_images = lambda response: ([], response)
        ad.filter_media_delivery_paths = lambda paths, session_key=None: paths
        asyncio.run(ad._extract_response_content(f"sorry MEDIA:{_image_file}", event, "sk", is_ephemeral_response=True))
        assert CHAT not in ad._fold_images

    def test_pending_postback_button_blocks_staging(self, _image_file):
        ad = _make_adapter()
        rid = ad._cache.register_pending(CHAT)
        ad._pending_buttons[CHAT] = rid
        event = _event()
        ad._extract_media = lambda response: ([str(_image_file)], response.replace(f"MEDIA:{_image_file}", ""))
        ad.extract_images = lambda response: ([], response)
        ad.filter_media_delivery_paths = lambda paths, session_key=None: paths
        ad.extract_local_files = lambda text: ([], text)
        ad.filter_local_delivery_paths = lambda paths, session_key=None: paths
        asyncio.run(ad._extract_response_content(f"answer MEDIA:{_image_file}", event, "sk", is_ephemeral_response=False))
        assert CHAT not in ad._fold_images

    def test_failed_combined_send_falls_back_to_separate_paths(self, _image_file):
        ad = _make_adapter()
        ad._client.reply = AsyncMock(side_effect=RuntimeError("reply rejected"))
        ad._client.push = AsyncMock(side_effect=RuntimeError("push rejected"))
        asyncio.run(_stage(ad, "here you go", _image_file))
        result = asyncio.run(ad.send(CHAT, "here you go"))
        assert not result.success
        assert CHAT not in ad._fold_images          # staging consumed either way
        assert CHAT not in ad._fold_delivered       # nothing marked delivered

    def test_attachment_pass_filters_delivered_images(self, _image_file):
        ad = _make_adapter()
        ad._fold_delivered[CHAT] = {str(_image_file)}
        extracted = _extracted(text="gone", media=[(str(_image_file), False)], local=["/tmp/other.bin"])
        event = _event()
        sent = {}

        async def fake_super(self, event, extracted, metadata, *, anything_sent, record_delivery):
            sent["media"] = list(extracted.media_files)
            sent["local"] = list(extracted.local_files)

        import unittest.mock as mock
        with mock.patch.object(_line.BasePlatformAdapter, "_deliver_attachments", new=fake_super):
            asyncio.run(ad._deliver_attachments(event, extracted, {}, anything_sent=True, record_delivery=lambda r: None))
        assert sent["media"] == []
        assert sent["local"] == ["/tmp/other.bin"]
        assert CHAT not in ad._fold_delivered

    def test_interim_send_does_not_consume_staging(self, _image_file):
        ad = _make_adapter()
        asyncio.run(_stage(ad, "here you go", _image_file))
        # A busy-ack heartbeat between extraction and the final text must not spend
        # the staged images on its own (metadata-marked interim) send.
        result = asyncio.run(ad.send(CHAT, "⏳ Working on it…", metadata={"_interim_send": True}))
        assert result.success
        assert CHAT in ad._fold_images  # staging survives for the real final send
        # The interim status did spend the reply token; the folded final still lands
        # everything in a single (push) call instead of text + separate metered image.
        final = asyncio.run(ad.send(CHAT, "here you go"))
        assert final.success
        ad._client.reply.assert_awaited_once()  # only the interim bubble used the token
        ad._client.push.assert_awaited_once()
        kinds = [m["type"] for m in ad._client.push.await_args.args[1]]
        assert kinds == ["text", "image"]
