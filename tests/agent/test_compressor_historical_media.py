"""Tests for post-compression historical-media stripping.

Port of Kilo-Org/kilocode#9434 (adapted for OpenAI-style message lists).
Without this pass, tail messages keep their original multi-MB base-64 image
payloads after context compression, and every subsequent request re-ships
them — sometimes breaching provider body-size limits and wedging the
session.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from agent.context_compressor import (
    ContextCompressor,
    _content_has_images,
    _is_image_part,
    _strip_historical_media,
    _strip_images_from_content,
)


IMG_URL = {
    "type": "image_url",
    "image_url": {"url": "data:image/png;base64," + ("A" * 1024)},
}
INPUT_IMG = {
    "type": "input_image",
    "image_url": "data:image/png;base64," + ("B" * 1024),
}
ANTHROPIC_IMG = {
    "type": "image",
    "source": {"type": "base64", "media_type": "image/png", "data": "C" * 1024},
}
TEXT = {"type": "text", "text": "hi"}
INPUT_TEXT = {"type": "input_text", "text": "hi"}


class TestIsImagePart:
    def test_openai_chat_shape(self):
        assert _is_image_part(IMG_URL) is True

    def test_openai_responses_shape(self):
        assert _is_image_part(INPUT_IMG) is True

    def test_anthropic_native_shape(self):
        assert _is_image_part(ANTHROPIC_IMG) is True

    def test_text_part_is_not_image(self):
        assert _is_image_part(TEXT) is False
        assert _is_image_part(INPUT_TEXT) is False

    def test_non_dict_rejected(self):
        assert _is_image_part("image") is False
        assert _is_image_part(None) is False
        assert _is_image_part(42) is False


class TestContentHasImages:
    def test_string_content(self):
        assert _content_has_images("a string") is False

    def test_empty_list(self):
        assert _content_has_images([]) is False

    def test_text_only_list(self):
        assert _content_has_images([TEXT, TEXT]) is False

    def test_list_with_image(self):
        assert _content_has_images([TEXT, IMG_URL]) is True

    def test_none(self):
        assert _content_has_images(None) is False


class TestStripImagesFromContent:
    def test_string_passthrough(self):
        assert _strip_images_from_content("hello") == "hello"

    def test_none_passthrough(self):
        assert _strip_images_from_content(None) is None

    def test_text_only_passthrough(self):
        parts = [TEXT, {"type": "text", "text": "world"}]
        assert _strip_images_from_content(parts) == parts

    def test_replaces_image_with_placeholder(self):
        parts = [TEXT, IMG_URL]
        out = _strip_images_from_content(parts)
        assert len(out) == 2
        assert out[0] == TEXT
        assert out[1] == {
            "type": "text",
            "text": "[Attached image — stripped after compression]",
        }

    def test_does_not_mutate_input(self):
        parts = [IMG_URL, TEXT]
        _ = _strip_images_from_content(parts)
        assert parts[0] is IMG_URL  # original list untouched
        assert parts[1] is TEXT

    def test_handles_all_three_shapes(self):
        parts = [IMG_URL, INPUT_IMG, ANTHROPIC_IMG, TEXT]
        out = _strip_images_from_content(parts)
        assert sum(1 for p in out if p.get("type") == "text") == 4
        assert not any(_is_image_part(p) for p in out)


class TestStripHistoricalMedia:
    def test_empty_passthrough(self):
        assert _strip_historical_media([]) == []

    def test_no_images_anywhere(self):
        msgs = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hey"},
            {"role": "user", "content": "bye"},
        ]
        assert _strip_historical_media(msgs) is msgs  # identity — no copy

    def test_single_image_user_only_first_message(self):
        # Only image-bearing user is the first message — nothing before it.
        msgs = [
            {"role": "user", "content": [TEXT, IMG_URL]},
            {"role": "assistant", "content": "ok"},
        ]
        out = _strip_historical_media(msgs)
        assert out is msgs  # no-op
        # Image still there.
        assert _content_has_images(out[0]["content"])

    def test_strips_older_user_image_keeps_newest(self):
        msgs = [
            {"role": "user", "content": [TEXT, IMG_URL]},     # old — strip
            {"role": "assistant", "content": "looked at it"},
            {"role": "user", "content": [TEXT, INPUT_IMG]},   # newest — keep
        ]
        out = _strip_historical_media(msgs)
        assert out is not msgs  # new list
        # First message's image was replaced
        assert not _content_has_images(out[0]["content"])
        # Newest user still has its image
        assert _content_has_images(out[2]["content"])

    def test_strips_assistant_and_tool_images_before_anchor(self):
        msgs = [
            {"role": "user", "content": [TEXT, IMG_URL]},          # old user
            {"role": "assistant", "content": [TEXT, IMG_URL]},     # old assistant
            {"role": "tool", "content": [TEXT, IMG_URL], "tool_call_id": "t1"},
            {"role": "user", "content": [TEXT, IMG_URL]},          # newest user — keep
        ]
        out = _strip_historical_media(msgs)
        for i in range(3):
            assert not _content_has_images(out[i]["content"]), f"msg {i} still has image"
        assert _content_has_images(out[3]["content"])

    def test_text_only_newest_user_still_strips_older_images(self):
        # The anchor is "newest user WITH images". If the newest user is
        # text-only, we fall back to the previous image-bearing user turn.
        msgs = [
            {"role": "user", "content": [TEXT, IMG_URL]},
            {"role": "assistant", "content": "ok"},
            {"role": "user", "content": [TEXT, IMG_URL]},  # anchor
            {"role": "assistant", "content": "done"},
            {"role": "user", "content": "follow-up text only"},
        ]
        out = _strip_historical_media(msgs)
        # First image-bearing user (index 0) was stripped — it was before the
        # newest image-bearing user (index 2).
        assert not _content_has_images(out[0]["content"])
        # Anchor (index 2) keeps its image.
        assert _content_has_images(out[2]["content"])

    def test_no_image_bearing_user_is_noop(self):
        msgs = [
            {"role": "user", "content": "first"},
            {"role": "assistant", "content": [TEXT, IMG_URL]},  # assistant image only
            {"role": "user", "content": "second"},
        ]
        out = _strip_historical_media(msgs)
        # No image-bearing user anchor → no stripping.
        assert out is msgs
        assert _content_has_images(out[1]["content"])

    def test_does_not_mutate_input_messages(self):
        msg0 = {"role": "user", "content": [TEXT, IMG_URL]}
        msg1 = {"role": "user", "content": [TEXT, IMG_URL]}
        msgs = [msg0, msg1]
        _ = _strip_historical_media(msgs)
        # Originals untouched
        assert _content_has_images(msg0["content"])
        assert _content_has_images(msg1["content"])

    def test_idempotent(self):
        msgs = [
            {"role": "user", "content": [TEXT, IMG_URL]},
            {"role": "assistant", "content": "k"},
            {"role": "user", "content": [TEXT, IMG_URL]},
        ]
        first = _strip_historical_media(msgs)
        second = _strip_historical_media(first)
        # Second pass is a no-op — no images left before the anchor.
        assert second is first

    def test_non_dict_messages_pass_through(self):
        msgs = [
            "not-a-dict",  # shouldn't crash
            {"role": "user", "content": [TEXT, IMG_URL]},
            {"role": "assistant", "content": "ok"},
            {"role": "user", "content": [TEXT, IMG_URL]},
        ]
        out = _strip_historical_media(msgs)
        assert out[0] == "not-a-dict"
        # Image-bearing user at index 1 is before the anchor (index 3) → stripped.
        assert not _content_has_images(out[1]["content"])


class TestCompressIntegration:
    """Verify the stripping runs inside ContextCompressor.compress()."""

    @pytest.fixture
    def compressor(self):
        with patch("agent.context_compressor.get_model_context_length", return_value=100_000):
            c = ContextCompressor(
                model="test/model",
                threshold_percent=0.50,
                protect_first_n=1,
                protect_last_n=2,
                quiet_mode=True,
            )
            return c

    def test_compress_strips_historical_images(self, compressor):
        # Enough messages to trigger the summarize path. protect_first_n=1 +
        # protect_last_n=2 + a middle window of at least 3 with a summary.
        msgs = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": [TEXT, IMG_URL]},           # old image-bearing user
            {"role": "assistant", "content": "looked at it"},
            {"role": "user", "content": "follow-up"},
            {"role": "assistant", "content": "ack"},
            {"role": "user", "content": "more"},
            {"role": "assistant", "content": "ok"},
            {"role": "user", "content": [TEXT, IMG_URL]},           # newest image-bearing user (tail)
            {"role": "assistant", "content": "done"},
        ]
        # Bypass the real LLM summary — return a stub so compress() proceeds.
        with patch.object(compressor, "_generate_summary", return_value="SUMMARY TEXT"):
            out = compressor.compress(msgs, current_tokens=60_000)

        # Newest user turn with image should still have it (it's in the tail).
        user_imgs = [m for m in out if m.get("role") == "user" and _content_has_images(m.get("content"))]
        assert len(user_imgs) == 1, (
            "Expected exactly one user message with images after compression "
            f"(the newest one); got {len(user_imgs)}"
        )
        # No assistant or tool messages should carry images either.
        for m in out:
            if m is user_imgs[0]:
                continue
            assert not _content_has_images(m.get("content")), (
                f"Stale image in {m.get('role')!r} message after compression"
            )
