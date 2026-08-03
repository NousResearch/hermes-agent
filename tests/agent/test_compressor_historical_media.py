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



    def test_text_part_is_not_image(self):
        assert _is_image_part(TEXT) is False
        assert _is_image_part(INPUT_TEXT) is False

    def test_non_dict_rejected(self):
        assert _is_image_part("image") is False
        assert _is_image_part(None) is False
        assert _is_image_part(42) is False


class TestContentHasImages:

    def test_empty_list(self):
        assert _content_has_images([]) is False



    def test_none(self):
        assert _content_has_images(None) is False


class TestStripImagesFromContent:



    def test_replaces_image_with_placeholder(self):
        parts = [TEXT, IMG_URL]
        out = _strip_images_from_content(parts)
        assert len(out) == 2
        assert out[0] == TEXT
        assert out[1] == {
            "type": "text",
            "text": "[Attached image — stripped after compression]",
        }


    def test_handles_all_three_shapes(self):
        parts = [IMG_URL, INPUT_IMG, ANTHROPIC_IMG, TEXT]
        out = _strip_images_from_content(parts)
        assert sum(1 for p in out if p.get("type") == "text") == 4
        assert not any(_is_image_part(p) for p in out)


class TestStripHistoricalMedia:
    def test_empty_passthrough(self):
        assert _strip_historical_media([]) == []








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
