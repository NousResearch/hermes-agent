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

    def test_strips_stale_tool_result_images_when_no_user_image_exists(self):
        """#89938: vision_analyze results are the only images in the session.

        Before this rule the anchor stayed at -1 and the list came back
        untouched, so every base64 blob rode along on every request.
        """
        msgs = [
            {"role": "user", "content": "look at these"},
            {"role": "tool", "tool_call_id": "a", "content": [TEXT, IMG_URL]},
            {"role": "tool", "tool_call_id": "b", "content": [TEXT, IMG_URL]},
            {"role": "tool", "tool_call_id": "c", "content": [TEXT, IMG_URL]},
        ]
        out = _strip_historical_media(msgs)

        assert not _content_has_images(out[1]["content"])
        assert not _content_has_images(out[2]["content"])
        # The newest tool image is what the model is reasoning about.
        assert _content_has_images(out[3]["content"])

    def test_first_message_user_image_no_longer_blocks_tool_stripping(self):
        """#89938's other half: ``anchor <= 0`` used to return early.

        One attachment on the opening message plus a run of vision tool calls
        is the exact reproduction in the report - and the old early return
        meant the 413 recovery compaction freed nothing.
        """
        msgs = [
            {"role": "user", "content": [TEXT, IMG_URL]},
            {"role": "tool", "tool_call_id": "a", "content": [TEXT, IMG_URL]},
            {"role": "tool", "tool_call_id": "b", "content": [TEXT, IMG_URL]},
        ]
        out = _strip_historical_media(msgs)

        # Rule 1b: newer tool images supersede the opening attachment, so
        # its bytes age out too — the row survives with a text placeholder.
        assert not _content_has_images(out[0]["content"])
        assert out[0]["role"] == "user"
        assert not _content_has_images(out[1]["content"])
        assert _content_has_images(out[2]["content"])

    def test_first_message_image_kept_when_it_is_the_only_image(self):
        """Rule 1b only fires when something newer supersedes the opener."""
        msgs = [
            {"role": "user", "content": [TEXT, IMG_URL]},
            {"role": "assistant", "content": "looked"},
            {"role": "tool", "tool_call_id": "a", "content": [TEXT]},
        ]
        assert _strip_historical_media(msgs) is msgs

    def test_multimodal_envelope_tool_results_age_out(self):
        """The native ``{_multimodal: True}`` dict envelope must strip too.

        vision_analyze hands back this shape before adapters unwrap it; the
        bare list matcher never saw it, so envelope-shaped results kept their
        base64 through every compaction (#89965's shape-coverage gap).
        """
        env = {
            "_multimodal": True,
            "text_summary": "a poster",
            "content": [TEXT, IMG_URL],
        }
        msgs = [
            {"role": "user", "content": "look"},
            {"role": "tool", "tool_call_id": "a", "content": dict(env), "api_content": "stale"},
            {"role": "tool", "tool_call_id": "b", "content": dict(env)},
        ]
        out = _strip_historical_media(msgs)

        # Older envelope collapses to its text summary; sidecar dropped.
        assert isinstance(out[1]["content"], str)
        assert "a poster" in out[1]["content"]
        assert "api_content" not in out[1]
        assert out[1]["tool_call_id"] == "a"
        # Newest envelope is the anchor and survives byte-for-byte.
        assert out[2] is msgs[2]

    def test_all_three_wire_shapes_strip_in_tool_results(self):
        """Chat Completions, Responses, and Anthropic-native parts all age."""
        for img in (IMG_URL, INPUT_IMG, ANTHROPIC_IMG):
            msgs = [
                {"role": "tool", "tool_call_id": "a", "content": [TEXT, dict(img)]},
                {"role": "tool", "tool_call_id": "b", "content": [TEXT, dict(img)]},
            ]
            out = _strip_historical_media(msgs)
            assert not _content_has_images(out[0]["content"]), img["type"]
            assert _content_has_images(out[1]["content"]), img["type"]

    def test_deterministic_double_run(self):
        """Running the strip twice yields byte-identical output."""
        import json

        msgs = [
            {"role": "user", "content": [TEXT, IMG_URL]},
            {"role": "tool", "tool_call_id": "a", "content": [TEXT, IMG_URL]},
            {"role": "tool", "tool_call_id": "b", "content": [TEXT, INPUT_IMG]},
        ]
        first = _strip_historical_media(msgs)
        second = _strip_historical_media(first)
        assert json.dumps(first, sort_keys=True) == json.dumps(second, sort_keys=True)
        assert second is first  # second pass is a no-op

    def test_newest_tool_image_survives_inside_the_protected_tail(self):
        msgs = [
            {"role": "user", "content": "hi"},
            {"role": "user", "content": [TEXT, IMG_URL]},
            {"role": "tool", "tool_call_id": "a", "content": [TEXT, IMG_URL]},
            {"role": "tool", "tool_call_id": "b", "content": [TEXT, IMG_URL]},
        ]
        out = _strip_historical_media(msgs)

        # The user anchor is index 1, so nothing before it changes and the
        # anchor itself is kept byte-for-byte (test_compressor_zero_user_guard
        # depends on that).
        assert out[1] is msgs[1]
        assert not _content_has_images(out[2]["content"])
        assert _content_has_images(out[3]["content"])

    def test_tool_image_before_the_user_anchor_is_still_stripped(self):
        """Rule 1 keeps precedence over rule 2 where the two disagree."""
        msgs = [
            {"role": "tool", "tool_call_id": "a", "content": [TEXT, IMG_URL]},
            {"role": "assistant", "content": "ok"},
            {"role": "user", "content": [TEXT, IMG_URL]},
        ]
        out = _strip_historical_media(msgs)

        # Index 0 is the newest *tool* image, but it sits before the user
        # anchor, which has always stripped it. That must not regress.
        assert not _content_has_images(out[0]["content"])
        assert _content_has_images(out[2]["content"])

    def test_unchanged_when_nothing_carries_images(self):
        msgs = [
            {"role": "user", "content": [TEXT]},
            {"role": "tool", "tool_call_id": "a", "content": [TEXT]},
        ]
        assert _strip_historical_media(msgs) is msgs

    def test_single_tool_image_is_left_alone(self):
        msgs = [
            {"role": "user", "content": "look"},
            {"role": "tool", "tool_call_id": "a", "content": [TEXT, IMG_URL]},
        ]
        assert _strip_historical_media(msgs) is msgs

    def test_idempotent_over_tool_images(self):
        msgs = [
            {"role": "tool", "tool_call_id": "a", "content": [TEXT, IMG_URL]},
            {"role": "tool", "tool_call_id": "b", "content": [TEXT, IMG_URL]},
        ]
        first = _strip_historical_media(msgs)
        assert first is not msgs
        assert _strip_historical_media(first) is first

    def test_stripped_tool_message_drops_its_api_content_sidecar(self):
        """Replaying the sidecar would resend the bytes the strip removed."""
        msgs = [
            {
                "role": "tool",
                "tool_call_id": "a",
                "content": [TEXT, IMG_URL],
                "api_content": "the exact multimodal bytes sent last turn",
            },
            {"role": "tool", "tool_call_id": "b", "content": [TEXT, IMG_URL]},
        ]
        out = _strip_historical_media(msgs)

        assert "api_content" not in out[0]
        # The input list is never mutated.
        assert "api_content" in msgs[0]

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

    def test_compress_frees_stale_vision_tool_results(self, compressor):
        """#89938 end to end: the 413 recovery compaction must free bytes.

        The reported session had one attachment on the opening message and a
        run of ``vision_analyze`` results after it. ``anchor <= 0`` made this
        pass a no-op, so every recovery compaction returned a body that was
        still multi-MB and the provider answered 413 again - seven times in
        thirteen minutes.
        """

        def call(idx: str):
            return [
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": idx,
                            "type": "function",
                            "function": {"name": "vision_analyze", "arguments": "{}"},
                        }
                    ],
                },
                {"role": "tool", "tool_call_id": idx, "content": [TEXT, IMG_URL]},
            ]

        msgs = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": [TEXT, IMG_URL]},  # the ONLY user image, first
            *call("a"),
            *call("b"),
            {"role": "user", "content": "and this one?"},
            *call("c"),
        ]
        with patch.object(compressor, "_generate_summary", return_value="SUMMARY TEXT"):
            out = compressor.compress(msgs, current_tokens=60_000)

        with_images = [m for m in out if isinstance(m, dict) and _content_has_images(m.get("content"))]
        # Exactly one survivor: the newest vision result. The opening
        # attachment is protect_first_n material and may or may not survive
        # the summary window, so assert on what must NOT be there instead.
        assert len(with_images) <= 2
        stale_tool_images = [
            m for m in out
            if isinstance(m, dict)
            and m.get("role") == "tool"
            and _content_has_images(m.get("content"))
            and m.get("tool_call_id") != "c"
        ]
        assert stale_tool_images == [], (
            "vision_analyze results a, b still carry base64 after compression"
        )
