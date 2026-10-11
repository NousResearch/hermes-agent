"""Tests for audio in post-compression historical-media stripping and the estimate shadow.

Native inbound clips arrive as OpenAI-style ``input_audio`` parts whose payload is base64 —
the same failure class as images: without this pass a stale clip rides every request after
compaction, and without the shadow its encoded bytes inflate every token estimate.

Covers the two halves of the audio media pass:

* ``agent.context_compressor`` — ``_is_audio_part`` / ``_strip_audio_from_content`` replace
  aged-out clips with ``[Attached audio — stripped after compression]`` (image placeholder
  wording is unchanged), on the same keep-newest timeline images use;
* ``agent.model_metadata._wire_message_shadow`` — an audio part collapses to
  ``{"type": …, "audio": "[stripped]"}`` so base64 never reaches the token estimate.
"""

from __future__ import annotations

import copy
import json
from unittest.mock import patch

import pytest

from agent.context_compressor import (
    ContextCompressor,
    _content_has_audio,
    _content_has_images,
    _content_has_media,
    _is_audio_part,
    _strip_audio_from_content,
    _strip_historical_media,
)

AUDIO_PLACEHOLDER = "[Attached audio — stripped after compression]"
IMAGE_PLACEHOLDER = "[Attached image — stripped after compression]"

TEXT = {"type": "text", "text": "hi"}
INPUT_TEXT = {"type": "input_text", "text": "hi"}
IMG_URL = {"type": "image_url", "image_url": {"url": "data:image/png;base64," + ("A" * 1024)}}
AUDIO = {"type": "input_audio", "input_audio": {"data": "Q" * 1024, "format": "wav"}}
GENERIC_AUDIO = {"type": "audio", "audio": {"data": "R" * 1024, "format": "mp3"}}


def _audio(data: str = "Q" * 1024) -> dict:
    return {"type": "input_audio", "input_audio": {"data": data, "format": "wav"}}


class TestIsAudioPart:
    def test_audio_parts(self):
        assert _is_audio_part(AUDIO) is True
        assert _is_audio_part(GENERIC_AUDIO) is True

    def test_non_audio_parts(self):
        assert _is_audio_part(TEXT) is False
        assert _is_audio_part(INPUT_TEXT) is False
        assert _is_audio_part(IMG_URL) is False

    def test_non_dict_rejected(self):
        assert _is_audio_part("input_audio") is False
        assert _is_audio_part(None) is False
        assert _is_audio_part(42) is False


class TestContentHasAudio:
    def test_detects_both_shapes(self):
        assert _content_has_audio([TEXT, AUDIO]) is True
        assert _content_has_audio([GENERIC_AUDIO]) is True

    def test_no_audio(self):
        assert _content_has_audio([TEXT, IMG_URL]) is False
        assert _content_has_audio([]) is False
        assert _content_has_audio(None) is False
        assert _content_has_audio("plain") is False

    def test_media_covers_image_and_audio(self):
        assert _content_has_media([TEXT, IMG_URL]) is True
        assert _content_has_media([TEXT, AUDIO]) is True
        assert _content_has_media([TEXT]) is False


class TestStripAudioFromContent:
    def test_replaces_audio_with_placeholder(self):
        parts = [TEXT, AUDIO]
        out = _strip_audio_from_content(parts)
        assert out == [
            TEXT,
            {"type": "text", "text": AUDIO_PLACEHOLDER},
        ]

    def test_image_placeholder_wording_unchanged(self):
        """The audio pass must not touch the image wording other tests pin."""
        from agent.context_compressor import _strip_images_from_content

        out = _strip_images_from_content([TEXT, IMG_URL])
        assert out[1] == {"type": "text", "text": IMAGE_PLACEHOLDER}
        # ...and the audio stripper leaves images alone.
        mixed = _strip_audio_from_content([TEXT, IMG_URL, AUDIO])
        assert mixed[1] == IMG_URL
        assert mixed[2] == {"type": "text", "text": AUDIO_PLACEHOLDER}

    def test_unchanged_when_no_audio(self):
        parts = [TEXT, IMG_URL]
        assert _strip_audio_from_content(parts) is parts
        assert _strip_audio_from_content(None) is None

    def test_generic_spelling_replaced(self):
        out = _strip_audio_from_content([GENERIC_AUDIO])
        assert out == [{"type": "text", "text": AUDIO_PLACEHOLDER}]


class TestStripHistoricalMediaAudio:
    def test_empty_passthrough(self):
        assert _strip_historical_media([]) == []

    def test_stale_audio_before_the_user_anchor_is_stripped(self):
        """Rule 1 with audio only: the clip ages out, the newest one survives."""
        msgs = [
            {"role": "user", "content": [TEXT, _audio("OLDOLDOLD")]},
            {"role": "assistant", "content": "heard it"},
            {"role": "user", "content": [TEXT, _audio("NEWNEWNEW")]},
        ]
        before = copy.deepcopy(msgs)
        out = _strip_historical_media(msgs)

        assert out is not msgs
        assert out[0]["content"] == [TEXT, {"type": "text", "text": AUDIO_PLACEHOLDER}]
        assert _content_has_audio(out[2]["content"]) is True
        # Input never mutated.
        assert msgs == before

    def test_audio_only_session_strips_without_any_image(self):
        """The regression this exists for: with no images anywhere the old image-only
        anchor logic early-returned and every clip rode every request forever."""
        msgs = [
            {"role": "user", "content": "listen to these"},
            {"role": "user", "content": [_audio("FIRST")]},
            {"role": "assistant", "content": "heard it"},
            {"role": "user", "content": [TEXT, _audio("SECOND")]},
        ]
        out = _strip_historical_media(msgs)

        assert not any(_content_has_images(m.get("content")) for m in out), (
            "this session must be image-free for the audio path to be what strips it"
        )
        assert _content_has_audio(out[1]["content"]) is False
        assert AUDIO_PLACEHOLDER in str(out[1]["content"])
        assert _content_has_audio(out[3]["content"]) is True

    def test_opening_audio_ages_out_once_a_newer_tool_result_exists(self):
        """Rule 1b applies to audio: an opening clip stops surviving when the model
        has moved on to newer tool traffic."""
        msgs = [
            {"role": "user", "content": [TEXT, AUDIO]},
            {"role": "assistant", "content": None, "tool_calls": [
                {"id": "a", "type": "function", "function": {"name": "vision_analyze", "arguments": "{}"}},
            ]},
            {"role": "tool", "tool_call_id": "a", "content": [TEXT, IMG_URL]},
        ]
        out = _strip_historical_media(msgs)

        assert _content_has_audio(out[0]["content"]) is False
        assert AUDIO_PLACEHOLDER in str(out[0]["content"])
        # The newest tool image is what the model is reasoning about.
        assert _content_has_images(out[2]["content"]) is True

    def test_newest_tool_audio_survives_older_ones_age_out(self):
        """Rule 2 on the audio timeline: keep only the newest tool-result clip."""
        msgs = [
            {"role": "tool", "tool_call_id": "a", "content": [TEXT, _audio("AAAA")]},
            {"role": "tool", "tool_call_id": "b", "content": [TEXT, _audio("BBBB")]},
        ]
        out = _strip_historical_media(msgs)

        assert _content_has_audio(out[0]["content"]) is False
        assert _content_has_audio(out[1]["content"]) is True

    def test_audio_only_tool_row_gets_placeholder_not_dropped(self):
        """An audio-only tool result keeps its row — dropping it orphans tool_call_id."""
        msgs = [
            {"role": "tool", "tool_call_id": "a", "content": [_audio("AAAA")]},
            {"role": "tool", "tool_call_id": "b", "content": [TEXT]},
        ]
        # A tool row before the newest media-bearing tool row must carry media itself to
        # be rewritten; without audio/image it is simply left alone.
        out = _strip_historical_media(msgs)
        assert out is msgs

        msgs = [
            {"role": "tool", "tool_call_id": "a", "content": [_audio("AAAA")]},
            {"role": "tool", "tool_call_id": "b", "content": [TEXT, _audio("BBBB")]},
        ]
        out = _strip_historical_media(msgs)
        assert out[0]["tool_call_id"] == "a"
        assert isinstance(out[0]["content"], list)
        assert out[0]["content"] == [{"type": "text", "text": AUDIO_PLACEHOLDER}]
        assert _content_has_audio(out[1]["content"]) is True

    def test_single_opening_clip_is_left_alone(self):
        """Rule 1b only fires when something newer supersedes the opener."""
        msgs = [
            {"role": "user", "content": [TEXT, AUDIO]},
            {"role": "assistant", "content": "ok"},
            {"role": "tool", "tool_call_id": "a", "content": [TEXT]},
        ]
        assert _strip_historical_media(msgs) is msgs

    def test_unchanged_when_nothing_carries_media(self):
        msgs = [
            {"role": "user", "content": [TEXT]},
            {"role": "tool", "tool_call_id": "a", "content": [TEXT]},
        ]
        assert _strip_historical_media(msgs) is msgs

    def test_idempotent_and_deterministic(self):
        msgs = [
            {"role": "user", "content": [TEXT, IMG_URL, AUDIO]},
            {"role": "tool", "tool_call_id": "a", "content": [TEXT, _audio("AAAA")]},
            {"role": "user", "content": [TEXT, _audio("BBBB")]},
        ]
        first = _strip_historical_media(msgs)
        second = _strip_historical_media(first)
        assert second is first  # second pass is a no-op
        assert json.dumps(first, sort_keys=True) == json.dumps(second, sort_keys=True)

    def test_stripped_row_drops_its_api_content_sidecar(self):
        """Replaying the sidecar would resend the bytes the strip removed."""
        msgs = [
            {
                "role": "user",
                "content": [TEXT, AUDIO],
                "api_content": "the exact multimodal bytes sent last turn",
            },
            {"role": "user", "content": [TEXT, _audio("NEW")]},
        ]
        out = _strip_historical_media(msgs)

        assert "api_content" not in out[0]
        assert "api_content" in msgs[0]  # input list is never mutated

    def test_non_dict_messages_pass_through(self):
        msgs = [
            "not-a-dict",
            {"role": "user", "content": [TEXT, AUDIO]},
            {"role": "assistant", "content": "ok"},
            {"role": "user", "content": [TEXT, AUDIO]},
        ]
        out = _strip_historical_media(msgs)
        assert out[0] == "not-a-dict"
        assert _content_has_audio(out[1]["content"]) is False

    def test_audio_only_multimodal_envelope_loses_the_clip_keeps_the_summary(self):
        env = {"_multimodal": True, "text_summary": "a voice note", "content": [TEXT, AUDIO]}
        msgs = [
            {"role": "user", "content": "and this one?"},
            {"role": "tool", "tool_call_id": "a", "content": dict(env), "api_content": "stale"},
            {"role": "tool", "tool_call_id": "b", "content": dict(env)},
        ]
        out = _strip_historical_media(msgs)

        body = out[1]["content"]
        assert isinstance(body, dict) and body["_multimodal"] is True
        assert body["text_summary"] == "a voice note"
        assert _content_has_audio(body["content"]) is False
        assert "api_content" not in out[1]
        # Newest envelope is the anchor and survives.
        assert out[2] is msgs[2]


class TestCompressIntegrationAudio:
    """The strip must run inside ContextCompressor.compress()."""

    @pytest.fixture
    def compressor(self):
        with patch("agent.context_compressor.get_model_context_length", return_value=100_000):
            return ContextCompressor(
                model="test/model",
                threshold_percent=0.50,
                protect_first_n=1,
                protect_last_n=2,
                quiet_mode=True,
            )

    def test_compress_strips_historical_audio(self, compressor):
        msgs = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": [TEXT, _audio("OLD")]},      # old clip
            {"role": "assistant", "content": "heard it"},
            {"role": "user", "content": "follow-up"},
            {"role": "assistant", "content": "ack"},
            {"role": "user", "content": "more"},
            {"role": "assistant", "content": "ok"},
            {"role": "user", "content": [TEXT, _audio("NEW")]},      # newest clip (tail)
            {"role": "assistant", "content": "done"},
        ]
        with patch.object(compressor, "_generate_summary", return_value="SUMMARY TEXT"):
            out = compressor.compress(msgs, current_tokens=60_000)

        carriers = [m for m in out if _content_has_audio(m.get("content"))]
        assert len(carriers) == 1, (
            "Expected exactly one message still carrying audio after compression "
            f"(the newest one); got {len(carriers)}"
        )
        assert carriers[0]["role"] == "user"


class TestWireMessageShadowAudio:
    """Base64 audio must never enter the token estimate (mirror of the image shadow test)."""

    @staticmethod
    def _big_audio() -> dict:
        # ~300 KB of base64 would be ~100K tokens as text.
        import base64
        import os

        return {
            "type": "input_audio",
            "input_audio": {
                "data": base64.b64encode(os.urandom(300_000)).decode(),
                "format": "wav",
            },
        }

    def test_audio_part_is_shadowed(self):
        from agent.model_metadata import _wire_message_shadow

        msg = {"role": "user", "content": [TEXT, self._big_audio()]}
        shadow = _wire_message_shadow(msg)

        assert shadow["content"][0] == TEXT
        assert shadow["content"][1] == {"type": "input_audio", "audio": "[stripped]"}
        assert "Q" * 100 not in json.dumps(shadow)

    def test_generic_audio_shape_is_shadowed(self):
        from agent.model_metadata import _wire_message_shadow

        msg = {"role": "user", "content": [GENERIC_AUDIO]}
        assert _wire_message_shadow(msg)["content"] == [{"type": "audio", "audio": "[stripped]"}]

    def test_image_marker_wording_unchanged(self):
        from agent.model_metadata import _wire_message_shadow

        msg = {"role": "user", "content": [IMG_URL]}
        assert _wire_message_shadow(msg)["content"] == [{"type": "image_url", "image": "[stripped]"}]

    def test_responses_output_parts_are_shadowed(self):
        from agent.model_metadata import _wire_message_shadow

        item = {
            "type": "function_call_output",
            "call_id": "call_1",
            "output": [{"type": "input_audio", "input_audio": {"data": "Z" * 4096, "format": "wav"}}],
        }
        assert _wire_message_shadow(item)["output"] == [
            {"type": "input_audio", "audio": "[stripped]"}
        ]

    def test_estimate_ignores_audio_bytes(self):
        from agent.model_metadata import estimate_messages_tokens_rough

        msg = {"role": "user", "content": [TEXT, self._big_audio()]}
        # Raw base64 would be ~100K tokens; the shadow keeps it in single digits.
        assert estimate_messages_tokens_rough([msg]) < 5_000

    def test_estimate_still_prices_text(self):
        from agent.model_metadata import estimate_messages_tokens_rough

        body = "audio clip transcript " * 500
        baseline = estimate_messages_tokens_rough([{"role": "user", "content": body}])
        assert baseline > 1_000
        with_audio = estimate_messages_tokens_rough(
            [{"role": "user", "content": [{"type": "text", "text": body}, AUDIO]}]
        )
        assert with_audio >= baseline
