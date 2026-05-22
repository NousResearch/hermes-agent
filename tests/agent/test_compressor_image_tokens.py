"""Tests for image-token accounting in the context compressor.

Covers the native-image-routing PR's companion change: the compressor's
multimodal message length counter now charges ~1600 tokens per attached
image part instead of 0, so tail-cut / prune decisions are accurate for
creative workflows that iterate on images across many turns.
"""

from __future__ import annotations

import pytest

from agent.context_compressor import (
    _CHARS_PER_TOKEN,
    _IMAGE_CHAR_EQUIVALENT,
    _IMAGE_TOKEN_ESTIMATE,
    _content_length_for_budget,
)


class TestContentLengthForBudget:
    def test_plain_string(self):
        assert _content_length_for_budget("hello world") == 11

    def test_empty_string(self):
        assert _content_length_for_budget("") == 0

    def test_none_coerces_to_zero(self):
        assert _content_length_for_budget(None) == 0

    def test_text_only_list(self):
        content = [
            {"type": "text", "text": "first"},
            {"type": "text", "text": "second"},
        ]
        assert _content_length_for_budget(content) == 5 + 6

    def test_single_image_part_charges_fixed_budget(self):
        content = [
            {"type": "text", "text": "look"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,XXXX"}},
        ]
        # 4 chars of text + 1 image at fixed char-equivalent
        assert _content_length_for_budget(content) == 4 + _IMAGE_CHAR_EQUIVALENT

    def test_image_url_raw_base64_is_not_counted_as_chars(self):
        """A 1MB base64 blob inside an image_url must NOT inflate token count.

        The flat image estimate is what the provider actually bills; the raw
        base64 is transport payload, not context tokens.
        """
        huge_url = "data:image/png;base64," + ("A" * 1_000_000)
        content = [
            {"type": "image_url", "image_url": {"url": huge_url}},
        ]
        # Exactly one image's worth, not 1M + something.
        assert _content_length_for_budget(content) == _IMAGE_CHAR_EQUIVALENT

    def test_multiple_image_parts(self):
        content = [
            {"type": "text", "text": "compare"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAA"}},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,BBB"}},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,CCC"}},
        ]
        assert _content_length_for_budget(content) == 7 + 3 * _IMAGE_CHAR_EQUIVALENT

    def test_openai_responses_input_image_shape(self):
        """Responses API uses type=input_image with top-level image_url string."""
        content = [
            {"type": "input_text", "text": "hey"},
            {"type": "input_image", "image_url": "data:image/png;base64,XX"},
        ]
        # input_text has .text "hey" (3 chars) + 1 image
        assert _content_length_for_budget(content) == 3 + _IMAGE_CHAR_EQUIVALENT

    def test_anthropic_native_image_shape(self):
        """Anthropic native shape: {type: image, source: {...}}."""
        content = [
            {"type": "text", "text": "hi"},
            {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "XX"}},
        ]
        assert _content_length_for_budget(content) == 2 + _IMAGE_CHAR_EQUIVALENT

    def test_bare_string_part_in_list(self):
        """Older code paths sometimes produce mixed list-of-strings content."""
        content = ["hello", {"type": "text", "text": "world"}]
        assert _content_length_for_budget(content) == 5 + 5

    def test_image_estimate_constant_is_reasonable(self):
        """Sanity-check the estimate aligns with real provider billing.

        Anthropic ≈ width*height/750 → ~1600 for 1000×1200.
        OpenAI GPT-4o high-detail 2048×2048 ≈ 1445.
        Gemini 258/tile × 6 tiles for a 2048×2048 ≈ 1548.
        Anything in the 800-2000 range is defensible. Enforce bounds so an
        accidental edit doesn't drop it to e.g. 16.
        """
        assert 800 <= _IMAGE_TOKEN_ESTIMATE <= 2500
        assert _IMAGE_CHAR_EQUIVALENT == _IMAGE_TOKEN_ESTIMATE * _CHARS_PER_TOKEN


class TestTokenBudgetWithImages:
    """Integration: the compressor's tail-cut decision now respects image cost."""

    def test_image_heavy_turns_count_toward_budget(self):
        """A tail with 5 image-bearing turns should blow past a 5K token budget."""
        from agent.context_compressor import ContextCompressor

        # Minimal compressor fixture — just enough to call _find_tail_cut_by_tokens
        cc = object.__new__(ContextCompressor)
        cc.tail_token_budget = 5000

        # Build 10 messages: 5 with images, 5 with short text. Without the
        # image-tokens fix, the compressor would think all 10 fit in 5K and
        # protect them all. With the fix, images alone cost 5 × 1600 = 8K,
        # so the tail should be trimmed.
        messages = [{"role": "system", "content": "sys"}]
        for i in range(5):
            messages.append({
                "role": "user",
                "content": [
                    {"type": "text", "text": f"turn {i}"},
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAA"}},
                ],
            })
            messages.append({
                "role": "assistant",
                "content": f"response {i}",
            })

        cut = cc._find_tail_cut_by_tokens(messages, head_end=0, token_budget=5000)

        # Budget is 5K, soft ceiling 7.5K. 5 images alone = 8000 image-tokens.
        # Walking backward, the compressor should stop before including all 5.
        # Exact cut depends on text lengths and min_tail, but it MUST be > 1
        # (at least some head-side messages should be compressible).
        assert cut > 1, (
            f"Expected image-heavy tail to be trimmed; compressor placed cut at "
            f"{cut} out of {len(messages)} (image tokens were likely ignored)."
        )
