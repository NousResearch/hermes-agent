"""Tests for image-token accounting in the context compressor.

Covers the native-image-routing PR's companion change: the compressor's
multimodal message length counter now charges ~1600 tokens per attached
image part instead of 0, so tail-cut / prune decisions are accurate for
creative workflows that iterate on images across many turns.
"""

from __future__ import annotations


from agent.context_compressor import (
    _CHARS_PER_TOKEN,
    _IMAGE_CHAR_EQUIVALENT,
    _IMAGE_TOKEN_ESTIMATE,
    _content_length_for_budget,
)


class TestContentLengthForBudget:
    def test_plain_string(self):
        assert _content_length_for_budget("hello world") == 11



    def test_text_only_list(self):
        content = [
            {"type": "text", "text": "first"},
            {"type": "text", "text": "second"},
        ]
        assert _content_length_for_budget(content) == 5 + 6







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
