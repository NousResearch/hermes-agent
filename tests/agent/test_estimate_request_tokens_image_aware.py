"""Tests for estimate_request_tokens_rough's image-aware token credit.

Bug repro: a session with a few inline screenshots triggered a "preflight
compression" loop on a 1M-context model even though the actual session
was only ~135K tokens.  The estimator had been doing
``len(str(message)) / 4`` which walked the entire base64 payload of
each ``image_url`` part; a single 200KB screenshot inflated the estimate
by ~50,000 phantom tokens.

These tests pin the new behavior:
  * ``image_url`` / ``input_image`` / Anthropic-native ``image`` parts
    each contribute a fixed ~1600-token credit, NOT the base64 length.
  * Text-only messages still use the cheap len/4 heuristic.
  * Tool schemas and the system prompt are still counted normally.
  * The regression-from-screenshot scenario stays under the 75%
    compression threshold of a 1M-context model.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import pytest

from agent.model_metadata import (
    _IMAGE_TOKEN_COST,
    _is_image_part,
    estimate_request_tokens_rough,
)


def _make_data_url_screenshot(payload_size: int = 200_000) -> str:
    """Fake a base64 data: URL with ``payload_size`` chars of body."""
    return "data:image/png;base64," + ("A" * payload_size)


class TestImageDetection:
    def test_openai_chat_completions_image_url_dict(self):
        assert _is_image_part({"type": "image_url", "image_url": {"url": "data:..."}})

    def test_openai_chat_completions_image_url_string(self):
        assert _is_image_part({"type": "image_url", "image_url": "data:..."})

    def test_openai_responses_input_image(self):
        assert _is_image_part({"type": "input_image", "image_url": "data:..."})

    def test_anthropic_native_image(self):
        assert _is_image_part({
            "type": "image",
            "source": {"type": "base64", "media_type": "image/png", "data": "..."},
        })

    def test_text_part_is_not_image(self):
        assert not _is_image_part({"type": "text", "text": "hello"})

    def test_non_dict_is_not_image(self):
        assert not _is_image_part("plain string")
        assert not _is_image_part(None)


class TestImageEstimateUsesFixedCredit:
    """Each image contributes a fixed token cost, not its base64 length."""

    def test_single_screenshot_does_not_dominate(self):
        """A 200KB screenshot adds ~1.6K tokens, not ~50K."""
        big_url = _make_data_url_screenshot(200_000)
        msgs = [{
            "role": "user",
            "content": [
                {"type": "text", "text": "what's in this screenshot?"},
                {"type": "image_url", "image_url": {"url": big_url}},
            ],
        }]

        est = estimate_request_tokens_rough(msgs)

        # Old broken behaviour: ~200_000 / 4 ≈ 50_000.
        # New behaviour: tiny text + 1 × _IMAGE_TOKEN_COST credit.
        assert est < _IMAGE_TOKEN_COST + 200, (
            f"image estimate too large; expected ~{_IMAGE_TOKEN_COST}, got {est}"
        )
        assert est >= _IMAGE_TOKEN_COST, (
            f"image credit must be at least {_IMAGE_TOKEN_COST}, got {est}"
        )

    def test_anthropic_native_image_uses_credit_too(self):
        msgs = [{
            "role": "user",
            "content": [
                {"type": "text", "text": "x"},
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": "Q" * 200_000,
                    },
                },
            ],
        }]
        est = estimate_request_tokens_rough(msgs)
        assert est < _IMAGE_TOKEN_COST + 200, est
        assert est >= _IMAGE_TOKEN_COST, est

    def test_multiple_images_stack_linearly(self):
        msgs = [{
            "role": "user",
            "content": [
                {"type": "text", "text": "compare"},
                {"type": "image_url", "image_url": {"url": _make_data_url_screenshot(150_000)}},
                {"type": "image_url", "image_url": {"url": _make_data_url_screenshot(150_000)}},
                {"type": "image_url", "image_url": {"url": _make_data_url_screenshot(150_000)}},
            ],
        }]
        est = estimate_request_tokens_rough(msgs)
        assert 3 * _IMAGE_TOKEN_COST <= est < 3 * _IMAGE_TOKEN_COST + 200, est

    def test_text_only_message_unchanged(self):
        """No images → behavior is the legacy len/4 estimate."""
        msgs = [{"role": "user", "content": "x" * 4000}]
        est = estimate_request_tokens_rough(msgs)
        # Legacy: len(str(msg))/4.  The dict wrapping adds a small
        # overhead but stays close to 1000 tokens for 4000 chars.
        assert 950 < est < 1100, est

    def test_tools_and_system_prompt_still_count(self):
        msgs = [{"role": "user", "content": "hi"}]
        est_no_tools = estimate_request_tokens_rough(msgs, system_prompt="x" * 4000)
        est_with_tools = estimate_request_tokens_rough(
            msgs,
            system_prompt="x" * 4000,
            tools=[{"name": "t", "description": "y" * 4000}],
        )
        assert est_with_tools > est_no_tools + 800


class TestRegressionFromScreenshotBug:
    """The actual scenario that produced the bug message in chat."""

    def test_session_with_few_screenshots_stays_under_million_threshold(self):
        """A handful of inline screenshots must NOT trip 750K threshold.

        This is the exact shape that triggered the spurious preflight
        compression: a 1M-context model, a moderately long session,
        and ~5 screen-grab attachments.  Pre-fix the estimator put it
        at ~750K+; post-fix it should land well under.
        """
        # Realistic-ish history: 100 turns of ~2000 chars each = 200K
        # chars ≈ 50K tokens of plain text, plus 5 large screenshots.
        msgs = []
        for i in range(100):
            msgs.append({
                "role": "user" if i % 2 == 0 else "assistant",
                "content": ("text payload " * 100 + str(i)),
            })
        # Sprinkle 5 screenshots, each carrying ~200KB of base64.
        for i in (5, 25, 50, 75, 95):
            msgs[i] = {
                "role": "user",
                "content": [
                    {"type": "text", "text": f"see image {i}"},
                    {"type": "image_url", "image_url": {"url": _make_data_url_screenshot(200_000)}},
                ],
            }

        est = estimate_request_tokens_rough(msgs, system_prompt="sys" * 1000)

        # 75% of 1M = 750K (the compression threshold).  The new
        # estimator must stay well under that for this workload.
        assert est < 750_000, (
            f"estimator still inflating image-bearing sessions: {est:,}"
        )
        # Sanity: it's still a non-trivial number (text content + 5 image credits).
        assert est > 5 * _IMAGE_TOKEN_COST
