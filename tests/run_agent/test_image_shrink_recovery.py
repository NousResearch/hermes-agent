"""Tests for reactive image-shrink recovery.

Covers the full chain for Anthropic's 5 MB per-image ceiling (and any
future provider that returns an image-too-large error):

  1. agent/error_classifier.py: 400 with "image exceeds 5 MB maximum"
     gets FailoverReason.image_too_large, not context_overflow.
  2. run_agent._try_shrink_image_parts_in_messages mutates the API
     payload in-place, re-encoding native data: URL image parts to fit
     under 4 MB using vision_tools._resize_image_for_vision.

The end-to-end wiring in the retry loop is not unit-tested here — it's
covered by the live E2E in the PR description. These tests lock in the
two pieces that matter independently: the classifier signal and the
payload rewriter.
"""

from __future__ import annotations

import base64
from pathlib import Path

import pytest

from agent.error_classifier import FailoverReason, classify_api_error


class _FakeApiError(Exception):
    """Stand-in for an openai.BadRequestError with status_code + body."""

    def __init__(self, status_code: int, message: str, body: dict | None = None):
        super().__init__(message)
        self.status_code = status_code
        self.body = body or {"error": {"message": message}}
        self.response = None  # required by some code paths


# ─── Classifier ──────────────────────────────────────────────────────────────


class TestImageTooLargeClassification:
    def test_anthropic_400_image_exceeds_message(self):
        """Anthropic's exact wording must classify as image_too_large, not context."""
        err = _FakeApiError(
            status_code=400,
            message=(
                "messages.0.content.1.image.source.base64: image exceeds 5 MB "
                "maximum: 12966600 bytes > 5242880 bytes"
            ),
        )
        result = classify_api_error(err, provider="anthropic", model="claude-sonnet-4-6")
        assert result.reason == FailoverReason.image_too_large
        assert result.retryable is True

    def test_generic_image_too_large_no_status(self):
        """No status_code path: message text alone triggers classification."""
        err = Exception("image too large for this endpoint")
        result = classify_api_error(err, provider="some-provider", model="some-model")
        assert result.reason == FailoverReason.image_too_large
        assert result.retryable is True

    def test_image_too_large_not_confused_with_context_overflow(self):
        """'image exceeds' must NOT be mis-classified as context_overflow.

        The context_overflow patterns include 'exceeds the limit' which is a
        superstring risk — verify the image-too-large check fires first.
        """
        err = _FakeApiError(
            status_code=400,
            message="image exceeds the limit for this model",
        )
        result = classify_api_error(err, provider="anthropic", model="claude-sonnet-4-6")
        assert result.reason == FailoverReason.image_too_large

    def test_regular_context_overflow_unaffected(self):
        """Context-overflow errors without image keywords still classify correctly."""
        err = _FakeApiError(
            status_code=400,
            message="prompt is too long: context length 300000 exceeds max of 200000",
        )
        result = classify_api_error(err, provider="anthropic", model="claude-sonnet-4-6")
        assert result.reason == FailoverReason.context_overflow


# ─── Shrink helper ───────────────────────────────────────────────────────────


def _big_png_data_url(size_kb: int) -> str:
    """Build a data URL with a plausible large base64 payload."""
    # Use real PNG header so MIME detection works; fill to target size.
    raw = b"\x89PNG\r\n\x1a\n" + b"X" * (size_kb * 1024)
    return "data:image/png;base64," + base64.b64encode(raw).decode("ascii")


def _make_agent():
    """Build a bare AIAgent for method-level testing, no provider setup."""
    from run_agent import AIAgent
    agent = object.__new__(AIAgent)
    agent.provider = "anthropic"
    agent.model = "claude-sonnet-4-6"
    return agent


class TestShrinkImagePartsHelper:
    def test_no_messages_returns_false(self):
        agent = _make_agent()
        assert agent._try_shrink_image_parts_in_messages([]) is False
        assert agent._try_shrink_image_parts_in_messages(None) is False

    def test_no_image_parts_returns_false(self):
        agent = _make_agent()
        msgs = [
            {"role": "user", "content": "plain text"},
            {"role": "assistant", "content": "ack"},
        ]
        assert agent._try_shrink_image_parts_in_messages(msgs) is False

    def test_small_image_part_not_shrunk(self, monkeypatch):
        """An image under 4 MB is left alone — shrink helper only touches oversized ones."""
        agent = _make_agent()
        small_url = _big_png_data_url(100)  # ~100 KB + b64 overhead

        resize_hits = {"count": 0}
        monkeypatch.setattr(
            "tools.vision_tools._resize_image_for_vision",
            lambda *a, **kw: resize_hits.__setitem__("count", resize_hits["count"] + 1) or small_url,
            raising=False,
        )

        msgs = [{
            "role": "user",
            "content": [
                {"type": "text", "text": "hi"},
                {"type": "image_url", "image_url": {"url": small_url}},
            ],
        }]
        assert agent._try_shrink_image_parts_in_messages(msgs) is False
        assert resize_hits["count"] == 0
        # URL unchanged.
        assert msgs[0]["content"][1]["image_url"]["url"] == small_url

    def test_oversized_image_url_dict_shape_rewritten(self, monkeypatch):
        """OpenAI chat.completions shape: {image_url: {url: data:...}}."""
        agent = _make_agent()
        oversized_url = _big_png_data_url(5000)  # ~5 MB raw → ~6.7 MB b64
        shrunk = "data:image/jpeg;base64," + "A" * 1000  # small

        def _fake_resize(path, mime_type=None, max_base64_bytes=None):
            return shrunk

        monkeypatch.setattr(
            "tools.vision_tools._resize_image_for_vision",
            _fake_resize,
            raising=False,
        )

        msgs = [{
            "role": "user",
            "content": [
                {"type": "text", "text": "look"},
                {"type": "image_url", "image_url": {"url": oversized_url}},
            ],
        }]
        changed = agent._try_shrink_image_parts_in_messages(msgs)
        assert changed is True
        assert msgs[0]["content"][1]["image_url"]["url"] == shrunk

    def test_oversized_input_image_string_shape_rewritten(self, monkeypatch):
        """OpenAI Responses shape: {type: input_image, image_url: "data:..."}."""
        agent = _make_agent()
        oversized_url = _big_png_data_url(5000)
        shrunk = "data:image/jpeg;base64," + "B" * 1000

        monkeypatch.setattr(
            "tools.vision_tools._resize_image_for_vision",
            lambda *a, **kw: shrunk,
            raising=False,
        )

        msgs = [{
            "role": "user",
            "content": [
                {"type": "input_text", "text": "look"},
                {"type": "input_image", "image_url": oversized_url},
            ],
        }]
        changed = agent._try_shrink_image_parts_in_messages(msgs)
        assert changed is True
        assert msgs[0]["content"][1]["image_url"] == shrunk

    def test_multiple_images_all_shrunk(self, monkeypatch):
        agent = _make_agent()
        big1 = _big_png_data_url(5000)
        big2 = _big_png_data_url(6000)
        shrunk = "data:image/jpeg;base64," + "C" * 500

        monkeypatch.setattr(
            "tools.vision_tools._resize_image_for_vision",
            lambda *a, **kw: shrunk,
            raising=False,
        )

        msgs = [{
            "role": "user",
            "content": [
                {"type": "text", "text": "compare"},
                {"type": "image_url", "image_url": {"url": big1}},
                {"type": "image_url", "image_url": {"url": big2}},
            ],
        }]
        changed = agent._try_shrink_image_parts_in_messages(msgs)
        assert changed is True
        assert msgs[0]["content"][1]["image_url"]["url"] == shrunk
        assert msgs[0]["content"][2]["image_url"]["url"] == shrunk

    def test_http_url_images_not_touched(self, monkeypatch):
        """Only data: URLs are candidates — http URLs are server-fetched."""
        agent = _make_agent()

        resize_hits = {"count": 0}
        monkeypatch.setattr(
            "tools.vision_tools._resize_image_for_vision",
            lambda *a, **kw: resize_hits.__setitem__("count", resize_hits["count"] + 1) or "shrunk",
            raising=False,
        )

        msgs = [{
            "role": "user",
            "content": [
                {"type": "text", "text": "at this url"},
                {"type": "image_url", "image_url": {"url": "https://example.com/big.png"}},
            ],
        }]
        assert agent._try_shrink_image_parts_in_messages(msgs) is False
        assert resize_hits["count"] == 0

    def test_shrink_failure_returns_false_and_leaves_url_intact(self, monkeypatch):
        """If re-encode fails, leave the URL alone so the caller surfaces the original error."""
        agent = _make_agent()
        oversized_url = _big_png_data_url(5000)

        monkeypatch.setattr(
            "tools.vision_tools._resize_image_for_vision",
            lambda *a, **kw: None,  # resize returned nothing usable
            raising=False,
        )

        msgs = [{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": oversized_url}},
            ],
        }]
        assert agent._try_shrink_image_parts_in_messages(msgs) is False
        assert msgs[0]["content"][0]["image_url"]["url"] == oversized_url

    def test_shrink_that_makes_it_bigger_rejected(self, monkeypatch):
        """If the 'shrink' somehow produces a larger payload, skip it."""
        agent = _make_agent()
        oversized_url = _big_png_data_url(5000)
        even_bigger = "data:image/png;base64," + "Z" * (10 * 1024 * 1024)

        monkeypatch.setattr(
            "tools.vision_tools._resize_image_for_vision",
            lambda *a, **kw: even_bigger,
            raising=False,
        )

        msgs = [{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": oversized_url}},
            ],
        }]
        assert agent._try_shrink_image_parts_in_messages(msgs) is False
        # Original URL still in place, not replaced by the bigger one.
        assert msgs[0]["content"][0]["image_url"]["url"] == oversized_url
