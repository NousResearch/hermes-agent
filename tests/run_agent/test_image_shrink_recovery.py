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
import sys
from types import SimpleNamespace


from agent.conversation_loop import _image_error_max_dimension
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

    def test_anthropic_many_image_dimension_limit(self):
        """OpenRouter-wrapped Anthropic many-image limits recover via shrink."""
        err = _FakeApiError(
            status_code=400,
            message=(
                "messages.21.content.43.image.source.base64.data: At least one "
                "of the image dimensions exceed max allowed size for many-image "
                "requests: 2000 pixels"
            ),
        )
        result = classify_api_error(err, provider="openrouter", model="anthropic/claude-opus-4.8")
        assert result.reason == FailoverReason.image_too_large
        assert result.retryable is True
        assert _image_error_max_dimension(err) == 2000


# ─── Shrink helper ───────────────────────────────────────────────────────────


def _big_png_data_url(size_kb: int) -> str:
    """Build a data URL with a plausible large base64 payload."""
    # Use real PNG header so MIME detection works; fill to target size.
    raw = b"\x89PNG\r\n\x1a\n" + b"X" * (size_kb * 1024)
    return "data:image/png;base64," + base64.b64encode(raw).decode("ascii")


def _install_fake_pillow(
    monkeypatch,
    size: tuple[int, int],
    *,
    shrunk_size: tuple[int, int] | None = None,
    sizes: list[tuple[int, int]] | None = None,
) -> None:
    """Install the tiny subset of Pillow used by the shrink preflight.

    The shrink helper decodes pixel dimensions twice for the dimension path:
    once on the *original* data URL (to decide it's oversized) and once on the
    *re-encoded* result (to confirm the downscale landed under the cap).  To
    model that honestly, ``_FakeImage`` can return a sequence of sizes across
    successive ``open()`` calls:

    * ``sizes=[...]``        — explicit per-call size list (clamped to last).
    * ``shrunk_size=(w, h)`` — shorthand for ``[size, shrunk_size]``: first
      decode is the oversized original, second is the in-cap re-encode.
    * neither                — every decode returns ``size`` (legacy behaviour).
    """
    call_count = {"n": 0}
    target_sizes = sizes or [
        size,
        shrunk_size if shrunk_size is not None else size,
    ]

    class _FakeImage:
        def __init__(self):
            self.size = target_sizes[min(call_count["n"], len(target_sizes) - 1)]
            call_count["n"] += 1

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    class _FakeImageModule:
        @staticmethod
        def open(_data):
            return _FakeImage()

    monkeypatch.setitem(sys.modules, "PIL", SimpleNamespace(Image=_FakeImageModule))
    monkeypatch.setitem(sys.modules, "PIL.Image", _FakeImageModule)


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

        def _fake_resize(path, mime_type=None, max_base64_bytes=None, max_dimension=None):
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

    def test_many_image_dimension_limit_rewritten(self, monkeypatch):
        """A 2000px many-image rejection must shrink images below the cap."""
        agent = _make_agent()
        # Original decodes oversized (2501px); the re-encode decodes in-cap.
        _install_fake_pillow(monkeypatch, (2501, 100), shrunk_size=(1500, 60))
        oversized_for_many = _big_png_data_url(100)
        shrunk = "data:image/jpeg;base64," + "M" * 1000
        seen = {}

        def _fake_resize(path, mime_type=None, max_base64_bytes=None, max_dimension=None):
            seen["max_dimension"] = max_dimension
            return shrunk

        monkeypatch.setattr(
            "tools.vision_tools._resize_image_for_vision",
            _fake_resize,
            raising=False,
        )

        msgs = [{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": oversized_for_many}},
            ],
        }]
        changed = agent._try_shrink_image_parts_in_messages(
            msgs,
            max_dimension=2000,
        )
        assert changed is True
        assert seen["max_dimension"] == 2000
        assert msgs[0]["content"][0]["image_url"]["url"] == shrunk

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

    def test_mixed_one_shrinkable_one_not_returns_false(self, monkeypatch):
        """Regression for the wedged-session incident (May 2026).

        When one oversized image shrinks but another oversized image can't,
        the helper must return False — retrying would re-send the surviving
        oversized payload and fail identically, burning the single retry on a
        no-op.  The original bug returned True after shrinking *any* part,
        which is what permanently wedged a session whose history held a 12 MB
        tool-result image alongside a freshly-loaded shrinkable one.
        """
        agent = _make_agent()
        shrinkable = _big_png_data_url(5000)
        unshrinkable = _big_png_data_url(6000)
        small = "data:image/jpeg;base64," + "C" * 500

        # _resize_image_for_vision returns small for the shrinkable input but
        # echoes the oversized payload back for the unshrinkable one.
        def fake_resize(path, *a, **kw):
            # The temp file written by the helper contains the decoded bytes;
            # distinguish by size — the 6000 KB source stays "big".
            try:
                size = path.stat().st_size
            except Exception:
                size = 0
            if size > 5500 * 1024:
                return unshrinkable  # can't reduce — echo oversized back
            return small

        monkeypatch.setattr(
            "tools.vision_tools._resize_image_for_vision",
            fake_resize,
            raising=False,
        )

        msgs = [{
            "role": "tool",
            "content": [
                {"type": "image_url", "image_url": {"url": shrinkable}},
                {"type": "image_url", "image_url": {"url": unshrinkable}},
            ],
        }]
        # One part shrank, one survived oversized → must NOT retry.
        assert agent._try_shrink_image_parts_in_messages(msgs) is False
        # The shrinkable one was still re-encoded (mutated in place).
        assert msgs[0]["content"][0]["image_url"]["url"] == small
        # The unshrinkable one is left as-is (caller surfaces original error).
        assert msgs[0]["content"][1]["image_url"]["url"] == unshrinkable

    # ------------------------------------------------------------------
    # #48013: the dimension path must accept a pixel-correct downscale even
    # when the re-encoded PNG grew in bytes.  Before the fix, the byte gate
    # (`len(resized) >= len(url)`) discarded the dimension-correct result and
    # left the image oversized, bricking the session on the Anthropic
    # many-image 2000px path.
    # ------------------------------------------------------------------

    def test_dimension_shrink_with_byte_growth_accepted(self, monkeypatch):
        """A dimension-driven shrink is accepted even if its bytes grow.

        Regression for #48013.  The original (2501px, under the 4 MB byte
        budget) is oversized on pixels only.  The re-encode lands at 1500px
        (in-cap) but is *larger in bytes* — the historical byte gate would
        reject it.  The fix keys the accept gate on the binding constraint
        (dimensions), so the pixel-correct result is kept.
        """
        agent = _make_agent()
        _install_fake_pillow(monkeypatch, (2501, 100), shrunk_size=(1500, 60))
        original_url = _big_png_data_url(100)  # ~100 KB → well under 4 MB
        # A *byte-larger* re-encode (the brick trigger): 200 KB payload.
        dimensionally_shrunk = "data:image/png;base64," + "G" * 200 * 1024
        seen = {}

        def _fake_resize(path, mime_type=None, max_base64_bytes=None, max_dimension=None):
            seen["max_dimension"] = max_dimension
            return dimensionally_shrunk

        monkeypatch.setattr(
            "tools.vision_tools._resize_image_for_vision",
            _fake_resize,
            raising=False,
        )

        msgs = [{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": original_url}},
            ],
        }]
        # The re-encode is byte-LARGER than the original — proves the byte gate
        # is no longer the rejection driver on the dimension path.
        assert len(dimensionally_shrunk) > len(original_url)
        assert agent._try_shrink_image_parts_in_messages(
            msgs, max_dimension=2000,
        ) is True
        assert seen["max_dimension"] == 2000
        assert msgs[0]["content"][0]["image_url"]["url"] == dimensionally_shrunk

    def test_dimension_shrink_failure_still_blocks_retry(self, monkeypatch):
        """A dimension-oversized image that stays oversized is unshrinkable.

        If the re-encode is *still* over the per-side cap, the helper must
        report no progress (return False) so the one-shot retry isn't burned
        re-sending a payload the provider already rejected.
        """
        agent = _make_agent()
        # Both decodes report oversized: original and re-encode are 2501px.
        _install_fake_pillow(monkeypatch, (2501, 100))
        original_url = _big_png_data_url(100)
        still_oversized = "data:image/png;base64," + "H" * 120 * 1024

        monkeypatch.setattr(
            "tools.vision_tools._resize_image_for_vision",
            lambda *a, **kw: still_oversized,
            raising=False,
        )

        msgs = [{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": original_url}},
            ],
        }]
        assert agent._try_shrink_image_parts_in_messages(
            msgs, max_dimension=2000,
        ) is False
        # Original left untouched — caller surfaces the provider's 400.
        assert msgs[0]["content"][0]["image_url"]["url"] == original_url

    def test_mixed_dimension_partial_progress_returns_false(self, monkeypatch):
        """Partial dimension-path progress must not falsely burn the retry.

        Two dimension-oversized images: the first re-encodes in-cap, the
        second stays oversized.  Even though one part changed, an oversized
        image survives, so retrying would 400 again — the helper must report
        False.  (Mirrors the byte-path
        ``test_mixed_one_shrinkable_one_not_returns_false`` invariant for the
        pixel axis.)
        """
        agent = _make_agent()
        # Decode order: img1 orig (2501) -> img1 re-encode (1500, in-cap) ->
        #               img2 orig (2501) -> img2 re-encode (2501, still over).
        _install_fake_pillow(
            monkeypatch,
            (2501, 100),
            sizes=[(2501, 100), (1500, 60), (2501, 100), (2501, 100)],
        )
        first = _big_png_data_url(100)
        second = _big_png_data_url(90)
        calls = {"n": 0}

        def _fake_resize(path, mime_type=None, max_base64_bytes=None, max_dimension=None):
            calls["n"] += 1
            if calls["n"] == 1:
                return "data:image/png;base64," + "G" * 200 * 1024  # in-cap
            return "data:image/png;base64," + "H" * 120 * 1024      # still over

        monkeypatch.setattr(
            "tools.vision_tools._resize_image_for_vision",
            _fake_resize,
            raising=False,
        )

        msgs = [{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": first}},
                {"type": "image_url", "image_url": {"url": second}},
            ],
        }]
        assert agent._try_shrink_image_parts_in_messages(
            msgs, max_dimension=2000,
        ) is False

    def test_byte_oversized_but_pixel_oversized_after_shrink_blocks_retry(self, monkeypatch):
        """Bytes-triggered shrink must ALSO honour the active per-side cap.

        Adversarial-review regression (#48013, round 2): an image over BOTH the
        4 MB byte budget AND the per-side pixel cap can be byte-shrunk yet stay
        over the cap (``_resize_image_for_vision`` returns a best-effort blob
        when it exhausts its halving budget on a very-high-aspect image).  The
        byte-path accept gate originally checked only ``len(resized) < len(url)``
        and reported success, so the caller retried and the provider re-rejected
        on dimensions — re-bricking the session.  The fix re-checks the pixel
        cap on the byte path too; a still-over-cap result must be unshrinkable.
        """
        agent = _make_agent()
        # On the BYTE path, _decode_pixels is called once — on the RESIZED blob.
        # Script that single decode to report still-over-cap dims (2560 > 2000).
        _install_fake_pillow(monkeypatch, (2560, 64), sizes=[(2560, 64)])
        # Over the 4 MB byte budget so the BYTE path is taken (triggered_by="bytes").
        oversized_url = _big_png_data_url(5000)  # ~5 MB raw → ~6.7 MB b64
        # Byte-SMALLER re-encode, but its decoded dims are still over the cap.
        byte_smaller_still_over = "data:image/png;base64," + "K" * 1000

        monkeypatch.setattr(
            "tools.vision_tools._resize_image_for_vision",
            lambda *a, **kw: byte_smaller_still_over,
            raising=False,
        )

        msgs = [{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": oversized_url}},
            ],
        }]
        # Bytes shrank, but the per-side cap is still violated → no real
        # progress; the helper must NOT report success (would burn the retry).
        assert len(byte_smaller_still_over) < len(oversized_url)
        assert agent._try_shrink_image_parts_in_messages(
            msgs, max_dimension=2000,
        ) is False
        # Original left in place — caller surfaces the provider's 400.
        assert msgs[0]["content"][0]["image_url"]["url"] == oversized_url

    def test_byte_oversized_with_no_dim_cap_accepts_byte_shrink(self, monkeypatch):
        """Bytes path with the default 8000px cap still accepts a byte shrink.

        Guards the fix above against over-reach: when no tight dimension cap is
        active (default 8000px) and the byte-shrunk re-encode is comfortably
        within it, the byte path must keep accepting on byte-shrinkage alone.
        """
        agent = _make_agent()
        # Byte path → single _decode_pixels call on the resized blob; report
        # in-cap dims so the byte-shrink is accepted under the default 8000 cap.
        _install_fake_pillow(monkeypatch, (1250, 50), sizes=[(1250, 50)])
        oversized_url = _big_png_data_url(5000)
        shrunk = "data:image/jpeg;base64," + "L" * 1000

        monkeypatch.setattr(
            "tools.vision_tools._resize_image_for_vision",
            lambda *a, **kw: shrunk,
            raising=False,
        )

        msgs = [{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": oversized_url}},
            ],
        }]
        # Default cap (8000) — no explicit max_dimension passed.
        assert agent._try_shrink_image_parts_in_messages(msgs) is True
        assert msgs[0]["content"][0]["image_url"]["url"] == shrunk
