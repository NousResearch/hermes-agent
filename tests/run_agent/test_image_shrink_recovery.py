"""Tests for reactive image-shrink recovery.

Covers the full chain for Anthropic's 5 MB per-image ceiling and
8000px-per-side dimension cap (bug #37677):

  1. agent/error_classifier.py: 400 with "image exceeds 5 MB maximum" or
     dimension-exceeded wording gets FailoverReason.image_too_large, not
     context_overflow.
  2. run_agent._try_shrink_image_parts_in_messages mutates the API
     payload in-place, re-encoding native data: URL image parts to fit
     under 4 MB / 7900px using vision_tools._resize_image_for_vision.
  3. vision_tools._resize_image_for_vision proportionally scales an image
     whose longest side exceeds _MAX_IMAGE_DIMENSION_PX BEFORE the
     byte-size shrink loop.

The end-to-end wiring in the retry loop is not unit-tested here — it's
covered by the live E2E in the PR description. These tests lock in the
pieces that matter independently: the classifier signal, the payload
rewriter, and the dimension-ceiling in the resize helper.
"""

from __future__ import annotations

import base64


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

    # ── Dimension-error wording (bug #37677) ─────────────────────────────

    def test_anthropic_dimension_cap_exceeded_classifies_as_image_too_large(self):
        """Dimension-cap wording must classify as image_too_large, not format_error.

        Anthropic returns HTTP 400 with "image dimensions exceed maximum" or
        "image width N exceeds maximum allowed dimension of 8000" when a tall
        screenshot passes the 5 MB byte-size check but violates the 8000px cap.
        """
        err = _FakeApiError(
            status_code=400,
            message=(
                "messages.0.content.1.image.source.base64: "
                "image dimensions exceed maximum: image width 8500 "
                "exceeds maximum allowed dimension of 8000"
            ),
        )
        result = classify_api_error(err, provider="anthropic", model="claude-sonnet-4-6")
        assert result.reason == FailoverReason.image_too_large
        assert result.retryable is True

    def test_image_width_exceeds_dimension_classifies_as_image_too_large(self):
        """Short 'image width … exceeds' wording must also classify correctly."""
        err = _FakeApiError(
            status_code=400,
            message="image width 9600 exceeds maximum allowed dimension of 8000",
        )
        result = classify_api_error(err, provider="anthropic", model="claude-sonnet-4-6")
        assert result.reason == FailoverReason.image_too_large
        assert result.retryable is True

    def test_maximum_dimension_generic_wording(self):
        """'maximum dimension' alone (generic provider wording) must route to image_too_large."""
        err = _FakeApiError(
            status_code=400,
            message="image exceeds the maximum dimension allowed by this endpoint",
        )
        result = classify_api_error(err, provider="some-provider", model="some-model")
        assert result.reason == FailoverReason.image_too_large
        assert result.retryable is True

    def test_dimension_error_no_status_code(self):
        """Dimension-error wording without a status code also routes correctly."""
        err = Exception("image dimensions exceed maximum: 9000x5000 > 8000px limit")
        result = classify_api_error(err, provider="anthropic", model="claude-sonnet-4-6")
        assert result.reason == FailoverReason.image_too_large
        assert result.retryable is True


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

    def test_dimension_oversized_image_triggers_shrink(self, monkeypatch):
        """An image small in bytes but wide/tall beyond 7900px must be shrunk.

        This is the core bug #37677 scenario: a tall screenshot under 5 MB
        that passes byte-size guards but exceeds the 8000px dimension cap.
        The shrink helper must treat it as oversized and call resize.
        """
        agent = _make_agent()
        # Build a data URL that is small in bytes (well under 4 MB) but
        # whose decoded bytes will look like an oversized-dimension image
        # to _exceeds_dimension_cap (mocked below).
        small_bytes_url = "data:image/png;base64," + base64.b64encode(
            b"\x89PNG\r\n\x1a\n" + b"X" * 512  # tiny, < 4 MB
        ).decode("ascii")

        shrunk = "data:image/jpeg;base64," + "S" * 1000

        # Patch _exceeds_dimension_cap to simulate an 8500×1080 image
        # (small bytes, large dims) being detected as over-cap.
        monkeypatch.setattr(
            "agent.conversation_compression._exceeds_dimension_cap",
            lambda url: True,
            raising=False,
        )
        monkeypatch.setattr(
            "tools.vision_tools._resize_image_for_vision",
            lambda *a, **kw: shrunk,
            raising=False,
        )

        msgs = [{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": small_bytes_url}},
            ],
        }]
        changed = agent._try_shrink_image_parts_in_messages(msgs)
        assert changed is True
        assert msgs[0]["content"][0]["image_url"]["url"] == shrunk

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


# ─── Pixel-dimension ceiling in _resize_image_for_vision (bug #37677) ────────


class TestDimensionResizeCeiling:
    """Unit tests for the _MAX_IMAGE_DIMENSION_PX pre-scale step added in
    tools/vision_tools._resize_image_for_vision.

    These tests use real Pillow calls (Pillow is now a required dependency)
    to create in-memory images that exceed the 7900px cap, pass them through
    _resize_image_for_vision, and assert the output image is within bounds.
    """

    def _make_png_bytes(self, width: int, height: int) -> bytes:
        """Produce a minimal valid PNG for the given dimensions."""
        try:
            from PIL import Image as _PIL
            import io
            img = _PIL.new("RGB", (width, height), color=(128, 128, 128))
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            return buf.getvalue()
        except ImportError:
            pytest.skip("Pillow not installed")

    def test_wide_image_scaled_to_dimension_cap(self, tmp_path):
        """A 9000×500 image (wide, small bytes) must be scaled to ≤7900px wide."""
        pytest = _import_pytest()
        try:
            from PIL import Image as _PIL
        except ImportError:
            pytest.skip("Pillow not installed")

        import io
        from tools.vision_tools import _resize_image_for_vision, _MAX_IMAGE_DIMENSION_PX

        png_bytes = self._make_png_bytes(9000, 500)
        img_path = tmp_path / "wide.png"
        img_path.write_bytes(png_bytes)

        result_url = _resize_image_for_vision(img_path, mime_type="image/png")
        assert result_url.startswith("data:")

        # Decode and check output dimensions.
        _, _, b64 = result_url.partition(",")
        raw = base64.b64decode(b64)
        with _PIL.open(io.BytesIO(raw)) as out_img:
            assert out_img.width <= _MAX_IMAGE_DIMENSION_PX, (
                f"Width {out_img.width} still exceeds {_MAX_IMAGE_DIMENSION_PX}px cap"
            )
            assert out_img.height <= _MAX_IMAGE_DIMENSION_PX

    def test_tall_image_scaled_to_dimension_cap(self, tmp_path):
        """A 500×9000 image (tall, small bytes) must be scaled to ≤7900px tall."""
        try:
            from PIL import Image as _PIL
        except ImportError:
            import pytest
            pytest.skip("Pillow not installed")

        import io
        from tools.vision_tools import _resize_image_for_vision, _MAX_IMAGE_DIMENSION_PX

        png_bytes = self._make_png_bytes(500, 9000)
        img_path = tmp_path / "tall.png"
        img_path.write_bytes(png_bytes)

        result_url = _resize_image_for_vision(img_path, mime_type="image/png")
        _, _, b64 = result_url.partition(",")
        raw = base64.b64decode(b64)
        with _PIL.open(io.BytesIO(raw)) as out_img:
            assert out_img.height <= _MAX_IMAGE_DIMENSION_PX, (
                f"Height {out_img.height} still exceeds {_MAX_IMAGE_DIMENSION_PX}px cap"
            )
            assert out_img.width <= _MAX_IMAGE_DIMENSION_PX

    def test_aspect_ratio_preserved_after_dimension_scale(self, tmp_path):
        """Proportional scaling must preserve aspect ratio to within 1px rounding."""
        try:
            from PIL import Image as _PIL
        except ImportError:
            import pytest
            pytest.skip("Pillow not installed")

        import io
        from tools.vision_tools import _resize_image_for_vision

        # 8500×2125 → ratio 4:1
        png_bytes = self._make_png_bytes(8500, 2125)
        img_path = tmp_path / "aspect.png"
        img_path.write_bytes(png_bytes)

        result_url = _resize_image_for_vision(img_path, mime_type="image/png")
        _, _, b64 = result_url.partition(",")
        raw = base64.b64decode(b64)
        with _PIL.open(io.BytesIO(raw)) as out_img:
            # Aspect ratio should remain close to 4:1 (within 1px float rounding).
            ratio = out_img.width / out_img.height
            assert abs(ratio - 4.0) < 0.1, f"Aspect ratio drifted: {ratio:.3f}"

    def test_image_within_dimension_cap_unchanged(self, tmp_path):
        """An image already within 7900px must NOT be scaled by the dimension step."""
        try:
            from PIL import Image as _PIL
        except ImportError:
            import pytest
            pytest.skip("Pillow not installed")

        import io
        from tools.vision_tools import _resize_image_for_vision, _MAX_IMAGE_DIMENSION_PX

        # 800×600 is well within the cap.
        png_bytes = self._make_png_bytes(800, 600)
        img_path = tmp_path / "small.png"
        img_path.write_bytes(png_bytes)

        result_url = _resize_image_for_vision(img_path, mime_type="image/png")
        _, _, b64 = result_url.partition(",")
        raw = base64.b64decode(b64)
        with _PIL.open(io.BytesIO(raw)) as out_img:
            # Must not be upscaled or otherwise distorted.
            assert out_img.width <= 800
            assert out_img.height <= 600


def _import_pytest():
    """Lazy pytest import so the module-level import stays clean."""
    import pytest as _pytest
    return _pytest
