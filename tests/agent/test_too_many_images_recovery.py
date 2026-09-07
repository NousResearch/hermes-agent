"""Reactive recovery for provider per-prompt image *count* limits.

vLLM's ``--limit-mm-per-prompt`` rejects a request that carries more images
than the deployment allows with a 400 like
``"At most 2 image(s) may be provided in one prompt."``.  Every image decoded
fine — the request simply has too many.  Before this fix the 400 fell through
to a non-retryable ``format_error`` and the session stalled.

Covers the three pieces independently:

  1. ``agent/error_classifier.py``: the count-limit 400 classifies as
     ``FailoverReason.too_many_images`` (retryable), not ``format_error`` /
     ``image_too_large`` / ``image_corrupt``.
  2. ``agent/conversation_loop._image_error_max_count``: parses the ceiling
     out of the error text, returning ``None`` when unparseable.
  3. ``agent/context_compressor.strip_images_to_count_limit``: strips the
     oldest image payloads (user uploads and tool results alike, protecting
     the newest user turn) on the per-call copy, leaving persisted history
     untouched.
"""

from __future__ import annotations

from agent.conversation_loop import _image_error_max_count
from agent.context_compressor import (
    _MAX_KEEP_TOOL_IMAGES,
    _count_image_parts,
    strip_images_to_count_limit,
)
from agent.error_classifier import FailoverReason, classify_api_error


class _FakeApiError(Exception):
    """Stand-in for an openai.BadRequestError with status_code + body."""

    def __init__(self, status_code: int, message: str, body: dict | None = None):
        super().__init__(message)
        self.status_code = status_code
        self.body = body or {"error": {"message": message}}
        self.response = None  # required by some code paths


_VLLM_COUNT_MSG = "At most 2 image(s) may be provided in one prompt. (parameter=image)"


# ─── Classifier ──────────────────────────────────────────────────────────────


class TestTooManyImagesClassification:
    def test_vllm_count_limit_400(self):
        """vLLM's exact wording must classify as too_many_images, retryable."""
        err = _FakeApiError(status_code=400, message=_VLLM_COUNT_MSG)
        result = classify_api_error(err, provider="vllm", model="qwen2.5-vl")
        assert result.reason == FailoverReason.too_many_images
        assert result.retryable is True

    def test_too_many_images_variant(self):
        err = _FakeApiError(
            status_code=400, message="Too many images in request (max 4)"
        )
        result = classify_api_error(err, provider="vllm", model="qwen2.5-vl")
        assert result.reason == FailoverReason.too_many_images

    def test_no_more_than_variant(self):
        err = _FakeApiError(
            status_code=400,
            message="No more than 4 images may be provided in one prompt",
        )
        result = classify_api_error(err, provider="vllm", model="qwen2.5-vl")
        assert result.reason == FailoverReason.too_many_images

    def test_image_too_large_not_misrouted(self):
        """A per-image *size* ceiling is image_too_large, not a count limit."""
        err = _FakeApiError(
            status_code=400,
            message="messages.0.content.1.image.source.base64: image exceeds 5 MB maximum",
        )
        result = classify_api_error(err, provider="anthropic", model="claude-sonnet-4-6")
        assert result.reason == FailoverReason.image_too_large
        assert result.reason != FailoverReason.too_many_images

    def test_image_corrupt_not_misrouted(self):
        """An undecodable-image 400 is image_corrupt, not a count limit."""
        err = _FakeApiError(status_code=400, message="Invalid PNG image.")
        result = classify_api_error(err, provider="xai", model="grok-5")
        assert result.reason == FailoverReason.image_corrupt
        assert result.reason != FailoverReason.too_many_images

    def test_sglang_exceeds_limit(self):
        """SGLang's ``limit_mm_data_per_request`` wording."""
        err = _FakeApiError(
            status_code=400,
            message="Image count 5 exceeds limit 2 per request.",
        )
        result = classify_api_error(err, provider="sglang", model="qwen2.5-vl")
        assert result.reason == FailoverReason.too_many_images
        assert result.retryable is True

    def test_unrelated_400_not_misrouted(self):
        """A generic 400 with no image-count phrasing must not match."""
        err = _FakeApiError(
            status_code=400, message="messages: unexpected role \"tool\" at index 3"
        )
        result = classify_api_error(err, provider="minimax", model="MiniMax-M3")
        assert result.reason != FailoverReason.too_many_images


# ─── _image_error_max_count ──────────────────────────────────────────────────


class TestImageErrorMaxCount:
    def test_parses_vllm_message(self):
        err = _FakeApiError(status_code=400, message=_VLLM_COUNT_MSG)
        assert _image_error_max_count(err) == 2

    def test_parses_no_more_than(self):
        err = _FakeApiError(
            status_code=400, message="No more than 4 images may be provided"
        )
        assert _image_error_max_count(err) == 4

    def test_parses_from_body(self):
        """The count can live on the error body, not just the message."""
        err = _FakeApiError(
            status_code=400,
            message="BadRequestError",
            body={"error": {"message": "At most 3 image(s) may be provided"}},
        )
        assert _image_error_max_count(err) == 3

    def test_parses_sglang_exceeds_limit(self):
        """SGLang: ``"Image count 5 exceeds limit 2 per request."`` → 2."""
        err = _FakeApiError(
            status_code=400,
            message="Image count 5 exceeds limit 2 per request.",
        )
        assert _image_error_max_count(err) == 2

    def test_unparseable_returns_none(self):
        err = _FakeApiError(
            status_code=400, message="Too many images in one prompt"
        )
        assert _image_error_max_count(err) is None

    def test_no_image_mention_returns_none(self):
        err = _FakeApiError(status_code=400, message="At most 2 tool calls allowed")
        assert _image_error_max_count(err) is None


# ─── strip_images_to_count_limit ─────────────────────────────────────────────


def _user_with_images(n: int, *, tag: str = "U") -> dict:
    parts = [{"type": "text", "text": f"{tag} text"}]
    for i in range(n):
        parts.append(
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{tag}{i}"},
            }
        )
    return {"role": "user", "content": parts}


def _tool_with_images(i: int) -> list[dict]:
    return [
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": f"call_{i}",
                    "type": "function",
                    "function": {
                        "name": "vision_analyze",
                        "arguments": f'{{"image_url":"shot{i}.png"}}',
                    },
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": f"call_{i}",
            "content": [
                {"type": "text", "text": f"Image attached natively shot {i}"},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,T{i}"},
                },
            ],
        },
    ]


def _total_images(messages: list[dict]) -> int:
    return sum(_count_image_parts(m.get("content")) for m in messages)


class TestStripImagesToCountLimit:
    def test_strips_oldest_keeps_newest(self):
        """4 images across user + tool, limit 2, last msg is a tool result.

        The newest user turn is NOT the last message, so nothing is protected
        and the two oldest images (the first user upload) are stripped.
        """
        msgs = [
            _user_with_images(2, tag="OLD"),
            *_tool_with_images(0),
            *_tool_with_images(1),
        ]
        assert _total_images(msgs) == 4
        stripped = strip_images_to_count_limit(msgs, 2)
        assert stripped == 2
        assert _total_images(msgs) == 2
        # The two tool images (newest) survive; the user upload is stripped.
        user = msgs[0]
        assert _count_image_parts(user["content"]) == 0
        assert _count_image_parts(msgs[2]["content"]) == 1  # tool 0
        assert _count_image_parts(msgs[4]["content"]) == 1  # tool 1

    def test_protected_user_turn_exceeds_limit_no_strip(self):
        """The newest user turn alone exceeds the limit → nothing to strip.

        User uploads 4 images in the latest message, limit 3: stripping the
        older images cannot get under the limit, so the caller surfaces the
        original error (returns 0, no mutation).
        """
        msgs = [
            _user_with_images(1, tag="OLD"),
            _user_with_images(4, tag="NEW"),
        ]
        assert _total_images(msgs) == 5
        stripped = strip_images_to_count_limit(msgs, 3)
        assert stripped == 0
        assert _total_images(msgs) == 5  # untouched

    def test_protected_user_turn_kept_older_stripped(self):
        """Newest user turn fits under the limit → strip older images only.

        Stripping is whole-message (oldest-first), consistent with the rest of
        the compressor: the older user turn's image parts are all removed,
        which is enough to get under the limit.
        """
        msgs = [
            _user_with_images(2, tag="OLD"),
            _user_with_images(1, tag="NEW"),
        ]
        assert _total_images(msgs) == 3
        stripped = strip_images_to_count_limit(msgs, 2)
        assert stripped == 2
        assert _total_images(msgs) <= 2
        assert _count_image_parts(msgs[0]["content"]) == 0  # older stripped
        assert _count_image_parts(msgs[1]["content"]) == 1  # protected kept

    def test_none_falls_back_to_constant(self):
        """max_images=None uses _MAX_KEEP_TOOL_IMAGES as the target."""
        msgs = [
            _user_with_images(1, tag="OLD"),
            *_tool_with_images(0),
            *_tool_with_images(1),
            *_tool_with_images(2),
            *_tool_with_images(3),
        ]
        # 1 user + 4 tool = 5 images; target = _MAX_KEEP_TOOL_IMAGES (3).
        assert _total_images(msgs) == 5
        stripped = strip_images_to_count_limit(msgs, None)
        assert _total_images(msgs) == _MAX_KEEP_TOOL_IMAGES
        assert stripped == 5 - _MAX_KEEP_TOOL_IMAGES

    def test_no_images_returns_zero(self):
        msgs = [{"role": "user", "content": "hello"}]
        assert strip_images_to_count_limit(msgs, 2) == 0

    def test_under_limit_returns_zero(self):
        msgs = [_user_with_images(1, tag="A")]
        assert strip_images_to_count_limit(msgs, 2) == 0

    def test_multimodal_envelope_tool_handled(self):
        """A vision_analyze _multimodal envelope is counted and stripped."""
        msgs = [
            _user_with_images(1, tag="OLD"),
            {
                "role": "tool",
                "tool_call_id": "call_x",
                "content": {
                    "_multimodal": True,
                    "content": [
                        {"type": "text", "text": "attached"},
                        {
                            "type": "image_url",
                            "image_url": {"url": "data:image/png;base64,ENV"},
                        },
                    ],
                    "text_summary": "screenshot summary",
                },
            },
        ]
        assert _total_images(msgs) == 2
        stripped = strip_images_to_count_limit(msgs, 1)
        assert stripped == 1
        # The user upload (older) is stripped; the envelope tool result kept.
        assert _count_image_parts(msgs[0]["content"]) == 0
        assert _count_image_parts(msgs[1]["content"]) == 1

    def test_does_not_rewrite_persisted_history(self):
        """The strip mutates only the passed list; a separate history is safe."""
        history = [_user_with_images(2, tag="OLD"), *_tool_with_images(0)]
        outbound = [dict(m) for m in history]  # shallow per-call clone
        strip_images_to_count_limit(outbound, 1)
        assert _total_images(history) == 3  # original untouched
        assert _total_images(outbound) == 1
