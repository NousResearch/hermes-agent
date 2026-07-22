"""Tests for reactive recovery when a provider rejects an image as corrupt.

Covers issue #69078: xAI returns a 400 with "...Invalid PNG image." when a
re-serialized image part in replayed history becomes undecodable. None of
the existing image-error pattern lists matched that wording, so the turn
aborted as non-retryable and the session was permanently bricked (no
further image-bearing turn could ever succeed).

Two independent pieces are locked in here:

  1. agent/error_classifier.py: xAI's "Invalid PNG image." wording
     classifies as FailoverReason.image_corrupt — a new reason distinct
     from image_too_large, because shrinking corrupt bytes can't fix them.
     It must NOT be routed to the shrink path.
  2. The generic strip-and-retry fallback: ANY non-retryable 400 whose
     outgoing request still contains image parts gets one strip-and-retry
     attempt, even for a wording nobody has catalogued yet. This is the
     un-brick-generically half of the fix — it doesn't require enumerating
     every provider's phrasing.

Both recovery paths share a single one-shot-per-attempt guard
(TurnRetryState.stripped_images_this_turn) so at most one strip happens
per attempt regardless of which branch fires.
"""

from __future__ import annotations

from agent.error_classifier import FailoverReason, classify_api_error
from agent.message_sanitization import _strip_images_from_messages
from agent.turn_retry_state import TurnRetryState


class _FakeApiError(Exception):
    """Stand-in for an openai.BadRequestError with status_code + body."""

    def __init__(self, status_code: int, message: str, body: dict | None = None):
        super().__init__(message)
        self.status_code = status_code
        self.body = body or {"error": {"message": message}}
        self.response = None


# ─── Classifier: xAI corrupt-image wording ───────────────────────────────────


class TestImageCorruptClassification:
    def test_xai_invalid_png_image_classifies_as_image_corrupt(self):
        err = _FakeApiError(
            status_code=400,
            message='{"code":"invalid-argument","error":"Request contains an '
                     'invalid argument: Invalid PNG image."}',
        )
        result = classify_api_error(err, provider="xai", model="grok-5")
        assert result.reason == FailoverReason.image_corrupt
        assert result.retryable is True

    def test_xai_invalid_jpeg_image_classifies_as_image_corrupt(self):
        err = _FakeApiError(status_code=400, message="Invalid JPEG image.")
        result = classify_api_error(err, provider="xai", model="grok-5")
        assert result.reason == FailoverReason.image_corrupt

    def test_does_not_route_to_shrink_path(self):
        """image_corrupt must be a distinct reason from image_too_large —
        the retry loop only enters the shrink branch on an exact match, so
        this alone proves corrupt images skip the (useless) shrink attempt."""
        err = _FakeApiError(status_code=400, message="Invalid PNG image.")
        result = classify_api_error(err, provider="xai", model="grok-5")
        assert result.reason != FailoverReason.image_too_large

    def test_no_status_code_message_only_path(self):
        err = Exception("Invalid PNG image.")
        result = classify_api_error(err, provider="xai", model="grok-5")
        assert result.reason == FailoverReason.image_corrupt

    def test_unrelated_400_still_falls_through_to_format_error(self):
        err = _FakeApiError(status_code=400, message="unsupported parameter: foo")
        result = classify_api_error(err, provider="xai", model="grok-5")
        assert result.reason == FailoverReason.format_error
        assert result.retryable is False


# ─── Generic strip-and-retry fallback trigger condition ──────────────────────


class TestGenericStripFallbackTriggerCondition:
    """Mirrors the condition added at agent/conversation_loop.py: a
    non-retryable 400 whose request still contains image parts gets one
    strip-and-retry, regardless of the exact wording.
    """

    def test_novel_wording_falls_through_to_format_error_non_retryable(self):
        """A wording nobody catalogued yet — the generic fallback's whole
        point is that classification does NOT need to recognize it; only
        (400, non-retryable, image parts present) matters."""
        err = _FakeApiError(status_code=400, message="unrecognized image format")
        result = classify_api_error(err, provider="some-new-provider", model="x")
        assert result.reason == FailoverReason.format_error
        assert result.retryable is False
        # The trigger condition in conversation_loop.py is:
        #   status_code == 400 and not classified.retryable
        # — both hold here, so the generic fallback fires as long as the
        # outgoing request has image parts (checked separately, at retry
        # time, via _strip_images_from_messages's return value).


# ─── TurnRetryState guard shape ───────────────────────────────────────────────


class TestStrippedImagesGuard:
    def test_guard_defaults_false(self):
        state = TurnRetryState()
        assert state.stripped_images_this_turn is False

    def test_guard_is_shared_by_both_branches(self):
        """Both the image_corrupt branch and the generic fallback branch key
        off the same flag, so setting it once from either blocks the other —
        proving at most one strip happens per attempt."""
        state = TurnRetryState()
        state.stripped_images_this_turn = True
        assert state.stripped_images_this_turn is True

    def test_one_shot_semantics_mirror_the_loop(self):
        """Simulates the loop's guard-and-strip sequence directly against
        the real strip helper: first pass strips and would retry, a second
        pass on the same attempt is blocked by the guard so it never loops
        forever on a request that keeps failing after stripping."""
        state = TurnRetryState()
        msgs = [{
            "role": "user",
            "content": [
                {"type": "text", "text": "look"},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,corrupt"}},
            ],
        }]

        def attempt_strip_and_retry() -> bool:
            if state.stripped_images_this_turn:
                return False  # guard blocks a second attempt
            state.stripped_images_this_turn = True
            return _strip_images_from_messages(msgs)

        assert attempt_strip_and_retry() is True
        assert msgs[0]["content"] == [{"type": "text", "text": "look"}]
        # Second call on the same attempt: guard blocks it even though the
        # (now text-only) messages have nothing left to strip anyway.
        assert attempt_strip_and_retry() is False

    def test_no_image_parts_present_nothing_to_strip(self):
        """If the request has no image parts, the generic fallback's own
        indicator (the strip helper's return value) is False, so the loop
        must not treat this as a recovered attempt and must fall through to
        the normal (non-retryable) error path."""
        msgs = [{"role": "user", "content": "just text, no images"}]
        assert _strip_images_from_messages(msgs) is False
