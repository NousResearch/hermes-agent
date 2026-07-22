"""Tests for reactive recovery when a provider rejects an image as corrupt.

Covers issue #69078: xAI returns a 400 with "...Invalid PNG image." when a
re-serialized image part in replayed history becomes undecodable. None of
the existing image-error pattern lists matched that wording, so the turn
aborted as non-retryable and the session was permanently bricked (no
further image-bearing turn could ever succeed).

The fix is deliberately narrow (see review history below): only a
recognized corruption wording routes to strip-and-retry.

  1. agent/error_classifier.py: xAI's "Invalid PNG image." wording
     classifies as FailoverReason.image_corrupt — a new reason distinct
     from image_too_large, because shrinking corrupt bytes can't fix them.
     It must NOT be routed to the shrink path, and it takes precedence
     over image_too_large when a message trips both pattern lists.
  2. agent/conversation_loop.py: the image_corrupt branch strips image
     parts via the existing _strip_images_from_messages helper and
     retries once.

An earlier revision of this fix also shipped a generic fallback — "any
non-retryable 400 with image parts present strips and retries" — without
requiring a recognized wording. Review (Sol xhigh) correctly flagged this
as too destructive: unrelated non-retryable 400s (bad tool schema,
unsupported parameter, billing, content policy) that merely happen to
carry image parts would also get their vision history silently erased and
retried, degrading sessions that were never bricked in the first place.
That branch was reverted; only the classifier-routed image_corrupt path
remains. Future provider wordings get added to _IMAGE_CORRUPT_PATTERNS
(a known-string addition) rather than reintroducing a blanket net.

No TurnRetryState guard is needed for the surviving branch: retrying only
happens when _strip_images_from_messages actually removed something, and
it removes every image_url/image/input_image part from the request in one
pass. A second image_corrupt hit on the retried (now text-only) request
finds nothing left to strip, so the branch reports no progress and falls
through to the normal error path — no separate one-shot flag required.
"""

from __future__ import annotations

import copy
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from agent.error_classifier import FailoverReason, classify_api_error
from agent.message_sanitization import _strip_images_from_messages


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

    def test_xai_unaligned_truncation_wording_classifies_as_image_corrupt(self):
        """xAI's second wire wording for the same corruption class — fires
        on unaligned base64 truncation instead of aligned (which returns
        "Invalid PNG image." instead). Same root cause, same recovery."""
        err = _FakeApiError(
            status_code=400,
            message='{"code":"invalid-argument","error":"The base64 string '
                     'of provided image cannot be decoded."}',
        )
        result = classify_api_error(err, provider="xai", model="grok-5")
        assert result.reason == FailoverReason.image_corrupt
        assert result.retryable is True

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
        """A non-retryable 400 with NO image-shaped wording at all must
        remain format_error — this is exactly the class of error the
        (reverted) generic fallback would have wrongly treated as
        image-related as long as the request happened to carry an image."""
        err = _FakeApiError(status_code=400, message="unsupported parameter: foo")
        result = classify_api_error(err, provider="xai", model="grok-5")
        assert result.reason == FailoverReason.format_error
        assert result.retryable is False

    def test_compound_message_prefers_corrupt_over_too_large(self):
        """P3 (Sol xhigh): a body matching BOTH corruption and too-large
        wording must classify as image_corrupt, not image_too_large — the
        classifier checks _IMAGE_CORRUPT_PATTERNS first specifically so a
        provider that mentions size alongside corruption still routes to
        strip (shrink cannot repair corrupt bytes) rather than shrink."""
        err = _FakeApiError(
            status_code=400,
            message="Invalid PNG image. Note: image exceeds 8000 pixels.",
        )
        result = classify_api_error(err, provider="xai", model="grok-5")
        assert result.reason == FailoverReason.image_corrupt
        assert result.reason != FailoverReason.image_too_large


# ─── Strip helper (reused, not reimplemented) ────────────────────────────────


class TestStripHelperBehaviorBackingTheBranch:
    """The image_corrupt branch in conversation_loop.py only retries when
    _strip_images_from_messages actually removed something — this is the
    mechanism that makes a separate one-shot guard unnecessary."""

    def test_strips_and_reports_progress(self):
        msgs = [{
            "role": "user",
            "content": [
                {"type": "text", "text": "look"},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,corrupt"}},
            ],
        }]
        assert _strip_images_from_messages(msgs) is True
        assert msgs[0]["content"] == [{"type": "text", "text": "look"}]

    def test_second_pass_on_already_stripped_messages_reports_no_progress(self):
        """After a successful strip, a hypothetical repeat call (mirroring
        what a second image_corrupt hit on the retried request would see)
        finds nothing left to remove — this is what stops the branch from
        looping without needing a dedicated retry-state flag."""
        msgs = [{"role": "user", "content": [{"type": "text", "text": "look"}]}]
        assert _strip_images_from_messages(msgs) is False

    def test_no_image_parts_present_nothing_to_strip(self):
        msgs = [{"role": "user", "content": "just text, no images"}]
        assert _strip_images_from_messages(msgs) is False


# ─── Integration: run_conversation recovers from a corrupt-image 400 ─────────


def _mock_response(content: str):
    msg = SimpleNamespace(content=content, tool_calls=None)
    choice = SimpleNamespace(message=msg, finish_reason="stop")
    return SimpleNamespace(choices=[choice], model="grok-5", usage=None)


def _make_agent():
    """Build a minimal AIAgent, mirroring the idiom in
    tests/run_agent/test_32646_fallback_429_after_timeout.py."""
    from run_agent import AIAgent

    with (
        patch("run_agent.get_tool_definitions", return_value=[]),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("run_agent.OpenAI", return_value=MagicMock()),
    ):
        agent = AIAgent(
            api_key="fx",  # unused — the OpenAI client is mocked below
            base_url="https://openrouter.ai/api/v1",
            provider="openrouter",
            model="x-ai/grok-5",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )
        agent.client = MagicMock()
        return agent


class TestRunConversationRecoversFromCorruptImage400:
    def test_strip_and_retry_succeeds_at_the_sequenced_provider_layer(self):
        """A prior turn's image part has gone corrupt on replay. The first
        attempt gets xAI's "Invalid PNG image." 400; the loop must strip
        the image and retry once, succeeding on the second attempt with the
        image no longer in the outgoing request.
        """
        history = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "what's in this screenshot?"},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": (
                                "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEA"
                                "AAABCAYAAAAfFcSJAAAADUlEQVR4nGNgYGBgAAAABQABpfZFQA"
                                "AAAABJRU5ErkJggg=="
                            )
                        },
                    },
                ],
            },
            {"role": "assistant", "content": "I see a browser window."},
        ]

        calls = []

        def fake_api_call(api_kwargs):
            # Snapshot — api_kwargs["messages"] is mutated in place by the
            # strip-and-retry recovery, so a bare reference would silently
            # reflect the post-strip state for every prior call too.
            calls.append(copy.deepcopy(api_kwargs.get("messages")))
            attempt = len(calls)
            if attempt == 1:
                raise _FakeApiError(
                    status_code=400,
                    message='{"code":"invalid-argument","error":"Invalid PNG image."}',
                )
            return _mock_response("The image was unreadable, but here's what I recall.")

        agent = _make_agent()
        agent._api_max_retries = 3

        with (
            patch.object(agent, "_interruptible_api_call", side_effect=fake_api_call),
            # The model is vision-capable in this scenario (that's *why* the
            # image reached the provider as a native image_url part in the
            # first place — the bug is corruption on replay, not routing).
            # Without this, an unrecognized model id defaults to text-mode
            # and the image never reaches the API as image_url at all.
            patch.object(agent, "_model_supports_vision", return_value=True),
            patch.object(agent, "_persist_session"),
            patch.object(agent, "_save_trajectory"),
            patch.object(agent, "_cleanup_task_resources"),
            patch("run_agent.OpenAI", return_value=MagicMock()),
            patch("agent.agent_runtime_helpers.time.sleep"),
            patch("agent.model_metadata.get_model_context_length", return_value=200000),
        ):
            result = agent.run_conversation(
                "did that look right?", conversation_history=history
            )

        assert result["completed"] is True
        assert result["final_response"] == (
            "The image was unreadable, but here's what I recall."
        )
        assert len(calls) == 2

        # First attempt still carried the (corrupted) image part.
        first_msgs = calls[0]
        assert any(
            isinstance(m.get("content"), list)
            and any(p.get("type") == "image_url" for p in m["content"])
            for m in first_msgs
        ), f"expected image_url part in first attempt, got: {first_msgs!r}"

        # Retry stripped it — no image_url parts anywhere in the request.
        second_msgs = calls[1]
        assert not any(
            isinstance(m.get("content"), list)
            and any(p.get("type") == "image_url" for p in m["content"])
            for m in second_msgs
        )
