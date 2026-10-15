"""Google's "Requests ending with a model turn are not supported." 400 on a
request ending with an image-bearing tool result (#139313).

``vision_analyze`` takes the native fast path on OpenRouter ``google/*``
models, so the image rides inside the ``role: "tool"`` message. When that
message is the LAST message of the request, Google rejects the call and
OpenRouter wraps the 400 as ``"Provider returned error"`` with the real
wording tucked inside ``error.metadata.raw``. Before the fix this text matched
no pattern, the turn read as a plain format error, and every vision turn
failed over to the (text-only) fallback provider.

The wording now classifies as ``multimodal_tool_content_unsupported`` so the
existing strip-and-retry recovery downgrades the trailing tool message to its
text part — the shape the upstream accepts — and records the
(provider, model) so later tool results downgrade without another 400.

The same text also fires for a thinking-only prefill leaving an assistant turn
at the end of the request (#75121); that path is prevented before the wire,
and the strip pass returning False for a request without image-bearing tool
messages keeps a slipped-through prefill from being "recovered" here.

See: https://github.com/NousResearch/hermes-agent/issues/139313
"""

from __future__ import annotations

import json

from agent.error_classifier import FailoverReason, classify_api_error


class _FakeApiError(Exception):
    """Stand-in for an openai.BadRequestError with status_code + body."""

    def __init__(self, status_code: int, message: str, body: dict | None = None):
        super().__init__(message)
        self.status_code = status_code
        self.body = body if body is not None else {"error": {"message": message}}
        self.response = None


def _make_agent(provider: str = "openrouter", model: str = "google/gemini-4-argon"):
    """Build a bare AIAgent for method-level testing, no provider setup."""
    from run_agent import AIAgent

    agent = object.__new__(AIAgent)
    agent.provider = provider
    agent.model = model
    return agent


# ─── Classification ──────────────────────────────────────────────────────────


class TestTrailingImageToolResultClassification:
    def test_openrouter_wrapped_400_classified_as_multimodal(self):
        """The exact body from the report: the Google wording only appears
        inside ``error.metadata.raw``, the outer message is the generic
        wrapper, so the classifier must unwrap it to see the rejection."""
        raw = json.dumps(
            {
                "error": {
                    "code": 400,
                    "message": "Requests ending with a model turn are not supported.",
                    "status": "INVALID_ARGUMENT",
                }
            }
        )
        e = _FakeApiError(
            400,
            "Error code: 400 - {'error': {'message': 'Provider returned error'}}",
            body={
                "error": {
                    "message": "Provider returned error",
                    "code": 400,
                    "metadata": {"raw": raw, "provider_name": "Google"},
                }
            },
        )
        result = classify_api_error(
            e, provider="openrouter", model="google/gemini-4-argon"
        )
        assert result.reason == FailoverReason.multimodal_tool_content_unsupported
        assert result.retryable is True

    def test_direct_400_wording_classified_as_multimodal(self):
        """The same rejection arriving unwrapped (native Gemini route, or a
        proxy that flattens the body): the wording sits in str(error)."""
        e = _FakeApiError(
            400,
            "400 INVALID_ARGUMENT: Requests ending with a model turn are not supported.",
        )
        result = classify_api_error(e, provider="gemini", model="gemini-4-argon")
        assert result.reason == FailoverReason.multimodal_tool_content_unsupported

    def test_unrelated_400_stays_format_error(self):
        """A generic 400 with neither the multimodal nor any other known
        wording must not ride the strip-and-retry path."""
        e = _FakeApiError(400, "Error code: 400 - invalid request: unknown field 'foo'")
        result = classify_api_error(
            e, provider="openrouter", model="google/gemini-4-argon"
        )
        assert result.reason != FailoverReason.multimodal_tool_content_unsupported


# ─── Recovery shape ──────────────────────────────────────────────────────────


class TestTrailingImageToolResultRecovery:
    def test_strip_downgrades_trailing_image_tool_message(self):
        """After the strip pass the request no longer ends on an image-bearing
        tool message — the accepted shape (report variant B)."""
        agent = _make_agent()
        msgs = [
            {"role": "user", "content": "analyze /tmp/test.png"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "t1",
                        "type": "function",
                        "function": {"name": "vision_analyze", "arguments": "{}"},
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "t1",
                "content": [
                    {"type": "text", "text": "Image loaded into your context."},
                    {
                        "type": "image_url",
                        "image_url": {"url": "data:image/png;base64,aGVsbG8="},
                    },
                ],
            },
        ]
        assert agent._try_strip_image_parts_from_tool_messages(msgs) is True
        assert msgs[-1]["role"] == "tool"
        assert isinstance(msgs[-1]["content"], str)
        assert "Image loaded into your context." in msgs[-1]["content"]
        assert (
            "openrouter",
            "google/gemini-4-argon",
        ) in agent._no_list_tool_content_models

    def test_strip_returns_false_when_no_image_tool_messages(self):
        """A request that ends on an assistant turn without any image-bearing
        tool message (the #75121 prefill shape) must not be "recovered": the
        strip pass finds nothing, returns False, the original error surfaces."""
        agent = _make_agent()
        msgs = [
            {"role": "user", "content": "continue your reasoning"},
            {"role": "assistant", "content": ""},
        ]
        assert agent._try_strip_image_parts_from_tool_messages(msgs) is False
        assert msgs[-1]["role"] == "assistant"
