"""Local-engine memory OOM must compress the context, not back off or bill the user.

oMLX, llama.cpp, vLLM, Ollama and MLX Metal / PyTorch MPS abort a request when the KV
cache or a device buffer cannot be allocated. The wording routinely contains "limit
exceeded" (e.g. "process memory limit exceeded", "kv cache limit exceeded"), which collides
with the usage-limit disambiguation: the failure was surfaced as a billing wall (told the user
to "add credits" for a local model with no billing) and then as ``overloaded`` — which only
backs off and re-sends the same oversized payload, or falls back to another local model with
the same memory ceiling, and never shrinks the context.

Behaviour contract:
  * a local-engine memory rejection classifies as ``FailoverReason.local_oom`` with
    ``should_compress=True`` (shrink the context — whose size IS the KV-cache size — and
    retry); it never rotates a credential and never falls back to another provider;
  * HTTP 402/429 wrapping such a message is still a memory problem, not a billing/quota wall;
  * a genuine account credit/usage limit is unchanged: still ``billing`` (retryable=False,
    should_rotate_credential=True).
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from agent.error_classifier import FailoverReason, classify_api_error


class MockAPIError(Exception):
    """Simulates an OpenAI SDK APIStatusError."""

    def __init__(self, message, status_code=None, body=None, headers=None):
        super().__init__(message)
        self.status_code = status_code
        self.body = body or {}
        self.response = SimpleNamespace(headers=headers or {})


# The exact message observed from a local oMLX/llama.cpp engine, plus the wordings the same
# family raises (oMLX prefill guard 0.5.6/0.5.7, KV-cache and device-allocation failures).
LOCAL_OOM_MESSAGES = [
    "Request aborted: process memory limit exceeded (usage 33.1 GB, ceiling 37.4 GB). "
    "Reduce context size or lower memory_guard_tier",
    "Prefill memory guard rejected request: Prefill would require ~13.87 GB peak, "
    "dynamic ceiling is 13.50 GB. Reduce context length or lower memory_guard_tier.",
    "predicted peak would exceed prefill safety cap 77.8GB. Reduce context length.",
    "memory_guard_tier exceeded",
    "kv cache allocation failed",
    "kv cache limit exceeded",
    "failed to allocate kv cache",
    "metal: allocation failed",
    "mps allocator: device out of memory",
    "cuda out of memory",
]

STATUS_CODES = (None, 400, 402, 429, 500)


@pytest.mark.parametrize("message", LOCAL_OOM_MESSAGES)
@pytest.mark.parametrize("status_code", STATUS_CODES)
def test_local_engine_oom_compresses_and_never_bills(message, status_code):
    kwargs = {} if status_code is None else {"status_code": status_code}
    result = classify_api_error(
        MockAPIError(message, **kwargs), provider="custom", model="gpt-oss-20b-M",
        base_url="http://127.0.0.1:8000/v1",
    )

    assert result.reason == FailoverReason.local_oom
    assert result.reason != FailoverReason.billing
    assert result.reason != FailoverReason.overloaded
    assert result.should_compress is True
    assert result.should_rotate_credential is False
    assert result.should_fallback is False
    assert result.retryable is True


def test_engine_memory_rejection_by_structured_code_compresses():
    """A proxy that flattened the wording leaves only the structured code (400 reads it, since
    _by_status runs before _by_error_code)."""
    result = classify_api_error(
        MockAPIError("Request failed.", status_code=400,
                     body={"error": {"message": "Request failed.", "code": "prefill_memory_exceeded"}}),
        provider="custom",
    )
    assert result.reason == FailoverReason.local_oom
    assert result.should_compress is True


def test_insufficient_credits_402_still_classifies_as_billing():
    """No over-reach: a real 402 credit exhaustion is still billing and still rotates."""
    result = classify_api_error(MockAPIError("Insufficient credits", status_code=402))
    assert result.reason == FailoverReason.billing
    assert result.retryable is False
    assert result.should_rotate_credential is True
    assert result.should_compress is False


def test_transient_usage_limit_still_is_rate_limit_not_billing():
    result = classify_api_error(
        MockAPIError("usage limit reached, try again in 5 minutes", status_code=402)
    )
    assert result.reason == FailoverReason.rate_limit
    assert result.reason != FailoverReason.billing


def test_genuine_context_overflow_still_compresses_as_context_overflow():
    """The new rule must not swallow a real window overflow."""
    result = classify_api_error(
        MockAPIError("This model's maximum context length is 200000 tokens. However, your "
                     "messages resulted in 250000 tokens.", status_code=400),
        provider="custom",
    )
    assert result.reason == FailoverReason.context_overflow
    assert result.should_compress is True
