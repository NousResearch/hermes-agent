"""A Gemini 429 must size the credential cooldown from the window Google itself names.

Cluster: Google's free-tier 429 carries the wait two ways — ``Please retry in 24.916159436s`` in
``error.message`` and a ``google.rpc.RetryInfo`` duration in ``error.details`` — and usually sends
no ``Retry-After`` header. Miss both and ``credential_pool`` falls back to its one-hour 429
default for a 25-second throttle, so a turn walks every key and then sits on the cross-provider
fallback chain for the rest of the hour.

The wait must reach the pool through ``GeminiAPIError.retry_after``, never through ``error.body``:
a body-level ``retry_after`` also feeds the in-turn retry loop, which has to rotate to another
credential instead of sleeping out the window on the key that just failed.
"""

from __future__ import annotations
import json
import time
import httpx
import pytest
from agent.gemini_native_adapter import gemini_http_error

GEMINI_429_MESSAGE = (
    "You exceeded your current quota, please check your plan and billing details. Quota exceeded for metric: "
    "generativelanguage.googleapis.com/generate_content_free_tier_input_token_count, limit: 250000, "
    "model: gemini-3.5-flash-lite. Please retry in 24.916159436s."
)


def _quota_429(retry_delay: str | None, *, message: str = GEMINI_429_MESSAGE) -> httpx.Response:
    details = [{"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": retry_delay}] if retry_delay else []
    body = {"error": {"code": 429, "message": message, "status": "RESOURCE_EXHAUSTED", "details": details}}
    return httpx.Response(
        429, content=json.dumps(body).encode(), headers={"content-type": "application/json"},
        request=httpx.Request("POST", "https://generativelanguage.googleapis.com/v1beta"),
    )


def test_retry_info_delay_sizes_the_error_window_without_a_header():
    err = gemini_http_error(_quota_429("24.916159436s"))
    assert err.retry_after == pytest.approx(24.916159436)


def test_a_retry_after_header_still_wins_over_the_structured_delay():
    response = _quota_429("24.9s")
    response.headers["Retry-After"] = "5"
    assert gemini_http_error(response).retry_after == pytest.approx(5.0)


@pytest.mark.parametrize("bad_delay", ["soon", "", "0s", "-4s"])
def test_an_unusable_retry_info_delay_is_ignored(bad_delay):
    """A malformed detail must not fabricate a window: None leaves the pool on its default."""
    assert gemini_http_error(_quota_429(bad_delay)).retry_after is None


def test_the_pool_reads_the_window_off_the_error_attribute():
    from agent.agent_runtime_helpers import extract_api_error_context

    context = extract_api_error_context(gemini_http_error(_quota_429("24.916159436s")))
    assert 0 < context["reset_at"] - time.time() <= 25


def test_the_prose_alone_still_sizes_the_window():
    """No RetryInfo (older free-tier bodies): the shared grammar must find ``retry in 24.91s``."""
    from agent.agent_runtime_helpers import extract_api_error_context

    context = extract_api_error_context(gemini_http_error(_quota_429(None)))
    assert 0 < context["reset_at"] - time.time() <= 25


def test_the_window_never_lands_on_the_error_body():
    """``body`` also drives in-turn backoff; the pool's window must stay off that path."""
    err = gemini_http_error(_quota_429("24.916159436s"))
    assert getattr(err, "body", None) is None
