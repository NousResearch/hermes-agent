"""Provider capacity retry hints must reach the retry wait (#145899)."""

from types import SimpleNamespace

import httpx
import openai
import pytest

from agent.retry_utils import RETRY_AFTER_CAP_S, provider_retry_after_seconds, reset_delay_from_message
from agent.turn_recovery import compute_error_backoff


@pytest.mark.parametrize("seconds", [42, 1.5, 900])
def test_capacity_message_drives_the_retry_wait(seconds):
    message = f"Upstream request failed: the cell is at capacity; retry shortly (after {seconds} s)"
    body = {"error": {"type": "server_error", "message": message}}
    response = httpx.Response(
        429, request=httpx.Request("POST", "https://example.test/v1/chat/completions"),
        json=body,
    )
    error = openai.RateLimitError(message, response=response, body=body)
    notices = []
    agent = SimpleNamespace(
        provider="opencode-go", api_key="test-key",
        _buffer_diagnostic_status=notices.append,
        _emit_diagnostic_status=notices.append,
        _emit_diagnostic_wait=notices.append,
        _client_log_context=lambda: "test route",
    )
    wait = compute_error_backoff(
        agent, error, retry_count=1, max_retries=3, is_rate_limited=True,
        is_zai_coding_overload=False, base_url="https://example.test/v1", model="test-model",
    )
    assert wait == min(seconds, RETRY_AFTER_CAP_S)
    assert reset_delay_from_message(message) == seconds


@pytest.mark.parametrize("headers, fields, message, expected", [
    ({"Retry-After": "12"}, {"retry_after": 18}, "retry shortly (after 42 s)", 12),
    ({}, {"retry_after": 18}, "retry shortly (after 42 s)", 18),
    ({"Retry-After": "0"}, {}, "retry shortly (after 42 s)", None),
    ({}, {}, "retry shortly (after 0 s)", None),
    ({}, {}, "retry shortly (after -4 s)", None),
    ({}, {}, "retry shortly (after 42 ms)", None),
    ({}, {}, "retry shortly (after 42 samples)", None),
    ({}, {}, "quota resets in 2 hours", None),
    ({}, {}, "retry after 2 seconds", 2),
])
def test_explicit_retry_hints_preserve_precedence_and_ignore_non_delays(headers, fields, message, expected):
    error = SimpleNamespace(
        response=SimpleNamespace(headers=headers),
        body={"error": {"message": message, **fields}},
    )
    assert provider_retry_after_seconds(error) == expected
