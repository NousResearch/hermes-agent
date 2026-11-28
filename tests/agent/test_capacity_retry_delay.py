"""A provider's free-text "retry shortly (after N s)" sets the wait before the next attempt (#145899)."""

from types import SimpleNamespace

import httpx
import openai
import pytest

from agent.retry_utils import RETRY_AFTER_CAP_S, provider_retry_after_seconds, reset_delay_from_message
from agent.turn_recovery import compute_error_backoff

_OPENCODE_CAPACITY = (
    "Upstream request failed: [ttft_admission_refused] the cell is at capacity; retry shortly (after {} s)"
)


def _opencode_429(seconds) -> openai.RateLimitError:
    message = _OPENCODE_CAPACITY.format(seconds)
    body = {"error": {"type": "server_error", "message": message}}
    response = httpx.Response(
        429, request=httpx.Request("POST", "https://opencode.ai/zen/go/v1/chat/completions"), json=body,
    )
    return openai.RateLimitError(f"Error code: 429 - {body}", response=response, body=body)


def _agent() -> SimpleNamespace:
    notices: list[str] = []
    return SimpleNamespace(
        provider="opencode-go", api_key="test-key", _notices=notices,
        _buffer_diagnostic_status=notices.append, _emit_diagnostic_status=notices.append,
        _emit_diagnostic_wait=notices.append, _client_log_context=lambda: "test route",
    )


class TestCapacityRetryHint:
    @pytest.mark.parametrize("seconds", [42, 1.5, 900])
    def test_opencode_capacity_429_waits_the_declared_time(self, seconds):
        wait = compute_error_backoff(
            _agent(), _opencode_429(seconds), retry_count=1, max_retries=3, is_rate_limited=True,
            is_zai_coding_overload=False, base_url="https://opencode.ai/zen/go/v1", model="deepseek-v4.1-flash",
        )
        assert wait == min(seconds, RETRY_AFTER_CAP_S)

    def test_free_text_parser_reads_the_capacity_grammar(self):
        assert reset_delay_from_message(_OPENCODE_CAPACITY.format(42)) == 42

    @pytest.mark.parametrize("headers, fields, message, expected", [
        # A structured cooldown outranks the text hint.
        ({"Retry-After": "12"}, {"retry_after": 18}, "retry shortly (after 42 s)", 12),
        ({}, {"retry_after": 18}, "retry shortly (after 42 s)", 18),
        # Grammars that already parsed keep parsing.
        ({}, {}, "Please retry after 2 seconds", 2),
        ({}, {}, "retry 7 secs", 7),
        # Not a usable wait.
        ({}, {}, "retry shortly (after 0 s)", None),
        ({}, {}, "retry shortly (after -4 s)", None),
        ({}, {}, "retry shortly (after 42 ms)", None),
        ({}, {}, "retry shortly (after 42 samples)", None),
        ({}, {}, "retry shortly", None),
        # A quota window is not a retry request.
        ({}, {}, "quota resets in 2 hours", None),
    ])
    def test_precedence_and_non_delays(self, headers, fields, message, expected):
        error = SimpleNamespace(
            response=SimpleNamespace(headers=headers), body={"error": {"message": message, **fields}},
        )
        assert provider_retry_after_seconds(error) == expected
