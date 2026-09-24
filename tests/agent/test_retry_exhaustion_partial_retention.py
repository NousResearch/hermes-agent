"""#119001: a 429 / network error that exhausts retries AFTER partial output was
delivered must retain that output instead of ending the turn error-only.

Repro: attempt 1 streams text then dies mid-stream (partial-stub path appends a
``_length_continuation_fragment`` + nudge to ``messages``); the continuation
attempt 429s before the stream starts and every retry re-hits the limit.
``build_api_request`` resets ``_current_streamed_assistant_text`` per attempt,
so the only record of the delivered text is the fragment rows — the terminal
builders below must recover it, flag ``partial``, and keep ``final_response``
distinct from ``error`` (the gateway/desktop retention contract).
"""
from __future__ import annotations

from agent.error_classifier import classify_api_error
from agent.turn_recovery import (
    max_retries_exhausted_result,
    nonretryable_client_error_result,
)


class _Agent:
    log_prefix = ""
    verbose = False
    verbose_logging = False
    provider = "openrouter"
    model = "m"
    base_url = "https://openrouter.ai/api/v1"
    _current_streamed_assistant_text = ""

    def _summarize_api_error(self, error):
        return str(error)

    def _has_pending_fallback(self):
        return False

    def _try_activate_fallback(self, **_kw):
        return False

    def __getattr__(self, name):
        return lambda *args, **kwargs: None


class _Http(Exception):
    def __init__(self, status_code, message):
        super().__init__(message)
        self.status_code = status_code


PARTIAL = "Here is the first half of the report, already shown to the user."


def _messages_with_fragment():
    return [
        {"role": "user", "content": "write a long report"},
        {
            "role": "assistant",
            "content": PARTIAL,
            "_length_continuation_fragment": True,
        },
        {
            "role": "user",
            "content": "continue where you left off",
            "_length_continuation_nudge": True,
        },
    ]


def _exhausted(messages, agent=None):
    error = _Http(429, "HTTP 429: RequestBurstTooFast — slow down traffic growth")
    classified = classify_api_error(error, provider="openrouter", model="m")
    assert classified.reason.value == "rate_limit"
    return max_retries_exhausted_result(
        agent or _Agent(), error, classified, max_retries=3, is_rate_limited=True,
        error_msg=str(error).lower(), api_kwargs=None, api_messages=[], messages=messages,
        conversation_history=None, api_call_count=3, approx_tokens=10, provider="openrouter",
        base_url="https://openrouter.ai/api/v1", model="m",
    )


def test_exhausted_429_keeps_delivered_partial():
    result = _exhausted(_messages_with_fragment())
    assert result.get("partial") is True
    assert PARTIAL in result["final_response"]
    # Gateway retention contract: final must differ from the error string.
    assert result["final_response"].strip() != str(result["error"]).strip()
    assert result["failure_reason"] == "rate_limit"
    assert result["failed"] is True


def test_exhausted_429_without_partial_is_unchanged():
    result = _exhausted([])
    assert "partial" not in result
    assert "/retry" in result["final_response"]
    assert result["failure_reason"] == "rate_limit"


def test_exhausted_429_live_accumulator_counts_as_delivered():
    agent = _Agent()
    agent._current_streamed_assistant_text = "fresh streamed text"
    result = _exhausted([], agent=agent)
    assert result.get("partial") is True
    assert "fresh streamed text" in result["final_response"]


def test_whitespace_only_fragment_is_not_partial():
    messages = _messages_with_fragment()
    messages[1] = dict(messages[1], content="   ")
    result = _exhausted(messages)
    assert "partial" not in result


def test_nonretryable_terminal_keeps_delivered_partial():
    error = _Http(400, "HTTP 400: Bad request")
    classified = classify_api_error(error, provider="openrouter", model="m")
    result = nonretryable_client_error_result(
        _Agent(), error, classified, status_code=400, api_kwargs=None, api_messages=[],
        messages=_messages_with_fragment(), conversation_history=None, api_call_count=1,
        approx_tokens=10, provider="openrouter",
        base_url="https://openrouter.ai/api/v1", model="m",
    )
    assert result.get("partial") is True
    assert PARTIAL in result["final_response"]
    assert result["final_response"].strip() != str(result["error"]).strip()


def test_exhausted_429_collapses_continuation_trail_into_one_assistant_row():
    messages = _messages_with_fragment()
    error = _Http(429, "HTTP 429: RequestBurstTooFast — slow down traffic growth")
    classified = classify_api_error(error, provider="openrouter", model="m")
    result = max_retries_exhausted_result(
        _Agent(), error, classified, max_retries=3, is_rate_limited=True,
        error_msg=str(error).lower(), api_kwargs=None, api_messages=[], messages=messages,
        conversation_history=None, api_call_count=3, approx_tokens=10, provider="openrouter",
        base_url="https://openrouter.ai/api/v1", model="m", current_turn_user_idx=0,
    )
    # No dangling synthetic nudge: the turn persists as user -> one assistant row.
    assert [m["role"] for m in messages] == ["user", "assistant"]
    assert messages[1]["content"] == PARTIAL
    assert not any(m.get("_length_continuation_nudge") for m in messages)
    assert result.get("partial") is True and PARTIAL in result["final_response"]
