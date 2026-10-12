"""#119001: an API error that ends the turn AFTER partial output was delivered must
retain that output instead of ending the turn error-only, and must not persist the
dangling continuation trail.

Repro: attempt 1 streams text then dies mid-stream (partial-stub path appends a
``_length_continuation_fragment`` + nudge to ``messages``); the continuation
attempt then fails before the stream starts. ``build_api_request`` resets
``_current_streamed_assistant_text`` per attempt, so the only record of the
delivered text is the fragment rows. Both terminal builders are driven through
``settle_unrecovered_error`` so the ``current_turn_user_idx`` threading is real.
"""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from agent.error_classifier import classify_api_error
from agent.turn_api_error import settle_unrecovered_error


class _Agent:
    log_prefix = ""
    verbose = False
    verbose_logging = False
    provider = "openrouter"
    model = "m"
    base_url = "https://openrouter.ai/api/v1"
    _current_streamed_assistant_text = ""
    _fallback_index = 0

    @staticmethod
    def _strip_think_blocks(text):
        return text

    def _summarize_api_error(self, error):
        return str(error)

    def _has_pending_fallback(self):
        return False

    def _try_activate_fallback(self, **_kw):
        return False

    def _try_recover_primary_transport(self, *_a, **_kw):
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
        {"role": "user", "content": "earlier turn"},
        {"role": "assistant", "content": "stale", "_length_continuation_fragment": True},
        {"role": "user", "content": "write a long report"},
        {"role": "assistant", "content": PARTIAL, "_length_continuation_fragment": True},
        {"role": "user", "content": "continue where you left off", "_length_continuation_nudge": True},
    ]


@pytest.mark.parametrize("status, message", [
    (429, "HTTP 429: RequestBurstTooFast — slow down traffic growth"),  # max_retries_exhausted_result
    (401, "HTTP 401: invalid api key"),  # nonretryable_client_error_result
])
def test_terminal_error_keeps_partial_and_collapses_this_turns_trail(status, message):
    messages = _messages_with_fragment()
    error = _Http(status, message)
    classified = classify_api_error(error, provider="openrouter", model="m")
    retry = SimpleNamespace(copilot_stale_cred_retry_attempted=False, primary_recovery_attempted=True)
    with patch("agent.conversation_loop._is_copilot_provider", lambda a: False), \
            patch("agent.turn_recovery_autorecover.auto_recover_after_exhaustion", lambda *a, **k: None):
        verdict = settle_unrecovered_error(
            _Agent(), api_error=error, classified=classified, _retry=retry, status_code=status,
            error_msg=message.lower(), is_context_length_error=False, is_rate_limited=status == 429,
            _is_zai_coding_overload=False, _provider="openrouter", _base="https://openrouter.ai/api/v1",
            _model="m", messages=messages, api_messages=[], api_kwargs=None, active_system_prompt="",
            conversation_history=None, approx_tokens=10, retry_count=3, max_retries=3,
            compression_attempts=0, api_call_count=3, current_turn_user_idx=2,
        )
    assert verdict.action == "return"
    result = verdict.result
    assert result.get("partial") is True and result["failed"] is True
    assert PARTIAL in result["final_response"] and "stale" not in result["final_response"]
    # Gateway retention contract: final must differ from the error string.
    assert result["final_response"].strip() != str(result["error"]).strip()
    # No dangling synthetic nudge: this turn persists as user -> one assistant row;
    # the earlier turn (before current_turn_user_idx) is left untouched.
    assert [m["role"] for m in messages] == ["user", "assistant", "user", "assistant"]
    assert messages[1]["content"] == "stale"
    assert (messages[3]["content"], messages[3]["finish_reason"]) == (PARTIAL, "error")
