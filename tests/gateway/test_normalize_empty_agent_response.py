"""Unit tests for persistence-failure-aware messaging in
``_normalize_empty_agent_response``.

When a turn is stopped because session persistence failed (SQLite lock
contention, disk exhaustion, ...), the user must NOT be told to /reset —
that destroys their conversation context and does nothing to fix storage.
They must also never see 'The request failed: None' when the gateway result
dict carries an explicit ``error: None``.
"""

import pytest

from gateway.run import _normalize_empty_agent_response


class TestPersistenceFailureRecoveryMessage:
    """Failed turns whose failure_reason marks a session-persistence
    failure get a dedicated recovery message: reassure the user their
    history is protected, tell them to resend — never suggest /reset."""

    def test_locked_persistence_failure_gets_recovery_message(self):
        agent_result = {
            "final_response": "",
            "failed": True,
            "failure_reason": "session_persistence_failed:locked",
            "error": "session storage was locked by another writer",
            "api_calls": 2,
        }

        response = _normalize_empty_agent_response(agent_result, "", history_len=10)

        assert "send it again" in response.lower()
        assert "/reset" not in response
        assert "unknown error" not in response.lower()

    def test_disk_persistence_failure_mentions_disk(self):
        agent_result = {
            "final_response": "",
            "failed": True,
            "failure_reason": "session_persistence_failed:disk",
            "error": "session storage write failed: disk full",
            "api_calls": 1,
        }

        response = _normalize_empty_agent_response(agent_result, "", history_len=10)

        assert "disk" in response.lower()
        assert "/reset" not in response
        assert "unknown error" not in response.lower()

    def test_unknown_cause_persistence_failure_still_avoids_reset(self):
        agent_result = {
            "final_response": "",
            "failed": True,
            "failure_reason": "session_persistence_failed:unknown",
            "error": "session storage failure",
            "api_calls": 1,
        }

        response = _normalize_empty_agent_response(agent_result, "", history_len=10)

        assert "/reset" not in response
        assert "send it again" in response.lower()

    def test_legacy_shape_error_text_mentioning_session_storage(self):
        """Legacy failed results carry no failure_reason but an error text
        naming session storage — they must get the same recovery message."""
        agent_result = {
            "final_response": "",
            "failed": True,
            "error": "turn stopped: session storage unavailable",
            "api_calls": 1,
        }

        response = _normalize_empty_agent_response(agent_result, "", history_len=10)

        assert "/reset" not in response
        assert "send it again" in response.lower()


class TestExplicitNoneErrorIsNoneSafe:
    """The gateway result dict is built with ``'error': holder.get('error')``
    and can carry an EXPLICIT None, which bypasses dict.get defaults."""

    def test_explicit_none_error_never_renders_none(self):
        agent_result = {
            "final_response": "",
            "failed": True,
            "error": None,
            "api_calls": 1,
        }

        response = _normalize_empty_agent_response(agent_result, "", history_len=10)

        assert "None" not in response
        # Non-persistence generic failures may legitimately say
        # 'unknown error' — the defect is rendering the literal None.
        assert "unknown error" in response.lower()


class TestGenericFailureRegression:
    """Non-persistence failures keep the existing byte-identical message."""

    def test_provider_error_still_formats_request_failed(self):
        agent_result = {
            "final_response": "",
            "failed": True,
            "error": "provider exploded",
            "api_calls": 1,
        }

        response = _normalize_empty_agent_response(agent_result, "", history_len=10)

        assert "The request failed: provider exploded" in response
        assert "/reset" in response

    def test_context_failure_branch_unchanged(self):
        agent_result = {
            "final_response": "",
            "failed": True,
            "error": "prompt exceeds context window",
            "api_calls": 1,
        }

        response = _normalize_empty_agent_response(agent_result, "", history_len=60)

        assert "context window" in response
        assert "/compact" in response


class TestNonempty400EnvelopeOverflowReply:
    """Failed turns that already carry the HTTP 400 envelope as
    ``final_response`` must still get the session-too-large rewrite.

    Production order is normalize then Telegram sanitizer. The empty-response
    rewriter used to return the 56-char envelope unchanged, and the sanitizer
    then replaced it with a generic 'provider failed' — so users never saw
    /compact even though the gateway already classified the turn as overflow.
    """

    _ENVELOPE = 'HTTP 400: {"object":"error","model":"deepseek-v4-flash"}'

    def _failed_400(self):
        return {
            "final_response": self._ENVELOPE,
            "failed": True,
            "error": self._ENVELOPE,
            "api_calls": 1,
        }

    def test_long_history_rewrites_envelope_to_session_too_large(self):
        response = _normalize_empty_agent_response(
            self._failed_400(), self._ENVELOPE, history_len=138,
        )
        assert "context window" in response.lower()
        assert "/compact" in response
        assert self._ENVELOPE not in response

    def test_telegram_sanitizer_keeps_overflow_rewrite(self):
        from gateway.run import _sanitize_gateway_final_response

        rewritten = _normalize_empty_agent_response(
            self._failed_400(), self._ENVELOPE, history_len=138,
        )
        delivered = _sanitize_gateway_final_response("telegram", rewritten)
        assert "/compact" in delivered
        assert "provider failed after retries" not in delivered.lower()
