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
    """Failed turns that already carry the HTTP 400 envelope as ``final_response`` must still get
    the session-too-large rewrite (chat sanitizers would otherwise turn the envelope into a generic
    'provider failed' reply); every other failure with text keeps that text."""

    _ENVELOPE = 'HTTP 400: {"object":"error","model":"deepseek-v4-flash"}'

    def _failed(self, text):
        return {"final_response": text, "failed": True, "error": text, "api_calls": 1}

    def test_long_history_rewrites_envelope_to_session_too_large(self):
        response = _normalize_empty_agent_response(
            self._failed(self._ENVELOPE), self._ENVELOPE, history_len=138,
        )
        assert "context window" in response.lower()
        assert "/compact" in response
        assert self._ENVELOPE not in response

    @pytest.mark.parametrize("text", [
        "Billing or credits exhausted: HTTP 429: You exceeded your current quota",
        "API call failed after 3 retries: HTTP 429 rate limit exceeded",
        "HTTP 401: invalid authentication token",
    ])
    def test_non_overflow_failure_keeps_its_own_text(self, text):
        assert _normalize_empty_agent_response(self._failed(text), text, history_len=138) == text

    def test_curated_overflow_text_from_the_agent_survives(self):
        text = "Context compression timed out without reducing this conversation. No messages were dropped."
        result = {**self._failed(text), "compression_exhausted": True}
        assert _normalize_empty_agent_response(result, text, history_len=138) == text
