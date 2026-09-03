"""Unit tests for persistence-failure-aware messaging in
``_normalize_empty_agent_response``.

When a turn is stopped because session persistence failed (SQLite lock
contention, disk exhaustion, ...), the user must NOT be told to /reset —
that destroys their conversation context and does nothing to fix storage.
They must also never see 'The request failed: None' when the gateway result
dict carries an explicit ``error: None``.
"""

import sys

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


class TestEmptySentinelNormalizedToFriendlyFallback:
    """The agent's '(empty)' terminal sentinel (conversation_loop.py sets
    final_response='(empty)' after the retry/fallback chain is exhausted)
    must never be delivered to a chat surface verbatim (#92924). It is a
    user-facing failure signal, so ``_normalize_empty_agent_response`` must
    treat it — and whitespace-padded variants of it — as an empty response
    and substitute the friendly fallback, exactly like a blank response."""

    def test_bare_empty_sentinel_becomes_friendly_fallback(self):
        agent_result = {"api_calls": 2}
        response = _normalize_empty_agent_response(agent_result, "(empty)", history_len=10)
        assert response != "(empty)"
        assert "no response was generated" in response

    def test_whitespace_padded_sentinel_becomes_friendly_fallback(self):
        agent_result = {"api_calls": 2}
        for padded in ("(empty)\n", " (empty) ", "(empty)\n\n"):
            response = _normalize_empty_agent_response(agent_result, padded, history_len=10)
            assert response != padded
            assert "(empty)" not in response
            assert "no response was generated" in response

    def test_whitespace_only_response_becomes_friendly_fallback(self):
        agent_result = {"api_calls": 3}
        response = _normalize_empty_agent_response(agent_result, "   \n  ", history_len=10)
        assert response.strip()
        assert "no response was generated" in response

    def test_sentinel_with_zero_api_calls_never_reaches_user(self):
        agent_result = {"api_calls": 0}
        response = _normalize_empty_agent_response(agent_result, "(empty)", history_len=10)
        assert response != "(empty)"
        assert "wasn't processed" in response

    def test_failed_turn_with_sentinel_uses_failure_message(self):
        agent_result = {"api_calls": 2, "failed": True, "error": "provider exploded"}
        response = _normalize_empty_agent_response(agent_result, "(empty)", history_len=10)
        assert "(empty)" not in response
        assert "The request failed: provider exploded" in response

    def test_real_text_still_passes_through(self):
        agent_result = {"api_calls": 1}
        assert _normalize_empty_agent_response(agent_result, "real answer", history_len=10) == "real answer"


class TestIsEmptyAgentSentinelHelper:
    """The shared sentinel classifier used by the gateway delivery chain."""

    def _helper(self):
        from gateway.run import _is_empty_agent_sentinel
        return _is_empty_agent_sentinel

    def test_recognizes_sentinel_variants(self):
        is_sentinel = self._helper()
        for value in (None, "", "   ", "\n", "(empty)", "(empty)\n", " (empty) "):
            assert is_sentinel(value), repr(value)

    def test_rejects_real_content(self):
        is_sentinel = self._helper()
        for value in ("hello", "(empty) but also words", "(emptyish)", "0", 123):
            assert not is_sentinel(value), repr(value)

    @staticmethod
    def _install_fake_agent_package(monkeypatch, tmp_path, anthropic_adapter_source):
        """Make `agent.anthropic_adapter` resolve to a fake, in-test module."""
        agent_dir = tmp_path / "agent"
        agent_dir.mkdir()
        (agent_dir / "__init__.py").write_text("", encoding="utf-8")
        (agent_dir / "anthropic_adapter.py").write_text(
            anthropic_adapter_source, encoding="utf-8"
        )
        monkeypatch.syspath_prepend(str(tmp_path))
        monkeypatch.delitem(sys.modules, "agent.anthropic_adapter", raising=False)
        monkeypatch.delitem(sys.modules, "agent", raising=False)

    def test_import_error_falls_back_to_literal(self, monkeypatch, tmp_path):
        """Standalone/test edge still degrades gracefully on ImportError.

        With the sentinel constant absent from the (fake) module, the lazy
        ``from ... import`` raises ImportError and the classifier must fall
        back to the ``"(empty)"`` literal (#92924 review: narrow to
        ImportError, keep the standalone edge working).
        """
        is_sentinel = self._helper()
        self._install_fake_agent_package(
            monkeypatch, tmp_path, "# no _EMPTY_TEXT_PLACEHOLDER here\n"
        )
        assert is_sentinel("(empty)")
        assert is_sentinel("   \n")
        assert not is_sentinel("real answer")

    def test_syntax_error_in_anthropic_adapter_surfaces(self, monkeypatch, tmp_path):
        """A genuine breakage in agent.anthropic_adapter must NOT be swallowed.

        A syntax error is not an ImportError, so the narrowed handler must let
        it propagate instead of silently pinning the fallback literal forever
        (#92924 review: real bugs surface loudly).
        """
        is_sentinel = self._helper()
        self._install_fake_agent_package(
            monkeypatch, tmp_path, "def broken(:\n"
        )
        with pytest.raises(SyntaxError):
            is_sentinel("(empty)")

