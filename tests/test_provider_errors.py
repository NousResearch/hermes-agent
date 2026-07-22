"""Tests for Phase 4c error classification and disposition policy."""

from __future__ import annotations

import inspect

import pytest

from agent.provider_errors import (
    DefaultProviderFailurePolicy,
    MiniMaxErrorClassifier,
    ProviderErrorCode,
    ProviderErrorFact,
    ProviderFailureDisposition,
    classify_missing_credentials,
)


# ---------------------------------------------------------------------------
# ProviderErrorCode
# ---------------------------------------------------------------------------


class TestProviderErrorCode:
    def test_is_closed_enum(self) -> None:
        from enum import Enum

        assert isinstance(ProviderErrorCode, type)
        assert issubclass(ProviderErrorCode, Enum)

    def test_values(self) -> None:
        names = {e.name for e in ProviderErrorCode}
        for required in (
            "AUTH_ERROR",
            "RATE_LIMIT_EXCEEDED",
            "QUOTA_EXCEEDED",
            "CONTEXT_LENGTH_EXCEEDED",
            "TRANSIENT_ERROR",
            "TIMEOUT",
            "INVALID_RESPONSE",
            "MISSING_CREDENTIALS",
        ):
            assert required in names

    def test_unknown_value_raises(self) -> None:
        with pytest.raises(ValueError):
            ProviderErrorCode("not_a_real_code")


# ---------------------------------------------------------------------------
# ProviderErrorFact
# ---------------------------------------------------------------------------


class TestProviderErrorFact:
    def test_has_only_documented_fields(self) -> None:
        import dataclasses

        field_names = {f.name for f in dataclasses.fields(ProviderErrorFact)}
        assert field_names == {
            "error_code",
            "reason",
            "http_status",
            "schema_version",
        }

    def test_no_disposition_field(self) -> None:
        import dataclasses

        field_names = {f.name for f in dataclasses.fields(ProviderErrorFact)}
        assert "disposition" not in field_names

    def test_no_retryable_or_blocked_fields(self) -> None:
        import dataclasses

        field_names = {f.name for f in dataclasses.fields(ProviderErrorFact)}
        assert "retryable" not in field_names
        assert "blocked" not in field_names

    def test_no_health_status_field(self) -> None:
        import dataclasses

        field_names = {f.name for f in dataclasses.fields(ProviderErrorFact)}
        assert "health_status" not in field_names


# ---------------------------------------------------------------------------
# MiniMaxErrorClassifier
# ---------------------------------------------------------------------------


class TestMiniMaxClassifier:
    def test_does_not_import_provider_failure_disposition(self) -> None:
        # The classifier's source must not contain references to the
        # disposition enum, which lives outside the classifier.
        src = inspect.getsource(MiniMaxErrorClassifier)
        # The classifier file itself imports ProviderFailureDisposition
        # (it lives in the same module), so we cannot grep the module
        # for the symbol. But the CLASS itself must not reference it
        # in its method bodies.
        for method_name in ("classify_http_response", "classify_exception"):
            method_src = inspect.getsource(
                getattr(MiniMaxErrorClassifier, method_name)
            )
            assert "ProviderFailureDisposition" not in method_src
            assert "ProviderHealthStatus" not in method_src
        # Sanity: file does mention ProviderErrorCode (expected).
        assert "ProviderErrorCode" in src

    def test_returns_none_for_200_ok_with_body(self) -> None:
        c = MiniMaxErrorClassifier()
        fact = c.classify_http_response(
            status_code=200, body={"choices": [{"message": {"content": "hi"}}]}
        )
        assert fact is None

    def test_200_ok_with_empty_body_returns_invalid_response(self) -> None:
        c = MiniMaxErrorClassifier()
        fact = c.classify_http_response(status_code=200, body=None)
        assert fact is not None
        assert fact.error_code is ProviderErrorCode.INVALID_RESPONSE

    def test_401_to_auth_error(self) -> None:
        c = MiniMaxErrorClassifier()
        fact = c.classify_http_response(status_code=401, body=None)
        assert fact.error_code is ProviderErrorCode.AUTH_ERROR
        assert fact.http_status == 401

    def test_403_to_auth_error(self) -> None:
        c = MiniMaxErrorClassifier()
        fact = c.classify_http_response(status_code=403, body=None)
        assert fact.error_code is ProviderErrorCode.AUTH_ERROR

    def test_429_to_rate_limit_exceeded(self) -> None:
        c = MiniMaxErrorClassifier()
        fact = c.classify_http_response(status_code=429, body=None)
        assert fact.error_code is ProviderErrorCode.RATE_LIMIT_EXCEEDED

    def test_402_to_quota_exceeded(self) -> None:
        c = MiniMaxErrorClassifier()
        fact = c.classify_http_response(status_code=402, body=None)
        assert fact.error_code is ProviderErrorCode.QUOTA_EXCEEDED

    def test_quota_body_hint_to_quota_exceeded(self) -> None:
        c = MiniMaxErrorClassifier()
        fact = c.classify_http_response(
            status_code=400,
            body=None,
            text="quota exceeded for monthly plan",
        )
        assert fact.error_code is ProviderErrorCode.QUOTA_EXCEEDED

    def test_400_to_context_length(self) -> None:
        c = MiniMaxErrorClassifier()
        fact = c.classify_http_response(status_code=400, body=None)
        assert fact.error_code is ProviderErrorCode.CONTEXT_LENGTH_EXCEEDED

    def test_413_to_context_length(self) -> None:
        c = MiniMaxErrorClassifier()
        fact = c.classify_http_response(status_code=413, body=None)
        assert fact.error_code is ProviderErrorCode.CONTEXT_LENGTH_EXCEEDED

    def test_context_body_hint_to_context_length(self) -> None:
        c = MiniMaxErrorClassifier()
        fact = c.classify_http_response(
            status_code=400,
            body=None,
            text="context_length exceeded",
        )
        assert fact.error_code is ProviderErrorCode.CONTEXT_LENGTH_EXCEEDED

    def test_5xx_to_transient_error(self) -> None:
        c = MiniMaxErrorClassifier()
        for code in (500, 502, 503, 504):
            fact = c.classify_http_response(status_code=code, body=None)
            assert fact.error_code is ProviderErrorCode.TRANSIENT_ERROR, code

    def test_timeout_exception_to_timeout(self) -> None:
        c = MiniMaxErrorClassifier()

        class _Timeout(Exception):
            pass

        fact = c.classify_exception(_Timeout("timed out"))
        assert fact.error_code is ProviderErrorCode.TIMEOUT

    def test_connection_error_to_timeout(self) -> None:
        c = MiniMaxErrorClassifier()
        fact = c.classify_exception(ConnectionError("refused"))
        assert fact.error_code is ProviderErrorCode.TIMEOUT

    def test_unknown_exception_to_invalid_response(self) -> None:
        c = MiniMaxErrorClassifier()
        fact = c.classify_exception(ValueError("???"))
        assert fact.error_code is ProviderErrorCode.INVALID_RESPONSE

    def test_unexpected_status_to_invalid_response(self) -> None:
        c = MiniMaxErrorClassifier()
        fact = c.classify_http_response(status_code=418, body=None)
        assert fact.error_code is ProviderErrorCode.INVALID_RESPONSE

    def test_classifier_returns_fact_not_disposition(self) -> None:
        c = MiniMaxErrorClassifier()
        fact = c.classify_http_response(status_code=401, body=None)
        assert isinstance(fact, ProviderErrorFact)
        # Fact does not have a `disposition` attribute.
        assert not hasattr(fact, "disposition")

    def test_classifier_never_logs_secret(self, capsys) -> None:
        c = MiniMaxErrorClassifier()
        c.classify_http_response(
            status_code=401,
            body=None,
            text="Bearer FAKEKEY_FAKEKEY_FAKEKEY",
        )
        captured = capsys.readouterr()
        assert "FAKEKEY_FAKEKEY_FAKEKEY" not in captured.out
        assert "FAKEKEY_FAKEKEY_FAKEKEY" not in captured.err


# ---------------------------------------------------------------------------
# DefaultProviderFailurePolicy
# ---------------------------------------------------------------------------


class TestDefaultPolicy:
    def test_auth_error_to_blocked(self) -> None:
        p = DefaultProviderFailurePolicy()
        fact = ProviderErrorFact(
            error_code=ProviderErrorCode.AUTH_ERROR, reason="x"
        )
        assert p.disposition(fact) is ProviderFailureDisposition.BLOCKED

    def test_quota_to_blocked(self) -> None:
        p = DefaultProviderFailurePolicy()
        fact = ProviderErrorFact(
            error_code=ProviderErrorCode.QUOTA_EXCEEDED, reason="x"
        )
        assert p.disposition(fact) is ProviderFailureDisposition.BLOCKED

    def test_rate_limit_to_blocked(self) -> None:
        p = DefaultProviderFailurePolicy()
        fact = ProviderErrorFact(
            error_code=ProviderErrorCode.RATE_LIMIT_EXCEEDED, reason="x"
        )
        assert p.disposition(fact) is ProviderFailureDisposition.BLOCKED

    def test_context_to_blocked(self) -> None:
        p = DefaultProviderFailurePolicy()
        fact = ProviderErrorFact(
            error_code=ProviderErrorCode.CONTEXT_LENGTH_EXCEEDED, reason="x"
        )
        assert p.disposition(fact) is ProviderFailureDisposition.BLOCKED

    def test_missing_credentials_to_blocked(self) -> None:
        p = DefaultProviderFailurePolicy()
        fact = ProviderErrorFact(
            error_code=ProviderErrorCode.MISSING_CREDENTIALS, reason="x"
        )
        assert p.disposition(fact) is ProviderFailureDisposition.BLOCKED

    def test_transient_to_retryable(self) -> None:
        p = DefaultProviderFailurePolicy()
        fact = ProviderErrorFact(
            error_code=ProviderErrorCode.TRANSIENT_ERROR, reason="x"
        )
        assert p.disposition(fact) is ProviderFailureDisposition.RETRYABLE

    def test_timeout_to_retryable(self) -> None:
        p = DefaultProviderFailurePolicy()
        fact = ProviderErrorFact(
            error_code=ProviderErrorCode.TIMEOUT, reason="x"
        )
        assert p.disposition(fact) is ProviderFailureDisposition.RETRYABLE

    def test_invalid_response_to_failed(self) -> None:
        p = DefaultProviderFailurePolicy()
        fact = ProviderErrorFact(
            error_code=ProviderErrorCode.INVALID_RESPONSE, reason="x"
        )
        assert p.disposition(fact) is ProviderFailureDisposition.FAILED

    def test_does_not_import_health_monitor(self) -> None:
        src = inspect.getsource(DefaultProviderFailurePolicy)
        assert "ProviderHealthMonitor" not in src

    def test_does_not_decide_provider_health_status(self) -> None:
        # The policy file imports ProviderHealthStatus only because the
        # module re-exports it; but the CLASS itself must not reference it.
        class_src = inspect.getsource(DefaultProviderFailurePolicy)
        assert "ProviderHealthStatus" not in class_src


# ---------------------------------------------------------------------------
# Missing credentials helper
# ---------------------------------------------------------------------------


class TestMissingCredentials:
    def test_classify_missing_credentials(self) -> None:
        fact = classify_missing_credentials()
        assert fact.error_code is ProviderErrorCode.MISSING_CREDENTIALS


# ---------------------------------------------------------------------------
# ProviderFailureDisposition
# ---------------------------------------------------------------------------


class TestDisposition:
    def test_is_closed_enum(self) -> None:
        names = {e.name for e in ProviderFailureDisposition}
        assert names == {"FAILED", "BLOCKED", "RETRYABLE"}

    def test_does_not_include_complete(self) -> None:
        assert not hasattr(ProviderFailureDisposition, "COMPLETE")