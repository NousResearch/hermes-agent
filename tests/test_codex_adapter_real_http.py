"""Unit tests for CodexAuthAdapter (Phase 4c1 real HTTP).

No real network. Uses FakeHttpTransport (defined here, not in
production code). Validates provider-specific wire-format handling
that is NOT covered by the common contract suite.
"""

from __future__ import annotations

import inspect
from dataclasses import FrozenInstanceError
from typing import Any

import pytest

from agent.provider_adapter import (
    LLMExecutionRequest,
    ProviderFailure,
)
from agent.provider_errors import ProviderErrorCode
from agent.provider_http import HttpResponse, HttpTransport
from agent.providers.codex_auth import CodexAuthAdapter


# ---------------------------------------------------------------------------
# FakeHttpTransport (test double, lives here, NOT in production code)
# ---------------------------------------------------------------------------


class _FakeResponse:
    def __init__(
        self,
        status_code: int,
        body: dict[str, Any] | None,
        text: str = "",
    ) -> None:
        self.status_code = status_code
        self._body = body if body is not None else {}
        self.text = text if text else ("" if body is None else str(body))

    def json(self) -> dict[str, Any]:
        return self._body


class FakeHttpTransport:
    """Test double. Records calls; returns canned responses."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self.responses: list[_FakeResponse | Exception] = []
        self._index = 0

    def queue(self, response: _FakeResponse | Exception) -> None:
        self.responses.append(response)

    def post(
        self,
        url: str,
        *,
        headers: dict,
        json: dict,
        timeout: float,
    ) -> HttpResponse:
        self.calls.append(
            {
                "url": url,
                "headers": dict(headers),
                "json": dict(json),
                "timeout": float(timeout),
            }
        )
        if self._index >= len(self.responses):
            raise AssertionError("FakeHttpTransport: no queued response")
        item = self.responses[self._index]
        self._index += 1
        if isinstance(item, Exception):
            raise item
        return item  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def fake_transport() -> FakeHttpTransport:
    return FakeHttpTransport()


@pytest.fixture
def adapter(fake_transport: FakeHttpTransport) -> CodexAuthAdapter:
    return CodexAuthAdapter(
        api_key="fake-key-for-codex-tests",
        http_transport=fake_transport,
    )


def _request() -> LLMExecutionRequest:
    return LLMExecutionRequest(
        request_id="req-codex-001",
        decision_id="dec-codex-001",
        prompt="Review this architecture.",
        max_tokens=10,
        temperature=0.0,
    )


def _completed_body(text: str = "REVIEW OK") -> dict[str, Any]:
    return {
        "choices": [{"message": {"content": text}}],
        "usage": {"prompt_tokens": 5, "completion_tokens": 7},
    }


# ---------------------------------------------------------------------------
# Architecture / wiring
# ---------------------------------------------------------------------------


class TestCodexArchitecture:
    def test_adapter_uses_http_transport_protocol(self) -> None:
        sig = inspect.signature(CodexAuthAdapter.__init__)
        assert "http_transport" in sig.parameters
        assert "error_classifier" in sig.parameters
        assert "failure_policy" in sig.parameters

    def test_fake_transport_satisfies_protocol(self, fake_transport) -> None:
        assert isinstance(fake_transport, HttpTransport)

    def test_adapter_does_not_import_httpx_in_core_logic(self) -> None:
        import agent.providers.codex_auth as m

        module_src = inspect.getsource(m)
        assert "import httpx" not in module_src

    def test_fake_transport_does_not_mutate_payload(self, fake_transport) -> None:
        fake_transport.queue(_FakeResponse(200, _completed_body()))
        headers = {"Authorization": "Bearer X", "Content-Type": "application/json"}
        body = {"model": "codex-auth-v1", "messages": [{"role": "user", "content": "hi"}]}
        headers_copy = dict(headers)
        body_copy = dict(body)
        fake_transport.post("http://x", headers=headers, json=body, timeout=5.0)
        assert headers == headers_copy
        assert body == body_copy


# ---------------------------------------------------------------------------
# Successful path
# ---------------------------------------------------------------------------


class TestCodexSuccessfulExecution:
    def test_200_ok_returns_completed(self, adapter, fake_transport) -> None:
        fake_transport.queue(_FakeResponse(200, _completed_body("REVIEW OK")))
        result = adapter.execute(_request())
        assert result.status == "completed"
        assert result.output_text == "REVIEW OK"
        assert result.failure is None

    def test_post_uses_bearer_auth_and_codex_endpoint(
        self, adapter, fake_transport
    ) -> None:
        fake_transport.queue(_FakeResponse(200, _completed_body()))
        adapter.execute(_request())
        call = fake_transport.calls[0]
        assert call["url"].endswith("/chat/completions")
        assert call["headers"]["Authorization"].startswith("Bearer ")

    def test_null_usage_handled(self, fake_transport) -> None:
        local = FakeHttpTransport()

        class _Resp:
            status_code = 200
            text = '{"choices":[{"message":{"content":"OK"}}],"usage":null}'

            def json(self) -> dict[str, Any]:
                return {
                    "choices": [{"message": {"content": "OK"}}],
                    "usage": None,
                }

        local.responses.append(_Resp())
        adapter = CodexAuthAdapter(
            api_key="fake-key", http_transport=local
        )
        result = adapter.execute(_request())
        assert result.status == "completed"
        assert result.output_text == "OK"
        assert result.input_tokens == 0
        assert result.output_tokens == 0


# ---------------------------------------------------------------------------
# Error mapping
# ---------------------------------------------------------------------------


class TestCodexErrorMapping:
    def test_401_to_blocked(self, fake_transport) -> None:
        fake_transport.queue(
            _FakeResponse(401, {"error": {"code": "invalid_api_key"}})
        )
        adapter = CodexAuthAdapter(
            api_key="fake-key", http_transport=fake_transport
        )
        result = adapter.execute(_request())
        assert result.status == "blocked"
        assert isinstance(result.failure, ProviderFailure)
        assert result.failure.error_code == ProviderErrorCode.AUTH_ERROR.value

    def test_429_to_blocked(self, fake_transport) -> None:
        fake_transport.queue(
            _FakeResponse(429, {"error": {"code": "rate_limited"}})
        )
        adapter = CodexAuthAdapter(
            api_key="fake-key", http_transport=fake_transport
        )
        result = adapter.execute(_request())
        assert result.status == "blocked"
        assert result.failure.error_code == ProviderErrorCode.RATE_LIMIT_EXCEEDED.value

    def test_402_or_billing_to_quota_blocked(self, fake_transport) -> None:
        fake_transport.queue(
            _FakeResponse(400, None, text="insufficient_quota: out of credits")
        )
        adapter = CodexAuthAdapter(
            api_key="fake-key", http_transport=fake_transport
        )
        result = adapter.execute(_request())
        assert result.status == "blocked"
        assert result.failure.error_code == ProviderErrorCode.QUOTA_EXCEEDED.value

    def test_413_to_context_blocked(self, fake_transport) -> None:
        fake_transport.queue(
            _FakeResponse(413, {"error": {"code": "context_length_exceeded"}})
        )
        adapter = CodexAuthAdapter(
            api_key="fake-key", http_transport=fake_transport
        )
        result = adapter.execute(_request())
        assert result.status == "blocked"
        assert result.failure.error_code == ProviderErrorCode.CONTEXT_LENGTH_EXCEEDED.value

    def test_5xx_to_failed_retryable(self, fake_transport) -> None:
        fake_transport.queue(_FakeResponse(503, None))
        adapter = CodexAuthAdapter(
            api_key="fake-key", http_transport=fake_transport
        )
        result = adapter.execute(_request())
        assert result.status == "failed"
        assert result.failure.error_code == ProviderErrorCode.TRANSIENT_ERROR.value
        assert result.failure.retryable is True

    def test_invalid_response_to_failed_non_retryable(self, fake_transport) -> None:
        fake_transport.queue(_FakeResponse(418, None))
        adapter = CodexAuthAdapter(
            api_key="fake-key", http_transport=fake_transport
        )
        result = adapter.execute(_request())
        assert result.status == "failed"
        assert result.failure.error_code == ProviderErrorCode.INVALID_RESPONSE.value
        assert result.failure.retryable is False

    def test_timeout_exception_to_failed_retryable(self, fake_transport) -> None:
        class _TimeoutExc(Exception):
            pass

        fake_transport.queue(_TimeoutExc("timed out"))
        adapter = CodexAuthAdapter(
            api_key="fake-key", http_transport=fake_transport
        )
        result = adapter.execute(_request())
        assert result.status == "failed"
        assert result.failure.error_code == ProviderErrorCode.TIMEOUT.value
        assert result.failure.retryable is True


# ---------------------------------------------------------------------------
# Missing credentials
# ---------------------------------------------------------------------------


class TestCodexMissingCredentials:
    def test_no_api_key_returns_blocked_missing_credentials(self, fake_transport) -> None:
        adapter = CodexAuthAdapter(api_key=None, http_transport=fake_transport)
        import os

        old = os.environ.pop("CODEX_AUTH_TOKEN", None)
        try:
            result = adapter.execute(_request())
            assert result.status == "blocked"
            assert (
                result.failure.error_code
                == ProviderErrorCode.MISSING_CREDENTIALS.value
            )
            assert result.failure.retryable is False
            assert fake_transport.calls == []
        finally:
            if old is not None:
                os.environ["CODEX_AUTH_TOKEN"] = old

    def test_adapter_does_not_store_api_key(self, fake_transport) -> None:
        adapter = CodexAuthAdapter(
            api_key="fake-key", http_transport=fake_transport
        )
        assert not hasattr(adapter, "api_key")
        assert not hasattr(adapter, "_api_key")


# ---------------------------------------------------------------------------
# Health snapshot
# ---------------------------------------------------------------------------


class TestCodexHealthSnapshot:
    def test_initial_health_is_unavailable_when_no_credentials(
        self, fake_transport
    ) -> None:
        adapter = CodexAuthAdapter(api_key=None, http_transport=fake_transport)
        import os

        old = os.environ.pop("CODEX_AUTH_TOKEN", None)
        try:
            h = adapter.health()
            assert h.consecutive_failures == 0
            assert h.last_error is None
            assert h.is_available is False
        finally:
            if old is not None:
                os.environ["CODEX_AUTH_TOKEN"] = old

    def test_completed_resets_failures(self, adapter, fake_transport) -> None:
        fake_transport.queue(
            _FakeResponse(400, None, text="insufficient_quota")
        )
        adapter.execute(_request())
        fake_transport.queue(_FakeResponse(200, _completed_body()))
        adapter.execute(_request())
        h = adapter.health()
        assert h.consecutive_failures == 0
        assert h.last_error is None


# ---------------------------------------------------------------------------
# Immutability
# ---------------------------------------------------------------------------


class TestCodexAdapterImmutability:
    def test_adapter_does_not_store_api_key(self, fake_transport) -> None:
        # Verified separately in TestCodexMissingCredentials. This test
        # only checks that the adapter does not expose `api_key` or
        # `_api_key` as a public/private attribute on the instance.
        adapter = CodexAuthAdapter(
            api_key="fake-key", http_transport=fake_transport
        )
        assert not hasattr(adapter, "api_key")
        assert not hasattr(adapter, "_api_key")