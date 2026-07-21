"""Tests for HttpTransport abstraction (no real network)."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pytest

from agent.provider_http import HttpResponse, HttpTransport, HttpxTransport


# ---------------------------------------------------------------------------
# Fake implementations
# ---------------------------------------------------------------------------


class _FakeResponse:
    def __init__(self, status_code: int, body: Mapping[str, Any]) -> None:
        self.status_code = status_code
        self.text = str(body)
        self._body = body

    def json(self) -> Mapping[str, Any]:
        return self._body


class FakeHttpTransport:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self.next_response: _FakeResponse | None = None
        self.next_exception: Exception | None = None

    def post(
        self,
        url: str,
        *,
        headers: Mapping[str, str],
        json: Mapping[str, Any],
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
        if self.next_exception is not None:
            raise self.next_exception
        if self.next_response is None:
            raise AssertionError("FakeHttpTransport: no response queued")
        return self.next_response  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Protocol conformance
# ---------------------------------------------------------------------------


class TestProtocol:
    def test_fake_satisfies_protocol(self) -> None:
        assert isinstance(FakeHttpTransport(), HttpTransport)

    def test_httpx_transport_satisfies_protocol(self) -> None:
        assert isinstance(HttpxTransport(), HttpTransport)


# ---------------------------------------------------------------------------
# Payload preservation
# ---------------------------------------------------------------------------


class TestTransportBehavior:
    def test_fake_does_not_mutate_payload(self) -> None:
        fake = FakeHttpTransport()
        fake.next_response = _FakeResponse(200, {"ok": True})
        headers = {"Authorization": "Bearer X", "Content-Type": "application/json"}
        body = {"model": "m", "messages": [{"role": "user", "content": "hi"}]}
        headers_copy = dict(headers)
        body_copy = dict(body)
        fake.post("http://example.com", headers=headers, json=body, timeout=5.0)
        assert headers == headers_copy
        assert body == body_copy

    def test_fake_preserves_payload_contents(self) -> None:
        fake = FakeHttpTransport()
        fake.next_response = _FakeResponse(200, {"ok": True})
        fake.post(
            "http://example.com",
            headers={"A": "1"},
            json={"k": "v"},
            timeout=1.0,
        )
        assert fake.calls[0]["headers"] == {"A": "1"}
        assert fake.calls[0]["json"] == {"k": "v"}
        assert fake.calls[0]["url"] == "http://example.com"
        assert fake.calls[0]["timeout"] == 1.0


# ---------------------------------------------------------------------------
# HttpxTransport
# ---------------------------------------------------------------------------


class TestHttpxTransport:
    def test_construction_does_not_import_httpx(self, monkeypatch) -> None:
        # Construction with timeout_s should NOT eagerly import httpx.
        # HttpxTransport lazy-imports on first post().
        import sys

        # Remove httpx if present.
        monkeypatch.setitem(sys.modules, "httpx", None)  # type: ignore[arg-type]
        t = HttpxTransport(timeout_s=10.0)
        # Construction succeeded without httpx.

    def test_post_without_httpx_raises_runtime_error(self, monkeypatch) -> None:
        import sys

        # Force httpx to be missing.
        monkeypatch.setitem(sys.modules, "httpx", None)  # type: ignore[arg-type]
        t = HttpxTransport(timeout_s=10.0)
        with pytest.raises(RuntimeError):
            t.post(
                "http://x",
                headers={},
                json={},
                timeout=1.0,
            )