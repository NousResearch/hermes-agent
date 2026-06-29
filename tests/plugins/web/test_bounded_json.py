"""Tests for bounded web-provider JSON response reads."""

from __future__ import annotations

import pytest


class _FakeStreamResponse:
    encoding = "utf-8"

    def __init__(self, chunks: list[bytes]) -> None:
        self._chunks = chunks

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def raise_for_status(self) -> None:
        return None

    def iter_bytes(self):
        yield from self._chunks


def test_httpx_json_request_streams_and_parses(monkeypatch):
    from plugins.web import _bounded_json

    captured = {}

    def fake_stream(method, url, **kwargs):
        captured["method"] = method
        captured["url"] = url
        captured["kwargs"] = kwargs
        return _FakeStreamResponse([b'{"ok": ', b"true}"])

    monkeypatch.setattr(_bounded_json.httpx, "stream", fake_stream)

    result = _bounded_json.httpx_json_request(
        "GET",
        "https://example.test/search",
        params={"q": "hermes"},
        max_bytes=32,
    )

    assert result == {"ok": True}
    assert captured["method"] == "GET"
    assert captured["url"] == "https://example.test/search"
    assert captured["kwargs"]["params"] == {"q": "hermes"}


def test_httpx_json_request_rejects_oversized_body(monkeypatch):
    from plugins.web import _bounded_json

    monkeypatch.setattr(
        _bounded_json.httpx,
        "stream",
        lambda *a, **kw: _FakeStreamResponse([b'{"big":"', b"x" * 64, b'"}']),
    )

    with pytest.raises(_bounded_json.WebProviderResponseTooLarge):
        _bounded_json.httpx_json_request(
            "GET",
            "https://example.test/search",
            max_bytes=16,
        )
