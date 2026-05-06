"""Tests for the Web via HTTP+SSE platform adapter.

Coverage focus:
* Requirements check returns a bool.
* Adapter ``__init__`` reads env / extra config defaults correctly.
* Auth secret comparison is constant-time and rejects mismatches.
* CORS allowlist enforces origin restrictions and preflight returns the
  right header bundle.
* ``send()`` to an unknown session returns a `no_active_stream` failure
  rather than raising.
* ``_extract_text`` pulls the conversation text out of typed message bodies.
* ``_redact_session`` masks correctly for short and long session ids.

Network-bound behaviours (binding aiohttp, accepting a real POST, streaming
``event: token`` frames) are exercised via lightweight aiohttp client tests
that go through the real ``connect()`` / ``disconnect()`` lifecycle on an
ephemeral port. They are gated behind aiohttp availability so the suite
still runs on minimal installs.
"""

from __future__ import annotations

import asyncio
import json
import os
from unittest import mock

import pytest

from gateway.config import Platform, PlatformConfig


def _make_config(**extra) -> PlatformConfig:
    return PlatformConfig(enabled=True, extra=extra)


# ---------------------------------------------------------------------------
# check_web_via_http_sse_requirements
# ---------------------------------------------------------------------------


class TestRequirements:
    def test_returns_bool(self) -> None:
        from gateway.platforms.web_via_http_sse import check_web_via_http_sse_requirements

        assert isinstance(check_web_via_http_sse_requirements(), bool)


# ---------------------------------------------------------------------------
# Adapter __init__ — env + extra config plumbing
# ---------------------------------------------------------------------------


class TestAdapterInit:
    def _make(self, **extra):
        from gateway.platforms.web_via_http_sse import WebViaHttpSsePlatformAdapter

        return WebViaHttpSsePlatformAdapter(_make_config(**extra))

    def test_extra_overrides_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("WEB_VIA_HTTP_SSE_PORT", "9000")
        adapter = self._make(port=8888)
        assert adapter._port == 8888

    def test_falls_back_to_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("WEB_VIA_HTTP_SSE_HOST", "127.0.0.1")
        monkeypatch.setenv("WEB_VIA_HTTP_SSE_PATH", "/chat")
        monkeypatch.setenv("WEB_VIA_HTTP_SSE_AUTH_SECRET", "s3cret")
        adapter = self._make()
        assert adapter._host == "127.0.0.1"
        assert adapter._path == "/chat"
        assert adapter._secret == "s3cret"

    def test_default_port_is_8644(self, monkeypatch: pytest.MonkeyPatch) -> None:
        for k in (
            "WEB_VIA_HTTP_SSE_PORT",
            "WEB_VIA_HTTP_SSE_HOST",
            "WEB_VIA_HTTP_SSE_PATH",
        ):
            monkeypatch.delenv(k, raising=False)
        adapter = self._make()
        assert adapter._port == 8644
        assert adapter._host == "0.0.0.0"
        assert adapter._path == "/web"

    def test_allowed_origins_csv_parsed_to_set(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("WEB_VIA_HTTP_SSE_ALLOWED_ORIGINS", raising=False)
        adapter = self._make(
            allowed_origins="https://a.example, https://b.example,  ,https://c.example"
        )
        assert adapter._allowed_origins == {
            "https://a.example",
            "https://b.example",
            "https://c.example",
        }

    def test_allowed_origins_empty_means_no_restriction_metadata(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("WEB_VIA_HTTP_SSE_ALLOWED_ORIGINS", raising=False)
        adapter = self._make()
        assert adapter._allowed_origins == set()

    def test_platform_attribute(self) -> None:
        adapter = self._make()
        assert adapter.platform == Platform.WEB_VIA_HTTP_SSE


# ---------------------------------------------------------------------------
# Auth check
# ---------------------------------------------------------------------------


class TestAuthCheck:
    def _adapter(self, secret: str):
        from gateway.platforms.web_via_http_sse import WebViaHttpSsePlatformAdapter

        return WebViaHttpSsePlatformAdapter(_make_config(auth_secret=secret))

    def test_no_secret_configured_accepts_anything(self) -> None:
        adapter = self._adapter("")
        request = mock.MagicMock()
        request.headers = {}
        assert adapter._check_auth(request) is True

    def test_correct_secret_passes(self) -> None:
        adapter = self._adapter("s3cret")
        request = mock.MagicMock()
        request.headers = {"X-Web-Auth-Secret": "s3cret"}
        assert adapter._check_auth(request) is True

    def test_wrong_secret_rejected(self) -> None:
        adapter = self._adapter("s3cret")
        request = mock.MagicMock()
        request.headers = {"X-Web-Auth-Secret": "wrong"}
        assert adapter._check_auth(request) is False

    def test_missing_header_when_secret_required_rejected(self) -> None:
        adapter = self._adapter("s3cret")
        request = mock.MagicMock()
        request.headers = {}
        assert adapter._check_auth(request) is False


# ---------------------------------------------------------------------------
# CORS allowlist
# ---------------------------------------------------------------------------


class TestCorsAllowlist:
    def _adapter(self, *origins: str):
        from gateway.platforms.web_via_http_sse import WebViaHttpSsePlatformAdapter

        return WebViaHttpSsePlatformAdapter(
            _make_config(allowed_origins=",".join(origins) if origins else "")
        )

    def test_no_origin_yields_empty_headers(self) -> None:
        adapter = self._adapter("https://app.example")
        assert adapter._origin_headers(None) == {}

    def test_unknown_origin_yields_empty_headers(self) -> None:
        adapter = self._adapter("https://app.example")
        assert adapter._origin_headers("https://attacker.example") == {}

    def test_known_origin_yields_full_cors_bundle(self) -> None:
        adapter = self._adapter("https://app.example")
        headers = adapter._origin_headers("https://app.example")
        assert headers["Access-Control-Allow-Origin"] == "https://app.example"
        assert "X-Web-Auth-Secret" in headers["Access-Control-Allow-Headers"]
        assert "POST" in headers["Access-Control-Allow-Methods"]
        assert "OPTIONS" in headers["Access-Control-Allow-Methods"]
        assert headers["Vary"] == "Origin"

    def test_empty_allowlist_treats_all_origins_as_allowed(self) -> None:
        adapter = self._adapter()
        headers = adapter._origin_headers("https://anywhere.example")
        assert headers.get("Access-Control-Allow-Origin") == "https://anywhere.example"


# ---------------------------------------------------------------------------
# send() to unknown session
# ---------------------------------------------------------------------------


class TestSendNoActiveStream:
    @pytest.mark.asyncio
    async def test_send_to_unknown_session_returns_failure(self) -> None:
        from gateway.platforms.web_via_http_sse import WebViaHttpSsePlatformAdapter

        adapter = WebViaHttpSsePlatformAdapter(_make_config())
        result = await adapter.send("nonexistent-session", "hello")
        assert result.success is False
        assert result.error == "no_active_stream"

    @pytest.mark.asyncio
    async def test_emit_proactive_to_unknown_session_returns_false(self) -> None:
        from gateway.platforms.web_via_http_sse import WebViaHttpSsePlatformAdapter

        adapter = WebViaHttpSsePlatformAdapter(_make_config())
        ok = await adapter.emit_proactive("nonexistent-session", "ping")
        assert ok is False

    @pytest.mark.asyncio
    async def test_emit_tool_calling_to_unknown_session_returns_false(self) -> None:
        from gateway.platforms.web_via_http_sse import WebViaHttpSsePlatformAdapter

        adapter = WebViaHttpSsePlatformAdapter(_make_config())
        ok = await adapter.emit_tool_calling("nonexistent-session", "some_tool")
        assert ok is False


# ---------------------------------------------------------------------------
# get_chat_info
# ---------------------------------------------------------------------------


class TestGetChatInfo:
    @pytest.mark.asyncio
    async def test_returns_dm_shape(self) -> None:
        from gateway.platforms.web_via_http_sse import WebViaHttpSsePlatformAdapter

        adapter = WebViaHttpSsePlatformAdapter(_make_config())
        info = await adapter.get_chat_info("abc123")
        assert info == {"name": "web-abc123", "type": "dm", "chat_id": "abc123"}


# ---------------------------------------------------------------------------
# Helpers — _extract_text + _redact_session
# ---------------------------------------------------------------------------


class TestExtractText:
    def test_text_string(self) -> None:
        from gateway.platforms.web_via_http_sse import WebViaHttpSsePlatformAdapter

        assert WebViaHttpSsePlatformAdapter._extract_text("text", "hello") == "hello"

    def test_text_non_string_returns_empty(self) -> None:
        from gateway.platforms.web_via_http_sse import WebViaHttpSsePlatformAdapter

        assert WebViaHttpSsePlatformAdapter._extract_text("text", 42) == ""

    def test_image_with_text(self) -> None:
        from gateway.platforms.web_via_http_sse import WebViaHttpSsePlatformAdapter

        result = WebViaHttpSsePlatformAdapter._extract_text(
            "image", {"text": "describe", "imageData": "..."}
        )
        assert result == "describe"

    def test_multipart_picks_text(self) -> None:
        from gateway.platforms.web_via_http_sse import WebViaHttpSsePlatformAdapter

        result = WebViaHttpSsePlatformAdapter._extract_text(
            "multipart", {"text": "label this image", "imageData": "..."}
        )
        assert result == "label this image"

    def test_image_without_text_returns_empty(self) -> None:
        from gateway.platforms.web_via_http_sse import WebViaHttpSsePlatformAdapter

        assert WebViaHttpSsePlatformAdapter._extract_text("image", {}) == ""


class TestRedactSession:
    def test_redacts_long_session(self) -> None:
        from gateway.platforms.web_via_http_sse import _redact_session

        assert _redact_session("abcdef1234567890") == "abcd***90"

    def test_redacts_short_session_to_mask(self) -> None:
        from gateway.platforms.web_via_http_sse import _redact_session

        assert _redact_session("abc") == "***"

    def test_redacts_empty_to_mask(self) -> None:
        from gateway.platforms.web_via_http_sse import _redact_session

        assert _redact_session("") == "***"


# ---------------------------------------------------------------------------
# End-to-end: bind, accept a POST, stream events, close
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_bind_post_stream_close(monkeypatch: pytest.MonkeyPatch) -> None:
    """Round-trip: connect → POST a message → see token + done events → disconnect.

    The agent's ``handle_message`` is replaced with a stub that calls
    ``adapter.send()`` immediately, simulating a one-shot reply.
    """
    pytest.importorskip("aiohttp")
    import aiohttp
    from gateway.platforms.web_via_http_sse import WebViaHttpSsePlatformAdapter

    monkeypatch.delenv("WEB_VIA_HTTP_SSE_PORT", raising=False)
    adapter = WebViaHttpSsePlatformAdapter(
        _make_config(host="127.0.0.1", port=0, auth_secret="abc")
    )

    async def fake_handle(event):
        # Reply immediately as if the agent finished synchronously.
        await adapter.send(event.source.chat_id, "hello back")

    adapter.handle_message = fake_handle  # type: ignore[assignment]
    # Find a free port: bind a transient socket, grab the port, close it,
    # and trust that the OS hasn't given the same port to anything else
    # in the few microseconds between close and aiohttp's listen.
    import socket

    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    adapter._port = port

    assert await adapter.connect() is True
    try:
        body = {
            "message": {"type": "text", "content": "hi"},
            "bank_id": "test-bank",
            "user_attribution": {"user_id": "u1", "display_name": "Alice"},
        }
        url = f"http://127.0.0.1:{port}/web/sess-1"
        async with aiohttp.ClientSession() as client:
            async with client.post(
                url,
                json=body,
                headers={"X-Web-Auth-Secret": "abc"},
            ) as resp:
                assert resp.status == 200
                assert resp.headers["Content-Type"].startswith("text/event-stream")
                events = []
                async for line in resp.content:
                    events.append(line.decode("utf-8"))
                    if len(events) > 16:  # safety cap
                        break
                joined = "".join(events)
                assert "event: token" in joined
                assert "event: done" in joined
                assert "hello back" in joined
    finally:
        await adapter.disconnect()


@pytest.mark.asyncio
async def test_bind_rejects_bad_secret(monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("aiohttp")
    import aiohttp
    from gateway.platforms.web_via_http_sse import WebViaHttpSsePlatformAdapter

    monkeypatch.delenv("WEB_VIA_HTTP_SSE_PORT", raising=False)
    adapter = WebViaHttpSsePlatformAdapter(
        _make_config(host="127.0.0.1", port=0, auth_secret="abc")
    )

    import socket

    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    adapter._port = port

    assert await adapter.connect() is True
    try:
        async with aiohttp.ClientSession() as client:
            async with client.post(
                f"http://127.0.0.1:{port}/web/sess-1",
                json={"message": {"type": "text", "content": "x"}, "bank_id": "b"},
                headers={"X-Web-Auth-Secret": "wrong"},
            ) as resp:
                assert resp.status == 401
    finally:
        await adapter.disconnect()
