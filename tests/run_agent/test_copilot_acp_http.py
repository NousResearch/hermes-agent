"""Tests for CopilotACPClient streamable-http transport mode."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from agent.copilot_acp_client import (
    CopilotACPClient,
    _HttpACPSession,
    _is_http_base_url,
    _normalize_http_base_url,
)


# ---------------------------------------------------------------------------
# Helper: URL detection and normalization
# ---------------------------------------------------------------------------

class TestIsHttpBaseUrl:
    def test_acp_http(self):
        assert _is_http_base_url("acp+http://localhost:3100") is True

    def test_acp_https(self):
        assert _is_http_base_url("acp+https://example.com") is True

    def test_acp_tcp(self):
        assert _is_http_base_url("acp+tcp://127.0.0.1:3100") is True

    def test_plain_http_not_matched(self):
        assert _is_http_base_url("http://localhost:3100") is False

    def test_acp_marker(self):
        assert _is_http_base_url("acp://copilot") is False

    def test_empty(self):
        assert _is_http_base_url("") is False


class TestNormalizeHttpBaseUrl:
    def test_acp_http_to_http(self):
        assert _normalize_http_base_url("acp+http://localhost:3100") == "http://localhost:3100"

    def test_acp_https_to_https(self):
        assert _normalize_http_base_url("acp+https://example.com") == "https://example.com"

    def test_acp_tcp_to_http(self):
        assert _normalize_http_base_url("acp+tcp://127.0.0.1:3100") == "http://127.0.0.1:3100"

    def test_trailing_slash_stripped(self):
        assert _normalize_http_base_url("acp+http://localhost:3100/") == "http://localhost:3100"


# ---------------------------------------------------------------------------
# Helper: mock SSE response stream
# ---------------------------------------------------------------------------

def _make_sse_response(events: list[dict]) -> list[str]:
    """Build raw SSE lines from a list of JSON-RPC events."""
    lines = [":ok"]
    for ev in events:
        lines.append("")
        lines.append("event: message")
        lines.append(f"data: {json.dumps(ev)}")
    lines.append("")
    return lines


class _FakeStreamResponse:
    """Minimal httpx-like stream response that yields lines."""

    def __init__(self, lines: list[str], status_code: int = 200):
        self.status_code = status_code
        self._lines = lines

    def raise_for_status(self):
        if self.status_code >= 400:
            raise Exception(f"HTTP {self.status_code}")

    def iter_lines(self):
        yield from self._lines

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass


# ---------------------------------------------------------------------------
# _HttpACPSession
# ---------------------------------------------------------------------------

class TestHttpACPSession:

    def test_requires_httpx(self):
        """When httpx is not available, constructor raises RuntimeError."""
        with patch("agent.copilot_acp_client.httpx", None):
            with pytest.raises(RuntimeError, match="httpx is required"):
                _HttpACPSession("http://localhost:3100")

    def test_connect_success(self):
        mock_response = MagicMock()
        mock_response.json.return_value = {"connectionId": "test-conn-id", "sessionToken": "tok"}
        mock_response.raise_for_status = MagicMock()

        mock_client = MagicMock()
        mock_client.post.return_value = mock_response
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)

        with patch("agent.copilot_acp_client.httpx") as mock_httpx:
            mock_httpx.Client.return_value = mock_client
            session = _HttpACPSession("http://localhost:3100")
            conn_id = session._connect()

        assert conn_id == "test-conn-id"
        mock_client.post.assert_called_once_with(
            "http://localhost:3100/api/v1/acp/connect", json={}
        )

    def test_connect_missing_connection_id(self):
        mock_response = MagicMock()
        mock_response.json.return_value = {}
        mock_response.raise_for_status = MagicMock()

        mock_client = MagicMock()
        mock_client.post.return_value = mock_response
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)

        with patch("agent.copilot_acp_client.httpx") as mock_httpx:
            mock_httpx.Client.return_value = mock_client
            session = _HttpACPSession("http://localhost:3100")
            with pytest.raises(RuntimeError, match="connectionId"):
                session._connect()

    def test_rpc_sse_detects_error(self):
        error_event = {
            "jsonrpc": "2.0",
            "id": 1,
            "error": {"code": -32601, "message": "Method not found"},
        }
        lines = _make_sse_response([error_event])

        fake_stream = _FakeStreamResponse(lines)
        mock_client = MagicMock()
        mock_client.stream.return_value = fake_stream
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)

        with patch("agent.copilot_acp_client.httpx") as mock_httpx:
            mock_httpx.Client.return_value = mock_client
            session = _HttpACPSession("http://localhost:3100")
            with pytest.raises(RuntimeError, match="Method not found"):
                session._rpc_sse("conn-id", "bad_method", {}, msg_id=1)


# ---------------------------------------------------------------------------
# CopilotACPClient transport auto-detection
# ---------------------------------------------------------------------------

class TestCopilotACPClientTransportSelection:

    def test_http_mode_uses_http_session(self):
        """When base_url is acp+http://, _run_prompt delegates to _HttpACPSession."""
        mock_session = MagicMock()
        mock_session.prompt.return_value = ("hello", "")

        with patch("agent.copilot_acp_client._HttpACPSession", return_value=mock_session) as mock_cls:
            client = CopilotACPClient(
                api_key="test",
                base_url="acp+http://localhost:3100",
            )
            text, reasoning = client._run_prompt("test prompt", model="gpt-4", timeout_seconds=30)

        mock_cls.assert_called_once_with("acp+http://localhost:3100", timeout=30)
        mock_session.prompt.assert_called_once_with("test prompt", model="gpt-4", timeout=30)
        assert text == "hello"

    def test_acp_tcp_mode_uses_http_session(self):
        """acp+tcp:// URLs also route to HTTP mode."""
        mock_session = MagicMock()
        mock_session.prompt.return_value = ("ok", "")

        with patch("agent.copilot_acp_client._HttpACPSession", return_value=mock_session):
            client = CopilotACPClient(base_url="acp+tcp://127.0.0.1:3100")
            text, _ = client._run_prompt("hi", timeout_seconds=10)

        assert text == "ok"

    def test_stdio_mode_spawns_subprocess(self):
        """When base_url is acp://, _run_prompt spawns a subprocess (not HTTP)."""
        client = CopilotACPClient(
            base_url="acp://copilot",
            command="/usr/bin/false",
            args=[],
        )
        # subprocess.Popen will fail since /usr/bin/false isn't an ACP server,
        # but the point is that _HttpACPSession is NOT used.
        with patch("agent.copilot_acp_client._HttpACPSession") as mock_http:
            with pytest.raises((RuntimeError, BrokenPipeError, OSError)):
                client._run_prompt("test", timeout_seconds=1)
            mock_http.assert_not_called()
