"""Tests for granola_client.py — Granola MCP Streamable HTTP client."""

import json
from unittest.mock import MagicMock, call, patch

import httpx
import pytest

from plugins.memory.tiered.granola_client import GRANOLA_MCP_URL, GranolaMcpClient


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ok_json_response(
    body: dict,
    extra_headers: dict[str, str] | None = None,
) -> MagicMock:
    """Mock httpx.Response for a successful JSON response."""
    headers = {"content-type": "application/json"}
    if extra_headers:
        headers.update(extra_headers)
    resp = MagicMock(spec=httpx.Response)
    resp.status_code = 200
    resp.is_success = True
    resp.raise_for_status = MagicMock()
    resp.json.return_value = body
    resp.text = json.dumps(body)
    resp.headers = headers
    return resp


def _sse_response(
    events: str,
    extra_headers: dict[str, str] | None = None,
) -> MagicMock:
    """Mock httpx.Response for an SSE response."""
    headers = {"content-type": "text/event-stream"}
    if extra_headers:
        headers.update(extra_headers)
    resp = MagicMock(spec=httpx.Response)
    resp.status_code = 200
    resp.is_success = True
    resp.raise_for_status = MagicMock()
    resp.text = events
    resp.headers = headers
    return resp


def _error_response(status: int, body: str = "") -> MagicMock:
    """Mock httpx.Response for an error."""
    resp = MagicMock(spec=httpx.Response)
    resp.status_code = status
    resp.is_success = False
    resp.text = body
    resp.headers = {}
    return resp


def _init_response(session_id: str | None = "session-123") -> MagicMock:
    """Standard initialize response."""
    headers = {}
    if session_id:
        headers["mcp-session-id"] = session_id
    return _ok_json_response(
        {"jsonrpc": "2.0", "id": "init", "result": {"capabilities": {}}},
        extra_headers=headers,
    )


def _notif_response() -> MagicMock:
    """Response for notifications/initialized (202 equivalent)."""
    resp = MagicMock(spec=httpx.Response)
    resp.status_code = 202
    resp.is_success = True
    resp.raise_for_status = MagicMock()
    resp.headers = {"content-type": "application/json"}
    resp.json.return_value = {}
    resp.text = ""
    return resp


# ===========================================================================
# TestInitialization
# ===========================================================================


class TestInitialization:
    """Tests for MCP session initialization handshake."""

    def test_sends_initialize_on_first_call_tool(self) -> None:
        """First callTool triggers initialize → notifications/initialized → tools/call."""
        get_token = MagicMock(return_value="test-oauth-token")
        client = GranolaMcpClient(get_token=get_token)

        with patch.object(client, "_http") as mock_http:
            mock_http.post.side_effect = [
                _init_response("session-123"),
                _notif_response(),
                _ok_json_response({"jsonrpc": "2.0", "id": "call", "result": {"meetings": []}}),
            ]

            client.call_tool("list_meetings", {})

            # 3 calls: initialize, notification, tools/call
            assert mock_http.post.call_count == 3

            # First call is initialize
            init_body = mock_http.post.call_args_list[0].kwargs.get(
                "json", mock_http.post.call_args_list[0][1].get("json", {})
            )
            assert init_body["method"] == "initialize"

            # Second call is initialized notification
            notif_body = mock_http.post.call_args_list[1].kwargs.get(
                "json", mock_http.post.call_args_list[1][1].get("json", {})
            )
            assert notif_body["method"] == "notifications/initialized"

            # Third call is tools/call
            tool_body = mock_http.post.call_args_list[2].kwargs.get(
                "json", mock_http.post.call_args_list[2][1].get("json", {})
            )
            assert tool_body["method"] == "tools/call"
            assert tool_body["params"]["name"] == "list_meetings"

    def test_captures_session_id_from_header(self) -> None:
        """Mcp-Session-Id from initialize response is used in subsequent calls."""
        get_token = MagicMock(return_value="test-token")
        client = GranolaMcpClient(get_token=get_token)

        with patch.object(client, "_http") as mock_http:
            mock_http.post.side_effect = [
                _init_response("session-456"),
                _notif_response(),
                _ok_json_response({"jsonrpc": "2.0", "id": "call", "result": {}}),
            ]

            client.call_tool("test_tool", {})

            # tools/call headers should include session ID
            tool_headers = mock_http.post.call_args_list[2].kwargs.get(
                "headers", mock_http.post.call_args_list[2][1].get("headers", {})
            )
            assert tool_headers.get("Mcp-Session-Id") == "session-456"

    def test_skips_notification_if_no_session_id(self) -> None:
        """If server doesn't return Mcp-Session-Id, skip notifications/initialized."""
        get_token = MagicMock(return_value="test-token")
        client = GranolaMcpClient(get_token=get_token)

        with patch.object(client, "_http") as mock_http:
            mock_http.post.side_effect = [
                _init_response(session_id=None),  # No session ID
                _ok_json_response({"jsonrpc": "2.0", "id": "call", "result": {}}),
            ]

            client.call_tool("test_tool", {})

            # Only 2 calls: initialize + tools/call (no notification)
            assert mock_http.post.call_count == 2


# ===========================================================================
# TestToolsCall
# ===========================================================================


class TestToolsCall:
    """Tests for tools/call request handling."""

    def _setup_initialized_client(self, mock_http: MagicMock) -> None:
        """Pre-populate mock_http with init + notification responses."""
        # Prepend init responses
        existing_effects = list(mock_http.post.side_effect or [])
        mock_http.post.side_effect = [
            _init_response("session-abc"),
            _notif_response(),
        ] + existing_effects

    def test_passes_tool_name_and_arguments(self) -> None:
        """tools/call sends name and arguments correctly."""
        get_token = MagicMock(return_value="test-token")
        client = GranolaMcpClient(get_token=get_token)

        with patch.object(client, "_http") as mock_http:
            mock_http.post.side_effect = [
                _init_response("session-abc"),
                _notif_response(),
                _ok_json_response(
                    {"jsonrpc": "2.0", "id": "call", "result": {"meetings": [{"id": "m1"}]}}
                ),
            ]

            result = client.call_tool("list_meetings", {"time_range": "last_week"})

            assert result == {"meetings": [{"id": "m1"}]}
            tool_body = mock_http.post.call_args_list[2].kwargs.get(
                "json", mock_http.post.call_args_list[2][1].get("json", {})
            )
            assert tool_body["params"]["arguments"]["time_range"] == "last_week"

    def test_authorization_header_includes_bearer_token(self) -> None:
        """Authorization header is set with Bearer token."""
        get_token = MagicMock(return_value="test-oauth-token")
        client = GranolaMcpClient(get_token=get_token)

        with patch.object(client, "_http") as mock_http:
            mock_http.post.side_effect = [
                _init_response("session-abc"),
                _notif_response(),
                _ok_json_response({"jsonrpc": "2.0", "id": "call", "result": {}}),
            ]

            client.call_tool("test_tool", {})

            tool_headers = mock_http.post.call_args_list[2].kwargs.get(
                "headers", mock_http.post.call_args_list[2][1].get("headers", {})
            )
            assert tool_headers["Authorization"] == "Bearer test-oauth-token"


# ===========================================================================
# TestSSEParsing
# ===========================================================================


class TestSSEParsing:
    """Tests for SSE response parsing."""

    def test_parses_sse_with_json_rpc_result(self) -> None:
        """SSE response with data: lines containing JSON-RPC result is parsed."""
        get_token = MagicMock(return_value="test-token")
        client = GranolaMcpClient(get_token=get_token)

        sse_text = (
            f'data: {json.dumps({"jsonrpc": "2.0", "id": "call", "result": {"answer": "yes"}})}\n\n'
        )

        with patch.object(client, "_http") as mock_http:
            mock_http.post.side_effect = [
                _init_response("session-sse"),
                _notif_response(),
                _sse_response(sse_text),
            ]

            result = client.call_tool("query", {"query": "test"})

            assert result == {"answer": "yes"}

    def test_handles_multiple_sse_events_skips_non_result(self) -> None:
        """Multiple SSE events — notifications are skipped, result is returned."""
        get_token = MagicMock(return_value="test-token")
        client = GranolaMcpClient(get_token=get_token)

        sse_text = (
            f'data: {json.dumps({"jsonrpc": "2.0", "method": "some_notification"})}\n'
            "\n"
            f'data: {json.dumps({"jsonrpc": "2.0", "id": "call", "result": {"data": "found"}})}\n'
            "\n"
        )

        with patch.object(client, "_http") as mock_http:
            mock_http.post.side_effect = [
                _init_response("session-sse"),
                _notif_response(),
                _sse_response(sse_text),
            ]

            result = client.call_tool("query", {})

            assert result == {"data": "found"}

    def test_raises_on_sse_json_rpc_error(self) -> None:
        """JSON-RPC error in SSE event stream raises RuntimeError."""
        get_token = MagicMock(return_value="test-token")
        client = GranolaMcpClient(get_token=get_token)

        sse_text = f'data: {json.dumps({"jsonrpc": "2.0", "id": "call", "error": {"code": -32000, "message": "Tool failed"}})}\n\n'

        with patch.object(client, "_http") as mock_http:
            mock_http.post.side_effect = [
                _init_response("session-sse"),
                _notif_response(),
                _sse_response(sse_text),
            ]

            with pytest.raises(RuntimeError, match="Tool failed"):
                client.call_tool("bad_tool", {})


# ===========================================================================
# TestSessionExpiry
# ===========================================================================


class TestSessionExpiry:
    """Tests for 404 session expired → reinit + retry."""

    def test_reinitializes_on_404_and_retries_once(self) -> None:
        """404 from tools/call triggers re-init and single retry."""
        get_token = MagicMock(return_value="test-token")
        client = GranolaMcpClient(get_token=get_token)

        with patch.object(client, "_http") as mock_http:
            mock_http.post.side_effect = [
                # First init
                _init_response("session-old"),
                _notif_response(),
                # tools/call gets 404
                _error_response(404),
                # Re-init
                _init_response("session-new"),
                _notif_response(),
                # Retry tools/call succeeds
                _ok_json_response({"jsonrpc": "2.0", "id": "call", "result": {"ok": True}}),
            ]

            result = client.call_tool("test_tool", {})

            assert result == {"ok": True}
            assert mock_http.post.call_count == 6


# ===========================================================================
# TestAuthFailure
# ===========================================================================


class TestAuthFailure:
    """Tests for 401 auth failure handling."""

    def test_raises_immediately_on_401_without_on_token_refresh(self) -> None:
        """401 without on_token_refresh callback raises immediately."""
        get_token = MagicMock(return_value="test-token")
        client = GranolaMcpClient(get_token=get_token)  # No on_token_refresh

        with patch.object(client, "_http") as mock_http:
            mock_http.post.side_effect = [
                _init_response("session-auth"),
                _notif_response(),
                _error_response(401, "token_expired"),
            ]

            with pytest.raises(RuntimeError, match="auth failed.*401"):
                client.call_tool("test", {})

            assert mock_http.post.call_count == 3

    def test_refreshes_token_and_retries_on_401_with_callback(self) -> None:
        """401 with on_token_refresh calls refresh, re-inits, retries."""
        # get_token is called 4 times:
        # 1. _initialize (first init)
        # 2. call_tool (first tools/call -> 401)
        # 3. _initialize (re-init after refresh)
        # 4. call_tool (retry tools/call -> success)
        token_sequence = iter(["old-token", "old-token", "refreshed-token", "refreshed-token"])
        get_token = MagicMock(side_effect=lambda: next(token_sequence))
        mock_refresh = MagicMock(return_value="refreshed-token")
        client = GranolaMcpClient(get_token=get_token, on_token_refresh=mock_refresh)

        with patch.object(client, "_http") as mock_http:
            mock_http.post.side_effect = [
                # First init succeeds
                _init_response("session-r1"),
                _notif_response(),
                # tools/call gets 401
                _error_response(401, "expired"),
                # Re-init after refresh
                _init_response("session-r2"),
                _notif_response(),
                # Retry succeeds
                _ok_json_response({"jsonrpc": "2.0", "id": "call", "result": {"ok": True}}),
            ]

            result = client.call_tool("test_tool", {})

            assert result == {"ok": True}
            mock_refresh.assert_called_once()

    def test_raises_if_refresh_also_gets_401(self) -> None:
        """After token refresh, if retry also gets 401, raises (no infinite loop)."""
        get_token = MagicMock(return_value="bad-token")
        mock_refresh = MagicMock(return_value="still-bad-token")
        client = GranolaMcpClient(get_token=get_token, on_token_refresh=mock_refresh)

        with patch.object(client, "_http") as mock_http:
            mock_http.post.side_effect = [
                # First init
                _init_response("session-f1"),
                _notif_response(),
                # First tools/call gets 401
                _error_response(401, "expired"),
                # Re-init after refresh
                _init_response("session-f2"),
                _notif_response(),
                # Retry also gets 401 — should NOT retry again
                _error_response(401, "still expired"),
            ]

            with pytest.raises(RuntimeError, match="auth failed.*401"):
                client.call_tool("test_tool", {})

            mock_refresh.assert_called_once()

    def test_refresh_attempted_resets_on_error(self) -> None:
        """refreshAttempted flag resets when on_token_refresh raises, allowing future retries."""
        call_count = 0

        def refresh_side_effect() -> str:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RuntimeError("refresh network error")
            return "new-token"

        get_token = MagicMock(return_value="old-token")
        mock_refresh = MagicMock(side_effect=refresh_side_effect)
        client = GranolaMcpClient(get_token=get_token, on_token_refresh=mock_refresh)

        with patch.object(client, "_http") as mock_http:
            mock_http.post.side_effect = [
                # First init
                _init_response("session-t1"),
                _notif_response(),
                # tools/call gets 401
                _error_response(401, "expired"),
            ]

            # First attempt: refresh raises
            with pytest.raises(RuntimeError, match="refresh network error"):
                client.call_tool("test_tool", {})

        # Second attempt: refresh succeeds (flag was reset by try/finally)
        with patch.object(client, "_http") as mock_http:
            mock_http.post.side_effect = [
                # Re-init (session was nulled)
                _init_response("session-t2"),
                _notif_response(),
                # tools/call gets 401 again
                _error_response(401, "expired again"),
                # After successful refresh: re-init + retry
                _init_response("session-t3"),
                _notif_response(),
                # Retry succeeds
                _ok_json_response({"jsonrpc": "2.0", "id": "call", "result": {"recovered": True}}),
            ]

            result = client.call_tool("test_tool", {})

            assert result == {"recovered": True}
            assert mock_refresh.call_count == 2

    def test_401_during_initialize_triggers_refresh(self) -> None:
        """401 during initialize (not just tools/call) triggers token refresh."""
        token_sequence = iter(["old-token", "refreshed-token", "refreshed-token"])
        get_token = MagicMock(side_effect=lambda: next(token_sequence))
        mock_refresh = MagicMock(return_value="refreshed-token")
        client = GranolaMcpClient(get_token=get_token, on_token_refresh=mock_refresh)

        with patch.object(client, "_http") as mock_http:
            mock_http.post.side_effect = [
                # First init gets 401
                _error_response(401, "expired"),
                # Re-init after refresh succeeds
                _init_response("session-ri"),
                _notif_response(),
                # tools/call succeeds
                _ok_json_response({"jsonrpc": "2.0", "id": "call", "result": {"data": "ok"}}),
            ]

            result = client.call_tool("test_tool", {})

            assert result == {"data": "ok"}
            mock_refresh.assert_called_once()


# ===========================================================================
# TestJsonRpcErrors
# ===========================================================================


class TestJsonRpcErrors:
    """Tests for JSON-RPC error handling in JSON responses."""

    def test_raises_on_json_rpc_error_in_json_response(self) -> None:
        """JSON-RPC error object in a 200 JSON response raises RuntimeError."""
        get_token = MagicMock(return_value="test-token")
        client = GranolaMcpClient(get_token=get_token)

        with patch.object(client, "_http") as mock_http:
            mock_http.post.side_effect = [
                _init_response("session-err"),
                _notif_response(),
                _ok_json_response({
                    "jsonrpc": "2.0",
                    "id": "call",
                    "error": {"code": -32602, "message": "Invalid params"},
                }),
            ]

            with pytest.raises(RuntimeError, match="Invalid params"):
                client.call_tool("bad_tool", {})
