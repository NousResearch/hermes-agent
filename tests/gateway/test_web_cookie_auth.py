"""Tests for cookie-based auth in the dashboard web server."""

import http.cookies
import hmac
import secrets
from unittest.mock import patch

import pytest


class TestCookieAuth:
    """Test that cookie-based auth works alongside existing token paths."""

    def test_session_cookie_name_defined(self) -> None:
        """The cookie name constant should be defined."""
        from hermes_cli.web_server import _SESSION_COOKIE_NAME
        assert _SESSION_COOKIE_NAME == "hermes_session"

    def test_has_valid_session_token_accepts_cookie(self) -> None:
        """_has_valid_session_token should accept cookie auth."""
        from hermes_cli.web_server import (
            _SESSION_COOKIE_NAME,
            _SESSION_TOKEN,
            _has_valid_session_token,
        )

        # Mock a request with the cookie set
        class MockRequest:
            def __init__(self):
                self.headers = {}
                self.cookies = {_SESSION_COOKIE_NAME: _SESSION_TOKEN}

        assert _has_valid_session_token(MockRequest())

    def test_has_valid_session_token_accepts_header(self) -> None:
        """_has_valid_session_token should accept header auth."""
        from hermes_cli.web_server import (
            _SESSION_HEADER_NAME,
            _SESSION_TOKEN,
            _has_valid_session_token,
        )

        class MockRequest:
            def __init__(self):
                self.headers = {_SESSION_HEADER_NAME: _SESSION_TOKEN}
                self.cookies = {}

        assert _has_valid_session_token(MockRequest())

    def test_has_valid_session_token_accepts_bearer(self) -> None:
        """_has_valid_session_token should accept Bearer auth."""
        from hermes_cli.web_server import _SESSION_TOKEN, _has_valid_session_token

        class MockRequest:
            def __init__(self):
                self.headers = {"authorization": f"Bearer {_SESSION_TOKEN}"}
                self.cookies = {}

        assert _has_valid_session_token(MockRequest())

    def test_has_valid_session_token_rejects_wrong_token(self) -> None:
        """_has_valid_session_token should reject invalid tokens."""
        from hermes_cli.web_server import _has_valid_session_token

        class MockRequest:
            def __init__(self):
                self.headers = {}
                self.cookies = {"hermes_session": "wrong-token"}

        assert not _has_valid_session_token(MockRequest())

    def test_has_valid_session_token_cookie_priority(self) -> None:
        """Cookie should work even when other auth methods are absent."""
        from hermes_cli.web_server import (
            _SESSION_COOKIE_NAME,
            _SESSION_TOKEN,
            _has_valid_session_token,
        )

        class MockRequest:
            def __init__(self):
                self.headers = {
                    "authorization": "Bearer wrong-token",
                }
                self.cookies = {_SESSION_COOKIE_NAME: _SESSION_TOKEN}

        # Header is wrong but cookie is right → should accept
        assert _has_valid_session_token(MockRequest())


class TestWsTokenExtraction:
    """Test WebSocket token extraction from various sources."""

    def test_extract_ws_token_from_query(self) -> None:
        """Should extract token from query parameter."""
        from hermes_cli.web_server import _extract_ws_token

        class MockWS:
            query_params = {"token": "test-token-123"}
            headers = {}

        assert _extract_ws_token(MockWS()) == "test-token-123"

    def test_extract_ws_token_from_cookie(self) -> None:
        """Should extract token from cookie header."""
        from hermes_cli.web_server import (
            _SESSION_COOKIE_NAME,
            _extract_ws_token,
        )

        class MockWS:
            query_params = {}
            headers = {"cookie": f"{_SESSION_COOKIE_NAME}=cookie-token-456"}

        assert _extract_ws_token(MockWS()) == "cookie-token-456"

    def test_extract_ws_token_from_cookie_multiple(self) -> None:
        """Should extract the right cookie when multiple are present."""
        from hermes_cli.web_server import (
            _SESSION_COOKIE_NAME,
            _extract_ws_token,
        )

        class MockWS:
            query_params = {}
            headers = {
                "cookie": f"other=value; {_SESSION_COOKIE_NAME}=right-token; foo=bar"
            }

        assert _extract_ws_token(MockWS()) == "right-token"

    def test_extract_ws_token_from_session_header(self) -> None:
        """Should extract token from X-Hermes-Session-Token header."""
        from hermes_cli.web_server import (
            _SESSION_HEADER_NAME,
            _extract_ws_token,
        )

        class MockWS:
            query_params = {}
            headers = {_SESSION_HEADER_NAME.lower(): "header-token-789"}

        assert _extract_ws_token(MockWS()) == "header-token-789"

    def test_extract_ws_token_from_bearer_header(self) -> None:
        """Should extract token from Authorization: Bearer header."""
        from hermes_cli.web_server import _extract_ws_token

        class MockWS:
            query_params = {}
            headers = {"authorization": "Bearer bearer-token-abc"}

        assert _extract_ws_token(MockWS()) == "bearer-token-abc"

    def test_extract_ws_token_empty_when_no_token(self) -> None:
        """Should return empty string when no token is present."""
        from hermes_cli.web_server import _extract_ws_token

        class MockWS:
            query_params = {}
            headers = {}

        assert _extract_ws_token(MockWS()) == ""

    def test_extract_ws_token_query_takes_priority(self) -> None:
        """Query param should take priority over cookie (legacy compat)."""
        from hermes_cli.web_server import (
            _SESSION_COOKIE_NAME,
            _extract_ws_token,
        )

        class MockWS:
            query_params = {"token": "query-token"}
            headers = {"cookie": f"{_SESSION_COOKIE_NAME}=cookie-token"}

        assert _extract_ws_token(MockWS()) == "query-token"
