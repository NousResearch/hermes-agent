"""Tests for granola_oauth.py — Granola OAuth token refresh (PKCE public client)."""

from unittest.mock import MagicMock, patch

import httpx
import pytest

from plugins.memory.tiered.granola_oauth import (
    GRANOLA_TOKEN_URL,
    GranolaTokenResult,
    refresh_granola_token,
)


class TestRefreshGranolaToken:
    """Tests for refresh_granola_token()."""

    @patch("plugins.memory.tiered.granola_oauth.httpx.post")
    def test_successful_refresh_with_token_rotation(self, mock_post: MagicMock) -> None:
        """Exchanges refresh token for new access token and captures rotated refresh token."""
        mock_resp = MagicMock(spec=httpx.Response)
        mock_resp.status_code = 200
        mock_resp.is_success = True
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {
            "access_token": "new-access-token",
            "expires_in": 3600,
            "token_type": "Bearer",
            "refresh_token": "rotated-rt",
        }
        mock_post.return_value = mock_resp

        result = refresh_granola_token(client_id="client-123", refresh_token="refresh-abc")

        assert isinstance(result, GranolaTokenResult)
        assert result.access_token == "new-access-token"
        assert result.expires_in == 3600
        assert result.new_refresh_token == "rotated-rt"

    @patch("plugins.memory.tiered.granola_oauth.httpx.post")
    def test_successful_refresh_without_rotation(self, mock_post: MagicMock) -> None:
        """Returns None for new_refresh_token when server does not rotate."""
        mock_resp = MagicMock(spec=httpx.Response)
        mock_resp.status_code = 200
        mock_resp.is_success = True
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {
            "access_token": "at",
            "expires_in": 3600,
            "token_type": "Bearer",
        }
        mock_post.return_value = mock_resp

        result = refresh_granola_token(client_id="c", refresh_token="r")

        assert result.access_token == "at"
        assert result.new_refresh_token is None

    @patch("plugins.memory.tiered.granola_oauth.httpx.post")
    def test_no_client_secret_in_request_body(self, mock_post: MagicMock) -> None:
        """PKCE public client — no client_secret should be sent."""
        mock_resp = MagicMock(spec=httpx.Response)
        mock_resp.status_code = 200
        mock_resp.is_success = True
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {
            "access_token": "tok",
            "expires_in": 3600,
            "token_type": "Bearer",
        }
        mock_post.return_value = mock_resp

        refresh_granola_token(client_id="c", refresh_token="r")

        call_args = mock_post.call_args
        # data= kwarg is the form-encoded dict
        sent_data = call_args.kwargs.get("data", call_args[1].get("data", {}))
        assert "client_secret" not in sent_data

    @patch("plugins.memory.tiered.granola_oauth.httpx.post")
    def test_http_error_with_json_error_body(self, mock_post: MagicMock) -> None:
        """Non-2xx with JSON error body raises with the error code from body."""
        mock_resp = MagicMock(spec=httpx.Response)
        mock_resp.status_code = 400
        mock_resp.is_success = False
        mock_resp.raise_for_status.side_effect = httpx.HTTPStatusError(
            "Bad Request", request=MagicMock(), response=mock_resp
        )
        mock_resp.json.return_value = {"error": "invalid_grant"}
        mock_resp.text = '{"error": "invalid_grant"}'
        mock_post.return_value = mock_resp

        with pytest.raises(RuntimeError, match="invalid_grant"):
            refresh_granola_token(client_id="c", refresh_token="bad")

    @patch("plugins.memory.tiered.granola_oauth.httpx.post")
    def test_http_error_with_non_json_body(self, mock_post: MagicMock) -> None:
        """Non-2xx with non-JSON body falls back to HTTP status code."""
        mock_resp = MagicMock(spec=httpx.Response)
        mock_resp.status_code = 503
        mock_resp.is_success = False
        mock_resp.raise_for_status.side_effect = httpx.HTTPStatusError(
            "Service Unavailable", request=MagicMock(), response=mock_resp
        )
        mock_resp.json.side_effect = ValueError("not JSON")
        mock_resp.text = "Service Unavailable"
        mock_post.return_value = mock_resp

        with pytest.raises(RuntimeError, match="HTTP 503"):
            refresh_granola_token(client_id="c", refresh_token="r")

    @patch("plugins.memory.tiered.granola_oauth.httpx.post")
    def test_correct_endpoint_and_content_type(self, mock_post: MagicMock) -> None:
        """Posts to Granola token endpoint with form-encoded content type."""
        mock_resp = MagicMock(spec=httpx.Response)
        mock_resp.status_code = 200
        mock_resp.is_success = True
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {
            "access_token": "tok",
            "expires_in": 3600,
            "token_type": "Bearer",
        }
        mock_post.return_value = mock_resp

        refresh_granola_token(client_id="c", refresh_token="r")

        call_args = mock_post.call_args
        # First positional arg is the URL
        assert call_args[0][0] == GRANOLA_TOKEN_URL
        # data= should contain the form fields
        sent_data = call_args.kwargs.get("data", call_args[1].get("data", {}))
        assert sent_data["grant_type"] == "refresh_token"
        assert sent_data["client_id"] == "c"
        assert sent_data["refresh_token"] == "r"

    @patch("plugins.memory.tiered.granola_oauth.httpx.post")
    def test_form_fields_sent_correctly(self, mock_post: MagicMock) -> None:
        """Verify all expected form fields are present in the request."""
        mock_resp = MagicMock(spec=httpx.Response)
        mock_resp.status_code = 200
        mock_resp.is_success = True
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {
            "access_token": "new-access-token",
            "expires_in": 3600,
            "token_type": "Bearer",
        }
        mock_post.return_value = mock_resp

        refresh_granola_token(client_id="client-123", refresh_token="refresh-abc")

        call_args = mock_post.call_args
        sent_data = call_args.kwargs.get("data", call_args[1].get("data", {}))
        assert sent_data["grant_type"] == "refresh_token"
        assert sent_data["client_id"] == "client-123"
        assert sent_data["refresh_token"] == "refresh-abc"
