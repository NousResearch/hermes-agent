"""Tests for Google Workspace OAuth setup script.

Focuses on exchange_auth_code() — specifically that it uses a direct HTTP
token exchange without PKCE so it works on headless systems.
"""

import json
import sys
import unittest.mock
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest


def _make_client_secret(tmp_path) -> Path:
    """Write a minimal client_secret.json and return its path."""
    data = {
        "installed": {
            "client_id": "test-client-id",
            "client_secret": "test-client-secret",
            "token_uri": "https://oauth2.googleapis.com/token",
            "auth_uri": "https://accounts.google.com/o/oauth2/auth",
            "redirect_uris": ["http://localhost"],
        }
    }
    p = tmp_path / "google_client_secret.json"
    p.write_text(json.dumps(data))
    return p


def _make_token_response() -> dict:
    return {
        "access_token": "ya29.test-access-token",
        "refresh_token": "1//test-refresh-token",
        "token_type": "Bearer",
        "expires_in": 3600,
    }


# ---------------------------------------------------------------------------
# exchange_auth_code
# ---------------------------------------------------------------------------

class TestExchangeAuthCode:

    def _run(self, tmp_path, code, token_response=None, http_error=None):
        """Import and call exchange_auth_code with mocked HTTP and paths."""
        import importlib, types

        # Dynamically import the setup module from its file path
        spec = importlib.util.spec_from_file_location(
            "setup",
            Path(__file__).parent.parent.parent
            / "skills/productivity/google-workspace/scripts/setup.py",
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        # Patch TOKEN_PATH and CLIENT_SECRET_PATH
        client_secret_path = _make_client_secret(tmp_path)
        token_path = tmp_path / "google_token.json"

        import urllib.error

        if http_error:
            mock_resp = MagicMock()
            mock_resp.read.return_value = b'{"error": "invalid_grant"}'
            urlopen_side_effect = urllib.error.HTTPError(
                url="https://oauth2.googleapis.com/token",
                code=400,
                msg="Bad Request",
                hdrs={},
                fp=MagicMock(read=lambda: b'{"error": "invalid_grant"}'),
            )
        else:
            resp_body = json.dumps(token_response or _make_token_response()).encode()
            mock_response = MagicMock()
            mock_response.read.return_value = resp_body
            mock_response.__enter__ = lambda s: s
            mock_response.__exit__ = MagicMock(return_value=False)
            urlopen_side_effect = mock_response

        with patch.object(mod, "TOKEN_PATH", token_path), \
             patch.object(mod, "CLIENT_SECRET_PATH", client_secret_path), \
             patch("urllib.request.urlopen",
                   side_effect=[urlopen_side_effect] if http_error else None,
                   return_value=urlopen_side_effect if not http_error else None):
            if http_error:
                with pytest.raises(SystemExit):
                    mod.exchange_auth_code(code)
                return None, token_path
            else:
                mod.exchange_auth_code(code)
                return token_path.read_text(), token_path

    def test_plain_code_produces_token_file(self, tmp_path):
        """A plain auth code results in a saved token file."""
        token_json, token_path = self._run(tmp_path, "4/test-auth-code")
        assert token_path.exists()
        data = json.loads(token_json)
        assert data["token"] == "ya29.test-access-token"
        assert data["refresh_token"] == "1//test-refresh-token"
        assert data["client_id"] == "test-client-id"
        assert data["client_secret"] == "test-client-secret"

    def test_url_code_is_extracted(self, tmp_path):
        """A full redirect URL has the code extracted from query params."""
        url_code = "http://localhost:1/?code=4/extracted-code&scope=gmail"
        token_json, token_path = self._run(tmp_path, url_code)
        assert token_path.exists()
        data = json.loads(token_json)
        assert data["token"] == "ya29.test-access-token"

    def test_no_pkce_in_request(self, tmp_path):
        """The token request does not include code_verifier or code_challenge."""
        import urllib.request as _urllib_request

        spec = __import__("importlib").util.spec_from_file_location(
            "setup2",
            Path(__file__).parent.parent.parent
            / "skills/productivity/google-workspace/scripts/setup.py",
        )
        mod = __import__("importlib").util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        client_secret_path = _make_client_secret(tmp_path)
        token_path = tmp_path / "google_token.json"

        captured_data = []

        resp_body = json.dumps(_make_token_response()).encode()
        mock_response = MagicMock()
        mock_response.read.return_value = resp_body
        mock_response.__enter__ = lambda s: s
        mock_response.__exit__ = MagicMock(return_value=False)

        original_urlopen = _urllib_request.urlopen

        def capturing_urlopen(req, **kwargs):
            captured_data.append(req.data)
            return mock_response

        with patch.object(mod, "TOKEN_PATH", token_path), \
             patch.object(mod, "CLIENT_SECRET_PATH", client_secret_path), \
             patch("urllib.request.urlopen", side_effect=capturing_urlopen):
            mod.exchange_auth_code("4/test-code")

        assert captured_data, "urlopen was not called"
        body = captured_data[0].decode()
        assert "code_verifier" not in body
        assert "code_challenge" not in body
        assert "grant_type=authorization_code" in body

    def test_http_error_exits_nonzero(self, tmp_path):
        """An HTTP error from the token endpoint causes a non-zero exit."""
        _, token_path = self._run(tmp_path, "4/bad-code", http_error=True)
        assert not token_path.exists()

    def test_saved_token_has_correct_keys(self, tmp_path):
        """Saved token JSON contains all keys needed by google.oauth2.credentials."""
        token_json, _ = self._run(tmp_path, "4/test-code")
        data = json.loads(token_json)
        for key in ("token", "refresh_token", "token_uri", "client_id", "client_secret", "scopes"):
            assert key in data, f"Missing key: {key}"

    def test_web_client_secret_format(self, tmp_path):
        """Also works with 'web' type client secrets (not just 'installed')."""
        data = {
            "web": {
                "client_id": "web-client-id",
                "client_secret": "web-client-secret",
                "token_uri": "https://oauth2.googleapis.com/token",
                "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                "redirect_uris": ["http://localhost"],
            }
        }
        client_secret_path = tmp_path / "google_client_secret.json"
        client_secret_path.write_text(json.dumps(data))
        token_path = tmp_path / "google_token.json"

        spec = __import__("importlib").util.spec_from_file_location(
            "setup3",
            Path(__file__).parent.parent.parent
            / "skills/productivity/google-workspace/scripts/setup.py",
        )
        mod = __import__("importlib").util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        resp_body = json.dumps(_make_token_response()).encode()
        mock_response = MagicMock()
        mock_response.read.return_value = resp_body
        mock_response.__enter__ = lambda s: s
        mock_response.__exit__ = MagicMock(return_value=False)

        with patch.object(mod, "TOKEN_PATH", token_path), \
             patch.object(mod, "CLIENT_SECRET_PATH", client_secret_path), \
             patch("urllib.request.urlopen", return_value=mock_response):
            mod.exchange_auth_code("4/test-code")

        data = json.loads(token_path.read_text())
        assert data["client_id"] == "web-client-id"


# ---------------------------------------------------------------------------
# get_auth_url
# ---------------------------------------------------------------------------

class TestGetAuthUrl:

    def _load_mod(self):
        import importlib
        spec = importlib.util.spec_from_file_location(
            "setup_url",
            Path(__file__).parent.parent.parent
            / "skills/productivity/google-workspace/scripts/setup.py",
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    def test_url_contains_no_pkce_params(self, tmp_path, capsys):
        """Auth URL must not contain code_challenge or code_challenge_method."""
        mod = self._load_mod()
        client_secret_path = _make_client_secret(tmp_path)

        with patch.object(mod, "CLIENT_SECRET_PATH", client_secret_path):
            mod.get_auth_url()

        out = capsys.readouterr().out.strip()
        assert "code_challenge" not in out
        assert "code_challenge_method" not in out

    def test_url_contains_required_params(self, tmp_path, capsys):
        """Auth URL contains client_id, redirect_uri, scope, response_type."""
        mod = self._load_mod()
        client_secret_path = _make_client_secret(tmp_path)

        with patch.object(mod, "CLIENT_SECRET_PATH", client_secret_path):
            mod.get_auth_url()

        out = capsys.readouterr().out.strip()
        assert "client_id=test-client-id" in out
        assert "response_type=code" in out
        assert "redirect_uri=" in out
        assert "scope=" in out
        assert "access_type=offline" in out
