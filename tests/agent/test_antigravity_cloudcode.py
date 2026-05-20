"""Tests for the google-antigravity OAuth + Antigravity Code Assist provider."""

from __future__ import annotations

import json
import os
import stat
import time
import threading
import urllib.parse
from io import BytesIO
from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def _isolate_env(monkeypatch, tmp_path):
    home = tmp_path / ".hermes"
    home.mkdir(parents=True)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(home))
    for key in (
        "HERMES_ANTIGRAVITY_CLIENT_ID",
        "HERMES_ANTIGRAVITY_CLIENT_SECRET",
        "HERMES_ANTIGRAVITY_CLI_PATH",
        "HERMES_ANTIGRAVITY_PROJECT_ID",
        "GOOGLE_CLOUD_PROJECT",
        "GOOGLE_CLOUD_PROJECT_ID",
        "LOCALAPPDATA",
        "APPDATA",
        "ProgramFiles",
        "ProgramFiles(x86)",
    ):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr("shutil.which", lambda _: None)
    try:
        from agent import antigravity_oauth

        antigravity_oauth._discovered_creds_cache.clear()
    except Exception:
        pass
    return home


class TestAntigravityCredentials:
    def test_save_load_uses_separate_file_and_0600_permissions(self):
        from agent.antigravity_oauth import (
            AntigravityCredentials,
            _credentials_path,
            load_credentials,
            save_credentials,
        )

        save_credentials(AntigravityCredentials(
            access_token="at",
            refresh_token="rt",
            expires_ms=int((time.time() + 3600) * 1000),
            email="user@example.com",
            project_id="proj-123",
        ))

        assert _credentials_path().name == "antigravity_oauth.json"
        loaded = load_credentials()
        assert loaded is not None
        assert loaded.refresh_token == "rt"
        assert loaded.project_id == "proj-123"
        if os.name != "nt":
            assert stat.S_IMODE(_credentials_path().stat().st_mode) == 0o600

    def test_env_override_client_id(self, monkeypatch):
        from agent.antigravity_oauth import _get_client_id

        monkeypatch.setenv("HERMES_ANTIGRAVITY_CLIENT_ID", "custom.apps.googleusercontent.com")
        assert _get_client_id() == "custom.apps.googleusercontent.com"

    def test_env_override_client_secret(self, monkeypatch):
        from agent.antigravity_oauth import _get_client_secret

        monkeypatch.setenv("HERMES_ANTIGRAVITY_CLIENT_SECRET", "custom-secret")
        assert _get_client_secret() == "custom-secret"

    def test_discovers_client_credentials_from_configured_agy_path(self, tmp_path, monkeypatch):
        from agent import antigravity_oauth

        fake_client_id = (
            "1071006060591-"
            + "fakefakefakefakefakefakefake"
            + ".apps.google"
            + "usercontent.com"
        )
        fake_client_secret = "GOC" + "SPX-" + "fake-secret-value-placeholde"
        fake_agy = tmp_path / "agy.exe"
        fake_agy.write_text(
            f'oauthClientId="{fake_client_id}";\n'
            f'oauthClientSecret="{fake_client_secret}";\n',
            encoding="utf-8",
        )
        monkeypatch.setenv("HERMES_ANTIGRAVITY_CLI_PATH", str(fake_agy))
        antigravity_oauth._discovered_creds_cache.clear()

        assert antigravity_oauth._get_client_id().startswith("1071006060591-")
        assert antigravity_oauth._get_client_secret() == fake_client_secret

    def test_missing_client_credentials_raise_with_setup_hint(self):
        from agent.antigravity_oauth import AntigravityOAuthError, _require_client_id

        with pytest.raises(AntigravityOAuthError) as exc_info:
            _require_client_id()
        assert exc_info.value.code == "antigravity_oauth_client_id_missing"
        assert "HERMES_ANTIGRAVITY_CLI_PATH" in str(exc_info.value)

    def test_pkce_challenge_is_s256(self):
        import base64
        import hashlib

        from agent.antigravity_oauth import _generate_pkce_pair

        verifier, challenge = _generate_pkce_pair()
        expected = base64.urlsafe_b64encode(
            hashlib.sha256(verifier.encode("ascii")).digest()
        ).rstrip(b"=").decode("ascii")
        assert challenge == expected
        assert 43 <= len(verifier) <= 128

    def test_exchange_code_posts_pkce_payload(self, monkeypatch):
        from agent import antigravity_oauth

        captured = {}

        def fake_post(url, data, timeout):
            captured.update({"url": url, "data": data, "timeout": timeout})
            return {"access_token": "at"}

        monkeypatch.setattr(antigravity_oauth, "_post_form", fake_post)
        monkeypatch.setenv("HERMES_ANTIGRAVITY_CLIENT_ID", "client.apps.googleusercontent.com")
        monkeypatch.setenv("HERMES_ANTIGRAVITY_CLIENT_SECRET", "secret")

        assert antigravity_oauth.exchange_code("code", "verifier", "http://localhost/cb") == {
            "access_token": "at"
        }
        assert captured["url"] == antigravity_oauth.TOKEN_ENDPOINT
        assert captured["data"]["grant_type"] == "authorization_code"
        assert captured["data"]["code_verifier"] == "verifier"
        assert captured["data"]["redirect_uri"] == "http://localhost/cb"
        assert captured["data"]["client_id"] == "client.apps.googleusercontent.com"
        assert captured["data"]["client_secret"] == "secret"

    def test_refresh_tries_discovered_client_secret_candidates(self, monkeypatch):
        from agent import antigravity_oauth
        from agent.antigravity_oauth import AntigravityOAuthError

        calls = []
        monkeypatch.setattr(
            antigravity_oauth,
            "_iter_client_credential_candidates",
            lambda: [
                ("client.apps.googleusercontent.com", "wrong-secret"),
                ("client.apps.googleusercontent.com", "right-secret"),
            ],
        )

        def fake_post(url, data, timeout):
            calls.append(data["client_secret"])
            if data["client_secret"] == "wrong-secret":
                raise AntigravityOAuthError(
                    "invalid client",
                    code="antigravity_oauth_invalid_client",
                )
            return {"access_token": "new-token", "expires_in": 3600}

        monkeypatch.setattr(antigravity_oauth, "_post_form", fake_post)

        assert antigravity_oauth.refresh_access_token("refresh-token")["access_token"] == "new-token"
        assert calls == ["wrong-secret", "right-secret"]

    def test_invalid_grant_refresh_clears_credentials(self, monkeypatch):
        from agent import antigravity_oauth
        from agent.antigravity_oauth import (
            AntigravityCredentials,
            AntigravityOAuthError,
            load_credentials,
            save_credentials,
        )

        save_credentials(AntigravityCredentials(
            access_token="expired",
            refresh_token="rt",
            expires_ms=int((time.time() - 3600) * 1000),
        ))

        def invalid_grant(_refresh_token):
            raise AntigravityOAuthError("revoked", code="antigravity_oauth_invalid_grant")

        monkeypatch.setattr(antigravity_oauth, "refresh_access_token", invalid_grant)
        with pytest.raises(AntigravityOAuthError, match="revoked"):
            antigravity_oauth.get_valid_access_token()
        assert load_credentials() is None

    def test_callback_handler_captures_code_on_handler_class(self):
        from agent.antigravity_oauth import CALLBACK_PATH, _OAuthCallbackHandler

        handler_cls = type("TestAntigravityOAuthCallbackHandler", (_OAuthCallbackHandler,), {})
        handler_cls.expected_state = "state-123"
        handler_cls.captured_code = None
        handler_cls.captured_error = None
        handler_cls.ready = threading.Event()

        handler = handler_cls.__new__(handler_cls)
        handler.path = CALLBACK_PATH + "?" + urllib.parse.urlencode({
            "state": "state-123",
            "code": "auth-code",
        })
        handler.wfile = BytesIO()
        responses = []
        headers = []
        handler.send_response = lambda code: responses.append(code)
        handler.send_header = lambda key, value: headers.append((key, value))
        handler.end_headers = lambda: None

        handler.do_GET()

        assert responses == [200]
        assert handler_cls.captured_code == "auth-code"
        assert handler_cls.captured_error is None
        assert handler_cls.ready.is_set()
        assert "captured_code" not in handler.__dict__


class TestAntigravityModelCatalog:
    def test_parse_agent_model_ids_prefers_recommended_group(self):
        from agent.antigravity_code_assist import parse_agent_model_ids

        payload = {
            "defaultAgentModelId": "gemini-3-flash-agent",
            "agentModelSorts": [
                {
                    "displayName": "Experimental",
                    "modelIds": ["tab_flash_lite_preview", "chat_23310"],
                },
                {
                    "displayName": "Recommended",
                    "modelIds": [
                        "gemini-3-flash-agent",
                        "gemini-3.5-flash-low",
                        "gemini-3.1-pro-high",
                        "gemini-pro-agent",
                        "claude-sonnet-4-6",
                    ],
                },
            ],
            "models": [{"id": "gpt-oss-120b-medium"}],
        }

        assert parse_agent_model_ids(payload) == [
            "gemini-3-flash-agent",
            "gemini-3.5-flash-low",
            "gemini-pro-agent",
            "claude-sonnet-4-6",
        ]

    def test_headers_include_antigravity_metadata(self):
        from agent.antigravity_code_assist import build_headers

        headers = build_headers("tok")
        assert headers["Authorization"] == "Bearer tok"
        assert headers["User-Agent"].startswith("antigravity/")
        assert headers["X-Goog-Api-Client"] == "google-cloud-sdk vscode_cloudshelleditor/0.1"
        metadata = json.loads(headers["Client-Metadata"])
        assert metadata["ideType"] == "ANTIGRAVITY"
        assert metadata["platform"] == "PLATFORM_UNSPECIFIED"


class TestAntigravityClient:
    def test_client_exposes_openai_interface(self):
        from agent.antigravity_cloudcode_adapter import AntigravityCloudCodeClient

        client = AntigravityCloudCodeClient(api_key="dummy")
        try:
            assert hasattr(client, "chat")
            assert hasattr(client.chat, "completions")
            assert callable(client.chat.completions.create)
        finally:
            client.close()

    def test_create_uses_antigravity_endpoint_and_headers(self, monkeypatch):
        from agent import antigravity_oauth
        from agent.antigravity_cloudcode_adapter import AntigravityCloudCodeClient
        from agent.antigravity_code_assist import ANTIGRAVITY_CODE_ASSIST_ENDPOINT

        monkeypatch.setattr(antigravity_oauth, "get_valid_access_token", lambda: "live-token")

        class _Response:
            status_code = 200

            def json(self):
                return {
                    "response": {
                        "candidates": [{
                            "content": {"parts": [{"text": "ok"}]},
                            "finishReason": "STOP",
                        }]
                    }
                }

        class _Http:
            def __init__(self):
                self.calls = []

            def post(self, url, json=None, headers=None):
                self.calls.append((url, json, headers))
                return _Response()

            def close(self):
                pass

        client = AntigravityCloudCodeClient(project_id="proj-123")
        client._http = _Http()
        try:
            result = client.chat.completions.create(
                model="gemini-3-flash-agent",
                messages=[{"role": "user", "content": "hi"}],
            )
        finally:
            client.close()

        assert result.choices[0].message.content == "ok"
        url, body, headers = client._http.calls[0]
        assert url == f"{ANTIGRAVITY_CODE_ASSIST_ENDPOINT}/v1internal:generateContent"
        assert body["project"] == "proj-123"
        assert body["model"] == "gemini-3-flash-agent"
        assert headers["Authorization"] == "Bearer live-token"
        assert json.loads(headers["Client-Metadata"])["ideType"] == "ANTIGRAVITY"


class TestAntigravityRegistration:
    def test_registry_entry_and_aliases(self):
        from hermes_cli.auth import PROVIDER_REGISTRY, resolve_provider

        assert "google-antigravity" in PROVIDER_REGISTRY
        assert PROVIDER_REGISTRY["google-antigravity"].auth_type == "oauth_external"
        assert resolve_provider("antigravity") == "google-antigravity"
        assert resolve_provider("antigravity-oauth") == "google-antigravity"
        assert resolve_provider("google-antigravity-oauth") == "google-antigravity"
        assert resolve_provider("agy") == "google-antigravity"

    def test_runtime_provider_raises_when_not_logged_in(self):
        from hermes_cli.auth import AuthError
        from hermes_cli.runtime_provider import resolve_runtime_provider

        with pytest.raises(AuthError) as exc_info:
            resolve_runtime_provider(requested="google-antigravity")
        assert exc_info.value.code == "antigravity_oauth_not_logged_in"

    def test_runtime_provider_returns_correct_shape_when_logged_in(self):
        from agent.antigravity_oauth import AntigravityCredentials, save_credentials
        from hermes_cli.runtime_provider import resolve_runtime_provider

        save_credentials(AntigravityCredentials(
            access_token="live-tok",
            refresh_token="rt",
            expires_ms=int((time.time() + 3600) * 1000),
            project_id="my-proj",
            email="t@e.com",
        ))

        result = resolve_runtime_provider(requested="google-antigravity")
        assert result["provider"] == "google-antigravity"
        assert result["api_mode"] == "chat_completions"
        assert result["api_key"] == "live-tok"
        assert result["base_url"] == "antigravity-pa://google"
        assert result["project_id"] == "my-proj"
        assert result["email"] == "t@e.com"

    def test_provider_model_ids_uses_live_antigravity_catalog(self, monkeypatch):
        from hermes_cli import models

        monkeypatch.setattr(
            models,
            "_fetch_antigravity_models",
            lambda force_refresh=False: ["gemini-3-flash-agent", "claude-sonnet-4-6"],
        )

        assert models.provider_model_ids("agy") == [
            "gemini-3-flash-agent",
            "claude-sonnet-4-6",
        ]

    def test_oauth_capable_set_includes_antigravity(self):
        from hermes_cli.auth_commands import _OAUTH_CAPABLE_PROVIDERS

        assert "google-antigravity" in _OAUTH_CAPABLE_PROVIDERS
