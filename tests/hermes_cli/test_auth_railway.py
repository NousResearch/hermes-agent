"""Railway OAuth device-code auth tests."""

from __future__ import annotations

import base64
import json
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace


def _write_auth_store(hermes_home, payload: dict) -> None:
    hermes_home.mkdir(parents=True, exist_ok=True)
    (hermes_home / "auth.json").write_text(json.dumps(payload, indent=2))


def _jwt_with_email(email: str) -> str:
    header = base64.urlsafe_b64encode(b'{"alg":"RS256","typ":"JWT"}').rstrip(b"=").decode()
    payload = base64.urlsafe_b64encode(json.dumps({"email": email}).encode()).rstrip(b"=").decode()
    return f"{header}.{payload}.signature"


def test_railway_oauth_client_credentials_resolve_from_hermes_dotenv(tmp_path, monkeypatch):
    hermes_home = tmp_path / "hermes"
    hermes_home.mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.delenv("RAILWAY_OAUTH_CLIENT_ID", raising=False)
    monkeypatch.delenv("RAILWAY_OAUTH_CLIENT_SECRET", raising=False)
    (hermes_home / ".env").write_text(
        "RAILWAY_OAUTH_CLIENT_ID=client-from-dotenv\n"
        "RAILWAY_OAUTH_CLIENT_SECRET=secret-from-dotenv\n"
    )

    from hermes_cli.auth import _railway_client_id, _railway_client_secret

    assert _railway_client_id() == "client-from-dotenv"
    assert _railway_client_secret() == "secret-from-dotenv"


def test_auth_add_railway_oauth_persists_refreshable_provider_state(tmp_path, monkeypatch):
    hermes_home = tmp_path / "hermes"
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    _write_auth_store(hermes_home, {"version": 1, "providers": {}, "active_provider": "openrouter"})
    token = _jwt_with_email("railway@example.com")

    def fake_login(**kwargs):
        assert kwargs["client_id"] == "client-123"
        assert kwargs["client_secret"] == "secret-456"
        assert "offline_access" in kwargs["scope"]
        assert kwargs["open_browser"] is False
        return {
            "issuer": "https://backboard.railway.com",
            "api_base_url": "https://backboard.railway.com/graphql/v2",
            "client_id": "client-123",
            "client_secret": "secret-456",
            "scope": "openid offline_access project:member ssh_keys",
            "token_type": "Bearer",
            "access_token": token,
            "refresh_token": "refresh-token",
            "obtained_at": "2026-05-07T00:00:00+00:00",
            "expires_at": "2026-05-07T01:00:00+00:00",
            "expires_in": 3600,
            "tls": {"insecure": False, "ca_bundle": None},
        }

    monkeypatch.setattr("hermes_cli.auth._railway_device_code_login", fake_login)

    from hermes_cli.auth_commands import auth_add_command

    auth_add_command(SimpleNamespace(
        provider="railway",
        auth_type="oauth",
        api_key=None,
        label="railway-dev",
        portal_url=None,
        inference_url=None,
        client_id="client-123",
        client_secret="secret-456",
        scope=None,
        no_browser=True,
        timeout=None,
        insecure=False,
        ca_bundle=None,
    ))

    payload = json.loads((hermes_home / "auth.json").read_text())
    assert payload["active_provider"] == "openrouter"

    state = payload["providers"]["railway"]
    assert state["access_token"] == token
    assert state["refresh_token"] == "refresh-token"
    assert state["client_id"] == "client-123"
    assert state["client_secret"] == "secret-456"
    assert state["label"] == "railway-dev"

    entries = payload["credential_pool"]["railway"]
    entry = next(item for item in entries if item["source"] == "device_code")
    assert entry["auth_type"] == "oauth"
    assert entry["label"] == "railway-dev"
    assert entry["refresh_token"] == "refresh-token"
    assert entry["client_secret"] == "secret-456"
    assert entry["api_base_url"] == "https://backboard.railway.com/graphql/v2"


def test_request_railway_device_code_uses_oidc_device_endpoint_with_refresh_scope():
    from hermes_cli.auth import _request_railway_device_code

    calls = []

    class _Response:
        status_code = 200

        def raise_for_status(self):
            return None

        def json(self):
            return {
                "device_code": "device-123",
                "user_code": "ABCD-EFGH",
                "verification_uri": "https://railway.com/device",
                "verification_uri_complete": "https://railway.com/device?user_code=ABCD-EFGH",
                "expires_in": 600,
                "interval": 5,
            }

    class _Client:
        def post(self, url, data=None, headers=None):
            calls.append((url, data, headers))
            return _Response()

    data = _request_railway_device_code(
        _Client(),
        issuer="https://backboard.railway.com",
        client_id="client-123",
        client_secret="secret-456",
        scope="openid offline_access project:member ssh_keys",
    )

    assert data["device_code"] == "device-123"
    url, form, headers = calls[0]
    assert url == "https://backboard.railway.com/oauth/device/auth"
    assert form["client_id"] == "client-123"
    assert form["client_secret"] == "secret-456"
    assert "offline_access" in form["scope"]
    assert form["prompt"] == "consent"
    assert headers["Content-Type"] == "application/x-www-form-urlencoded"


def test_poll_railway_device_token_uses_client_secret_post(monkeypatch):
    from hermes_cli.auth import _poll_railway_device_token

    monkeypatch.setattr("hermes_cli.auth.time.sleep", lambda _seconds: None)
    calls = []

    class _Response:
        status_code = 200

        def json(self):
            return {
                "access_token": "access",
                "refresh_token": "refresh",
                "expires_in": 3600,
            }

    class _Client:
        def post(self, url, data=None, headers=None):
            calls.append((url, data, headers))
            return _Response()

    payload = _poll_railway_device_token(
        _Client(),
        issuer="https://backboard.railway.com",
        client_id="client-123",
        client_secret="secret-456",
        device_code="device-123",
        expires_in=600,
        poll_interval=1,
    )

    assert payload["access_token"] == "access"
    url, form, headers = calls[0]
    assert url == "https://backboard.railway.com/oauth/token"
    assert form["grant_type"] == "urn:ietf:params:oauth:grant-type:device_code"
    assert form["client_id"] == "client-123"
    assert form["client_secret"] == "secret-456"
    assert form["device_code"] == "device-123"
    assert headers["Content-Type"] == "application/x-www-form-urlencoded"


def test_refresh_railway_access_token_uses_client_secret_post():
    from hermes_cli.auth import _refresh_railway_access_token

    calls = []

    class _Response:
        status_code = 200

        def json(self):
            return {
                "access_token": "new-access",
                "refresh_token": "new-refresh",
                "expires_in": 3600,
            }

    class _Client:
        def post(self, url, data=None, headers=None):
            calls.append((url, data, headers))
            return _Response()

    payload = _refresh_railway_access_token(
        _Client(),
        issuer="https://backboard.railway.com",
        client_id="client-123",
        client_secret="secret-456",
        refresh_token="old-refresh",
    )

    assert payload["access_token"] == "new-access"
    url, form, headers = calls[0]
    assert url == "https://backboard.railway.com/oauth/token"
    assert form["grant_type"] == "refresh_token"
    assert form["client_id"] == "client-123"
    assert form["client_secret"] == "secret-456"
    assert form["refresh_token"] == "old-refresh"
    assert headers["Content-Type"] == "application/x-www-form-urlencoded"


def test_resolve_railway_runtime_credentials_refreshes_and_rotates_refresh_token(tmp_path, monkeypatch):
    hermes_home = tmp_path / "hermes"
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    expired = datetime.now(timezone.utc) - timedelta(minutes=5)
    _write_auth_store(
        hermes_home,
        {
            "version": 1,
            "providers": {
                "railway": {
                    "issuer": "https://backboard.railway.com",
                    "api_base_url": "https://backboard.railway.com/graphql/v2",
                    "client_id": "client-123",
                    "client_secret": "secret-456",
                    "scope": "openid offline_access project:member ssh_keys",
                    "token_type": "Bearer",
                    "access_token": "old-access",
                    "refresh_token": "old-refresh",
                    "expires_at": expired.isoformat(),
                }
            },
            "credential_pool": {},
            "active_provider": "openrouter",
        },
    )

    def fake_refresh(client, *, issuer, client_id, client_secret, refresh_token):
        assert issuer == "https://backboard.railway.com"
        assert client_id == "client-123"
        assert client_secret == "secret-456"
        assert refresh_token == "old-refresh"
        return {
            "access_token": "new-access",
            "refresh_token": "new-refresh",
            "expires_in": 3600,
            "token_type": "Bearer",
            "scope": "openid offline_access project:member ssh_keys",
        }

    monkeypatch.setattr("hermes_cli.auth._refresh_railway_access_token", fake_refresh)

    from hermes_cli.auth import resolve_railway_runtime_credentials

    creds = resolve_railway_runtime_credentials(timeout_seconds=1)

    assert creds["provider"] == "railway"
    assert creds["api_key"] == "new-access"
    assert creds["access_token"] == "new-access"
    assert creds["refresh_token"] == "new-refresh"
    assert creds["base_url"] == "https://backboard.railway.com/graphql/v2"

    saved = json.loads((hermes_home / "auth.json").read_text())
    state = saved["providers"]["railway"]
    assert state["access_token"] == "new-access"
    assert state["refresh_token"] == "new-refresh"
    assert state["expires_at"] != expired.isoformat()
    assert saved["active_provider"] == "openrouter"

def test_credential_pool_refreshes_railway_device_code_entry(tmp_path, monkeypatch):
    hermes_home = tmp_path / "hermes"
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    expired = datetime.now(timezone.utc) - timedelta(minutes=5)
    future = datetime.now(timezone.utc) + timedelta(hours=1)
    _write_auth_store(
        hermes_home,
        {
            "version": 1,
            "providers": {
                "railway": {
                    "issuer": "https://backboard.railway.com",
                    "api_base_url": "https://backboard.railway.com/graphql/v2",
                    "client_id": "client-123",
                    "client_secret": "secret-456",
                    "scope": "openid offline_access project:member ssh_keys",
                    "token_type": "Bearer",
                    "access_token": "old-access",
                    "refresh_token": "old-refresh",
                    "expires_at": expired.isoformat(),
                    "tls": {"insecure": False, "ca_bundle": None},
                }
            },
            "credential_pool": {},
            "active_provider": "openrouter",
        },
    )

    def fake_refresh(state, **kwargs):
        assert kwargs["force_refresh"] is True
        assert state["client_id"] == "client-123"
        assert state["client_secret"] == "secret-456"
        updated = dict(state)
        updated.update({
            "access_token": "new-access",
            "refresh_token": "new-refresh",
            "expires_at": future.isoformat(),
        })
        return updated

    monkeypatch.setattr("hermes_cli.auth.refresh_railway_oauth_from_state", fake_refresh)

    from agent.credential_pool import load_pool

    pool = load_pool("railway")
    selected = pool.select()

    assert selected is not None
    assert selected.access_token == "new-access"
    assert selected.refresh_token == "new-refresh"
    assert selected.expires_at == future.isoformat()

    saved = json.loads((hermes_home / "auth.json").read_text())
    assert saved["providers"]["railway"]["access_token"] == "new-access"
    assert saved["providers"]["railway"]["refresh_token"] == "new-refresh"
    assert saved["credential_pool"]["railway"][0]["access_token"] == "new-access"

def test_sync_railway_shared_variables_from_env_upserts_all_dotenv_keys(tmp_path, monkeypatch):
    hermes_home = tmp_path / "hermes"
    hermes_home.mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    (hermes_home / ".env").write_text(
        "OPENROUTER_API_KEY=or-key\n"
        "ANTHROPIC_API_KEY='ant-key'\n"
        "CUSTOM_VALUE=foo=bar\n"
        "EMPTY_VALUE=\n"
        "# ignored comment\n"
    )

    calls = []

    class _Response:
        status_code = 200

        def json(self):
            return {"data": {"variableCollectionUpsert": True}}

    class _Client:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def post(self, url, json=None, headers=None):
            calls.append((url, json, headers))
            return _Response()

    def client_factory(**kwargs):
        assert kwargs["timeout"] is not None
        return _Client()

    from hermes_cli.auth import sync_railway_shared_variables_from_env

    result = sync_railway_shared_variables_from_env(
        project_id="project-123",
        environment_id="environment-456",
        access_token="access-token",
        client_factory=client_factory,
    )

    assert result["count"] == 4
    assert result["project_id"] == "project-123"
    assert result["environment_id"] == "environment-456"
    url, payload, headers = calls[0]
    assert url == "https://backboard.railway.com/graphql/v2"
    assert headers["Authorization"] == "Bearer access-token"
    input_payload = payload["variables"]["input"]
    assert input_payload["projectId"] == "project-123"
    assert input_payload["environmentId"] == "environment-456"
    assert "serviceId" not in input_payload
    assert input_payload["variables"] == {
        "OPENROUTER_API_KEY": "or-key",
        "ANTHROPIC_API_KEY": "ant-key",
        "CUSTOM_VALUE": "foo=bar",
        "EMPTY_VALUE": "",
    }


def test_sync_railway_shared_variables_from_env_resolves_oauth_token_and_ids_from_dotenv(tmp_path, monkeypatch):
    hermes_home = tmp_path / "hermes"
    hermes_home.mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    (hermes_home / ".env").write_text(
        "RAILWAY_PROJECT_ID=project-from-env\n"
        "RAILWAY_ENVIRONMENT_ID=environment-from-env\n"
        "OPENROUTER_API_KEY=or-key\n"
    )
    monkeypatch.setattr(
        "hermes_cli.auth.resolve_railway_runtime_credentials",
        lambda **kwargs: {"access_token": "resolved-access", "api_base_url": "https://backboard.railway.com/graphql/v2"},
    )
    calls = []

    class _Response:
        status_code = 200

        def json(self):
            return {"data": {"variableCollectionUpsert": True}}

    class _Client:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def post(self, url, json=None, headers=None):
            calls.append((url, json, headers))
            return _Response()

    from hermes_cli.auth import sync_railway_shared_variables_from_env

    result = sync_railway_shared_variables_from_env(client_factory=lambda **kwargs: _Client())

    assert result["count"] == 3
    input_payload = calls[0][1]["variables"]["input"]
    assert input_payload["projectId"] == "project-from-env"
    assert input_payload["environmentId"] == "environment-from-env"
    assert calls[0][2]["Authorization"] == "Bearer resolved-access"


def test_auth_remove_railway_device_code_clears_auth_store_and_blocks_reseed(tmp_path, monkeypatch):
    hermes_home = tmp_path / "hermes"
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    _write_auth_store(
        hermes_home,
        {
            "version": 1,
            "providers": {
                "railway": {
                    "issuer": "https://backboard.railway.com",
                    "api_base_url": "https://backboard.railway.com/graphql/v2",
                    "client_id": "client-123",
                    "scope": "openid offline_access project:member ssh_keys",
                    "token_type": "Bearer",
                    "access_token": "access-token",
                    "refresh_token": "refresh-token",
                    "expires_at": (datetime.now(timezone.utc) + timedelta(hours=1)).isoformat(),
                }
            },
            "credential_pool": {},
            "active_provider": "openrouter",
        },
    )

    from agent.credential_pool import load_pool
    from hermes_cli.auth_commands import auth_remove_command

    assert len(load_pool("railway").entries()) == 1

    auth_remove_command(SimpleNamespace(provider="railway", target="1"))

    saved = json.loads((hermes_home / "auth.json").read_text())
    assert "railway" not in saved.get("providers", {})
    assert "device_code" in saved.get("suppressed_sources", {}).get("railway", [])
    assert load_pool("railway").entries() == []


def test_railway_env_keys_are_known_to_env_sanitizer():
    from hermes_cli.config import _sanitize_env_lines

    sanitized = _sanitize_env_lines([
        "RAILWAY_OAUTH_CLIENT_ID=client-123RAILWAY_OAUTH_CLIENT_SECRET=secret-456\n",
        "RAILWAY_PROJECT_ID=project-123RAILWAY_ENVIRONMENT_ID=environment-456\n",
    ])

    assert sanitized == [
        "RAILWAY_OAUTH_CLIENT_ID=client-123\n",
        "RAILWAY_OAUTH_CLIENT_SECRET=secret-456\n",
        "RAILWAY_PROJECT_ID=project-123\n",
        "RAILWAY_ENVIRONMENT_ID=environment-456\n",
    ]


def test_auth_railway_sync_env_command_wires_arguments(monkeypatch):
    calls = []

    def fake_sync(**kwargs):
        calls.append(kwargs)
        return {"count": 2, "keys": ["A", "B"], "project_id": "project-123", "environment_id": "environment-456"}

    monkeypatch.setattr("hermes_cli.auth.sync_railway_shared_variables_from_env", fake_sync)

    from hermes_cli.auth_commands import auth_railway_command

    auth_railway_command(SimpleNamespace(
        railway_action="sync-env",
        project_id="project-123",
        environment_id="environment-456",
        env_path="/tmp/hermes.env",
        timeout=9.0,
    ))

    assert calls == [{
        "project_id": "project-123",
        "environment_id": "environment-456",
        "env_path": "/tmp/hermes.env",
        "timeout_seconds": 9.0,
    }]
