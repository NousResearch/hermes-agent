"""Regression tests for Microsoft Graph delegated OAuth setup."""

import importlib.util
import json
import stat
import sys
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import pytest


SCRIPT_DIR = (
    Path(__file__).resolve().parents[2]
    / "skills/productivity/microsoft-graph-mail/scripts"
)
AUTH_PATH = SCRIPT_DIR / "microsoft_auth.py"
HELPER_PATH = SCRIPT_DIR / "_hermes_home.py"


@pytest.fixture
def auth_module(monkeypatch, tmp_path):
    monkeypatch.syspath_prepend(str(SCRIPT_DIR))
    spec = importlib.util.spec_from_file_location("microsoft_auth_test", AUTH_PATH)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    monkeypatch.setattr(module, "CLIENT_CONFIG_PATH", tmp_path / "microsoft_graph_client.json")
    monkeypatch.setattr(module, "TOKEN_PATH", tmp_path / "microsoft_graph_token.json")
    monkeypatch.setattr(module, "PENDING_AUTH_PATH", tmp_path / "microsoft_graph_oauth_pending.json")
    return module


def test_configure_client_stores_public_app_metadata_only(auth_module):
    auth_module.configure_client(
        "client-id",
        tenant="tenant-id",
        redirect_uri="http://localhost:1",
    )

    saved = json.loads(auth_module.CLIENT_CONFIG_PATH.read_text())
    assert saved == {
        "client_id": "client-id",
        "tenant": "tenant-id",
        "redirect_uri": "http://localhost:1",
        "auth_flow": "authorization_code_pkce",
    }
    assert "client_secret" not in saved


def test_get_auth_url_persists_state_and_pkce(auth_module, monkeypatch, capsys):
    auth_module.configure_client("client-id", tenant="organizations")
    monkeypatch.setattr(auth_module, "_new_state", lambda: "saved-state")
    monkeypatch.setattr(auth_module, "_new_code_verifier", lambda: "saved-verifier")

    auth_module.get_auth_url()

    url = capsys.readouterr().out.strip().splitlines()[-1]
    parsed = urlparse(url)
    params = parse_qs(parsed.query)
    assert parsed.scheme == "https"
    assert parsed.netloc == "login.microsoftonline.com"
    assert parsed.path == "/organizations/oauth2/v2.0/authorize"
    assert params["client_id"] == ["client-id"]
    assert params["response_type"] == ["code"]
    assert params["redirect_uri"] == ["http://localhost:1"]
    assert params["state"] == ["saved-state"]
    assert params["code_challenge_method"] == ["S256"]
    assert params["code_challenge"] == [auth_module._code_challenge("saved-verifier")]
    assert params["scope"] == ["offline_access User.Read Mail.Read"]

    pending = json.loads(auth_module.PENDING_AUTH_PATH.read_text())
    assert pending["state"] == "saved-state"
    assert pending["code_verifier"] == "saved-verifier"
    assert pending["client_id"] == "client-id"
    assert stat.S_IMODE(auth_module.PENDING_AUTH_PATH.stat().st_mode) == 0o600


def test_get_auth_url_encodes_tenant_path_segment(auth_module, monkeypatch, capsys):
    auth_module.configure_client("client-id", tenant="tenant/with?bad\nline")
    capsys.readouterr()
    monkeypatch.setattr(auth_module, "_new_state", lambda: "saved-state")
    monkeypatch.setattr(auth_module, "_new_code_verifier", lambda: "saved-verifier")

    auth_module.get_auth_url()

    url = capsys.readouterr().out.strip()
    parsed = urlparse(url)
    assert "\n" not in url
    assert parsed.path == "/tenant%2Fwith%3Fbad%0Aline/oauth2/v2.0/authorize"


def test_request_token_encodes_tenant_path_segment(auth_module, monkeypatch):
    captured = {}

    class FakeResponse:
        status_code = 200

        def json(self):
            return {"access_token": "access-token"}

    def fake_post(url, data, headers, timeout):
        captured["url"] = url
        captured["data"] = data
        captured["headers"] = headers
        captured["timeout"] = timeout
        return FakeResponse()

    monkeypatch.setattr(auth_module, "httpx", type("FakeHttpx", (), {"post": staticmethod(fake_post)}))

    response = auth_module._request_token("tenant/with?bad\nline", {"grant_type": "authorization_code"})

    assert captured["url"] == "https://login.microsoftonline.com/tenant%2Fwith%3Fbad%0Aline/oauth2/v2.0/token"
    assert captured["headers"] == {"Content-Type": "application/x-www-form-urlencoded"}
    assert captured["timeout"] == 30
    assert response == {"access_token": "access-token"}


def test_exchange_auth_code_reuses_pending_pkce_without_secret(auth_module, monkeypatch):
    auth_module.configure_client("client-id", tenant="tenant-id")
    auth_module.PENDING_AUTH_PATH.write_text(
        json.dumps({"state": "saved-state", "code_verifier": "saved-verifier", "client_id": "client-id", "tenant": "tenant-id", "redirect_uri": "http://localhost:1"})
    )
    captured = {}

    def fake_request_token(tenant, data):
        captured["tenant"] = tenant
        captured["data"] = data
        return {
            "access_token": "access-token",
            "refresh_token": "refresh-token",
            "expires_in": 3600,
            "scope": "Mail.Read User.Read",
        }

    monkeypatch.setattr(auth_module, "_request_token", fake_request_token)

    auth_module.exchange_auth_code("http://localhost:1/?code=auth-code&state=saved-state")

    assert captured["tenant"] == "tenant-id"
    assert captured["data"]["grant_type"] == "authorization_code"
    assert captured["data"]["code"] == "auth-code"
    assert captured["data"]["code_verifier"] == "saved-verifier"
    assert captured["data"]["client_id"] == "client-id"
    assert "client_secret" not in captured["data"]

    saved = json.loads(auth_module.TOKEN_PATH.read_text())
    assert saved["access_token"] == "access-token"
    assert saved["refresh_token"] == "refresh-token"
    assert saved["client_id"] == "client-id"
    assert saved["tenant"] == "tenant-id"
    assert saved["redirect_uri"] == "http://localhost:1"
    assert saved["expires_at"] > 0
    assert not auth_module.PENDING_AUTH_PATH.exists()


def test_exchange_auth_code_rejects_state_mismatch(auth_module, capsys):
    auth_module.configure_client("client-id")
    auth_module.PENDING_AUTH_PATH.write_text(
        json.dumps({"state": "saved-state", "code_verifier": "saved-verifier"})
    )

    with pytest.raises(SystemExit):
        auth_module.exchange_auth_code("http://localhost:1/?code=auth-code&state=wrong")

    assert "state mismatch" in capsys.readouterr().out.lower()
    assert not auth_module.TOKEN_PATH.exists()


def test_exchange_auth_code_rejects_missing_state_from_callback_url(auth_module, capsys):
    auth_module.configure_client("client-id")
    auth_module.PENDING_AUTH_PATH.write_text(
        json.dumps({"state": "saved-state", "code_verifier": "saved-verifier", "client_id": "client-id"})
    )

    with pytest.raises(SystemExit):
        auth_module.exchange_auth_code("http://localhost:1/?code=auth-code")

    assert "full microsoft redirect url" in capsys.readouterr().out.lower()
    assert not auth_module.TOKEN_PATH.exists()


def test_exchange_auth_code_rejects_missing_mail_scope(auth_module, monkeypatch, capsys):
    auth_module.configure_client("client-id")
    auth_module.PENDING_AUTH_PATH.write_text(
        json.dumps({"state": "saved-state", "code_verifier": "saved-verifier", "client_id": "client-id"})
    )
    monkeypatch.setattr(
        auth_module,
        "_request_token",
        lambda tenant, data: {
            "access_token": "access-token",
            "refresh_token": "refresh-token",
            "expires_in": 3600,
            "scope": "User.Read",
        },
    )

    with pytest.raises(SystemExit):
        auth_module.exchange_auth_code("http://localhost:1/?code=auth-code&state=saved-state")

    out = capsys.readouterr().out
    assert "Mail.Read" in out
    assert not auth_module.TOKEN_PATH.exists()
    assert auth_module.PENDING_AUTH_PATH.exists()


def test_check_auth_refreshes_expired_token(auth_module, monkeypatch):
    auth_module.configure_client("client-id", tenant="tenant-id")
    auth_module.TOKEN_PATH.write_text(
        json.dumps(
            {
                "access_token": "old-token",
                "refresh_token": "refresh-token",
                "expires_at": 1,
                "scope": "Mail.Read User.Read",
                "tenant": "tenant-id",
            }
        )
    )

    def fake_request_token(tenant, data):
        assert tenant == "tenant-id"
        assert data["grant_type"] == "refresh_token"
        assert data["refresh_token"] == "refresh-token"
        assert "client_secret" not in data
        return {
            "access_token": "new-token",
            "expires_in": 3600,
            "scope": "Mail.Read User.Read",
        }

    monkeypatch.setattr(auth_module, "_request_token", fake_request_token)

    assert auth_module.check_auth() is True
    saved = json.loads(auth_module.TOKEN_PATH.read_text())
    assert saved["access_token"] == "new-token"
    assert saved["refresh_token"] == "refresh-token"


def test_exchange_auth_code_writes_token_with_owner_only_permissions(auth_module, monkeypatch):
    auth_module.configure_client("client-id", tenant="tenant-id")
    auth_module.PENDING_AUTH_PATH.write_text(
        json.dumps({"state": "saved-state", "code_verifier": "saved-verifier", "client_id": "client-id", "tenant": "tenant-id", "redirect_uri": "http://localhost:1"})
    )
    monkeypatch.setattr(
        auth_module,
        "_request_token",
        lambda tenant, data: {
            "access_token": "access-token",
            "refresh_token": "refresh-token",
            "expires_in": 3600,
            "scope": "Mail.Read User.Read",
        },
    )

    auth_module.exchange_auth_code("http://localhost:1/?code=auth-code&state=saved-state")

    assert stat.S_IMODE(auth_module.TOKEN_PATH.stat().st_mode) == 0o600


def test_get_valid_access_token_refreshes_quietly(auth_module, monkeypatch, capsys):
    auth_module.configure_client("client-id", tenant="tenant-id")
    auth_module.TOKEN_PATH.write_text(
        json.dumps(
            {
                "access_token": "old-token",
                "refresh_token": "refresh-token",
                "expires_at": 1,
                "scope": "Mail.Read User.Read",
                "tenant": "tenant-id",
            }
        )
    )
    capsys.readouterr()
    monkeypatch.setattr(
        auth_module,
        "_request_token",
        lambda tenant, data: {
            "access_token": "new-token",
            "expires_in": 3600,
            "scope": "Mail.Read User.Read",
        },
    )

    assert auth_module.get_valid_access_token() == "new-token"
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""


def test_auth_error_callback_is_sanitized(auth_module, capsys):
    with pytest.raises(SystemExit):
        auth_module._extract_code_and_state(
            "http://localhost:1/?error=access_denied&error_description=line1%0Aline2"
        )

    out = capsys.readouterr().out
    assert "line1 line2" in out
    assert "line1\nline2" not in out


def test_exchange_auth_code_uses_client_id_from_pending_auth(auth_module, monkeypatch):
    auth_module.configure_client("new-client-id", tenant="new-tenant")
    auth_module.PENDING_AUTH_PATH.write_text(
        json.dumps(
            {
                "state": "saved-state",
                "code_verifier": "saved-verifier",
                "client_id": "original-client-id",
                "tenant": "original-tenant",
                "redirect_uri": "http://localhost:1",
            }
        )
    )
    captured = {}

    def fake_request_token(tenant, data):
        captured["tenant"] = tenant
        captured["data"] = data
        return {
            "access_token": "access-token",
            "refresh_token": "refresh-token",
            "expires_in": 3600,
            "scope": "Mail.Read User.Read",
        }

    monkeypatch.setattr(auth_module, "_request_token", fake_request_token)

    auth_module.exchange_auth_code("http://localhost:1/?code=auth-code&state=saved-state")

    assert captured["tenant"] == "original-tenant"
    assert captured["data"]["client_id"] == "original-client-id"

    saved = json.loads(auth_module.TOKEN_PATH.read_text())
    assert saved["client_id"] == "original-client-id"
    assert saved["tenant"] == "original-tenant"
    assert saved["redirect_uri"] == "http://localhost:1"


def test_refresh_token_uses_stored_client_id(auth_module, monkeypatch):
    auth_module.configure_client("new-client-id", tenant="tenant-id")
    auth_module.TOKEN_PATH.write_text(
        json.dumps(
            {
                "access_token": "old-token",
                "refresh_token": "refresh-token",
                "expires_at": 1,
                "scope": "Mail.Read User.Read",
                "tenant": "tenant-id",
                "client_id": "original-client-id",
            }
        )
    )
    captured = {}

    def fake_request_token(tenant, data):
        captured["tenant"] = tenant
        captured["data"] = data
        return {
            "access_token": "new-token",
            "expires_in": 3600,
            "scope": "Mail.Read User.Read",
        }

    monkeypatch.setattr(auth_module, "_request_token", fake_request_token)

    assert auth_module.refresh_token(emit_status=False) is True
    assert captured["tenant"] == "tenant-id"
    assert captured["data"]["client_id"] == "original-client-id"


def test_helper_fallback_normalizes_relative_hermes_home(monkeypatch, tmp_path):
    monkeypatch.setitem(sys.modules, "hermes_constants", None)
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    monkeypatch.chdir(workspace)
    monkeypatch.setenv("HERMES_HOME", "../profiles/../custom-hermes")

    spec = importlib.util.spec_from_file_location("microsoft_hermes_home_test", HELPER_PATH)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    assert module.get_hermes_home() == (tmp_path / "custom-hermes").resolve()