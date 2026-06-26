"""Tests for the Hindsight Cloud OAuth client (hermes_cli.hindsight_oauth).

Covers PKCE correctness, the loopback authorization flow (register → authorize
→ callback → token exchange), CSRF state validation, credential persistence
(0600), and refresh-token rotation including the dead-refresh-token path. All
HTTP is mocked — no live network.
"""

import base64
import hashlib
import json

import pytest

import hermes_cli.hindsight_oauth as ho


def _fake_jwt(claims: dict) -> str:
    """Build an unsigned JWT-shaped token whose payload carries *claims*."""
    def b64(obj: dict) -> str:
        return base64.urlsafe_b64encode(json.dumps(obj).encode()).rstrip(b"=").decode()

    return f"{b64({'alg': 'RS256'})}.{b64(claims)}.signature"


# --- PKCE ------------------------------------------------------------------

def test_generate_pkce_is_s256():
    verifier, challenge = ho._generate_pkce()
    expected = (
        base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest())
        .rstrip(b"=")
        .decode()
    )
    assert challenge == expected
    assert "=" not in challenge  # base64url, unpadded


def test_org_id_from_jwt_reads_claim():
    token = _fake_jwt({"org_id": "org_xyz", "sub": "user_1"})
    assert ho._org_id_from_jwt(token) == "org_xyz"
    assert ho._org_id_from_jwt("not-a-jwt") == ""


# --- Full login flow -------------------------------------------------------

def _patch_loopback(monkeypatch, result):
    """Patch the loopback server so the test controls the callback result dict."""
    monkeypatch.setattr(
        ho, "_start_loopback_server", lambda: (object(), result, object(), 54321)
    )
    monkeypatch.setattr(ho, "_stop_loopback_server", lambda server, thread: None)


def test_full_login_flow_persists_credentials(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    captured: dict = {}
    result = {"code": None, "state": None, "error": None, "error_description": None}
    _patch_loopback(monkeypatch, result)

    def fake_post_json(url, payload, timeout=ho._HTTP_TIMEOUT):
        captured["register_url"] = url
        captured["register_payload"] = payload
        return {"client_id": "hmc_test", "token_endpoint_auth_method": "none"}

    monkeypatch.setattr(ho, "_post_json", fake_post_json)

    def fake_open(auth_url, open_browser):
        from urllib.parse import parse_qs, urlparse

        query = parse_qs(urlparse(auth_url).query)
        captured["authorize_query"] = query
        # Simulate the browser round-trip landing on the loopback callback.
        result["code"] = "authcode123"
        result["state"] = query["state"][0]

    monkeypatch.setattr(ho, "_open_browser", fake_open)

    def fake_post_form(url, fields, timeout=ho._HTTP_TIMEOUT):
        captured["token_url"] = url
        captured["token_fields"] = fields
        return {
            "access_token": _fake_jwt({"org_id": "org_42", "org_name": "Acme Corp"}),
            "refresh_token": "hrt_abc",
            "expires_in": 3600,
            "scope": "openid profile email offline_access",
        }

    monkeypatch.setattr(ho, "_post_form", fake_post_form)

    creds = ho.run_hindsight_oauth_login("https://api.example.com")

    # Registration shape: native loopback client, no secret requested.
    assert captured["register_url"] == "https://api.example.com/oauth/register"
    assert captured["register_payload"]["client_name"] == "Hermes Agent"
    assert captured["register_payload"]["redirect_uris"] == [
        "http://127.0.0.1:54321/callback"
    ]

    # Authorization params: PKCE S256, matching client + redirect.
    query = captured["authorize_query"]
    assert query["code_challenge_method"] == ["S256"]
    assert query["client_id"] == ["hmc_test"]
    assert query["redirect_uri"] == ["http://127.0.0.1:54321/callback"]
    assert query["response_type"] == ["code"]

    # Token exchange shape: authorization_code + code_verifier, NO client_secret.
    fields = captured["token_fields"]
    assert captured["token_url"] == "https://api.example.com/oauth/token"
    assert fields["grant_type"] == "authorization_code"
    assert fields["code"] == "authcode123"
    assert fields["client_id"] == "hmc_test"
    assert fields["redirect_uri"] == "http://127.0.0.1:54321/callback"
    assert fields["code_verifier"]
    assert "client_secret" not in fields

    # Persisted credentials.
    assert creds["access_token"]
    assert creds["refresh_token"] == "hrt_abc"
    assert creds["org_id"] == "org_42"
    assert creds["org_name"] == "Acme Corp"
    assert creds["api_url"] == "https://api.example.com"
    assert creds["expires_at_ms"] > ho._now_ms()

    path = ho.credentials_path()
    assert path.exists()
    assert (path.stat().st_mode & 0o777) == 0o600
    assert ho.read_credentials()["refresh_token"] == "hrt_abc"


def test_state_mismatch_aborts_without_persisting(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    result = {"code": None, "state": None, "error": None, "error_description": None}
    _patch_loopback(monkeypatch, result)
    monkeypatch.setattr(ho, "_post_json", lambda url, payload, timeout=30.0: {"client_id": "hmc"})

    def fake_open(auth_url, open_browser):
        result["code"] = "code"
        result["state"] = "WRONG-STATE"  # not the generated state

    monkeypatch.setattr(ho, "_open_browser", fake_open)
    # Token endpoint must never be reached.
    monkeypatch.setattr(
        ho, "_post_form", lambda *a, **k: pytest.fail("token exchange should not run")
    )

    with pytest.raises(ho.HindsightOAuthError, match="state mismatch"):
        ho.run_hindsight_oauth_login("https://api.example.com")
    assert not ho.credentials_path().exists()


def test_authorization_error_is_surfaced(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    result = {"code": None, "state": None, "error": None, "error_description": None}
    _patch_loopback(monkeypatch, result)
    monkeypatch.setattr(ho, "_post_json", lambda url, payload, timeout=30.0: {"client_id": "hmc"})

    def fake_open(auth_url, open_browser):
        result["error"] = "access_denied"
        result["error_description"] = "user declined"

    monkeypatch.setattr(ho, "_open_browser", fake_open)

    with pytest.raises(ho.HindsightOAuthError, match="access_denied"):
        ho.run_hindsight_oauth_login("https://api.example.com")
    assert not ho.credentials_path().exists()


# --- Credential storage ----------------------------------------------------

def test_write_credentials_is_0600(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    ho.write_credentials({"access_token": "x", "refresh_token": "y"})
    path = ho.credentials_path()
    assert path.exists()
    assert (path.stat().st_mode & 0o777) == 0o600


def test_read_credentials_missing_returns_none(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    assert ho.read_credentials() is None


# --- Refresh ---------------------------------------------------------------

def _store(monkeypatch, tmp_path, **overrides):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    creds = {
        "client_id": "hmc",
        "access_token": "old-token",
        "refresh_token": "hrt_old",
        "expires_at_ms": ho._now_ms() - 1,  # expired
        "scope": "openid",
        "api_url": "https://api.example.com",
        "org_id": "org_1",
    }
    creds.update(overrides)
    ho.write_credentials(creds)


def test_valid_token_returned_without_refresh(tmp_path, monkeypatch):
    _store(monkeypatch, tmp_path, access_token="good", expires_at_ms=ho._now_ms() + 3_600_000)
    monkeypatch.setattr(
        ho, "refresh_access_token", lambda *a, **k: pytest.fail("should not refresh")
    )
    assert ho.get_valid_access_token() == "good"


def test_expired_token_is_refreshed_and_rotated(tmp_path, monkeypatch):
    _store(monkeypatch, tmp_path)
    seen = {}

    def fake_refresh(api_url, client_id, refresh_token):
        seen["refresh_token"] = refresh_token
        seen["client_id"] = client_id
        return {
            "access_token": _fake_jwt({"org_id": "org_1"}),
            "refresh_token": "hrt_new",
            "expires_in": 3600,
        }

    monkeypatch.setattr(ho, "refresh_access_token", fake_refresh)

    token = ho.get_valid_access_token("https://api.example.com")
    stored = ho.read_credentials()
    assert seen["refresh_token"] == "hrt_old"
    assert seen["client_id"] == "hmc"
    assert stored["refresh_token"] == "hrt_new"  # rotated + rewritten
    assert token == stored["access_token"]
    assert stored["org_id"] == "org_1"


def test_invalid_grant_clears_credentials(tmp_path, monkeypatch):
    _store(monkeypatch, tmp_path, refresh_token="hrt_dead")

    def fake_refresh(*a, **k):
        raise ho.HindsightOAuthError('400 Bad Request: {"error":"invalid_grant"}')

    monkeypatch.setattr(ho, "refresh_access_token", fake_refresh)

    with pytest.raises(ho.HindsightReauthRequired):
        ho.get_valid_access_token("https://api.example.com")
    assert ho.read_credentials() is None  # dead token wiped


def test_get_valid_access_token_no_credentials_returns_none(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    assert ho.get_valid_access_token("https://api.example.com") is None
