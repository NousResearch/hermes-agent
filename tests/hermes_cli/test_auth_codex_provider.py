"""Tests for Codex auth — tokens stored in Hermes auth store (~/.hermes/auth.json)."""

import json
import time
import base64
import http.client
import socket
import threading
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import parse_qs, urlparse

import pytest

from hermes_cli.auth import (
    AuthError,
    DEFAULT_CODEX_BASE_URL,
    CODEX_OAUTH_REDIRECT_URI,
    CODEX_OAUTH_SCOPE,
    CODEX_OAUTH_ORIGINATOR,
    CODEX_OAUTH_CALLBACK_PORT,
    CODEX_OAUTH_CALLBACK_PATH,
    PROVIDER_REGISTRY,
    _codex_pkce_verifier,
    _codex_pkce_challenge,
    _codex_build_authorize_url,
    _codex_parse_authorization_response,
    _codex_exchange_authorization_code,
    _codex_extract_account_id,
    _codex_create_callback_server,
    _codex_wait_for_callback_server,
    _read_codex_tokens,
    _save_codex_tokens,
    _import_codex_cli_tokens,
    _login_openai_codex,
    refresh_codex_oauth_pure,
    resolve_codex_runtime_credentials,
    resolve_provider,
)


def _setup_hermes_auth(hermes_home: Path, *, access_token: str = "access", refresh_token: str = "refresh"):
    """Write Codex tokens into the Hermes auth store."""
    hermes_home.mkdir(parents=True, exist_ok=True)
    auth_store = {
        "version": 1,
        "active_provider": "openai-codex",
        "providers": {
            "openai-codex": {
                "tokens": {
                    "access_token": access_token,
                    "refresh_token": refresh_token,
                },
                "last_refresh": "2026-02-26T00:00:00Z",
                "auth_mode": "chatgpt",
            },
        },
    }
    auth_file = hermes_home / "auth.json"
    auth_file.write_text(json.dumps(auth_store, indent=2))
    return auth_file


def _jwt_with_exp(exp_epoch: int) -> str:
    payload = {"exp": exp_epoch}
    encoded = base64.urlsafe_b64encode(json.dumps(payload).encode("utf-8")).rstrip(b"=").decode("utf-8")
    return f"h.{encoded}.s"


def test_read_codex_tokens_success(tmp_path, monkeypatch):
    hermes_home = tmp_path / "hermes"
    _setup_hermes_auth(hermes_home)
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    data = _read_codex_tokens()
    assert data["tokens"]["access_token"] == "access"
    assert data["tokens"]["refresh_token"] == "refresh"


def test_read_codex_tokens_missing(tmp_path, monkeypatch):
    hermes_home = tmp_path / "hermes"
    hermes_home.mkdir(parents=True, exist_ok=True)
    # Empty auth store
    (hermes_home / "auth.json").write_text(json.dumps({"version": 1, "providers": {}}))
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    with pytest.raises(AuthError) as exc:
        _read_codex_tokens()
    assert exc.value.code == "codex_auth_missing"


def test_resolve_codex_runtime_credentials_missing_access_token(tmp_path, monkeypatch):
    hermes_home = tmp_path / "hermes"
    _setup_hermes_auth(hermes_home, access_token="")
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    with pytest.raises(AuthError) as exc:
        resolve_codex_runtime_credentials()
    assert exc.value.code == "codex_auth_missing_access_token"
    assert exc.value.relogin_required is True


def test_resolve_codex_runtime_credentials_refreshes_expiring_token(tmp_path, monkeypatch):
    hermes_home = tmp_path / "hermes"
    expiring_token = _jwt_with_exp(int(time.time()) - 10)
    _setup_hermes_auth(hermes_home, access_token=expiring_token, refresh_token="refresh-old")
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    called = {"count": 0}

    def _fake_refresh(tokens, timeout_seconds):
        called["count"] += 1
        return {"access_token": "access-new", "refresh_token": "refresh-new"}

    monkeypatch.setattr("hermes_cli.auth._refresh_codex_auth_tokens", _fake_refresh)

    resolved = resolve_codex_runtime_credentials()

    assert called["count"] == 1
    assert resolved["api_key"] == "access-new"


def test_resolve_codex_runtime_credentials_force_refresh(tmp_path, monkeypatch):
    hermes_home = tmp_path / "hermes"
    _setup_hermes_auth(hermes_home, access_token="access-current", refresh_token="refresh-old")
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    called = {"count": 0}

    def _fake_refresh(tokens, timeout_seconds):
        called["count"] += 1
        return {"access_token": "access-forced", "refresh_token": "refresh-new"}

    monkeypatch.setattr("hermes_cli.auth._refresh_codex_auth_tokens", _fake_refresh)

    resolved = resolve_codex_runtime_credentials(force_refresh=True, refresh_if_expiring=False)

    assert called["count"] == 1
    assert resolved["api_key"] == "access-forced"


def test_resolve_provider_explicit_codex_does_not_fallback(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    assert resolve_provider("openai-codex") == "openai-codex"


def test_save_codex_tokens_roundtrip(tmp_path, monkeypatch):
    hermes_home = tmp_path / "hermes"
    hermes_home.mkdir(parents=True, exist_ok=True)
    (hermes_home / "auth.json").write_text(json.dumps({"version": 1, "providers": {}}))
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    _save_codex_tokens({"access_token": "at123", "refresh_token": "rt456"})
    data = _read_codex_tokens()

    assert data["tokens"]["access_token"] == "at123"
    assert data["tokens"]["refresh_token"] == "rt456"


def test_import_codex_cli_tokens(tmp_path, monkeypatch):
    codex_home = tmp_path / "codex-cli"
    codex_home.mkdir(parents=True, exist_ok=True)
    (codex_home / "auth.json").write_text(json.dumps({
        "tokens": {"access_token": "cli-at", "refresh_token": "cli-rt"},
    }))
    monkeypatch.setenv("CODEX_HOME", str(codex_home))

    tokens = _import_codex_cli_tokens()
    assert tokens is not None
    assert tokens["access_token"] == "cli-at"
    assert tokens["refresh_token"] == "cli-rt"


def test_import_codex_cli_tokens_missing(tmp_path, monkeypatch):
    monkeypatch.setenv("CODEX_HOME", str(tmp_path / "nonexistent"))
    assert _import_codex_cli_tokens() is None


def test_codex_tokens_not_written_to_shared_file(tmp_path, monkeypatch):
    """Verify _save_codex_tokens writes only to Hermes auth store, not ~/.codex/."""
    hermes_home = tmp_path / "hermes"
    codex_home = tmp_path / "codex-cli"
    hermes_home.mkdir(parents=True, exist_ok=True)
    codex_home.mkdir(parents=True, exist_ok=True)

    (hermes_home / "auth.json").write_text(json.dumps({"version": 1, "providers": {}}))
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.setenv("CODEX_HOME", str(codex_home))

    _save_codex_tokens({"access_token": "hermes-at", "refresh_token": "hermes-rt"})

    # ~/.codex/auth.json should NOT exist — _save_codex_tokens only touches Hermes store
    assert not (codex_home / "auth.json").exists()

    # Hermes auth store should have the tokens
    data = _read_codex_tokens()
    assert data["tokens"]["access_token"] == "hermes-at"


def test_resolve_returns_hermes_auth_store_source(tmp_path, monkeypatch):
    hermes_home = tmp_path / "hermes"
    _setup_hermes_auth(hermes_home)
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    creds = resolve_codex_runtime_credentials()
    assert creds["source"] == "hermes-auth-store"
    assert creds["provider"] == "openai-codex"
    assert creds["base_url"] == DEFAULT_CODEX_BASE_URL


class _StubHTTPResponse:
    def __init__(self, status_code: int, payload):
        self.status_code = status_code
        self._payload = payload
        self.text = json.dumps(payload) if isinstance(payload, (dict, list)) else str(payload)

    def json(self):
        if isinstance(self._payload, Exception):
            raise self._payload
        return self._payload


class _StubHTTPClient:
    def __init__(self, response):
        self._response = response

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def post(self, *args, **kwargs):
        return self._response


def _patch_httpx(monkeypatch, response):
    def _factory(*args, **kwargs):
        return _StubHTTPClient(response)

    monkeypatch.setattr("hermes_cli.auth.httpx.Client", _factory)


def test_refresh_parses_openai_nested_error_shape_refresh_token_reused(monkeypatch):
    """OpenAI returns {"error": {"code": "refresh_token_reused", "message": "..."}}
    — parser must surface relogin_required and the dedicated message.
    """
    response = _StubHTTPResponse(
        401,
        {
            "error": {
                "message": "Your refresh token has already been used to generate a new access token. Please try signing in again.",
                "type": "invalid_request_error",
                "param": None,
                "code": "refresh_token_reused",
            }
        },
    )
    _patch_httpx(monkeypatch, response)

    with pytest.raises(AuthError) as exc_info:
        refresh_codex_oauth_pure("a-tok", "r-tok")

    err = exc_info.value
    assert err.code == "refresh_token_reused"
    assert err.relogin_required is True
    # The existing dedicated branch should override the message with actionable guidance.
    assert "already consumed by another client" in str(err)


def test_refresh_parses_openai_nested_error_shape_generic_code(monkeypatch):
    """Nested error with arbitrary code still surfaces code + message."""
    response = _StubHTTPResponse(
        400,
        {
            "error": {
                "message": "Invalid client credentials.",
                "type": "invalid_request_error",
                "code": "invalid_client",
            }
        },
    )
    _patch_httpx(monkeypatch, response)

    with pytest.raises(AuthError) as exc_info:
        refresh_codex_oauth_pure("a-tok", "r-tok")

    err = exc_info.value
    assert err.code == "invalid_client"
    assert "Invalid client credentials." in str(err)


def test_refresh_parses_oauth_spec_flat_error_shape_invalid_grant(monkeypatch):
    """Fallback path: OAuth spec-shape {"error": "invalid_grant", "error_description": "..."}
    must still map to relogin_required=True via the existing code set.
    """
    response = _StubHTTPResponse(
        400,
        {
            "error": "invalid_grant",
            "error_description": "Refresh token is expired or revoked.",
        },
    )
    _patch_httpx(monkeypatch, response)

    with pytest.raises(AuthError) as exc_info:
        refresh_codex_oauth_pure("a-tok", "r-tok")

    err = exc_info.value
    assert err.code == "invalid_grant"
    assert err.relogin_required is True
    assert "Refresh token is expired or revoked." in str(err)


def test_refresh_falls_back_to_generic_message_on_unparseable_body(monkeypatch):
    """No JSON body → generic 'with status 401' message; 401 always forces relogin."""
    response = _StubHTTPResponse(401, ValueError("not json"))
    _patch_httpx(monkeypatch, response)

    with pytest.raises(AuthError) as exc_info:
        refresh_codex_oauth_pure("a-tok", "r-tok")

    err = exc_info.value
    assert err.code == "codex_refresh_failed"
    # 401/403 from the token endpoint always means the refresh token is
    # invalid/expired — force relogin even without a parseable error body.
    assert err.relogin_required is True
    assert "status 401" in str(err)


def test_login_openai_codex_force_new_login_skips_existing_reuse_prompt(monkeypatch):
    called = {"device_login": 0}

    monkeypatch.setattr(
        "hermes_cli.auth.resolve_codex_runtime_credentials",
        lambda: {"base_url": DEFAULT_CODEX_BASE_URL},
    )
    monkeypatch.setattr(
        "hermes_cli.auth._import_codex_cli_tokens",
        lambda: {"access_token": "cli-at", "refresh_token": "cli-rt"},
    )
    monkeypatch.setattr(
        "hermes_cli.auth._codex_device_code_login",
        lambda: {
            "tokens": {"access_token": "fresh-at", "refresh_token": "fresh-rt"},
            "last_refresh": "2026-04-01T00:00:00Z",
            "base_url": DEFAULT_CODEX_BASE_URL,
        },
    )

    def _fake_save(tokens, last_refresh=None, **kwargs):
        called["device_login"] += 1
        called["tokens"] = dict(tokens)
        called["last_refresh"] = last_refresh
        called["save_kwargs"] = kwargs

    monkeypatch.setattr("hermes_cli.auth._save_codex_tokens", _fake_save)
    monkeypatch.setattr("hermes_cli.auth._update_config_for_provider", lambda *args, **kwargs: "/tmp/config.yaml")
    monkeypatch.setattr(
        "builtins.input",
        lambda prompt="": (_ for _ in ()).throw(AssertionError("force_new_login should not prompt for reuse/import")),
    )

    _login_openai_codex(
        SimpleNamespace(auth_method="device_code"),
        PROVIDER_REGISTRY["openai-codex"],
        force_new_login=True,
        auth_method="device_code",
    )

    assert called["device_login"] == 1
    assert called["tokens"]["access_token"] == "fresh-at"



def _jwt_with_claims(claims: dict) -> str:
    payload = base64.urlsafe_b64encode(json.dumps(claims).encode("utf-8")).rstrip(b"=").decode("utf-8")
    return f"h.{payload}.s"


def test_codex_pkce_verifier_and_challenge_shape():
    verifier = _codex_pkce_verifier()
    assert 43 <= len(verifier) <= 128
    assert all(ch.isalnum() or ch in "-._~" for ch in verifier)
    assert _codex_pkce_challenge("dBjftJeZ4CVP-mB92K27uhbUJU1p1r_wW1gFWFOEjXk") == "E9Melhoa2OwvFrEMTJguCHaoeK1t8URWbuGJSstw-cM"


def test_codex_authorize_url_uses_upstream_codex_pkce_params():
    url = _codex_build_authorize_url(code_challenge="challenge", state="state123")
    parsed = urlparse(url)
    query = {key: values[0] for key, values in parse_qs(parsed.query).items()}

    assert url.startswith("https://auth.openai.com/oauth/authorize?")
    assert query["response_type"] == "code"
    assert query["redirect_uri"] == CODEX_OAUTH_REDIRECT_URI
    assert query["scope"] == CODEX_OAUTH_SCOPE
    assert query["code_challenge"] == "challenge"
    assert query["code_challenge_method"] == "S256"
    assert query["id_token_add_organizations"] == "true"
    assert query["codex_cli_simplified_flow"] == "true"
    assert query["originator"] == CODEX_OAUTH_ORIGINATOR
    assert query["state"] == "state123"
    assert "code_verifier" not in query


def test_codex_manual_redirect_url_parses_and_verifies_state():
    parsed = _codex_parse_authorization_response(
        f"{CODEX_OAUTH_REDIRECT_URI}?code=abc&state=good",
        expected_state="good",
    )
    assert parsed == {"code": "abc", "state_verified": True}


def test_codex_manual_code_only_warn_path():
    parsed = _codex_parse_authorization_response("abc123", expected_state="expected")
    assert parsed == {"code": "abc123", "state_verified": False}


def test_codex_manual_redirect_rejects_state_mismatch():
    with pytest.raises(AuthError) as exc:
        _codex_parse_authorization_response(
            f"{CODEX_OAUTH_REDIRECT_URI}?code=abc&state=bad",
            expected_state="good",
        )
    assert exc.value.code == "codex_oauth_state_mismatch"


def test_codex_manual_redirect_error_is_actionable():
    with pytest.raises(AuthError) as exc:
        _codex_parse_authorization_response(
            f"{CODEX_OAUTH_REDIRECT_URI}?error=access_denied&error_description=Denied&state=good",
            expected_state="good",
        )
    assert exc.value.code == "codex_authorization_denied"
    assert "access_denied" in str(exc.value)


def test_codex_manual_redirect_missing_code():
    with pytest.raises(AuthError) as exc:
        _codex_parse_authorization_response(
            f"{CODEX_OAUTH_REDIRECT_URI}?state=good",
            expected_state="good",
        )
    assert exc.value.code == "codex_manual_redirect_missing_code"


def test_codex_extract_account_id_from_access_token():
    token = _jwt_with_claims({"chatgpt_account_id": "acct_123", "exp": int(time.time()) + 3600})
    assert _codex_extract_account_id(token) == "acct_123"


def test_codex_extract_account_id_failure():
    with pytest.raises(AuthError) as exc:
        _codex_extract_account_id(_jwt_with_claims({"exp": int(time.time()) + 3600}))
    assert exc.value.code == "codex_account_id_extraction_failed"


def test_codex_token_exchange_request_and_metadata(monkeypatch):
    access = _jwt_with_claims({"account_id": "acct_456", "exp": int(time.time()) + 7200})
    response = _StubHTTPResponse(200, {"access_token": access, "refresh_token": "refresh-new", "expires_in": 3600})
    calls = []

    class _Client(_StubHTTPClient):
        def post(self, *args, **kwargs):
            calls.append((args, kwargs))
            return response

    monkeypatch.setattr("hermes_cli.auth.httpx.Client", lambda *args, **kwargs: _Client(response))
    tokens = _codex_exchange_authorization_code(authorization_code="code123", code_verifier="verifier123")

    assert tokens["access_token"] == access
    assert tokens["refresh_token"] == "refresh-new"
    assert tokens["account_id"] == "acct_456"
    assert tokens["accountId"] == "acct_456"
    assert tokens["expires_at_ms"] > int(time.time() * 1000)
    data = calls[0][1]["data"]
    assert data["grant_type"] == "authorization_code"
    assert data["code"] == "code123"
    assert data["code_verifier"] == "verifier123"
    assert data["redirect_uri"] == CODEX_OAUTH_REDIRECT_URI


def test_codex_token_exchange_error(monkeypatch):
    response = _StubHTTPResponse(400, {"error": "invalid_grant", "error_description": "bad code"})
    _patch_httpx(monkeypatch, response)
    with pytest.raises(AuthError) as exc:
        _codex_exchange_authorization_code(authorization_code="bad", code_verifier="verifier")
    assert exc.value.code == "invalid_grant"
    assert "bad code" in str(exc.value)


def test_refresh_codex_oauth_pure_preserves_metadata(monkeypatch):
    access = _jwt_with_claims({"account_id": "acct_ref", "exp": int(time.time()) + 7200})
    response = _StubHTTPResponse(200, {"access_token": access, "refresh_token": "rotated", "expires_in": 3600})
    _patch_httpx(monkeypatch, response)
    refreshed = refresh_codex_oauth_pure("old-access", "old-refresh")
    assert refreshed["access_token"] == access
    assert refreshed["refresh_token"] == "rotated"
    assert refreshed["account_id"] == "acct_ref"
    assert refreshed["expires_at_ms"] > int(time.time() * 1000)


def test_codex_extract_account_id_from_nested_auth_claim():
    token = _jwt_with_claims({"https://api.openai.com/auth": {"chatgpt_account_id": "acct_nested"}})
    assert _codex_extract_account_id(token) == "acct_nested"


def test_codex_extract_account_id_malformed_and_non_jwt_fail_readably():
    for token in ("not-a-jwt", "a.b", "a.%%% .c"):
        with pytest.raises(AuthError) as exc:
            _codex_extract_account_id(token)
        assert exc.value.code == "codex_account_id_extraction_failed"


def test_codex_callback_server_port_1455_free_starts_and_closes():
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        if sock.connect_ex(("127.0.0.1", CODEX_OAUTH_CALLBACK_PORT)) == 0:
            pytest.skip("callback port 1455 is already in use on this test host")
    finally:
        sock.close()
    server = _codex_create_callback_server("state-ok")
    try:
        assert server.server_address[1] == CODEX_OAUTH_CALLBACK_PORT
    finally:
        server.server_close()


def test_codex_callback_server_port_1455_occupied_fails_cleanly():
    blocker = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        blocker.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            blocker.bind(("127.0.0.1", CODEX_OAUTH_CALLBACK_PORT))
        except OSError:
            pytest.skip("callback port 1455 is already in use on this test host")
        blocker.listen(1)
        with pytest.raises(OSError):
            _codex_create_callback_server("state-ok")
    finally:
        blocker.close()


def _serve_single_callback(server, result_holder):
    result_holder["result"] = _codex_wait_for_callback_server(server, 3)


def _request_callback(port: int, path: str):
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=3)
    try:
        conn.request("GET", path)
        return conn.getresponse().status
    finally:
        conn.close()


def test_codex_callback_accepts_success_and_closes_server():
    server = _codex_create_callback_server("good")
    result_holder = {}
    thread = threading.Thread(target=_serve_single_callback, args=(server, result_holder))
    thread.start()
    status = _request_callback(CODEX_OAUTH_CALLBACK_PORT, f"{CODEX_OAUTH_CALLBACK_PATH}?code=abc&state=good")
    thread.join(timeout=5)

    assert status == 200
    assert result_holder["result"] == {"code": "abc", "state": "good"}
    with pytest.raises(OSError):
        _request_callback(CODEX_OAUTH_CALLBACK_PORT, f"{CODEX_OAUTH_CALLBACK_PATH}?code=again&state=good")


def test_codex_callback_rejects_wrong_state_missing_code_error_and_wrong_path():
    cases = [
        (f"{CODEX_OAUTH_CALLBACK_PATH}?code=abc&state=bad", 400, "state_mismatch"),
        (f"{CODEX_OAUTH_CALLBACK_PATH}?state=good", 400, "missing_code"),
        (f"{CODEX_OAUTH_CALLBACK_PATH}?error=access_denied&error_description=Denied&state=good", 400, "access_denied"),
        ("/wrong?code=abc&state=good", 404, None),
    ]
    for path, expected_status, expected_error in cases:
        server = _codex_create_callback_server("good")
        result_holder = {}
        thread = threading.Thread(target=_serve_single_callback, args=(server, result_holder))
        thread.start()
        status = _request_callback(CODEX_OAUTH_CALLBACK_PORT, path)
        thread.join(timeout=5)

        assert status == expected_status
        if expected_error:
            assert result_holder["result"]["error"] == expected_error
        else:
            assert result_holder["result"] == {}


def test_login_browser_manual_redirect_persists_credentials_and_returns(tmp_path, monkeypatch):
    hermes_home = tmp_path / "hermes"
    hermes_home.mkdir(parents=True, exist_ok=True)
    (hermes_home / "auth.json").write_text(json.dumps({"version": 1, "providers": {}}))
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.setattr("hermes_cli.auth._codex_pkce_pair", lambda: ("verifier", "challenge", "state-ok"))
    monkeypatch.setattr("hermes_cli.auth._is_remote_session", lambda: True)
    monkeypatch.setattr("hermes_cli.auth._update_config_for_provider", lambda *args, **kwargs: str(hermes_home / "config.yaml"))
    access = _jwt_with_claims({"https://api.openai.com/auth": {"chatgpt_account_id": "acct_manual"}, "exp": int(time.time()) + 3600})

    calls = {}
    def _fake_exchange(*, authorization_code, code_verifier, **kwargs):
        calls["authorization_code"] = authorization_code
        calls["code_verifier"] = code_verifier
        return {
            "access_token": access,
            "refresh_token": "refresh_manual",
            "account_id": "acct_manual",
            "accountId": "acct_manual",
            "expires": int(time.time()) + 3600,
        }

    monkeypatch.setattr("hermes_cli.auth._codex_exchange_authorization_code", _fake_exchange)
    monkeypatch.setattr("builtins.input", lambda prompt="": f"{CODEX_OAUTH_REDIRECT_URI}?code=manual-code&state=state-ok")

    _login_openai_codex(
        SimpleNamespace(auth_method="browser_oauth_pkce", no_browser=True),
        PROVIDER_REGISTRY["openai-codex"],
        force_new_login=True,
        auth_method="browser_oauth_pkce",
    )

    data = _read_codex_tokens()
    assert calls == {"authorization_code": "manual-code", "code_verifier": "verifier"}
    assert data["tokens"]["refresh_token"] == "refresh_manual"
    assert data["tokens"]["account_id"] == "acct_manual"
    assert data["auth_mode"] == "browser_oauth_pkce"


def test_refresh_persists_rotated_refresh_token_and_new_pool_reads_it(tmp_path, monkeypatch):
    hermes_home = tmp_path / "hermes"
    expired = _jwt_with_claims({"https://api.openai.com/auth": {"chatgpt_account_id": "acct_old"}, "exp": int(time.time()) - 60})
    _setup_hermes_auth(hermes_home, access_token=expired, refresh_token="refresh_old")
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    access_new = _jwt_with_claims({"https://api.openai.com/auth": {"chatgpt_account_id": "acct_new"}, "exp": int(time.time()) + 7200})
    response = _StubHTTPResponse(200, {"access_token": access_new, "refresh_token": "refresh_new", "expires_in": 3600})
    calls = []

    class _Client(_StubHTTPClient):
        def post(self, *args, **kwargs):
            calls.append(kwargs)
            return response

    monkeypatch.setattr("hermes_cli.auth.httpx.Client", lambda *args, **kwargs: _Client(response))

    resolved = resolve_codex_runtime_credentials(refresh_if_expiring=True)
    assert resolved["api_key"] == access_new
    assert calls[0]["json"]["refresh_token"] == "refresh_old"
    assert calls[0]["json"]["grant_type"] == "refresh_token"

    stored = json.loads((hermes_home / "auth.json").read_text())
    tokens = stored["providers"]["openai-codex"]["tokens"]
    assert tokens["access_token"] == access_new
    assert tokens["refresh_token"] == "refresh_new"
    assert tokens["account_id"] == "acct_new"
    assert stored["providers"]["openai-codex"]["expires_at_ms"] > int(time.time() * 1000)

    from agent.credential_pool import load_pool

    pool = load_pool("openai-codex")
    entry = pool.select()
    assert entry is not None
    assert entry.refresh_token == "refresh_new"
    assert entry.access_token == access_new
