"""Hardening tests for Meta OAuth: expiry, mint contract, paywall, quarantine, URLs, aux refresh."""

import pytest


class _Resp:
    def __init__(self, status_code, payload, text=""):
        self.status_code = status_code
        self._payload = payload
        self.text = text

    def json(self):
        if isinstance(self._payload, Exception):
            raise self._payload
        return self._payload


class _Client:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def post(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        return self.responses.pop(0)


def test_trusted_url_requires_https_meta_host():
    from hermes_cli.auth_meta import _trusted_http_url

    assert _trusted_http_url("https://auth.meta.com/device?code=X") is not None
    assert _trusted_http_url("https://meta.com/setup") is not None
    assert _trusted_http_url("http://auth.meta.com/device") is None
    assert _trusted_http_url("https://evil.example.com/device") is None
    assert _trusted_http_url("https://meta.com.evil.example.com/device") is None
    assert _trusted_http_url("https://notmeta.com/device") is None


def test_mint_prefers_server_expires_in(monkeypatch):
    from hermes_cli.auth_meta import mint_meta_api_key

    monkeypatch.setattr("hermes_cli.auth_meta.time.time", lambda: 1_000.0)
    client = _Client([_Resp(200, {"api_key": "k", "expires_in": 3600})])
    result = mint_meta_api_key("identity", client)
    assert result["expires_at_ms"] == 1_000_000 + 3_600_000


def test_mint_prefers_server_expires_at_ms(monkeypatch):
    from hermes_cli.auth_meta import mint_meta_api_key

    monkeypatch.setattr("hermes_cli.auth_meta.time.time", lambda: 1_000.0)
    client = _Client([_Resp(200, {"api_key": "k", "expires_at_ms": 9_999_000})])
    result = mint_meta_api_key("identity", client)
    assert result["expires_at_ms"] == 9_999_000


def test_mint_onboard_body_only_when_onboarding():
    import json

    from hermes_cli.auth_meta import mint_meta_api_key

    client = _Client([_Resp(200, {"api_key": "k"})])
    mint_meta_api_key("identity", client, onboard=True)
    assert json.loads(client.calls[0][1]["content"]) == {"onboard": True}

    client = _Client([_Resp(200, {"api_key": "k"})])
    mint_meta_api_key("identity", client)
    assert json.loads(client.calls[0][1]["content"]) == {}


def test_mint_403_paywall_is_distinct_and_not_relogin():
    from hermes_cli.auth_meta import mint_meta_api_key

    client = _Client([_Resp(403, {"require_payment": True, "action_url": "https://auth.meta.com/billing"})])
    with pytest.raises(Exception) as caught:
        mint_meta_api_key("identity", client)
    assert caught.value.code == "meta_subscription_inactive"
    assert caught.value.relogin_required is False
    assert "auth.meta.com/billing" in str(caught.value)


def test_mint_403_inactive_subscription_flag_is_paywall():
    from hermes_cli.auth_meta import mint_meta_api_key

    client = _Client([_Resp(403, {"is_subs_active": False})])
    with pytest.raises(Exception) as caught:
        mint_meta_api_key("identity", client)
    assert caught.value.code == "meta_subscription_inactive"
    assert caught.value.relogin_required is False


def test_mint_403_without_paywall_fields_stays_session_expired():
    from hermes_cli.auth_meta import mint_meta_api_key

    client = _Client([_Resp(403, {"error_description": "bad session"})])
    with pytest.raises(Exception) as caught:
        mint_meta_api_key("identity", client)
    assert caught.value.code == "meta_session_expired"
    assert caught.value.relogin_required is True


def test_quarantine_saves_under_auth_store_lock(monkeypatch):
    import contextlib

    from hermes_cli.auth_constants import AuthError
    from hermes_cli import auth_meta

    locked = []

    @contextlib.contextmanager
    def fake_lock(*args, **kwargs):
        locked.append(True)
        yield

    saved = {}
    monkeypatch.setattr("hermes_cli.auth._auth_store_lock", fake_lock)
    monkeypatch.setattr("hermes_cli.auth._load_auth_store", lambda: {"providers": {}})
    monkeypatch.setattr("hermes_cli.auth._load_provider_state", lambda store, pid: {"tokens": {"access_token": "a", "refresh_token": "r"}})
    monkeypatch.setattr("hermes_cli.auth._store_provider_state", lambda store, pid, state, set_active=True: saved.update(state))
    monkeypatch.setattr("hermes_cli.auth._save_auth_store", lambda store: saved.setdefault("_saved", True))
    monkeypatch.setattr("hermes_cli.auth._last_auth_error_marker", lambda *a, **k: {"code": "x"})

    auth_meta._quarantine_meta_oauth_tokens(AuthError("boom", provider="meta-oauth", code="meta_session_expired", relogin_required=True))
    assert locked, "quarantine must save under _auth_store_lock"
    assert saved["tokens"] == {}


def test_error_detail_is_truncated():
    from hermes_cli.auth_meta import _error_detail

    detail = _error_detail({"error_description": "x" * 500})
    assert len(detail) <= 202


def test_aux_refreshers_include_meta_oauth(monkeypatch):
    from agent import auxiliary_client

    assert "meta-oauth" in auxiliary_client._CREDENTIAL_REFRESHERS
    monkeypatch.setattr(auxiliary_client, "load_pool", lambda _p: None)
    monkeypatch.setattr(
        "hermes_cli.auth.resolve_meta_oauth_runtime_credentials",
        lambda **_k: {"api_key": "fresh-key"},
    )
    assert auxiliary_client._refresh_meta_oauth_credentials() is True


def test_poll_slow_down_then_success(monkeypatch):
    from hermes_cli.auth_meta import poll_for_identity_token

    monkeypatch.setattr("hermes_cli.auth_device_flow.time.sleep", lambda _s: None)
    client = _Client([
        _Resp(400, {"error": "slow_down"}),
        _Resp(200, {"access_token": "identity-token"}),
    ])
    assert poll_for_identity_token(client, "device-code", expires_in=60, poll_interval=1) == "identity-token"


def test_poll_expired_token_requires_relogin(monkeypatch):
    from hermes_cli.auth_meta import poll_for_identity_token

    monkeypatch.setattr("hermes_cli.auth_device_flow.time.sleep", lambda _s: None)
    client = _Client([_Resp(400, {"error": "expired_token"})])
    with pytest.raises(Exception) as caught:
        poll_for_identity_token(client, "device-code", expires_in=60, poll_interval=1)
    assert caught.value.relogin_required is True


def test_mint_5xx_is_mint_failed_not_relogin():
    from hermes_cli.auth_meta import mint_meta_api_key

    client = _Client([_Resp(500, {"error_description": "upstream boom"})])
    with pytest.raises(Exception) as caught:
        mint_meta_api_key("identity", client)
    assert caught.value.code == "api_key_mint_failed"
    assert caught.value.relogin_required is False


def test_poll_honours_cancellation(monkeypatch):
    from hermes_cli.auth_meta import poll_for_identity_token

    monkeypatch.setattr("hermes_cli.auth_device_flow.time.sleep", lambda _s: None)
    client = _Client([_Resp(400, {"error": "authorization_pending"})] * 5)
    with pytest.raises(Exception) as caught:
        poll_for_identity_token(client, "device-code", expires_in=60, poll_interval=1, cancelled=lambda: True)
    assert not isinstance(caught.value, TypeError)
    assert len(client.calls) == 0
