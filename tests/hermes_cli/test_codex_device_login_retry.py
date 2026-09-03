"""Regression test: _codex_device_code_login must survive transient
network errors (e.g. GFW resets on Cloudflare-fronted endpoints) during the
device-auth polling loop instead of aborting the whole login.

Previously a single httpx.TransportError raised from the poll POST killed the
flow with an unhandled exception. The patched loop catches httpx.HTTPError and
retries on the next poll cycle.
"""

import time

import httpx
import pytest

from hermes_cli import auth as auth_mod


class _FakeResponse:
    def __init__(self, status_code, json_data=None):
        self.status_code = status_code
        self._json = json_data or {}

    def json(self):
        return self._json


class _FakeClient:
    """Scripted httpx.Client:

    1. usercode request      -> 200 (device code issued)
    2. token poll #1         -> raises httpx.ConnectError (transient reset)
    3. token poll #2         -> 200 (user finished login)
    4. token exchange        -> 200 (access/refresh tokens)
    """

    post_calls = []  # class-level shared call log

    def __init__(self, *args, **kwargs):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def post(self, url, **kwargs):
        self.post_calls.append(url)
        if url.endswith("/api/accounts/deviceauth/usercode"):
            return _FakeResponse(
                200,
                {"user_code": "TEST-CODE", "device_auth_id": "daid", "interval": "3"},
            )
        if url.endswith("/api/accounts/deviceauth/token"):
            if sum(1 for u in self.post_calls if u.endswith("/token")) == 1:
                raise httpx.ConnectError("simulated transient network reset")
            return _FakeResponse(
                200, {"authorization_code": "authcode", "code_verifier": "verifier"}
            )
        # token exchange
        return _FakeResponse(200, {"access_token": "at", "refresh_token": "rt"})


def test_codex_device_login_retries_transient_poll_error(monkeypatch):
    _FakeClient.post_calls = []
    monkeypatch.setattr(auth_mod.httpx, "Client", _FakeClient)
    monkeypatch.setattr("time.sleep", lambda _: None)

    creds = auth_mod._codex_device_code_login()

    assert creds["tokens"]["access_token"] == "at"
    assert creds["tokens"]["refresh_token"] == "rt"
    assert creds["auth_mode"] == "chatgpt"
    assert creds["source"] == "device-code"

    token_polls = [
        u for u in _FakeClient.post_calls if u.endswith("/api/accounts/deviceauth/token")
    ]
    assert len(token_polls) == 2  # first attempt reset, second succeeded


def test_codex_device_login_times_out_cleanly_on_persistent_errors(monkeypatch):
    class _FlakyClient(_FakeClient):
        def post(self, url, **kwargs):
            self.post_calls.append(url)
            if url.endswith("/api/accounts/deviceauth/token"):
                raise httpx.ConnectError("repeated simulated resets")
            return _FakeResponse(
                200,
                {"user_code": "TEST-CODE", "device_auth_id": "daid", "interval": "3"},
            )

    _FakeClient.post_calls = []
    monkeypatch.setattr(auth_mod.httpx, "Client", _FlakyClient)
    monkeypatch.setattr("time.sleep", lambda _: None)

    # Drive the real-time max_wait (15 min) to expiry after one loop pass
    # instead of letting the test spin for 15 minutes.
    real_mono = time.monotonic
    anchor: dict = {"value": None}

    def fake_monotonic():
        if anchor["value"] is None:
            anchor["value"] = real_mono()
            return anchor["value"]
        return anchor["value"] + 60 * 16  # beyond max_wait

    monkeypatch.setattr("time.monotonic", fake_monotonic)

    with pytest.raises(auth_mod.AuthError) as exc_info:
        auth_mod._codex_device_code_login()

    # Persistent network failure -> clean, actionable timeout error,
    # NOT a raw httpx exception.
    assert exc_info.value.code == "device_code_timeout"
