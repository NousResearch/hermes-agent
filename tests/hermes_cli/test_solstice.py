"""Tests for the Solstice OAuth flow (PKCE loopback, NAS-brokered exchange).

Verifies the enable gate, PKCE pair generation, token store round-trip,
NAS exchange/refresh contract, runtime resolution, and the login flow
against a fake NAS broker.
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler
from socketserver import TCPServer
from types import SimpleNamespace
from unittest import mock

import pytest

import hermes_cli.solstice as aa


class _FakeNAS:
    """Fake NAS broker: records exchange/refresh bodies, returns canned tokens."""

    def __init__(self, exchange_payload=None, refresh_payload=None):
        self.exchange_body = None
        self.refresh_body = None
        self.config_payload = {
            "client_id": "solstice-client-123",
            "authorize_url": "https://accounts.google.com/o/oauth2/v2/auth",
            "scope": "openid email profile https://www.googleapis.com/auth/generative-language.peruserquota",
        }
        self.exchange_payload = exchange_payload or {
            "access_token": "at-1",
            "refresh_token": "rt-1",
            "expires_in": 3600,
            "token_type": "Bearer",
        }
        self.refresh_payload = refresh_payload or {
            "access_token": "at-2",
            "refresh_token": "rt-2",
            "expires_in": 3600,
        }

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):  # noqa: N802
                if self.path == aa.SOLSTICE_NAS_CONFIG_PATH:
                    data = json.dumps(outer.config_payload).encode()
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(data)))
                    self.end_headers()
                    self.wfile.write(data)
                else:
                    self.send_response(404)
                    self.end_headers()

            def do_POST(self):  # noqa: N802
                body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
                obj = json.loads(body)
                if self.path == aa.SOLSTICE_NAS_EXCHANGE_PATH:
                    outer.exchange_body = obj
                    payload = outer.exchange_payload
                elif self.path == aa.SOLSTICE_NAS_REFRESH_PATH:
                    outer.refresh_body = obj
                    payload = outer.refresh_payload
                else:
                    self.send_response(404)
                    self.end_headers()
                    return
                data = json.dumps(payload).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def log_message(self, format, *args):  # noqa: A002
                return

        self.server = TCPServer(("127.0.0.1", 0), Handler)
        outer = self
        self.port = self.server.server_address[1]
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def close(self):
        self.server.shutdown()
        self.server.server_close()


@pytest.fixture
def fake_nas():
    srv = _FakeNAS()
    yield srv
    srv.close()


@pytest.fixture
def broker_env(fake_nas, monkeypatch):
    """Point the broker at the fake NAS and skip the HTTPS guard (test-only)."""
    monkeypatch.setattr(aa, "solstice_nas_base_url", lambda: fake_nas.base_url)
    monkeypatch.setattr(aa, "_validate_broker_url", lambda url: url.rstrip("/"))
    # NAS broker calls require a valid Nous token; provide a fake one.
    monkeypatch.setattr(
        aa, "_nous_bearer_header", lambda: {"Authorization": "Bearer fake-nous-token"}
    )
    return fake_nas


@pytest.fixture
def clean_home(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    yield


def test_enable_gate(monkeypatch):
    # Provider is enabled by default; no local client_id gate anymore
    # (NAS owns the client config, discovered at login).
    monkeypatch.delenv("SOLSTICE_CLIENT_ID", raising=False)
    assert aa.solstice_enabled() is True


def test_client_id_comes_from_nas_discovery(monkeypatch):
    # The login flow fetches client config from NAS discovery, not local env.
    monkeypatch.delenv("SOLSTICE_CLIENT_ID", raising=False)
    assert hasattr(aa, "_fetch_solstice_config")


def test_pkce_pair_shape():
    verifier, challenge, state = aa._solstice_pkce_pair()
    assert verifier and challenge and state
    assert "." not in challenge  # base64url, no padding
    assert len(verifier) >= 43


def test_broker_rejects_plain_http():
    with pytest.raises(aa._AuthError, match="HTTPS"):
        aa._validate_broker_url("http://127.0.0.1:1234")


def test_exchange_contract(broker_env, fake_nas):
    payload = aa._exchange_code("code-xyz", "verifier-abc", "http://127.0.0.1:9999/cb")
    assert payload["access_token"] == "at-1"
    assert fake_nas.exchange_body == {
        "code": "code-xyz",
        "code_verifier": "verifier-abc",
        "redirect_uri": "http://127.0.0.1:9999/cb",
    }


def test_token_store_round_trip(clean_home, monkeypatch):
    monkeypatch.setenv("SOLSTICE_CLIENT_ID", "client-123")
    tokens = {
        "access_token": "at-1",
        "refresh_token": "rt-1",
        "token_type": "Bearer",
        "expires_in": 3600,
    }
    aa._save_solstice_tokens(tokens, redirect_uri="http://127.0.0.1:9999/cb")
    stored = aa._read_solstice_tokens()
    assert stored["tokens"]["access_token"] == "at-1"
    assert stored["redirect_uri"] == "http://127.0.0.1:9999/cb"
    assert stored["auth_mode"] == "oauth_pkce"


def test_refresh_contract(broker_env, fake_nas):
    rotated = aa._refresh_tokens("rt-1")
    assert rotated["access_token"] == "at-2"
    assert fake_nas.refresh_body == {"refresh_token": "rt-1"}


def test_runtime_resolution_refreshes_via_broker(clean_home, broker_env, monkeypatch):
    monkeypatch.setenv("SOLSTICE_CLIENT_ID", "client-123")
    tokens = {
        "access_token": "at-1",
        "refresh_token": "rt-1",
        "token_type": "Bearer",
        "expires_in": 3600,
    }
    aa._save_solstice_tokens(tokens)
    creds = aa.resolve_solstice_runtime_credentials(force_refresh=True)
    assert creds["api_key"] == "at-2"
    assert creds["base_url"] == "https://generativelanguage.googleapis.com/v1alpha"
    assert creds["api_mode"] == "chat_completions"
    assert creds["provider"] == "solstice"


def test_login_flow(clean_home, broker_env, monkeypatch):
    """End-to-end login: PKCE pair → loopback callback → NAS exchange → store."""
    monkeypatch.setenv("SOLSTICE_CLIENT_ID", "client-123")

    state_holder: dict = {}
    real_pair = aa._solstice_pkce_pair

    def fake_pair():
        v, c, s = real_pair()
        state_holder["expected_state"] = s
        return v, c, s

    def fake_wait(port, path, *, timeout_seconds=180.0):
        return {
            "code": "auth-code-123",
            "state": state_holder["expected_state"],
            "error": None,
        }

    args = SimpleNamespace(no_browser=True, timeout=30.0)
    with (
        mock.patch.object(aa, "_solstice_pkce_pair", fake_pair),
        mock.patch.object(aa, "_wait_for_loopback_callback", fake_wait),
    ):
        aa._login_solstice(args, None)

    creds = aa.resolve_solstice_runtime_credentials(refresh_if_expiring=False)
    assert creds["api_key"] == "at-1"


def test_login_rejects_state_mismatch(clean_home, broker_env, monkeypatch):
    monkeypatch.setenv("SOLSTICE_CLIENT_ID", "client-123")

    def fake_wait(port, path, *, timeout_seconds=180.0):
        return {"code": "x", "state": "wrong-state", "error": None}

    args = SimpleNamespace(no_browser=True, timeout=30.0)
    with mock.patch.object(aa, "_wait_for_loopback_callback", fake_wait):
        with pytest.raises(aa._AuthError, match="state mismatch"):
            aa._login_solstice(args, None)


def test_status_reflects_gate(monkeypatch):
    monkeypatch.delenv("SOLSTICE_CLIENT_ID", raising=False)
    status = aa.get_solstice_status()
    # Provider is always configured now (NAS owns the client config); logged_in
    # reflects whether tokens are stored.
    assert status["logged_in"] is False
    assert status["configured"] is True


def test_runtime_resolution_quarantines_on_terminal_refresh_failure(
    clean_home, monkeypatch
):
    """A terminal refresh failure clears dead tokens and marks relogin required."""
    monkeypatch.setenv("SOLSTICE_CLIENT_ID", "client-123")
    tokens = {
        "access_token": "at-1",  # not a JWT → would not auto-refresh; use force
        "refresh_token": "rt-expired",
        "token_type": "Bearer",
        "expires_in": 3600,
    }
    aa._save_solstice_tokens(tokens)

    # Make the broker return a 400 invalid_grant on refresh.
    class _FailingNAS:
        base_url = None

        def __init__(self):
            import socketserver

            class H(BaseHTTPRequestHandler):
                def do_POST(self):  # noqa: N802
                    self.rfile.read(int(self.headers.get("Content-Length", 0)))
                    body = b'{"error": "invalid_grant"}'
                    self.send_response(400)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)

                def log_message(self, format, *args):  # noqa: A002
                    return

            self.server = socketserver.TCPServer(("127.0.0.1", 0), H)
            self.base_url = f"http://127.0.0.1:{self.server.server_address[1]}"
            threading.Thread(target=self.server.serve_forever, daemon=True).start()

        def close(self):
            self.server.shutdown()
            self.server.server_close()

    nas = _FailingNAS()
    monkeypatch.setattr(aa, "solstice_nas_base_url", lambda: nas.base_url)
    monkeypatch.setattr(aa, "_validate_broker_url", lambda url: url.rstrip("/"))
    monkeypatch.setattr(
        aa, "_nous_bearer_header", lambda: {"Authorization": "Bearer fake-nous-token"}
    )

    from hermes_cli.auth import AuthError

    with pytest.raises(AuthError) as excinfo:
        aa.resolve_solstice_runtime_credentials(force_refresh=True)
    assert excinfo.value.relogin_required is True
    nas.close()

    # Dead tokens cleared from the store → next resolution fails fast.
    creds = aa.resolve_solstice_runtime_credentials()
    assert creds["api_key"] == ""  # quarantined: no usable token remains
