"""Unit tests for the generic-OIDC / Nous-Portal caller-identity token resolver.

Covers gateway.relay._resolve_relay_identity_token() — the canonical resolver
shared by the runtime self-provision path and the `hermes gateway enroll` CLI.

Two modes:
  1. Generic OAuth2 client_credentials when gateway.idp.token_url (or
     GATEWAY_RELAY_IDP_TOKEN_URL) is configured (air-gapped / self-hosted-IdP).
  2. Nous Portal (resolve_nous_access_token) otherwise — the default.

The HTTP POST and the Nous resolver are monkeypatched; these prove the mode
SELECTION, the client_credentials request shape, and the fail-closed paths.
"""

from __future__ import annotations

import io
import json

import pytest

import gateway.relay as relay


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for k in (
        "GATEWAY_RELAY_IDP_TOKEN_URL",
        "GATEWAY_RELAY_IDP_CLIENT_ID",
        "GATEWAY_RELAY_IDP_CLIENT_SECRET",
        "GATEWAY_RELAY_IDP_SCOPE",
    ):
        monkeypatch.delenv(k, raising=False)
    # Never read config.yaml off disk by default.
    monkeypatch.setattr("gateway.run._load_gateway_config", lambda: {}, raising=False)


def test_defaults_to_nous_portal_when_no_idp_configured(monkeypatch):
    called = {}

    def fake_resolve():
        called["yes"] = True
        return "nous-portal-token"

    monkeypatch.setattr(
        "hermes_cli.auth.resolve_nous_access_token", fake_resolve, raising=False
    )
    assert relay._resolve_relay_identity_token() == "nous-portal-token"
    assert called == {"yes": True}


def test_client_credentials_via_env(monkeypatch):
    monkeypatch.setenv("GATEWAY_RELAY_IDP_TOKEN_URL", "https://idp.test/token")
    monkeypatch.setenv("GATEWAY_RELAY_IDP_CLIENT_ID", "agent-client")
    monkeypatch.setenv("GATEWAY_RELAY_IDP_CLIENT_SECRET", "shh")
    monkeypatch.setenv("GATEWAY_RELAY_IDP_SCOPE", "connector.provision")

    captured = {}

    def fake_urlopen(req, timeout=None):
        captured["url"] = req.full_url
        captured["method"] = req.get_method()
        captured["body"] = req.data.decode()
        captured["headers"] = {k.lower(): v for k, v in req.headers.items()}
        return io.BytesIO(json.dumps({"access_token": "idp-workload-token"}).encode())

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    token = relay._resolve_relay_identity_token()
    assert token == "idp-workload-token"
    assert captured["url"] == "https://idp.test/token"
    assert captured["method"] == "POST"
    # client_credentials grant, form-encoded, with all fields.
    assert "grant_type=client_credentials" in captured["body"]
    assert "client_id=agent-client" in captured["body"]
    assert "client_secret=shh" in captured["body"]
    assert "scope=connector.provision" in captured["body"]
    assert captured["headers"]["content-type"] == "application/x-www-form-urlencoded"


def test_client_credentials_via_config_yaml(monkeypatch):
    monkeypatch.setattr(
        "gateway.run._load_gateway_config",
        lambda: {
            "gateway": {
                "idp": {
                    "token_url": "https://idp.test/token",
                    "client_id": "cfg-client",
                    "client_secret": "cfg-secret",
                }
            }
        },
        raising=False,
    )

    def fake_urlopen(req, timeout=None):
        body = req.data.decode()
        assert "client_id=cfg-client" in body
        assert "client_secret=cfg-secret" in body
        # No scope configured -> not sent.
        assert "scope=" not in body
        return io.BytesIO(json.dumps({"access_token": "cfg-token"}).encode())

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    assert relay._resolve_relay_identity_token() == "cfg-token"


def test_env_token_url_takes_precedence_over_config(monkeypatch):
    monkeypatch.setenv("GATEWAY_RELAY_IDP_TOKEN_URL", "https://env.test/token")
    monkeypatch.setenv("GATEWAY_RELAY_IDP_CLIENT_ID", "env-client")
    monkeypatch.setenv("GATEWAY_RELAY_IDP_CLIENT_SECRET", "env-secret")
    monkeypatch.setattr(
        "gateway.run._load_gateway_config",
        lambda: {"gateway": {"idp": {"token_url": "https://cfg.test/token"}}},
        raising=False,
    )

    def fake_urlopen(req, timeout=None):
        assert req.full_url == "https://env.test/token"
        return io.BytesIO(json.dumps({"access_token": "t"}).encode())

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    assert relay._resolve_relay_identity_token() == "t"


def test_raises_when_client_creds_missing(monkeypatch):
    monkeypatch.setenv("GATEWAY_RELAY_IDP_TOKEN_URL", "https://idp.test/token")
    # No client_id / client_secret.
    with pytest.raises(RuntimeError, match="client_id/client_secret missing"):
        relay._resolve_relay_identity_token()


def test_raises_when_no_access_token_in_response(monkeypatch):
    monkeypatch.setenv("GATEWAY_RELAY_IDP_TOKEN_URL", "https://idp.test/token")
    monkeypatch.setenv("GATEWAY_RELAY_IDP_CLIENT_ID", "c")
    monkeypatch.setenv("GATEWAY_RELAY_IDP_CLIENT_SECRET", "s")

    def fake_urlopen(req, timeout=None):
        return io.BytesIO(json.dumps({"token_type": "Bearer"}).encode())

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    with pytest.raises(RuntimeError, match="no access_token"):
        relay._resolve_relay_identity_token()
