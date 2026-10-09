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


def test_raises_when_no_access_token_in_response(monkeypatch):
    monkeypatch.setenv("GATEWAY_RELAY_IDP_TOKEN_URL", "https://idp.test/token")
    monkeypatch.setenv("GATEWAY_RELAY_IDP_CLIENT_ID", "c")
    monkeypatch.setenv("GATEWAY_RELAY_IDP_CLIENT_SECRET", "s")

    def fake_urlopen(req, timeout=None):
        return io.BytesIO(json.dumps({"token_type": "Bearer"}).encode())

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    with pytest.raises(RuntimeError, match="no access_token"):
        relay._resolve_relay_identity_token()
