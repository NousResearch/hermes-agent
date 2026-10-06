"""Unit tests for the generic-OIDC / Nous-Portal caller-identity token resolver.

Covers gateway.relay._resolve_relay_identity_token() — the canonical resolver
shared by the runtime self-provision path and the `hermes gateway enroll` CLI.

Three modes:
  1. Generic OAuth2 client_credentials when gateway.idp.token_url (or
     GATEWAY_RELAY_IDP_TOKEN_URL) is configured WITH client credentials
     (air-gapped / self-hosted-IdP).
  1b. Ambient token endpoint when token_url is configured WITHOUT client
     credentials: plain GET, body is the token (raw JWT or JSON envelope).
     The metadata-server pattern (e.g. Domino's $DOMINO_API_PROXY/access-token).
  2. Nous Portal (resolve_nous_access_token) otherwise — the default.

The HTTP calls and the Nous resolver are monkeypatched; these prove the mode
SELECTION, the request shapes, and the fail-closed paths.
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


# ---------------------------------------------------------------------------
# Mode 1b — ambient token endpoint (token_url set, NO client credentials).
# The metadata-server pattern: plain GET, response body IS the token, either
# a raw JWT string (Domino's $DOMINO_API_PROXY/access-token) or a JSON
# envelope ({"access_token": ...}).
# ---------------------------------------------------------------------------

_FAKE_JWT = "eyJhbGciOiJSUzI1NiJ9.eyJzdWIiOiJ1c2VyIn0.c2ln"


def test_ambient_get_when_no_client_credentials(monkeypatch):
    """token_url without client_id/secret selects a plain GET, not a raise."""
    monkeypatch.setenv("GATEWAY_RELAY_IDP_TOKEN_URL", "https://proxy.local/access-token")

    captured = {}

    def fake_urlopen(req, timeout=None):
        captured["url"] = req.full_url
        captured["method"] = req.get_method()
        captured["body"] = req.data
        return io.BytesIO((_FAKE_JWT + "\n").encode())

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    token = relay._resolve_relay_identity_token()
    assert token == _FAKE_JWT  # raw body, whitespace-trimmed
    assert captured["url"] == "https://proxy.local/access-token"
    assert captured["method"] == "GET"
    assert captured["body"] is None  # no form payload on the ambient path


def test_ambient_accepts_json_envelope(monkeypatch):
    """Ambient endpoints that return {"access_token": ...} JSON also work."""
    monkeypatch.setenv("GATEWAY_RELAY_IDP_TOKEN_URL", "https://proxy.local/access-token")

    def fake_urlopen(req, timeout=None):
        return io.BytesIO(json.dumps({"access_token": _FAKE_JWT}).encode())

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    assert relay._resolve_relay_identity_token() == _FAKE_JWT


@pytest.mark.parametrize(
    "envelope",
    [
        {"access_token": 12345678901234567890123456789012},  # number, not string
        {"access_token": True},  # boolean: str() would coerce to 'True'
        {"access_token": {"nested": "x"}},  # object
        {"access_token": ""},  # empty string
        {"access_token": None},
        {"token": "wrong-field-name"},
    ],
)
def test_ambient_rejects_non_string_envelope_values(monkeypatch, envelope):
    """access_token in a JSON envelope must be a non-empty STRING — the same
    contract as the client_credentials path. No str() coercion of numbers,
    booleans, or objects into 'tokens'."""
    monkeypatch.setenv("GATEWAY_RELAY_IDP_TOKEN_URL", "https://proxy.local/access-token")

    def fake_urlopen(req, timeout=None):
        return io.BytesIO(json.dumps(envelope).encode())

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    with pytest.raises(RuntimeError, match="ambient"):
        relay._resolve_relay_identity_token()


def test_ambient_rejects_non_token_body(monkeypatch):
    """A body that is neither a JWT-ish string nor a token envelope fails loudly."""
    monkeypatch.setenv("GATEWAY_RELAY_IDP_TOKEN_URL", "https://proxy.local/access-token")

    def fake_urlopen(req, timeout=None):
        return io.BytesIO(b"<html>404 not found</html>")

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    with pytest.raises(RuntimeError, match="ambient"):
        relay._resolve_relay_identity_token()


def test_ambient_rejects_empty_body(monkeypatch):
    monkeypatch.setenv("GATEWAY_RELAY_IDP_TOKEN_URL", "https://proxy.local/access-token")

    def fake_urlopen(req, timeout=None):
        return io.BytesIO(b"   \n")

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    with pytest.raises(RuntimeError, match="ambient"):
        relay._resolve_relay_identity_token()


def test_ambient_rejects_short_plaintext_error_words(monkeypatch):
    """A terse plain-text error body (e.g. 'unauthorized') must not be
    returned as a credential — it matches the base64url alphabet but is
    neither a JWT nor plausibly an opaque token."""
    monkeypatch.setenv("GATEWAY_RELAY_IDP_TOKEN_URL", "https://proxy.local/access-token")

    for body in ("unauthorized", "error", "access_denied", "null", "forbidden"):
        def fake_urlopen(req, timeout=None, _b=body):
            return io.BytesIO(_b.encode())

        monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
        with pytest.raises(RuntimeError, match="ambient"):
            relay._resolve_relay_identity_token()


def test_ambient_accepts_long_opaque_token(monkeypatch):
    """Non-JWT opaque bearer tokens (long random strings) still work."""
    monkeypatch.setenv("GATEWAY_RELAY_IDP_TOKEN_URL", "https://proxy.local/access-token")
    opaque = "v2_9a8b7c6d5e4f3a2b1c0d9e8f7a6b5c4d3e2f1a0b"

    def fake_urlopen(req, timeout=None):
        return io.BytesIO(opaque.encode())

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    assert relay._resolve_relay_identity_token() == opaque


def test_client_credentials_still_selected_when_creds_present(monkeypatch):
    """Presence of client creds keeps the POST grant — ambient never hijacks it."""
    monkeypatch.setenv("GATEWAY_RELAY_IDP_TOKEN_URL", "https://idp.test/token")
    monkeypatch.setenv("GATEWAY_RELAY_IDP_CLIENT_ID", "agent-client")
    monkeypatch.setenv("GATEWAY_RELAY_IDP_CLIENT_SECRET", "shh")

    captured = {}

    def fake_urlopen(req, timeout=None):
        captured["method"] = req.get_method()
        return io.BytesIO(json.dumps({"access_token": "cc-token"}).encode())

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    assert relay._resolve_relay_identity_token() == "cc-token"
    assert captured["method"] == "POST"


def test_partial_credentials_client_id_only_raises_without_get(monkeypatch):
    """client_id without client_secret is a misconfig: loud error, no ambient GET."""
    monkeypatch.setenv("GATEWAY_RELAY_IDP_TOKEN_URL", "https://idp.test/token")
    monkeypatch.setenv("GATEWAY_RELAY_IDP_CLIENT_ID", "agent-client")

    def fake_urlopen(req, timeout=None):
        raise AssertionError("no HTTP request may be issued on partial credentials")

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    with pytest.raises(RuntimeError, match="client_secret missing"):
        relay._resolve_relay_identity_token()


def test_partial_credentials_client_secret_only_raises_without_get(monkeypatch):
    """client_secret without client_id is a misconfig: loud error, no ambient GET."""
    monkeypatch.setenv("GATEWAY_RELAY_IDP_TOKEN_URL", "https://idp.test/token")
    monkeypatch.setenv("GATEWAY_RELAY_IDP_CLIENT_SECRET", "shh")

    def fake_urlopen(req, timeout=None):
        raise AssertionError("no HTTP request may be issued on partial credentials")

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    with pytest.raises(RuntimeError, match="client_id missing"):
        relay._resolve_relay_identity_token()


def test_ambient_via_config_yaml(monkeypatch):
    """Ambient mode also engages when token_url comes from config.yaml, not env."""
    monkeypatch.setattr(
        "gateway.run._load_gateway_config",
        lambda: {"gateway": {"idp": {"token_url": "https://proxy.local/access-token"}}},
        raising=False,
    )

    def fake_urlopen(req, timeout=None):
        assert req.get_method() == "GET"
        return io.BytesIO(_FAKE_JWT.encode())

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    assert relay._resolve_relay_identity_token() == _FAKE_JWT
