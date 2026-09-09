"""Provider identity is display metadata; pool IDs and explicit labels remain authoritative."""
import base64
import json
from types import SimpleNamespace

import httpx

from agent.credential_pool import label_from_token, load_pool
from hermes_cli import auth_commands
from hermes_cli.web_routers import oauth


def jwt(claims):
    payload = base64.urlsafe_b64encode(json.dumps(claims).encode()).decode().rstrip("=")
    return f"header.{payload}.signature"


def test_codex_identity_survives_exchange_and_pool_reload(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    tokens = {"access_token": jwt({"sub": "account-a"}), "refresh_token": "refresh-a",
              "id_token": jwt({"email": "first@example.test"})}
    real_client = httpx.Client
    monkeypatch.setattr(httpx, "Client", lambda **kw: real_client(
        transport=httpx.MockTransport(lambda request: httpx.Response(200, json=tokens)), **kw))
    exchanged = oauth._codex_exchange_tokens(httpx, {"authorization_code": "code", "code_verifier": "verifier"})
    monkeypatch.setattr(auth_commands.auth_mod, "_codex_device_code_login", lambda: {"tokens": exchanged})
    pool = load_pool("openai-codex")
    first = auth_commands._add_credential(SimpleNamespace(label=None), "openai-codex", pool, "oauth")
    assert first.label == "first@example.test"
    tokens.update(access_token=jwt({"https://api.openai.com/profile": {"email": "second@example.test"}}),
                  refresh_token="refresh-b", id_token="")
    exchanged.update(tokens)
    second = auth_commands._add_credential(SimpleNamespace(label="Work account"), "openai-codex", pool, "oauth")
    reloaded = {entry.id: entry for entry in load_pool("openai-codex").entries()}
    assert first.id != second.id
    assert reloaded[first.id].label == first.label
    assert reloaded[second.id].label == "Work account"
    assert reloaded[first.id].access_token == first.access_token
    assert reloaded[second.id].access_token == tokens["access_token"]


def test_identity_claims_are_optional_and_never_used_as_account_ids():
    assert label_from_token(jwt({"https://api.openai.com/profile": {"email": " a@example.test "}}), "fallback") == "a@example.test"
    for claims in ({}, [], {"email": 123}, {"https://api.openai.com/profile": "invalid"}, {"email": "  "}):
        assert label_from_token(jwt(claims), "fallback", id_token="malformed") == "fallback"
    assert label_from_token("opaque", "fallback") == "fallback"
    assert label_from_token(jwt({"email": "access@example.test"}), "fallback",
                            id_token=jwt({"email": "identity@example.test"})) == "identity@example.test"
