"""Provider identity is display metadata; pool IDs and explicit labels remain authoritative."""
import asyncio
import base64
import json
from types import SimpleNamespace

import httpx

from agent.credential_pool import label_from_token, load_pool
from hermes_cli import auth_codex, auth_commands
from hermes_cli.web_routers import oauth, ops


def jwt(claims):
    payload = base64.urlsafe_b64encode(json.dumps(claims).encode()).decode().rstrip("=")
    return f"header.{payload}.signature"


def test_codex_identity_survives_exchange_and_pool_reload(tmp_path, monkeypatch, caplog, capsys):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    tokens = {"access_token": jwt({"sub": "account-a"}), "refresh_token": "refresh-a",
              "id_token": jwt({"email": "first@example.test"})}
    real_client = httpx.Client
    monkeypatch.setattr(httpx, "Client", lambda **kw: real_client(
        transport=httpx.MockTransport(lambda request: httpx.Response(200, json=tokens)), **kw))
    exchanged = oauth._codex_exchange_tokens(httpx, {"authorization_code": "code", "code_verifier": "verifier"})
    monkeypatch.setattr(auth_codex, "_codex_request_device_code", lambda *args: {
        "user_code": "CODE", "device_auth_id": "device", "interval": 1})
    monkeypatch.setattr(auth_codex, "_codex_poll_authorization_code", lambda *args, **kw: {})
    monkeypatch.setattr(auth_codex, "_codex_exchange_authorization_code", lambda *args: exchanged)
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

    # Refresh rotates secrets, not identity or the native account selector.
    def refresh(access_token, refresh_token, **kwargs):
        assert access_token == first.access_token
        assert refresh_token == first.refresh_token
        return {"access_token": "rotated-access-secret", "refresh_token": "rotated-refresh-secret"}

    monkeypatch.setattr(auth_commands.auth_mod, "refresh_codex_oauth_pure", refresh)
    monkeypatch.setattr(auth_codex, "refresh_codex_oauth_pure", refresh)
    refreshed = pool._refresh_entry(first, force=True)
    assert refreshed is not None
    assert (refreshed.id, refreshed.label) == (first.id, first.label)
    assert refreshed.access_token == "rotated-access-secret"
    assert load_pool("openai-codex").entries()[0].label == first.label

    # Disconnect/reconnect one login leaves the other account and its custom name intact.
    assert pool.remove_index(1).id == first.id
    exchanged.update(access_token=jwt({"sub": "account-a-reconnected"}),
                     refresh_token="reconnected-refresh-secret",
                     id_token=jwt({"email": first.label}))
    reconnected = auth_commands._add_credential(SimpleNamespace(label=None), "openai-codex", pool, "oauth")
    assert reconnected.id != first.id and reconnected.label == first.label
    listed = asyncio.run(ops.list_credential_pool())
    rows = next(provider["entries"] for provider in listed["providers"] if provider["provider"] == "openai-codex")
    assert {row["id"]: row["label"] for row in rows} == {
        second.id: "Work account", reconnected.id: first.label}
    visible = json.dumps(listed) + caplog.text + capsys.readouterr().out
    for secret in (first.access_token, first.refresh_token, second.access_token,
                   second.refresh_token, refreshed.access_token, refreshed.refresh_token,
                   reconnected.access_token, reconnected.refresh_token, exchanged["id_token"]):
        assert secret not in visible
    assert all("id_token" not in row and "access_token" not in row and "refresh_token" not in row for row in rows)


def test_identity_claims_are_optional_and_never_used_as_account_ids():
    assert label_from_token(jwt({"https://api.openai.com/profile": {"email": " a@example.test "}}), "fallback") == "a@example.test"
    for claims in ({}, [], {"email": 123}, {"https://api.openai.com/profile": "invalid"}, {"email": "  "}):
        assert label_from_token(jwt(claims), "fallback", id_token="malformed") == "fallback"
    assert label_from_token("opaque", "fallback") == "fallback"
    assert label_from_token(jwt({"email": "access@example.test"}), "fallback",
                            id_token=jwt({"email": "identity@example.test"})) == "identity@example.test"
