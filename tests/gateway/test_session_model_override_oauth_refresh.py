"""A live xAI /model override follows same-account rotations, including its keyed fast path.

Regressions for #108122 and #108197: exercise the turn resolver and actual auth store,
not only the override-overlay helper that the normal keyed path bypasses.
"""

from __future__ import annotations

import base64
import json
from pathlib import Path

import httpx
import pytest
from openai import OpenAI

from gateway.run import GatewayRunner, _profile_runtime_scope


_XAI_URL = "https://api.x.ai/v1"


def _token(account: str, generation: int, *, expires_at: int = 4_102_444_800) -> str:
    def encode(value: dict) -> str:
        return base64.urlsafe_b64encode(json.dumps(value).encode()).decode().rstrip("=")

    return ".".join((encode({"alg": "none"}), encode({
        "sub": account, "exp": expires_at, "fixture_generation": generation,
    }), "fixture-signature"))


def _write_auth(home: Path, token: str) -> None:
    home.mkdir(parents=True, exist_ok=True)
    (home / "config.yaml").write_text(
        "model:\n  provider: xai-oauth\n  default: grok-default\n", encoding="utf-8",
    )
    (home / "auth.json").write_text(json.dumps({
        "version": 1,
        "providers": {"xai-oauth": {
            "auth_mode": "oauth_device_code",
            "tokens": {"access_token": token, "refresh_token": "fixture-refresh"},
        }},
        "credential_pool": {"xai-oauth": [{
            "id": "fixture-account", "label": "fixture-account", "auth_type": "oauth",
            "source": "device_code", "priority": 0, "access_token": token,
            "refresh_token": "fixture-refresh", "base_url": _XAI_URL,
        }]},
    }), encoding="utf-8")


def _override(token: str) -> dict:
    return {
        "model": "grok-selected", "provider": "xai-oauth", "requested_provider": "xai-oauth",
        "api_key": token, "base_url": _XAI_URL, "api_mode": "codex_responses",
        "request_overrides": {"store": False}, "capabilities": {"vision": True},
        "max_tokens": 1234,
    }


def _runner(overrides: dict) -> GatewayRunner:
    runner = object.__new__(GatewayRunner)
    runner.session_store = None
    runner.config = None
    runner._session_model_overrides = overrides
    return runner


def _request_authorization(runtime: dict, model: str) -> str:
    authorization = []

    def respond(request: httpx.Request) -> httpx.Response:
        authorization.append(request.headers["authorization"])
        assert request.url == f"{_XAI_URL}/responses"
        assert json.loads(request.content)["model"] == model
        return httpx.Response(200, json={
            "id": "resp_fixture", "object": "response", "created_at": 0,
            "status": "completed", "model": model, "output": [],
        })

    with httpx.Client(transport=httpx.MockTransport(respond)) as transport:
        with OpenAI(api_key=runtime["api_key"], base_url=runtime["base_url"],
                    http_client=transport, max_retries=0) as client:
            client.responses.create(model=model, input="fixture request")
    return authorization[0]


def test_keyed_xai_override_adopts_each_rotation_from_its_own_profile(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    profiles = {account: tmp_path / account for account in ("account-a", "account-b")}
    runner = _runner({account: _override(_token(account, 0, expires_at=1)) for account in profiles})
    signatures = []
    for account, generation in (("account-a", 1), ("account-b", 1), ("account-a", 2)):
        token = _token(account, generation)
        home = profiles[account]
        _write_auth(home, token)
        with _profile_runtime_scope(home):
            model, runtime = runner._resolve_session_agent_runtime(session_key=account)

        assert model == "grok-selected"
        assert runtime["api_key"] == token
        assert runtime["provider"] == runtime["requested_provider"] == "xai-oauth"
        assert runtime["api_mode"] == "codex_responses"
        assert runtime["request_overrides"] == {"store": False}
        assert runtime["capabilities"] == {"vision": True}
        assert runtime["max_tokens"] == 1234
        assert runtime["credential_pool"].entry_id_for_api_key(token) == "fixture-account"
        assert runner._session_model_overrides[account]["api_key"] == token
        assert _request_authorization(runtime, model) == f"Bearer {token}"
        if account == "account-a":
            signatures.append(runner._agent_config_signature(model, runtime, [], ""))

    assert signatures[0] != signatures[1], "a cached client must not survive under the old bearer signature"
    assert runner._session_model_overrides["account-b"]["api_key"] == _token("account-b", 1)


def _empty_auth(home: Path) -> None:
    (home / "auth.json").write_text("{}", encoding="utf-8")


def _replace_pool_account(home: Path) -> None:
    auth = json.loads((home / "auth.json").read_text(encoding="utf-8"))
    auth["credential_pool"]["xai-oauth"][0].update(
        source="manual:device_code", access_token=_token("account-b", 1),
    )
    (home / "auth.json").write_text(json.dumps(auth), encoding="utf-8")


@pytest.mark.parametrize("store_account,override_changes,mutate_auth", [
    pytest.param("account-b", {}, None, id="different-account"),
    pytest.param("account-a", {"api_key": "fixture-static-key"}, None, id="opaque-explicit-key"),
    pytest.param("account-a", {"base_url": "https://proxy.invalid/v1"}, None, id="foreign-endpoint"),
    pytest.param("account-a", {"base_url": "http://api.x.ai/v1"}, None, id="plaintext-endpoint"),
    pytest.param("account-a", {"provider": "xai", "requested_provider": "xai", "api_mode": "chat_completions"},
                 None, id="static-provider"),
    pytest.param("account-a", {"requested_provider": "custom:private-xai"}, None, id="custom-requested-provider"),
    pytest.param("account-a", {}, _empty_auth, id="missing-auth"),
    pytest.param("account-a", {}, _replace_pool_account, id="pool-other-account"),
])
def test_xai_override_preserves_unowned_or_unavailable_credentials(
    tmp_path, monkeypatch, store_account, override_changes, mutate_auth,
):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    home = tmp_path / "profile"
    _write_auth(home, _token(store_account, 1))
    if mutate_auth is not None:
        mutate_auth(home)
    override = {**_override(_token("account-a", 0)), **override_changes, "credential_pool": object()}
    snapshot = dict(override)
    runner = _runner({"session": override})

    with _profile_runtime_scope(home):
        model, runtime = runner._resolve_session_agent_runtime(session_key="session")

    assert model == "grok-selected"
    assert {key: runtime[key] for key in snapshot if key != "model"} == {
        key: value for key, value in snapshot.items() if key != "model"
    }
    assert runner._session_model_overrides["session"] == snapshot
