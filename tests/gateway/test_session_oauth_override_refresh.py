"""Native OAuth /model overrides follow the live profile credential."""

from __future__ import annotations

import base64
import json
from pathlib import Path
from unittest.mock import patch

import httpx
import pytest
from openai import OpenAI

from agent.secret_scope import set_multiplex_active
from gateway.run import GatewayRunner, _profile_runtime_scope


_CODEX_URL = "https://chatgpt.com/backend-api/codex"


def _fake_codex_token(account: str, version: str) -> str:
    def encode(payload: dict) -> str:
        raw = json.dumps(payload, separators=(",", ":")).encode()
        return base64.urlsafe_b64encode(raw).rstrip(b"=").decode()

    return ".".join(
        (
            encode({"alg": "none"}),
            encode(
                {
                    "sub": f"subject-{account}",
                    "exp": 4_102_444_800,
                    "https://api.openai.com/auth": {"chatgpt_account_id": account},
                    "fixture_version": version,
                }
            ),
            "fake-signature",
        )
    )


def _write_codex_pool(home: Path, token: str) -> None:
    home.mkdir(parents=True, exist_ok=True)
    (home / "config.yaml").write_text(
        "model:\n  provider: openai-codex\n  default: gpt-fixture-codex\n",
        encoding="utf-8",
    )
    (home / "auth.json").write_text(
        json.dumps(
            {
                "version": 1,
                "credential_pool": {
                    "openai-codex": [
                        {
                            "id": "fixture-account",
                            "label": "fixture-account",
                            "auth_type": "oauth",
                            "priority": 0,
                            "source": "manual:device_code",
                            "access_token": token,
                            "refresh_token": "fake-refresh-token",
                            "base_url": _CODEX_URL,
                        }
                    ]
                },
            }
        ),
        encoding="utf-8",
    )


def _override(stale_token: str) -> dict:
    return {
        "model": "gpt-fixture-codex",
        "provider": "openai-codex",
        "api_key": stale_token,
        "base_url": _CODEX_URL,
        "api_mode": "codex_responses",
        "max_tokens": 12_345,
        "request_overrides": {"store": False},
        "capabilities": {"openai_native_compaction": True},
    }


def _sdk_request(runtime: dict, model: str) -> tuple[str, str, str]:
    captured: dict[str, str] = {}

    def respond(request: httpx.Request) -> httpx.Response:
        captured["authorization"] = request.headers["authorization"]
        captured["path"] = request.url.path
        captured["model"] = json.loads(request.content)["model"]
        return httpx.Response(
            200,
            request=request,
            json={
                "id": "resp_fixture",
                "object": "response",
                "created_at": 0,
                "status": "completed",
                "model": model,
                "output": [],
            },
        )

    with httpx.Client(transport=httpx.MockTransport(respond)) as http_client:
        with OpenAI(
            api_key=runtime["api_key"],
            base_url=runtime["base_url"],
            http_client=http_client,
            max_retries=0,
        ) as client:
            client.responses.create(model=model, input="fixture request")
    return captured["authorization"], captured["path"], captured["model"]


def test_native_oauth_override_uses_current_profile_token_after_each_rotation(tmp_path):
    """Real resolver + profile stores: A -> B -> rotated A reaches the SDK boundary."""
    home_a, home_b = tmp_path / "profile-a", tmp_path / "profile-b"
    token_a1 = _fake_codex_token("fixture-account-a", "v1")
    token_a2 = _fake_codex_token("fixture-account-a", "v2")
    token_b1 = _fake_codex_token("fixture-account-b", "v1")
    _write_codex_pool(home_a, token_a1)
    _write_codex_pool(home_b, token_b1)

    runner = object.__new__(GatewayRunner)
    runner.session_store = None
    runner._session_model_overrides = {
        "session-a": _override(_fake_codex_token("fixture-account-a", "stale")),
        "session-b": _override(_fake_codex_token("fixture-account-b", "stale")),
    }

    set_multiplex_active(True)
    try:
        with _profile_runtime_scope(home_a):
            model_a1, runtime_a1 = runner._resolve_session_agent_runtime(session_key="session-a")
        with _profile_runtime_scope(home_b):
            model_b1, runtime_b1 = runner._resolve_session_agent_runtime(session_key="session-b")

        _write_codex_pool(home_a, token_a2)
        with _profile_runtime_scope(home_a):
            model_a2, runtime_a2 = runner._resolve_session_agent_runtime(session_key="session-a")
    finally:
        set_multiplex_active(False)

    assert [runtime_a1["api_key"], runtime_b1["api_key"], runtime_a2["api_key"]] == [token_a1, token_b1, token_a2]
    assert model_a1 == model_b1 == model_a2 == "gpt-fixture-codex"
    assert runtime_a2["max_tokens"] == 12_345
    assert runtime_a2["request_overrides"] == {"store": False}
    assert runtime_a2["capabilities"] == {"openai_native_compaction": True}
    assert runner._agent_config_signature(model_a1, runtime_a1, [], "") != runner._agent_config_signature(
        model_a2, runtime_a2, [], ""
    )
    stored = runner._session_model_overrides["session-a"]
    assert stored["api_key"] == token_a2
    assert runtime_a2["credential_pool"].entry_id_for_api_key(token_a2) == "fixture-account"
    _, reapplied = runner._apply_session_model_override(
        "session-a", "global-model", {"api_key": "fake-global-key"}
    )
    assert reapplied["api_key"] == token_a2
    assert reapplied["credential_pool"] is runtime_a2["credential_pool"]
    assert _sdk_request(runtime_a2, model_a2) == (
        f"Bearer {token_a2}",
        "/backend-api/codex/responses",
        "gpt-fixture-codex",
    )


@pytest.mark.parametrize("provider,base_url,config", [
    ("custom", "https://custom.example/v1", {}),
    ("openai-codex", "https://proxy.example/backend-api/codex", {}),
    ("openai-codex", "https://proxy.example/backend-api/codex", {
        "providers": {"openai-codex": {"base_url": "https://proxy.example/backend-api/codex"}},
    }),
    ("openai-codex", "https://chatgpt.com/not-the-codex-route", {}),
])
def test_oauth_refresh_preserves_explicit_custom_and_proxy_keys(provider, base_url, config):
    runner = object.__new__(GatewayRunner)
    runner.session_store = None
    runner._session_model_overrides = {"session": {
        **_override("fake-explicit-key"), "provider": provider, "base_url": base_url,
    }}
    with patch("gateway.run._resolve_gateway_model", return_value="global-model"), patch(
        "gateway.run._credential_pool_for_provider", return_value=None
    ), patch(
        "gateway.run._resolve_runtime_agent_kwargs_for_provider",
        side_effect=AssertionError("static overrides must not re-resolve credentials"),
    ):
        model, runtime = runner._resolve_session_agent_runtime(session_key="session", user_config=config)
    assert (model, runtime["api_key"], runtime["base_url"]) == (
        "gpt-fixture-codex", "fake-explicit-key", base_url,
    )


@pytest.mark.parametrize("change,error", [
    ({"requested_provider": "different-codex-route"}, "different route identity"),
    ({"provider": "custom"}, "different route identity"),
    ({"base_url": "https://proxy.example/backend-api/codex"}, "different route identity"),
    ({"api_key": ""}, "no access token"),
    (None, "OAuth credentials.*unavailable"),
])
def test_native_oauth_refresh_fails_without_replaying_stale_or_default_credentials(change, error):
    runner = object.__new__(GatewayRunner)
    runner.session_store = None
    runner._session_model_overrides = {"session": _override("fake-old-token")}
    fresh = {**_override("fake-fresh-token"), "requested_provider": "openai-codex"}
    fresh.update(change or {})
    with patch("gateway.run._resolve_gateway_model", return_value="global-model"), patch(
        "gateway.run._resolve_runtime_agent_kwargs_for_provider", return_value=fresh,
        side_effect=RuntimeError("fixture auth missing") if change is None else None,
    ), patch(
        "gateway.run._resolve_runtime_agent_kwargs",
        side_effect=AssertionError("unsafe OAuth refresh must not fall back to the default provider"),
    ):
        with pytest.raises(RuntimeError, match=error):
            runner._resolve_session_agent_runtime(session_key="session")
    assert runner._session_model_overrides["session"]["api_key"] == "fake-old-token"


def test_native_oauth_override_follows_profile_account_selection(tmp_path):
    """A /model choice does not pin an account or override the profile's pool policy."""
    first = _fake_codex_token("fixture-account-a", "v1")
    second = _fake_codex_token("fixture-account-b", "v1")
    _write_codex_pool(tmp_path, first)
    runner = object.__new__(GatewayRunner)
    runner.session_store = None
    runner._session_model_overrides = {"session": _override(first)}
    set_multiplex_active(True)
    try:
        with _profile_runtime_scope(tmp_path):
            runner._resolve_session_agent_runtime(session_key="session")
        _write_codex_pool(tmp_path, second)
        with _profile_runtime_scope(tmp_path):
            model, runtime = runner._resolve_session_agent_runtime(session_key="session")
    finally:
        set_multiplex_active(False)
    assert model == "gpt-fixture-codex"
    assert runtime["api_key"] == second
    assert runtime["credential_pool"].entry_id_for_api_key(second) == "fixture-account"


@pytest.mark.parametrize("old_token", ["fake-opaque-old-token", ""])
def test_native_xai_oauth_override_refreshes_opaque_or_missing_tokens(old_token):
    from hermes_cli.providers import get_provider
    definition = get_provider("xai-oauth", allow_network=False)
    assert definition is not None
    base_url = definition.base_url
    runner = object.__new__(GatewayRunner)
    runner.session_store = None
    runner._session_model_overrides = {"session": {
        "model": "grok-fixture", "provider": "xai-oauth", "api_key": old_token,
        "base_url": base_url, "api_mode": "chat_completions",
    }}
    pool = object()
    fresh = {"provider": "xai-oauth", "api_key": "fake-opaque-new-token",
             "base_url": base_url, "credential_pool": pool, "api_mode": "chat_completions"}
    with patch("gateway.run._resolve_gateway_model", return_value="global-model"), patch(
        "gateway.run._resolve_runtime_agent_kwargs_for_provider", return_value=fresh,
    ):
        model, runtime = runner._resolve_session_agent_runtime(session_key="session")
    assert model == "grok-fixture"
    assert runtime["api_key"] == "fake-opaque-new-token"
    assert runtime["credential_pool"] is pool
