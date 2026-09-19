"""Account-scoped Codex catalogs follow runtime's usable pool credential."""

import base64
import json
import time

import httpx
import pytest


@pytest.fixture
def codex_accounts(tmp_path, monkeypatch):
    from hermes_cli.auth import DEFAULT_CODEX_BASE_URL

    home = tmp_path / "hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("CODEX_HOME", str(tmp_path / "codex"))
    (home / "config.yaml").write_text(
        "model:\n  provider: openai-codex\n  default: gpt-5.6-sol\n",
        encoding="utf-8",
    )
    tokens = {}
    for account in ("preferred", "backup", "revoked"):
        claims = {
            "exp": int(time.time()) + 3600,
            "https://api.openai.com/auth": {"chatgpt_account_id": account},
        }
        payload = (
            base64.urlsafe_b64encode(json.dumps(claims).encode()).decode().rstrip("=")
        )
        tokens[account] = f"header.{payload}.signature"

    store = {
        "version": 1,
        "active_provider": "openai-codex",
        "providers": {
            "openai-codex": {
                "tokens": {
                    "access_token": tokens["revoked"],
                    "refresh_token": "revoked-refresh",
                }
            }
        },
        "credential_pool": {
            "openai-codex": [
                {
                    "id": account,
                    "label": account,
                    "priority": priority,
                    "auth_type": "oauth",
                    "source": "device_code"
                    if account == "revoked"
                    else "manual:device_code",
                    "access_token": tokens[account],
                    "refresh_token": f"{account}-refresh",
                    "base_url": DEFAULT_CODEX_BASE_URL,
                }
                for priority, account in enumerate(tokens)
            ]
        },
    }
    auth_file = home / "auth.json"
    auth_file.write_text(json.dumps(store), encoding="utf-8")
    return home, auth_file, store, tokens


@pytest.mark.parametrize("preferred_dead", [False, True])
def test_picker_and_setup_discover_with_runtime_pool_account(
    codex_accounts,
    monkeypatch,
    preferred_dead,
):
    from hermes_cli import auth, models
    from hermes_cli.model_setup_flows import _model_flow_openai_codex
    from hermes_cli.runtime_provider import resolve_runtime_provider

    _home, auth_file, store, tokens = codex_accounts
    if preferred_dead:
        store["credential_pool"]["openai-codex"][0].update(
            last_status="dead",
            last_status_at=time.time(),
            last_error_reason="invalid_grant",
        )
        auth_file.write_text(json.dumps(store), encoding="utf-8")
    selected = "backup" if preferred_dead else "preferred"
    runtime = resolve_runtime_provider(requested="openai-codex")
    assert runtime["api_key"] == tokens[selected]
    # The legacy singleton remains independently readable but is revoked upstream.
    assert auth.resolve_codex_runtime_credentials()["api_key"] == tokens["revoked"]
    requested_accounts = []

    def models_endpoint(url, *, headers, timeout):
        account = headers["ChatGPT-Account-Id"]
        requested_accounts.append(account)
        assert headers["Authorization"] == f"Bearer {tokens[account]}"
        if account == "revoked":
            return httpx.Response(401, json={"error": {"code": "token_revoked"}})
        return httpx.Response(
            200, json={"models": [{"slug": "gpt-6-astra", "priority": 0}]}
        )

    monkeypatch.setattr(httpx, "get", models_endpoint)
    picker_models = models.provider_model_ids("openai-codex")
    setup_models = []

    def capture_selection(model_ids, **kwargs):
        setup_models.extend(model_ids)
        assert kwargs["confirm_api_key"] == runtime["api_key"]
        return None  # Exercise discovery without saving a model choice.

    monkeypatch.setattr("builtins.input", lambda _prompt="": "1")
    monkeypatch.setattr(auth, "_prompt_model_selection", capture_selection)
    _model_flow_openai_codex({}, current_model="gpt-5.6-sol")

    assert requested_accounts == [selected, selected]
    assert picker_models == setup_models
    assert picker_models == ["gpt-6-astra", "gpt-6-astra-900k"]


@pytest.mark.parametrize("status", [200, 401])
def test_preferred_account_entitlement_is_never_borrowed_from_other_state(
    codex_accounts,
    monkeypatch,
    tmp_path,
    status,
):
    from hermes_cli import models

    _home, _auth_file, _store, tokens = codex_accounts
    codex_home = tmp_path / "codex"
    codex_home.mkdir()
    (codex_home / "config.toml").write_text('model = "gpt-6-astra"\n', encoding="utf-8")
    (codex_home / "models_cache.json").write_text(
        json.dumps({"models": [{"slug": "gpt-6-astra"}, {"slug": "gpt-6-astra-900k"}]}),
        encoding="utf-8",
    )
    requests = []

    def models_endpoint(url, *, headers, timeout):
        account = headers["ChatGPT-Account-Id"]
        requests.append(account)
        assert headers["Authorization"] == f"Bearer {tokens[account]}"
        if account == "preferred":
            return httpx.Response(status, json={"models": [{"slug": "gpt-5.6-sol"}]})
        return httpx.Response(200, json={"models": [{"slug": "gpt-6-astra"}]})

    monkeypatch.setattr(httpx, "get", models_endpoint)
    discovered = models.provider_model_ids("openai-codex")

    assert requests == ["preferred"]
    assert "gpt-6-astra" not in discovered
    assert "gpt-6-astra-900k" not in discovered
