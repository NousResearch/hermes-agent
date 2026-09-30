"""Pooled OAuth plugins use their own Responses route for auxiliary work too."""

import pytest
import time

from providers import ProviderProfile, register_provider


@pytest.fixture
def oauth_provider(monkeypatch, tmp_path):
    import providers
    from hermes_cli import auth

    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
    provider = "fixture-oauth-responses"
    register_provider(ProviderProfile(
        name=provider, auth_type="oauth_external", api_mode="codex_responses",
        base_url="https://api.openai.com/v1", auth_handler=lambda *a: True,
        refresh_credential=lambda entry: {"access_token": "rotated-bearer",
                                          "expires_at_ms": int(time.time() * 1000) + 3600000},
    ))
    yield provider
    providers._REGISTRY.pop(provider, None)
    auth.PROVIDER_REGISTRY.pop(provider, None)


def test_pooled_oauth_main_and_aux_resolve_the_same_credentials(oauth_provider):
    from agent.auxiliary_client import CodexAuxiliaryClient, resolve_provider_client
    from agent.credential_pool import PooledCredential, load_pool
    from hermes_cli.runtime_provider import resolve_runtime_provider

    load_pool(oauth_provider).add_entry(PooledCredential(
        provider=oauth_provider, id="account-a", label="Account A", auth_type="oauth",
        priority=0, source="manual:fixture", access_token="account-bearer",
        refresh_token="refresh-bearer",
    ))
    runtime = resolve_runtime_provider(requested=oauth_provider, target_model="account-model")
    client, model = resolve_provider_client(oauth_provider, model="account-model")
    assert isinstance(client, CodexAuxiliaryClient)
    assert model == "account-model"
    assert client.api_key == runtime["api_key"] == "account-bearer"
    assert str(client.base_url).rstrip("/") == runtime["base_url"] == "https://api.openai.com/v1"
    assert runtime["api_mode"] == "codex_responses"
    client._real_client.close()


def test_missing_oauth_credentials_never_fall_through_to_another_provider(oauth_provider):
    from hermes_cli.auth_constants import AuthError
    from hermes_cli.runtime_provider import resolve_runtime_provider

    with pytest.raises(AuthError, match=oauth_provider):
        resolve_runtime_provider(requested=oauth_provider, target_model="account-model")


def test_oauth_plugin_refreshes_an_expired_selected_token_before_inference(oauth_provider):
    from agent.credential_pool import PooledCredential, load_pool
    from hermes_cli.runtime_provider import resolve_runtime_provider

    load_pool(oauth_provider).add_entry(PooledCredential(
        provider=oauth_provider, id="expired", label="Account", auth_type="oauth",
        priority=0, source="manual:fixture", access_token="expired-bearer",
        refresh_token="refresh-bearer", expires_at_ms=1,
    ))
    runtime = resolve_runtime_provider(requested=oauth_provider, target_model="account-model")
    assert runtime["api_key"] == "rotated-bearer"


def test_provider_credential_eligibility_prevents_cross_account_rotation(oauth_provider):
    from agent.credential_pool import PooledCredential, load_pool
    from providers import get_provider_profile

    profile = get_provider_profile(oauth_provider)
    profile.credential_is_eligible = lambda entry: entry.id == "selected"
    pool = load_pool(oauth_provider)
    for credential_id in ("other", "selected"):
        pool.add_entry(PooledCredential(
            provider=oauth_provider, id=credential_id, label=credential_id, auth_type="oauth",
            priority=0, source="manual:fixture", access_token=f"bearer-{credential_id}",
        ))
    assert pool.select().id == "selected"
    assert pool.mark_exhausted_and_rotate(status_code=429, api_key_hint="bearer-selected") is None
    assert {entry.id for entry in load_pool(oauth_provider).entries()} == {"other", "selected"}


def test_refresh_that_revokes_inference_permission_is_not_leased(oauth_provider):
    from agent.credential_pool import PooledCredential, load_pool
    from providers import get_provider_profile

    profile = get_provider_profile(oauth_provider)
    profile.credential_is_eligible = lambda entry: bool(entry.extra.get("authorized"))
    profile.refresh_credential = lambda entry: {"access_token": "new-bearer", "authorized": False,
                                               "expires_at_ms": int(time.time() * 1000) + 3600000}
    pool = load_pool(oauth_provider)
    pool.add_entry(PooledCredential(
        provider=oauth_provider, id="revoked", label="Account", auth_type="oauth",
        priority=0, source="manual:fixture", access_token="old-bearer",
        refresh_token="refresh-bearer", expires_at_ms=1, extra={"authorized": True},
    ))
    assert pool.select() is None


def test_terminal_refresh_clears_credentials_through_provider_hook(oauth_provider):
    from agent.credential_pool import PooledCredential, load_pool
    from hermes_cli.auth_constants import AuthError
    from providers import get_provider_profile

    def rejected(entry):
        raise AuthError("Sign in again", provider=oauth_provider, code="invalid_grant", relogin_required=True)

    profile = get_provider_profile(oauth_provider)
    profile.refresh_credential = rejected
    profile.clear_credential = lambda entry: {"access_token": "", "refresh_token": "",
                                              "registration": {"client_id": "registered-client"}}
    pool = load_pool(oauth_provider)
    pool.add_entry(PooledCredential(
        provider=oauth_provider, id="revoked", label="Account", auth_type="oauth",
        priority=0, source="manual:fixture", access_token="old-bearer",
        refresh_token="refresh-bearer", expires_at_ms=1,
    ))
    assert pool.select() is None
    stored = load_pool(oauth_provider).entries()[0]
    assert not stored.access_token and not stored.refresh_token
    assert stored.extra["registration"]["client_id"] == "registered-client"


def test_live_pool_adopts_reauthentication_after_terminal_cleanup(oauth_provider):
    from dataclasses import replace
    from agent.credential_pool import PooledCredential, load_pool
    from hermes_cli.auth import write_credential_pool
    from providers import get_provider_profile

    profile = get_provider_profile(oauth_provider)
    profile.credential_is_eligible = lambda entry: bool(entry.access_token)
    pool = load_pool(oauth_provider)
    dead = PooledCredential(provider=oauth_provider, id="account", label="Account", auth_type="oauth",
                            priority=0, source="manual:fixture", access_token="", last_status="dead")
    pool.add_entry(dead)
    fresh = replace(dead, access_token="reauthenticated-bearer", refresh_token="new-refresh",
                    last_status="ok", expires_at_ms=int(time.time() * 1000) + 3600000)
    write_credential_pool(oauth_provider, [fresh.to_dict()])
    assert pool.select().access_token == "reauthenticated-bearer"


def test_chatgpt_profile_switches_main_aux_and_live_catalog_together(monkeypatch, tmp_path):
    """Use actual imports and auth stores; only replace the outbound model HTTP request."""
    import io
    import json
    from agent.auxiliary_client import resolve_provider_client
    from agent.secret_scope import is_multiplex_active, set_multiplex_active
    from hermes_cli.models import _profile_live_catalog
    from hermes_cli.runtime_provider import resolve_runtime_provider
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override

    provider = "openai-chatgpt"
    homes = {}
    for account in ("A", "B"):
        home = tmp_path / account
        home.mkdir()
        homes[account] = home
        (home / "auth.json").write_text(json.dumps({
            "version": 1,
            "providers": {provider: {"active_credential_id": account}},
            "credential_pool": {provider: [
                {"id": account, "provider": provider, "label": account, "auth_type": "oauth",
                 "priority": 0, "source": "manual:fixture", "access_token": f"token-{account}",
                 "expires_at_ms": int(time.time() * 1000) + 3600000,
                 "base_url": "https://api.openai.com/v1",
                 "chatgpt": {"scopes": ["chatgpt.tokens.use.direct"]}},
                {"id": "inactive", "provider": provider, "label": "inactive", "auth_type": "oauth",
                 "priority": -1, "source": "manual:fixture", "access_token": "must-not-use",
                 "chatgpt": {"scopes": ["chatgpt.tokens.use.direct"]}},
            ]},
        }))
        (home / "config.yaml").write_text("model:\n  provider: openai-chatgpt\n  api_mode: chat_completions\n")
    monkeypatch.setenv("HERMES_HOME", str(homes["A"]))
    seen = []

    def catalog(request, **kwargs):
        account = request.get_header("Authorization").removeprefix("Bearer token-")
        seen.append(account)
        return io.BytesIO(json.dumps({"models": [{"slug": f"model-{account}",
                              "display_name": f"Model {account}", "visibility": "list"}]}).encode())

    monkeypatch.setattr("hermes_cli.urllib_security.open_credentialed_url", catalog)
    was_multiplex = is_multiplex_active()
    set_multiplex_active(True)
    try:
        for account in ("A", "B", "A"):
            token = set_hermes_home_override(homes[account])
            try:
                runtime = resolve_runtime_provider(requested=provider, target_model=f"model-{account}")
                client, model = resolve_provider_client(provider, model=f"model-{account}", api_mode="chat_completions")
                try:
                    from agent.auxiliary_client import CodexAuxiliaryClient
                    assert runtime["api_mode"] == "codex_responses"
                    assert isinstance(client, CodexAuxiliaryClient)
                    assert runtime["api_key"] == client.api_key == f"token-{account}"
                    assert model == f"model-{account}"
                    assert _profile_live_catalog(provider) == [f"model-{account}"]
                finally:
                    getattr(client, "_real_client", client).close()
            finally:
                reset_hermes_home_override(token)
    finally:
        set_multiplex_active(was_multiplex)
    assert seen == ["A", "B", "A"]


@pytest.mark.parametrize("terminal", ["completed", "eof", "incomplete-object"])
def test_chatgpt_auxiliary_wire_and_stream_contract(monkeypatch, terminal):
    from types import SimpleNamespace
    from agent.auxiliary_client import _CodexCompletionsAdapter

    monkeypatch.setattr("hermes_cli.auth_chatgpt.assert_active_access_token", lambda token: None, raising=False)

    captured = []
    final = SimpleNamespace(status="completed", output=[SimpleNamespace(
        type="message", content=[SimpleNamespace(type="output_text", text="summary")])], usage=None)

    def create(**kwargs):
        captured.append(kwargs)
        if terminal == "incomplete-object":
            return SimpleNamespace(status="incomplete", output=[], usage=None)
        events = [SimpleNamespace(type="response.output_item.done", item=final.output[0])]
        if terminal == "completed":
            events.append(SimpleNamespace(type="response.completed", response=final))
        return iter(events)

    client = SimpleNamespace(base_url="https://api.openai.com/v1", _hermes_aux_effective_provider="openai-chatgpt",
                             responses=SimpleNamespace(create=create))
    adapter = _CodexCompletionsAdapter(client, "account-model")
    kwargs = {"messages": [{"role": "system", "content": "Summarize"}, {"role": "user", "content": "Hi"}],
              "tools": [{"type": "function", "function": {"name": "lookup", "parameters": {"type": "object", "properties": {}}}}],
              "temperature": 0.3, "max_tokens": 200, "extra_body": {"prompt_cache_retention": "24h"}}
    if terminal == "completed":
        assert adapter.create(**kwargs).choices[0].message.content == "summary"
    else:
        with pytest.raises(RuntimeError):
            adapter.create(**kwargs)
    wire = captured[0]
    # Large input/tools are carried in extra_body by the SDK transform bypass.
    body = {**wire, **wire.get("extra_body", {})}
    assert body["store"] is False and body["stream"] is True
    assert body["tools"][0]["type"] == "namespace"
    assert not {"temperature", "max_output_tokens", "prompt_cache_retention"} & body.keys()


@pytest.mark.parametrize("code,status", [("subscription_sharing_usage_limit_exceeded", 429),
                                        ("chatpass_v2_scope_not_authorized", 401),
                                        ("chatgpt_session_changed", None)])
def test_chatgpt_auxiliary_terminal_error_cannot_rotate_or_fallback(monkeypatch, code, status):
    from types import SimpleNamespace
    import agent.auxiliary_client as auxiliary

    if code == "chatgpt_session_changed":
        from hermes_cli.auth_constants import AuthError
        error = AuthError("Reinitialize this session", provider="openai-chatgpt", code=code)
    else:
        error = RuntimeError(code)
        error.status_code = status
        error.body = {"error": {"code": code, "message": code}}
    from agent.error_classifier import classify_api_error
    classified = classify_api_error(error, provider="openai-chatgpt")
    assert not classified.retryable and not classified.should_fallback
    client = SimpleNamespace(_hermes_aux_effective_provider="openai-chatgpt")

    def forbidden(*args, **kwargs):
        pytest.fail("A terminal ChatGPT plan error reached credential rotation or provider fallback")

    monkeypatch.setattr(auxiliary, "_ladder_credential_rungs", forbidden)
    monkeypatch.setattr(auxiliary, "_ladder_provider_fallback", forbidden)
    ladder = auxiliary._aux_recovery_ladder(
        error, client=client, kwargs={}, task="compression", async_mode=False,
        base_info="https://api.openai.com/v1", resolved_provider="openai-chatgpt",
        resolved_model="account-model", resolved_base_url="https://api.openai.com/v1",
        resolved_api_key="fixture-bearer", resolved_api_mode="codex_responses",
        final_model="account-model", max_tokens=None, main_runtime=None, route_info=None,
    )
    with pytest.raises(type(error)) as exc:
        next(ladder)
    assert exc.value is error


@pytest.mark.parametrize("change", ["logout", "scope", "active"])
def test_cached_chatgpt_auxiliary_client_rechecks_account_before_dispatch(monkeypatch, tmp_path, change):
    import json
    from agent.auxiliary_client import resolve_provider_client
    from hermes_cli.auth_constants import AuthError

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    row = {"id": "account", "provider": "openai-chatgpt", "auth_type": "oauth", "priority": 0,
           "source": "manual:fixture", "access_token": "cached-bearer", "label": "Account",
           "chatgpt": {"scopes": ["chatgpt.tokens.use.direct"]}}
    store = {"version": 1, "providers": {"openai-chatgpt": {"active_credential_id": "account"}},
             "credential_pool": {"openai-chatgpt": [row]}}
    auth_path = tmp_path / "auth.json"
    auth_path.write_text(json.dumps(store))
    client, _ = resolve_provider_client("openai-chatgpt", model="account-model")
    if change == "logout":
        row["access_token"] = ""
    elif change == "scope":
        row["chatgpt"]["scopes"] = []
    else:
        store["providers"]["openai-chatgpt"]["active_credential_id"] = "another-account"
    auth_path.write_text(json.dumps(store))
    monkeypatch.setattr(client._real_client.responses, "create", lambda **kw: pytest.fail("Stale account reached HTTP"))
    try:
        with pytest.raises(AuthError):
            client.chat.completions.create(messages=[{"role": "user", "content": "Hello"}])
    finally:
        client._real_client.close()
