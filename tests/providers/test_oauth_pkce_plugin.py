"""Behavior tests for declarative OAuth PKCE model-provider plugins."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock
from urllib.parse import parse_qs, urlencode, urlparse
from urllib.request import urlopen

import pytest

from providers.base import OAuthPKCEConfig, ProviderProfile


REPO_ROOT = Path(__file__).resolve().parents[2]


def _oauth_config(scope: str = "inference:invoke offline_access") -> OAuthPKCEConfig:
    return OAuthPKCEConfig(
        client_id="public-client",
        authorization_url="https://auth.example.test/oauth/authorize",
        token_url="https://auth.example.test/oauth/token",
        scope=scope,
    )


def test_user_oauth_plugin_auto_extends_provider_registry(tmp_path):
    hermes_home = tmp_path / "hermes"
    plugin_dir = hermes_home / "plugins" / "model-providers" / "example-oauth"
    plugin_dir.mkdir(parents=True)
    (plugin_dir / "__init__.py").write_text(
        "from providers import OAuthPKCEConfig, ProviderProfile, register_provider\n"
        "register_provider(ProviderProfile(\n"
        "    name='example-oauth', display_name='Example OAuth',\n"
        "    auth_type='oauth_pkce', api_mode='chat_completions',\n"
        "    base_url='https://api.example.test/v1',\n"
        "    oauth=OAuthPKCEConfig(\n"
        "        client_id='public-client',\n"
        "        authorization_url='https://auth.example.test/oauth/authorize',\n"
        "        token_url='https://auth.example.test/oauth/token',\n"
        "        scope='inference:invoke offline_access',\n"
        "    ),\n"
        "))\n",
        encoding="utf-8",
    )
    (plugin_dir / "plugin.yaml").write_text(
        "name: example-oauth\nkind: model-provider\nversion: 1.0.0\n",
        encoding="utf-8",
    )
    env = dict(os.environ)
    env["HERMES_HOME"] = str(hermes_home)
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import json; from hermes_cli.auth import PROVIDER_REGISTRY; "
                "p=PROVIDER_REGISTRY['example-oauth']; "
                "print(json.dumps({'auth_type':p.auth_type,'client_id':p.client_id,"
                "'base_url':p.inference_base_url}))"
            ),
        ],
        cwd=os.fspath(REPO_ROOT),
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    assert json.loads(result.stdout) == {
        "auth_type": "oauth_pkce",
        "client_id": "public-client",
        "base_url": "https://api.example.test/v1",
    }


def test_refresh_provider_oauth_pkce_preserves_rotating_token(monkeypatch):
    from hermes_cli import auth

    response = SimpleNamespace(
        status_code=200,
        json=lambda: {
            "access_token": "new-access",
            "refresh_token": "new-refresh",
            "expires_in": 3600,
        },
    )
    monkeypatch.setattr(auth.httpx, "post", Mock(return_value=response))

    result = auth.refresh_provider_oauth_pkce(
        "example-oauth",
        _oauth_config(),
        "old-refresh",
    )

    assert result["access_token"] == "new-access"
    assert result["refresh_token"] == "new-refresh"
    assert result["expires_at_ms"] > int(time.time() * 1000)


@pytest.mark.parametrize("scope", ["inference:invoke offline_access", ""])
def test_login_provider_oauth_pkce_completes_loopback_flow(monkeypatch, scope):
    from hermes_cli import auth

    response = SimpleNamespace(
        status_code=200,
        json=lambda: {
            "access_token": "access-token",
            "refresh_token": "refresh-token",
            "expires_in": 3600,
        },
    )
    post = Mock(return_value=response)
    monkeypatch.setattr(auth.httpx, "post", post)

    def _complete_authorization(authorize_url: str) -> bool:
        params = parse_qs(urlparse(authorize_url).query, keep_blank_values=True)
        if scope:
            assert params["scope"] == [scope]
        else:
            assert "scope" not in params
        callback = params["redirect_uri"][0]
        query = urlencode({"code": "auth-code", "state": params["state"][0]})
        with urlopen(f"{callback}?{query}", timeout=2) as callback_response:
            assert callback_response.status == 200
        return True

    monkeypatch.setattr(auth.webbrowser, "open", _complete_authorization)

    result = auth.login_provider_oauth_pkce(
        "example-oauth",
        _oauth_config(scope),
        timeout_seconds=5,
    )

    assert result["access_token"] == "access-token"
    token_request = post.call_args.kwargs["data"]
    assert token_request["grant_type"] == "authorization_code"
    assert token_request["code"] == "auth-code"
    assert token_request["code_verifier"]
    assert token_request["redirect_uri"].startswith("http://127.0.0.1:")


def test_auth_add_uses_plugin_pkce_flow_and_persists_pool(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))

    from hermes_cli import auth
    from hermes_cli.auth_commands import auth_add_command

    profile = ProviderProfile(
        name="example-oauth",
        display_name="Example OAuth",
        auth_type="oauth_pkce",
        base_url="https://api.example.test/v1",
        oauth=_oauth_config(),
    )
    auth.PROVIDER_REGISTRY[profile.name] = auth.ProviderConfig(
        id=profile.name,
        name=profile.display_name,
        auth_type=profile.auth_type,
        inference_base_url=profile.base_url,
        client_id=profile.oauth.client_id,
        scope=profile.oauth.scope,
        extra={"oauth": profile.oauth},
    )
    monkeypatch.setattr(
        auth,
        "login_provider_oauth_pkce",
        lambda *args, **kwargs: {
            "access_token": "access-token",
            "refresh_token": "refresh-token",
            "expires_at_ms": int(time.time() * 1000) + 3_600_000,
            "scope": profile.oauth.scope,
            "token_type": "Bearer",
        },
    )
    try:
        auth_add_command(SimpleNamespace(
            provider=profile.name,
            auth_type="oauth",
            label="work",
            api_key=None,
            no_browser=True,
            timeout=None,
        ))
        payload = json.loads(
            (tmp_path / "hermes" / "auth.json").read_text(encoding="utf-8")
        )
        entry = payload["credential_pool"][profile.name][0]
        assert entry["source"] == "manual:oauth_pkce"
        assert entry["access_token"] == "access-token"
        assert entry["refresh_token"] == "refresh-token"
        assert entry["client_id"] == "public-client"
    finally:
        auth.PROVIDER_REGISTRY.pop(profile.name, None)


def test_generic_pool_refresh_uses_profile_oauth_config(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
    from agent.credential_pool import AUTH_TYPE_OAUTH, CredentialPool, PooledCredential
    from hermes_cli import auth
    from providers import _REGISTRY, register_provider

    profile = ProviderProfile(
        name="example-oauth",
        auth_type="oauth_pkce",
        base_url="https://api.example.test/v1",
        oauth=_oauth_config(),
    )
    register_provider(profile)
    entry = PooledCredential(
        provider=profile.name,
        id="oauth1",
        label="work",
        auth_type=AUTH_TYPE_OAUTH,
        priority=0,
        source="manual:oauth_pkce",
        access_token="old-access",
        refresh_token="old-refresh",
        expires_at_ms=int(time.time() * 1000) - 1,
    )
    monkeypatch.setattr(
        auth,
        "refresh_provider_oauth_pkce",
        lambda *args, **kwargs: {
            "access_token": "new-access",
            "refresh_token": "new-refresh",
            "expires_at_ms": int(time.time() * 1000) + 3_600_000,
        },
    )
    try:
        refreshed = CredentialPool(profile.name, [entry]).select()
        assert refreshed is not None
        assert refreshed.access_token == "new-access"
        assert refreshed.refresh_token == "new-refresh"
    finally:
        _REGISTRY.pop(profile.name, None)


def test_generic_pool_unknown_expiry_refreshes_after_auth_failure(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
    from agent.credential_pool import AUTH_TYPE_OAUTH, CredentialPool, PooledCredential
    from hermes_cli import auth
    from providers import _REGISTRY, register_provider

    profile = ProviderProfile(
        name="example-oauth",
        auth_type="oauth_pkce",
        base_url="https://api.example.test/v1",
        oauth=_oauth_config(),
    )
    register_provider(profile)
    entry = PooledCredential(
        provider=profile.name,
        id="oauth1",
        label="work",
        auth_type=AUTH_TYPE_OAUTH,
        priority=0,
        source="manual:oauth_pkce",
        access_token="old-access",
        refresh_token="old-refresh",
        expires_at_ms=None,
    )
    refresh = Mock(return_value={
        "access_token": "new-access",
        "refresh_token": "new-refresh",
        "expires_at_ms": None,
    })
    monkeypatch.setattr(auth, "refresh_provider_oauth_pkce", refresh)
    try:
        pool = CredentialPool(profile.name, [entry])
        assert pool._entry_needs_refresh(entry) is False

        refreshed = pool.try_refresh_matching(credential_id=entry.id)

        assert refreshed is not None
        assert refreshed.access_token == "new-access"
        refresh.assert_called_once()
    finally:
        _REGISTRY.pop(profile.name, None)


def test_plugin_oauth_profile_lookup_failure_is_not_silenced(monkeypatch):
    from agent.credential_pool import AUTH_TYPE_OAUTH, CredentialPool, PooledCredential
    import providers

    entry = PooledCredential(
        provider="example-oauth",
        id="oauth1",
        label="work",
        auth_type=AUTH_TYPE_OAUTH,
        priority=0,
        source="manual:oauth_pkce",
        access_token="access",
        refresh_token="refresh",
    )
    pool = CredentialPool(entry.provider, [entry])
    monkeypatch.setattr(
        providers,
        "get_provider_profile",
        Mock(side_effect=RuntimeError("provider discovery failed")),
    )

    with pytest.raises(RuntimeError, match="provider discovery failed"):
        pool._entry_needs_refresh(entry)
    with pytest.raises(RuntimeError, match="provider discovery failed"):
        pool._refresh_entry(entry, force=False)


def test_generic_oauth_auth_status_rejects_expired_access_token(monkeypatch):
    from agent import credential_pool
    from hermes_cli import auth

    provider = "example-oauth-status"
    entry = SimpleNamespace(
        access_token="expired-access",
        refresh_token="refresh-token",
        expires_at_ms=1,
    )
    monkeypatch.setattr(
        credential_pool,
        "load_pool",
        lambda target: SimpleNamespace(peek=lambda: entry),
    )
    auth.PROVIDER_REGISTRY[provider] = auth.ProviderConfig(
        id=provider,
        name="Example OAuth",
        auth_type="oauth_pkce",
    )
    try:
        status = auth.get_auth_status(provider)

        assert status["logged_in"] is False
        assert status["needs_refresh"] is True
        assert status["expired"] is True
        assert status["expires_at_ms"] == 1
    finally:
        auth.PROVIDER_REGISTRY.pop(provider, None)


def test_auth_status_command_reports_expired_oauth_token(monkeypatch, capsys):
    from hermes_cli import auth
    from hermes_cli.auth_commands import auth_status_command

    monkeypatch.setattr(
        auth,
        "get_auth_status",
        lambda provider: {"logged_in": False, "needs_refresh": True},
    )

    auth_status_command(SimpleNamespace(provider="example-oauth"))

    assert capsys.readouterr().out.strip() == (
        "example-oauth: access token expired (refresh needed)"
    )
