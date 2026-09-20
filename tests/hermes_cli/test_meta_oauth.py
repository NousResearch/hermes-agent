"""Meta Muse subscription OAuth tests."""


def test_meta_oauth_constants_present():
    from hermes_cli import auth_constants as C

    assert C.META_OAUTH_CLIENT_ID == "1031625952748946"
    assert C.META_DEVICE_AUTHORIZATION_URL.startswith("https://auth.meta.com/")
    assert C.META_DEVICE_TOKEN_URL.startswith("https://auth.meta.com/")
    assert C.META_API_KEY_MINT_URL == "https://api.meta.ai/muse-code/key"
    assert C.META_API_KEY_LIFETIME_MS == 24 * 60 * 60 * 1000


def test_meta_trusted_url_rejects_non_http():
    from hermes_cli.auth_meta import _trusted_http_url

    assert _trusted_http_url("https://auth.meta.com/oauth/device/?code=X") is not None
    assert _trusted_http_url("file:///etc/passwd") is None
    assert _trusted_http_url(None) is None


def test_meta_error_detail_prefers_description():
    from hermes_cli.auth_meta import _error_detail

    assert _error_detail({"error_description": "slow"}) == ": slow"
    assert _error_detail({}) == ""


def test_meta_positive_number_rejects_non_positive_values():
    from hermes_cli.auth_meta import _positive_number

    assert _positive_number(2) == 2
    assert _positive_number(0) is None
    assert _positive_number("2") is None


class _MetaResponse:
    def __init__(self, status_code, payload, text=""):
        self.status_code = status_code
        self._payload = payload
        self.text = text

    def json(self):
        if isinstance(self._payload, Exception):
            raise self._payload
        return self._payload

    def raise_for_status(self):
        raise RuntimeError(f"HTTP {self.status_code}")


class _MetaClient:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def post(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        return self.responses.pop(0)


def test_meta_device_authorization_success():
    from hermes_cli.auth_meta import start_device_authorization

    client = _MetaClient([_MetaResponse(200, {
        "device_code": "device-code",
        "user_code": "ABCD-EFGH",
        "verification_uri": "https://auth.meta.com/device",
        "verification_uri_complete": "https://auth.meta.com/device?code=ABCD-EFGH",
        "interval": 3,
        "expires_in": 600,
    })])

    result = start_device_authorization(client)

    assert result == {
        "device_code": "device-code",
        "user_code": "ABCD-EFGH",
        "verification_uri": "https://auth.meta.com/device?code=ABCD-EFGH",
        "interval": 3,
        "expires_in": 600,
    }
    assert client.calls[0][1]["data"] == {"client_id": "1031625952748946"}


def test_meta_device_authorization_rejects_untrusted_verification_url():
    import pytest
    from hermes_cli.auth_constants import AuthError
    from hermes_cli.auth_meta import start_device_authorization

    client = _MetaClient([_MetaResponse(200, {
        "device_code": "device-code", "user_code": "ABCD",
        "verification_uri": "file:///etc/passwd", "expires_in": 600, "interval": 3,
    })])

    with pytest.raises(AuthError, match="missing fields"):
        start_device_authorization(client)


def test_meta_poll_returns_identity_on_second_try(monkeypatch):
    from hermes_cli.auth_meta import poll_for_identity_token

    monkeypatch.setattr("hermes_cli.auth_device_flow.time.sleep", lambda _seconds: None)
    client = _MetaClient([
        _MetaResponse(400, {"error": "authorization_pending"}),
        _MetaResponse(200, {"access_token": "identity-token"}),
    ])

    assert poll_for_identity_token(client, "device-code", expires_in=60, poll_interval=1) == "identity-token"
    assert len(client.calls) == 2
    assert client.calls[0][1]["data"]["grant_type"] == "urn:ietf:params:oauth:grant-type:device_code"


def test_meta_poll_access_denied_requires_relogin(monkeypatch):
    import pytest
    from hermes_cli.auth_constants import AuthError
    from hermes_cli.auth_meta import poll_for_identity_token

    client = _MetaClient([_MetaResponse(400, {"error": "access_denied"})])

    with pytest.raises(AuthError, match="denied") as caught:
        poll_for_identity_token(client, "device-code", expires_in=60, poll_interval=1)
    assert caught.value.relogin_required is True


def test_meta_mint_returns_oauth_credential(monkeypatch):
    from hermes_cli.auth_meta import mint_meta_api_key

    monkeypatch.setattr("hermes_cli.auth_meta.time.time", lambda: 1_000.0)
    client = _MetaClient([_MetaResponse(200, {"api_key": "LLM|runtime-key"})])

    result = mint_meta_api_key("identity-token", client)

    assert result["access_token"] == "LLM|runtime-key"
    assert result["refresh_token"] == "identity-token"
    assert result["expires_at_ms"] == 1_000_000 + 24 * 60 * 60 * 1000
    assert client.calls[0][1]["headers"]["x-api-version"] == "1.0.0"
    assert client.calls[0][1]["content"] == b"{}"


def test_meta_mint_401_is_terminal_relogin():
    import pytest
    from hermes_cli.auth_meta import mint_meta_api_key

    client = _MetaClient([_MetaResponse(401, {"error_description": "session expired"})])

    with pytest.raises(Exception, match="Re-authenticate") as caught:
        mint_meta_api_key("identity-token", client)
    assert caught.value.relogin_required is True


def test_meta_mint_missing_key_reports_action_url():
    import pytest
    from hermes_cli.auth_meta import mint_meta_api_key

    client = _MetaClient([_MetaResponse(200, {
        "action_url": "https://auth.meta.com/setup",
    })])

    with pytest.raises(Exception, match="https://auth.meta.com/setup"):
        mint_meta_api_key("identity-token", client)


def test_meta_save_and_read_roundtrip(tmp_path, monkeypatch):
    import json
    from hermes_cli.auth_meta import _read_meta_oauth_tokens, _save_meta_oauth_tokens

    hermes_home = tmp_path / "hermes"
    hermes_home.mkdir(parents=True, exist_ok=True)
    (hermes_home / "auth.json").write_text(json.dumps({"version": 1, "providers": {}}))
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    _save_meta_oauth_tokens({
        "access_token": "at-1", "refresh_token": "rt-1",
        "expires_at_ms": 1_000_000,
    })
    data = _read_meta_oauth_tokens()
    assert data["tokens"]["access_token"] == "at-1"
    assert data["tokens"]["refresh_token"] == "rt-1"


def test_meta_read_missing_requires_relogin(tmp_path, monkeypatch):
    import json
    import pytest
    from hermes_cli.auth_constants import AuthError
    from hermes_cli.auth_meta import _read_meta_oauth_tokens

    hermes_home = tmp_path / "hermes"
    hermes_home.mkdir(parents=True, exist_ok=True)
    (hermes_home / "auth.json").write_text(json.dumps({"version": 1, "providers": {}}))
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    with pytest.raises(AuthError) as caught:
        _read_meta_oauth_tokens()
    assert caught.value.relogin_required is True


def test_meta_quarantine_clears_dead_tokens(tmp_path, monkeypatch):
    import json
    from hermes_cli.auth_constants import AuthError
    from hermes_cli.auth_meta import (
        _quarantine_meta_oauth_tokens, _read_meta_oauth_tokens, _save_meta_oauth_tokens,
    )

    hermes_home = tmp_path / "hermes"
    hermes_home.mkdir(parents=True, exist_ok=True)
    (hermes_home / "auth.json").write_text(json.dumps({"version": 1, "providers": {}}))
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    _save_meta_oauth_tokens({
        "access_token": "at-1", "refresh_token": "rt-1", "expires_at_ms": 1_000_000,
    })

    _quarantine_meta_oauth_tokens(AuthError("dead", provider="meta-oauth", code="x", relogin_required=True))

    import pytest
    with pytest.raises(AuthError):
        _read_meta_oauth_tokens()


def test_meta_oauth_registered():
    from hermes_cli.auth import PROVIDER_REGISTRY

    assert PROVIDER_REGISTRY["meta-oauth"].auth_type == "oauth_external"
    assert PROVIDER_REGISTRY["meta-oauth"].inference_base_url == "https://api.meta.ai/v1"


def test_meta_subscription_aliases_resolve_to_oauth():
    from hermes_cli.auth import _plugin_aliases, resolve_provider

    aliases = _plugin_aliases()
    assert aliases.get("meta-subscription") == "meta-oauth"
    assert aliases.get("muse-subscription") == "meta-oauth"
    # Plugin keeps its own `meta` alias — subscription must not clobber it.
    assert aliases.get("meta") == "meta-ai"
    assert resolve_provider("meta-oauth") == "meta-oauth"
    assert resolve_provider("meta-subscription") == "meta-oauth"


def test_meta_oauth_flow_wired():
    from hermes_cli.auth import OAUTH_PROVIDER_FLOWS

    assert "meta-oauth" in OAUTH_PROVIDER_FLOWS
    flow = OAUTH_PROVIDER_FLOWS["meta-oauth"]
    assert flow.resolve_fn == "resolve_meta_oauth_runtime_credentials"
    assert flow.status_fn == "get_meta_oauth_auth_status"


def test_meta_device_code_login_mints_and_returns_tokens(monkeypatch):
    from hermes_cli import auth_meta as meta

    monkeypatch.setattr(meta, "start_device_authorization", lambda client=None: {
        "device_code": "device-code", "user_code": "ABCD-EFGH",
        "verification_uri": "https://auth.meta.com/device?code=ABCD-EFGH",
        "interval": 1, "expires_in": 600,
    })
    monkeypatch.setattr(meta, "poll_for_identity_token", lambda client, code, **kwargs: "identity-token")
    monkeypatch.setattr(meta, "mint_meta_api_key", lambda identity, client=None: {
        "access_token": "at-minted", "refresh_token": identity, "expires_at_ms": 9_999_999,
    })
    monkeypatch.setattr("hermes_cli.auth_device_flow.time.sleep", lambda _s: None)

    result = meta._meta_oauth_device_code_login(open_browser=False)
    assert result["tokens"]["access_token"] == "at-minted"
    assert result["tokens"]["refresh_token"] == "identity-token"
    assert result["base_url"] == "https://api.meta.ai/v1"


def test_meta_login_saves_tokens(tmp_path, monkeypatch):
    import json
    from types import SimpleNamespace
    from hermes_cli.auth_meta import _login_meta_oauth

    hermes_home = tmp_path / "hermes"
    hermes_home.mkdir(parents=True, exist_ok=True)
    (hermes_home / "auth.json").write_text(json.dumps({"version": 1, "providers": {}}))
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.setattr(
        "hermes_cli.auth_meta._meta_oauth_device_code_login",
        lambda **kwargs: {
            "tokens": {"access_token": "at-1", "refresh_token": "rt-1", "expires_at_ms": 9_999_999},
            "base_url": "https://api.meta.ai/v1", "last_refresh": "2026-09-20T00:00:00Z",
        },
    )
    monkeypatch.setattr(
        "hermes_cli.auth._update_config_for_provider", lambda *a, **k: "config.toml",
    )

    _login_meta_oauth(SimpleNamespace(no_browser=True, timeout=3), None, force_new_login=True)

    raw = json.loads((hermes_home / "auth.json").read_text())
    assert raw["providers"]["meta-oauth"]["tokens"]["access_token"] == "at-1"


def test_meta_auth_commands_wired():
    from hermes_cli.auth_commands import (
        _OAUTH_ADD_SPECS, _OAUTH_CAPABLE_PROVIDERS, _normalize_provider,
    )

    assert "meta-oauth" in _OAUTH_CAPABLE_PROVIDERS
    assert _normalize_provider("meta-subscription") == "meta-oauth"
    assert _normalize_provider("muse-subscription") == "meta-oauth"
    assert "meta-oauth" in _OAUTH_ADD_SPECS


def test_meta_catalog_lists_subscription():
    from hermes_cli.models import provider_model_ids
    from hermes_cli.models_catalog_static import CANONICAL_PROVIDERS, _PROVIDER_LABELS

    assert "meta-oauth" in _PROVIDER_LABELS
    assert any(p.slug == "meta-oauth" for p in CANONICAL_PROVIDERS)
    models = provider_model_ids("meta-oauth")
    assert "muse-spark-1.3" in models
    assert "muse-spark-1.3-contributor" in models
    assert "muse-spark-1.2" not in models


def _seed_meta_auth_store(hermes_home, tokens=None, last_refresh="2026-09-20T00:00:00Z"):
    import json

    store = {
        "version": 1,
        "providers": {
            "meta-oauth": {
                "tokens": tokens or {
                    "access_token": "store-at", "refresh_token": "store-rt",
                    "expires_at_ms": 9_999_999_999_999,
                },
                "last_refresh": last_refresh, "auth_mode": "oauth_device_code",
            }
        },
    }
    (hermes_home / "auth.json").write_text(json.dumps(store))


def test_meta_env_key_wins_over_auth_store(tmp_path, monkeypatch):
    from hermes_cli.auth_meta import resolve_meta_oauth_runtime_credentials

    hermes_home = tmp_path / "hermes"
    hermes_home.mkdir(parents=True, exist_ok=True)
    _seed_meta_auth_store(hermes_home)
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    for var in ("MODEL_API_KEY", "META_API_KEY", "META_MODEL_API_KEY"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("META_API_KEY", "env-key-1")

    creds = resolve_meta_oauth_runtime_credentials(refresh_if_expiring=False)

    assert creds["api_key"] == "env-key-1"
    assert creds["source"] == "env"


def test_meta_pool_seeds_from_auth_store(tmp_path, monkeypatch):
    from agent.credential_pool import load_pool

    hermes_home = tmp_path / "hermes"
    hermes_home.mkdir(parents=True, exist_ok=True)
    _seed_meta_auth_store(hermes_home)
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    for var in ("MODEL_API_KEY", "META_API_KEY", "META_MODEL_API_KEY"):
        monkeypatch.delenv(var, raising=False)

    pool = load_pool("meta-oauth")

    assert pool.has_credentials()
    entry = pool._entries[0]
    assert entry.source == "device_code"
    assert entry.access_token == "store-at"
    assert entry.base_url == "https://api.meta.ai/v1"


def test_meta_pool_refresh_uses_access_refresh_convention(monkeypatch):
    from hermes_cli.auth_meta import refresh_meta_oauth_pure

    monkeypatch.setattr("hermes_cli.auth_meta.time.time", lambda: 2_000.0)
    client = _MetaClient([_MetaResponse(200, {"api_key": "fresh-key"})])

    result = refresh_meta_oauth_pure("stale-access", "rt-identity", client=client)

    assert client.calls[0][1]["headers"]["Authorization"] == "Bearer rt-identity"
    assert result["access_token"] == "fresh-key"
    assert result["refresh_token"] == "rt-identity"


def test_meta_runtime_provider_resolves_oauth(tmp_path, monkeypatch):
    from hermes_cli.runtime_provider import resolve_runtime_provider

    hermes_home = tmp_path / "hermes"
    hermes_home.mkdir(parents=True, exist_ok=True)
    _seed_meta_auth_store(hermes_home)
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    for var in ("MODEL_API_KEY", "META_API_KEY", "META_MODEL_API_KEY"):
        monkeypatch.delenv(var, raising=False)

    runtime = resolve_runtime_provider(requested="meta-oauth")

    assert runtime["provider"] == "meta-oauth"
    assert runtime["api_mode"] == "codex_responses"
    assert runtime["base_url"] == "https://api.meta.ai/v1"
    assert runtime["api_key"] == "store-at"


def test_meta_oauth_transport_carries_retention():
    import agent.transports.codex  # noqa: F401
    from agent.transports import get_transport

    transport = get_transport("codex_responses")
    kw = transport.build_kwargs(
        model="muse-spark-1.3",
        messages=[{"role": "user", "content": "Hi"}],
        tools=[],
        base_url="https://api.meta.ai/v1",
        session_id="sid",
    )

    assert kw.get("prompt_cache_retention") == "24h"

def test_meta_refresh_remints_from_identity(monkeypatch):
    from hermes_cli.auth_meta import refresh_meta_oauth_pure

    monkeypatch.setattr("hermes_cli.auth_meta.time.time", lambda: 2_000.0)

    class _MintClient:
        def post(self, *args, **kwargs):
            return _MetaResponse(200, {"api_key": "at-new"})

    result = refresh_meta_oauth_pure("stale-access", "rt-identity", client=_MintClient())
    assert result["access_token"] == "at-new"
    assert result["refresh_token"] == "rt-identity"
    assert result["expires_at_ms"] == 2_000_000 + 24 * 60 * 60 * 1000
    assert result["last_refresh"]


def test_meta_resolve_uses_minted_key(tmp_path, monkeypatch):
    import json
    import time as _time
    from hermes_cli.auth_meta import (
        _save_meta_oauth_tokens, resolve_meta_oauth_runtime_credentials,
    )

    hermes_home = tmp_path / "hermes"
    hermes_home.mkdir(parents=True, exist_ok=True)
    (hermes_home / "auth.json").write_text(json.dumps({"version": 1, "providers": {}}))
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    _save_meta_oauth_tokens({
        "access_token": "at-live", "refresh_token": "rt-live",
        "expires_at_ms": int(_time.time() * 1000) + 20 * 60 * 60 * 1000,
    })

    creds = resolve_meta_oauth_runtime_credentials()
    assert creds["api_key"] == "at-live"
    assert creds["provider"] == "meta-oauth"
    assert creds["base_url"] == "https://api.meta.ai/v1"


def test_meta_resolve_refreshes_expiring_key(tmp_path, monkeypatch):
    import json
    import time as _time
    from hermes_cli.auth_meta import (
        _save_meta_oauth_tokens, resolve_meta_oauth_runtime_credentials,
    )

    hermes_home = tmp_path / "hermes"
    hermes_home.mkdir(parents=True, exist_ok=True)
    (hermes_home / "auth.json").write_text(json.dumps({"version": 1, "providers": {}}))
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    _save_meta_oauth_tokens({
        "access_token": "at-old", "refresh_token": "rt-identity",
        "expires_at_ms": int(_time.time() * 1000) - 1_000,
    })

    def _fake_refresh(access_token, refresh_token, **kwargs):
        assert refresh_token == "rt-identity"
        return {
            "access_token": "at-new", "refresh_token": "rt-identity",
            "expires_at_ms": int(_time.time() * 1000) + 24 * 60 * 60 * 1000,
            "last_refresh": "now",
        }

    monkeypatch.setattr("hermes_cli.auth_meta.refresh_meta_oauth_pure", _fake_refresh)
    creds = resolve_meta_oauth_runtime_credentials()
    assert creds["api_key"] == "at-new"

