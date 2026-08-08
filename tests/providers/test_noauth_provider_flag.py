"""Tests for auth_type="none" (no-auth provider profiles).

No-auth providers (e.g. free tiers that reject Authorization headers
entirely) must resolve with an empty api_key in every credential path —
the CLI runtime resolver, the auxiliary client router, and the client
bootstrap — while auth-requiring providers keep the existing key gate.
"""

import pytest

from providers import ProviderProfile
import providers
from providers import get_provider_profile


@pytest.fixture(autouse=True)
def isolate_provider_registry():
    registry = providers._REGISTRY.copy()
    aliases = providers._ALIASES.copy()
    provider_list_cache = (
        None
        if providers._PROVIDER_LIST_CACHE is None
        else list(providers._PROVIDER_LIST_CACHE)
    )
    discovered = providers._discovered

    from hermes_cli.auth import PROVIDER_REGISTRY
    auth_registry = PROVIDER_REGISTRY.copy()

    yield

    providers._REGISTRY.clear()
    providers._REGISTRY.update(registry)
    providers._ALIASES.clear()
    providers._ALIASES.update(aliases)
    providers._PROVIDER_LIST_CACHE = provider_list_cache
    providers._discovered = discovered
    PROVIDER_REGISTRY.clear()
    PROVIDER_REGISTRY.update(auth_registry)


@pytest.fixture()
def noauth_profile():
    profile = ProviderProfile(
        name="noauth-preview",
        aliases=("np",),
        base_url="https://preview.example.com/v1",
        auth_type="none",
    )
    providers.register_provider(profile)
    _register_in_registry(profile)
    return profile


@pytest.fixture()
def keyed_profile():
    profile = ProviderProfile(
        name="keyed-preview",
        aliases=("kp",),
        env_vars=("KEYED_PREVIEW_KEY",),
        base_url="https://keyed.example.com/v1",
        auth_type="api_key",
    )
    providers.register_provider(profile)
    _register_in_registry(profile)
    return profile


def _register_in_registry(profile):
    """Mirror auth.py's PROVIDER_REGISTRY auto-extend for test profiles."""
    from hermes_cli.auth import PROVIDER_REGISTRY, ProviderConfig

    if profile.name in PROVIDER_REGISTRY:
        return
    PROVIDER_REGISTRY[profile.name] = ProviderConfig(
        id=profile.name,
        name=profile.display_name or profile.name,
        auth_type=profile.auth_type,
        inference_base_url=profile.base_url,
        api_key_env_vars=profile.env_vars if profile.auth_type == "api_key" else (),
    )
    for alias in profile.aliases:
        if alias not in PROVIDER_REGISTRY:
            PROVIDER_REGISTRY[alias] = PROVIDER_REGISTRY[profile.name]


class TestNoAuthProvider:
    def test_auth_type_defaults_to_api_key(self):
        assert ProviderProfile(name="x").auth_type == "api_key"

    def test_resolver_returns_empty_key_for_noauth(self, noauth_profile):
        from hermes_cli.auth import resolve_api_key_provider_credentials

        creds = resolve_api_key_provider_credentials("noauth-preview")
        assert creds["api_key"] == ""
        assert creds["source"] == "no-auth"

    def test_resolver_sanitizes_env_key_for_noauth(self, noauth_profile, monkeypatch):
        """Even if an env var is set, auth_type=none must force api_key=''.
        Free tiers reject Authorization headers entirely."""
        monkeypatch.setenv("NOAUTH_PREVIEW_KEY", "should-never-be-sent")
        from hermes_cli.auth import resolve_api_key_provider_credentials

        creds = resolve_api_key_provider_credentials("noauth-preview")
        assert creds["api_key"] == ""

    def test_resolver_keeps_gate_for_keyed_provider(self, keyed_profile, monkeypatch):
        from hermes_cli.auth import resolve_api_key_provider_credentials

        # A keyed provider with its env var set must keep the key — the
        # no-auth sanitization must not apply to api_key providers.
        monkeypatch.setenv("KEYED_PREVIEW_KEY", "real-key")
        creds = resolve_api_key_provider_credentials("keyed-preview")
        assert creds["api_key"] == "real-key"
        assert creds["source"] == "KEYED_PREVIEW_KEY"

    def test_runtime_resolves_noauth_without_auth_error(self, noauth_profile):
        from hermes_cli.runtime_provider import resolve_runtime_provider

        runtime = resolve_runtime_provider(requested="noauth-preview")
        assert runtime["provider"] == "noauth-preview"
        assert runtime["api_key"] == ""
        assert runtime["base_url"] == "https://preview.example.com/v1"

    def test_runtime_drops_explicit_api_key_for_noauth(self, noauth_profile):
        """explicit_api_key must be discarded for auth_type=none providers."""
        from hermes_cli.runtime_provider import resolve_runtime_provider

        runtime = resolve_runtime_provider(
            requested="noauth-preview",
            explicit_api_key="must-not-leak",
        )
        assert runtime["api_key"] == ""

    def test_runtime_raises_auth_error_for_keyed_provider(self, keyed_profile):
        from hermes_cli.auth import AuthError
        from hermes_cli.runtime_provider import resolve_runtime_provider

        with pytest.raises(AuthError):
            resolve_runtime_provider(requested="keyed-preview")

    def test_auxiliary_client_builds_noauth_client(self, noauth_profile):
        from agent.auxiliary_client import resolve_provider_client

        client, model = resolve_provider_client("noauth-preview", "noauth-model")
        assert client is not None
        assert client.api_key == ""
        assert str(client.base_url).startswith("https://preview.example.com")

    def test_auxiliary_client_drops_explicit_api_key_for_noauth(self, noauth_profile):
        """explicit_api_key must be discarded for auth_type=none providers."""
        from agent.auxiliary_client import resolve_provider_client

        client, model = resolve_provider_client(
            "noauth-preview", "noauth-model", explicit_api_key="must-not-leak"
        )
        assert client is not None
        assert client.api_key == ""

    def test_auxiliary_client_returns_none_for_keyed_provider(self, keyed_profile):
        from agent.auxiliary_client import resolve_provider_client

        client, model = resolve_provider_client("keyed-preview", "keyed-model")
        assert (client, model) == (None, None)

    def test_status_shows_configured_for_noauth(self, noauth_profile):
        from hermes_cli.auth import get_api_key_provider_status

        status = get_api_key_provider_status("noauth-preview")
        assert status["configured"] is True
        assert status["logged_in"] is True

    def test_noauth_registers_without_env_vars(self):
        """auth_type=none must register in PROVIDER_REGISTRY even without env_vars."""
        profile = ProviderProfile(
            name="noauth-no-env",
            base_url="https://noenv.example.com/v1",
            auth_type="none",
        )
        providers.register_provider(profile)
        _register_in_registry(profile)

        from hermes_cli.auth import PROVIDER_REGISTRY
        assert "noauth-no-env" in PROVIDER_REGISTRY

    def test_fetch_models_called_without_api_key(self, noauth_profile, monkeypatch):
        """models.py must call fetch_models for auth_type=none even with empty key."""
        from hermes_cli.models import provider_model_ids

        calls = []

        def _fake_fetch_models(*args, **kwargs):
            calls.append((args, kwargs))
            return ["live-model"]

        # Replace the profile's fetch_models with a recording stub so the test
        # proves the hook actually runs for auth_type=none (the outer gate
        # `auth_type in ("api_key", "none")` + inner `api_key or none` path).
        monkeypatch.setattr(noauth_profile, "fetch_models", _fake_fetch_models)
        ids = provider_model_ids("noauth-preview", force_refresh=True)
        assert "live-model" in ids
        assert calls, "fetch_models was never called for auth_type=none"
