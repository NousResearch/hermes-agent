"""Plugin ProviderProfiles resolve wherever a provider NAME is resolved at switch time (#69576).

Discovery is real (a plugin dir under the test HERMES_HOME); only the remote model probe is stubbed.
"""

from __future__ import annotations

import sys

import pytest
import yaml

from hermes_cli import auth as auth_mod
from hermes_cli.model_switch import switch_model
from hermes_cli.providers import resolve_provider_full
from hermes_constants import get_hermes_home

KEY = "test-profile-credential"
ACCEPT = {"accepted": True, "persist": True, "recognized": True, "message": None}


@pytest.fixture
def install_profile(monkeypatch):
    """Write one model-provider plugin, rediscover, and mirror it into the auth registry."""
    import providers as profiles

    monkeypatch.setattr(profiles, "_REGISTRY", dict(profiles._REGISTRY))
    monkeypatch.setattr(profiles, "_ALIASES", dict(profiles._ALIASES))
    monkeypatch.setattr(profiles, "_PROVIDER_LIST_CACHE", None)
    monkeypatch.setattr("agent.models_dev.fetch_models_dev", lambda *a, **kw: {})
    monkeypatch.setattr("hermes_cli.models_validate.validate_requested_model", lambda *a, **kw: ACCEPT)
    auth_before = dict(auth_mod.PROVIDER_REGISTRY)

    def _install(name: str, *, aliases=(), base_url: str, env_var: str, api_mode: str = "chat_completions"):
        monkeypatch.setenv(env_var, KEY)
        monkeypatch.setattr(profiles, "_discovered", False)
        monkeypatch.delitem(sys.modules, f"_hermes_user_provider_{name}", raising=False)
        plugin_dir = get_hermes_home() / "plugins" / "model-providers" / name
        plugin_dir.mkdir(parents=True, exist_ok=True)
        (plugin_dir / "plugin.yaml").write_text(
            f"name: {name}\nkind: model-provider\nversion: 0.0.1\ndescription: switch fixture\n", encoding="utf-8")
        (plugin_dir / "__init__.py").write_text(
            "from providers import register_provider\nfrom providers.base import ProviderProfile\n"
            f"register_provider(ProviderProfile(name={name!r}, aliases={tuple(aliases)!r}, auth_type='api_key',\n"
            f"    env_vars=({env_var!r},), base_url={base_url!r}, api_mode={api_mode!r}))\n", encoding="utf-8")
        auth_mod.sync_plugin_provider_registry()
        return profiles.get_provider_profile(name)

    yield _install
    auth_mod.PROVIDER_REGISTRY.clear()
    auth_mod.PROVIDER_REGISTRY.update(auth_before)


def test_alias_switch_carries_the_profiles_canonical_identity(install_profile):
    """`/model --provider <alias>` lands on the profile's own name (persisted config, credential and
    runtime lookups all key on it); a user-configured block for that name still wins over the profile."""
    install_profile("testgw", aliases=("testgw-alias",), base_url="https://gw.example.test/v1",
                    env_var="TESTGW_API_KEY", api_mode="anthropic_messages")

    pdef = resolve_provider_full("testgw-alias", user_providers={}, custom_providers=[])
    assert (pdef.id, pdef.source) == ("testgw", "plugin-profile")
    result = switch_model("test-model", current_provider="openrouter", current_model="m",
                          explicit_provider="testgw-alias", user_providers={}, custom_providers=[])
    assert result.success, result.error_message
    assert (result.target_provider, result.api_mode, result.api_key) == ("testgw", "anthropic_messages", KEY)

    user = {"testgw": {"name": "Configured", "base_url": "https://cfg.example.test/v1", "key_env": "TESTGW_API_KEY"}}
    assert resolve_provider_full("testgw-alias", user_providers=user).source == "user-config"


def test_runtime_endpoint_profile_is_a_known_provider_but_placeholders_are_not(install_profile):
    """A profile that mints its endpoint at runtime (empty base_url) resolves on the last rung, so the
    switch accepts it; the bare ``custom`` placeholder and its local aliases still fall through to the
    custom_providers / current-endpoint handling."""
    install_profile("mintgw", base_url="", env_var="MINTGW_API_KEY")

    pdef = resolve_provider_full("mintgw", user_providers={}, custom_providers=[])
    assert (pdef.id, pdef.base_url, pdef.source) == ("mintgw", "", "plugin-profile")
    result = switch_model("test-model", current_provider="openrouter", current_model="m",
                          explicit_provider="mintgw", user_providers={}, custom_providers=[])
    assert result.success, result.error_message
    assert (result.target_provider, result.api_key) == ("mintgw", KEY)
    for placeholder in ("custom", "ollama", "custom:nope"):
        assert resolve_provider_full(placeholder, user_providers={}, custom_providers=[]) is None


def test_persisted_alias_config_applies_to_the_canonical_runtime_provider(install_profile):
    """config.yaml written under an alias (``provider: testgw-alias``) is the same provider at runtime:
    its base_url override applies when the canonical name is requested."""
    from hermes_cli import runtime_provider as rp

    install_profile("testgw", aliases=("testgw-alias",), base_url="https://gw.example.test/v1", env_var="TESTGW_API_KEY")
    (get_hermes_home() / "config.yaml").write_text(yaml.safe_dump(
        {"model": {"provider": "testgw-alias", "default": "test-model", "base_url": "https://cfg.example.test/v1"}}),
        encoding="utf-8")

    runtime = rp.resolve_runtime_provider(requested="testgw")
    assert (runtime["provider"], runtime["base_url"]) == ("testgw", "https://cfg.example.test/v1")
