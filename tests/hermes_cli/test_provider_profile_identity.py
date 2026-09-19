"""ProviderProfile identity/config regressions through discovery, CLI and runtime.

The on-disk discovery fixture is adapted from david-bowiegxw's #53054 tests
(commit fcbc30e5defc112a66ef834f6e2e4e46d3cbcec4). The current auth bridge
and transport fallback remain upstream-owned; these tests cover residual
identity and configuration behavior.
"""

from __future__ import annotations

import sys

import pytest
import yaml

from hermes_cli import auth as auth_mod
from hermes_cli import runtime_provider as rp
from hermes_cli.model_switch import switch_model
from hermes_cli.providers import get_provider, resolve_provider_full
from hermes_constants import get_hermes_home


PLUGIN_NAME = "testgw"
PLUGIN_ALIAS = "testgw-alias"
PLUGIN_OTHER_ALIAS = "testgw-other"
PLUGIN_ENV_VAR = "TESTGW_API_KEY"
PLUGIN_URL_VAR = "TESTGW_BASE_URL"
PLUGIN_BASE_URL = "https://gw.example.com/api/coding"
ENV_BASE_URL = "https://env.example.test/api/coding"
CONFIG_BASE_URL = "https://configured.example.test/api/coding"
TEST_KEY = "test-profile-credential"


@pytest.fixture(params=[PLUGIN_BASE_URL, ""], ids=["default-url", "env-only-url"])
def registered_plugin_provider(request, monkeypatch):
    """Discover an actual plugin file, then use the production auth bridge.

    Keep previously discovered bundled profiles and restore shared auth state;
    only this fixture's imported module is removed on teardown.
    """
    import providers as profiles

    monkeypatch.setattr(profiles, "_REGISTRY", dict(profiles._REGISTRY))
    monkeypatch.setattr(profiles, "_ALIASES", dict(profiles._ALIASES))
    monkeypatch.setattr(profiles, "_PROVIDER_LIST_CACHE", None)
    monkeypatch.setattr(profiles, "_discovered", False)
    auth_before = dict(auth_mod.PROVIDER_REGISTRY)
    module_name = "_hermes_user_provider_testgw"
    monkeypatch.delitem(sys.modules, module_name, raising=False)
    monkeypatch.setattr("agent.models_dev.fetch_models_dev", lambda *a, **kw: {})
    monkeypatch.setenv(PLUGIN_ENV_VAR, TEST_KEY)
    monkeypatch.delenv(PLUGIN_URL_VAR, raising=False)
    monkeypatch.delenv("TESTGW_URL", raising=False)
    if not request.param:
        monkeypatch.setenv(PLUGIN_URL_VAR, ENV_BASE_URL)

    plugin_dir = get_hermes_home() / "plugins" / "model-providers" / PLUGIN_NAME
    plugin_dir.mkdir(parents=True)
    (plugin_dir / "plugin.yaml").write_text(
        f"name: {PLUGIN_NAME}\nkind: model-provider\nversion: 0.0.1\n"
        "description: Provider identity regression fixture\n"
    )
    (plugin_dir / "__init__.py").write_text(
        "from providers import register_provider\n"
        "from providers.base import ProviderProfile\n\n"
        "register_provider(ProviderProfile(\n"
        f"    name={PLUGIN_NAME!r},\n"
        f"    aliases={(PLUGIN_ALIAS, PLUGIN_OTHER_ALIAS)!r},\n"
        "    display_name='Test Gateway',\n"
        "    api_mode='anthropic_messages',\n"
        f"    env_vars={(PLUGIN_URL_VAR, PLUGIN_ENV_VAR, 'TESTGW_URL')!r},\n"
        f"    base_url={request.param!r},\n"
        "    auth_type='api_key',\n"
        "))\n"
        "register_provider(ProviderProfile(\n"
        "    name='foreign-provider', env_vars=('FOREIGN_TESTGW_API_KEY',),\n"
        "    base_url='https://foreign.example.test/v1', api_mode='codex_responses',\n"
        "))\n"
        "register_provider(ProviderProfile(\n"
        "    name='testgw-process', auth_type='external_process',\n"
        "    base_url='acp+tcp://127.0.0.1:56789', env_vars=(),\n"
        "))\n"
    )
    try:
        for profile in profiles.list_providers():
            if profile.name not in auth_mod.PROVIDER_REGISTRY:
                auth_mod._register_plugin_provider(profile)
        profile = profiles.get_provider_profile(PLUGIN_NAME)
        assert profile is not None
        config = auth_mod.PROVIDER_REGISTRY[PLUGIN_NAME]
        assert config is auth_mod.PROVIDER_REGISTRY[PLUGIN_ALIAS]
        assert config.api_key_env_vars == (PLUGIN_ENV_VAR,)
        assert config.base_url_env_var == PLUGIN_URL_VAR
        yield profile
    finally:
        auth_mod.PROVIDER_REGISTRY.clear()
        auth_mod.PROVIDER_REGISTRY.update(auth_before)
        sys.modules.pop(module_name, None)


def _write_model_config(provider, **settings):
    (get_hermes_home() / "config.yaml").write_text(
        yaml.safe_dump({"model": {"provider": provider, "default": "test-model", **settings}})
    )


@pytest.mark.parametrize("requested", [PLUGIN_NAME, PLUGIN_ALIAS, PLUGIN_OTHER_ALIAS])
@pytest.mark.parametrize("override", [None, "canonical", "raw"])
def test_residual_profile_cli_identity(registered_plugin_provider, monkeypatch, requested, override):
    profile = registered_plugin_provider
    _write_model_config(PLUGIN_NAME)
    # Validation is an external model probe; discovery, auth, config and the
    # switch's runtime credential resolution are all real.
    monkeypatch.setattr(
        "hermes_cli.models_validate.validate_requested_model",
        lambda *a, **kw: {"accepted": True, "persist": True, "recognized": True, "message": None},
    )
    user_providers = {}
    expected_url = profile.base_url or ENV_BASE_URL
    expected_id = profile.name
    if override:
        user_providers[profile.name] = {
            "name": "Configured Gateway", "base_url": CONFIG_BASE_URL,
            "key_env": PLUGIN_ENV_VAR,
        }
        expected_url = CONFIG_BASE_URL
        if override == "raw":
            user_providers[requested] = {
                "name": "Raw Override", "base_url": "https://raw.example.test/api/coding",
                "key_env": PLUGIN_ENV_VAR,
            }
            expected_url = user_providers[requested]["base_url"]
            expected_id = requested

    result = switch_model(
        "test-model", current_provider="openrouter", current_model="previous-model",
        current_api_key="foreign-provider-key", explicit_provider=requested,
        user_providers=user_providers, custom_providers=[],
    )
    assert result.success, result.error_message
    assert result.target_provider == expected_id
    assert result.base_url == expected_url
    assert result.api_key == TEST_KEY

    definition = resolve_provider_full(requested, user_providers=user_providers)
    assert definition is not None
    assert definition.id == expected_id
    assert definition.api_key_env_vars == (PLUGIN_ENV_VAR,)
    if not override:
        assert definition.base_url_env_var == PLUGIN_URL_VAR
        assert definition.source == "plugin-profile"
        assert result.api_mode == profile.api_mode
    else:
        assert definition.source == "user-config"
        assert definition.base_url == expected_url


@pytest.mark.parametrize("route", ["pooled", "explicit-key", "no-pool"])
@pytest.mark.parametrize(
    "requested,persisted",
    [
        (PLUGIN_NAME, PLUGIN_NAME),
        (PLUGIN_NAME, PLUGIN_ALIAS),
        (PLUGIN_ALIAS, PLUGIN_NAME),
        (PLUGIN_ALIAS, PLUGIN_OTHER_ALIAS),
        (PLUGIN_OTHER_ALIAS, "foreign-provider"),
        (PLUGIN_OTHER_ALIAS, ""),
    ],
)
def test_residual_profile_runtime_config(registered_plugin_provider, monkeypatch, route, requested, persisted):
    from agent.credential_pool import CredentialPool

    profile = registered_plugin_provider
    # A URL without a configured provider deliberately selects the bare-custom
    # bypass. Leave it absent when testing the no-provider api_mode contract.
    _write_model_config(persisted, base_url=CONFIG_BASE_URL if persisted else "", api_mode="chat_completions")
    kwargs = {}
    if route == "explicit-key":
        kwargs["explicit_api_key"] = TEST_KEY
    elif route == "no-pool":
        # Force the fallback route with a real empty pool; credentials still
        # resolve through the real provider config and scoped environment.
        monkeypatch.setattr(rp, "load_pool", lambda provider: CredentialPool(provider, []))
    runtime = rp.resolve_runtime_provider(requested=requested, **kwargs)

    same_profile = persisted in (PLUGIN_NAME, PLUGIN_ALIAS, PLUGIN_OTHER_ALIAS)
    expected_url = profile.base_url or ENV_BASE_URL
    if same_profile and (route == "no-pool" or (route == "pooled" and profile.base_url)):
        expected_url = CONFIG_BASE_URL
    assert runtime["provider"] == profile.name
    assert runtime["requested_provider"] == requested
    assert runtime["base_url"] == expected_url
    assert runtime["api_key"] == TEST_KEY
    assert runtime["api_mode"] == ("chat_completions" if same_profile or not persisted else profile.api_mode)
    if route == "pooled":
        assert isinstance(runtime["credential_pool"], CredentialPool)
        assert runtime["source"] == f"env:{PLUGIN_ENV_VAR}"
    else:
        assert "credential_pool" not in runtime
        if route == "explicit-key":
            assert runtime["source"] == "explicit"


@pytest.mark.parametrize("route", ["pooled", "explicit-key", "no-pool", "explicit-url"])
def test_residual_profile_url_precedence(registered_plugin_provider, monkeypatch, route):
    from agent.credential_pool import CredentialPool

    profile = registered_plugin_provider
    _write_model_config(PLUGIN_ALIAS, base_url=CONFIG_BASE_URL)
    monkeypatch.setenv(PLUGIN_URL_VAR, "https://api.openai.com/v1")
    kwargs = {}
    if route in ("explicit-key", "explicit-url"):
        kwargs["explicit_api_key"] = TEST_KEY
    if route == "explicit-url":
        kwargs["explicit_base_url"] = PLUGIN_BASE_URL
    if route == "no-pool":
        monkeypatch.setattr(rp, "load_pool", lambda provider: CredentialPool(provider, []))

    runtime = rp.resolve_runtime_provider(requested=PLUGIN_OTHER_ALIAS, **kwargs)
    expected_url = {
        "pooled": "https://api.openai.com/v1", "explicit-key": "https://api.openai.com/v1",
        "no-pool": CONFIG_BASE_URL, "explicit-url": PLUGIN_BASE_URL,
    }[route]
    assert runtime["provider"] == profile.name
    assert runtime["base_url"] == expected_url
    assert runtime["api_mode"] == (
        "codex_responses" if route in ("pooled", "explicit-key") else profile.api_mode
    )


def test_residual_profile_url_is_not_a_credential(registered_plugin_provider, monkeypatch):
    from hermes_cli.auth import AuthError

    monkeypatch.delenv(PLUGIN_ENV_VAR)
    monkeypatch.setenv(PLUGIN_URL_VAR, ENV_BASE_URL)
    definition = get_provider(PLUGIN_ALIAS, allow_network=False)
    assert definition is not None
    assert definition.api_key_env_vars == (PLUGIN_ENV_VAR,)
    with pytest.raises(AuthError, match="No usable credentials"):
        rp.resolve_runtime_provider(requested=PLUGIN_ALIAS)


def test_residual_external_process_route_keeps_process_auth(registered_plugin_provider, monkeypatch):
    """Control: resolving an external profile does not require an API key or launch it."""
    _write_model_config(PLUGIN_ALIAS, base_url=CONFIG_BASE_URL, api_mode="anthropic_messages")
    monkeypatch.setattr(
        "hermes_cli.models_validate.validate_requested_model",
        lambda *a, **kw: {"accepted": True, "persist": True, "recognized": True, "message": None},
    )
    definition = get_provider("testgw-process", allow_network=False)
    assert definition is not None
    assert definition.auth_type == "external_process"
    assert definition.api_key_env_vars == ()
    runtime = rp.resolve_runtime_provider(requested=definition.id)
    assert runtime["source"] == "process"
    assert runtime["api_key"] == definition.id
    assert runtime["base_url"] == definition.base_url
    assert runtime["api_mode"] == "chat_completions"
    result = switch_model(
        "test-model", current_provider=PLUGIN_NAME, current_model="previous-model",
        explicit_provider=definition.id, user_providers={}, custom_providers=[],
    )
    assert result.success, result.error_message
    assert result.target_provider == definition.id
    assert result.api_key == runtime["api_key"]
    assert result.base_url == runtime["base_url"]
