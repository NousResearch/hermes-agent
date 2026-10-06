"""litco.config_guard: a matter host's model route is product-controlled (LKP-1014)."""

from __future__ import annotations

import pytest

from litco.config_guard import ProductModelGuardError, assert_product_controlled

PRODUCT_CONFIG_TEXT = 'model:\n  default: "moonshotai/kimi-k3"\n  provider: "openrouter"\n'
PRODUCT_CONFIG = {"model": {"default": "moonshotai/kimi-k3", "provider": "openrouter"}}
PRODUCT_ENV = {"LITCO_MATTER_ID": "m1", "OPENROUTER_API_KEY": "sk-or-v1-SECRET"}


def _refused(config=PRODUCT_CONFIG, text=PRODUCT_CONFIG_TEXT, env=PRODUCT_ENV, auth=None) -> str:
    with pytest.raises(ProductModelGuardError) as info:
        assert_product_controlled(config, text, env, auth)
    return str(info.value)


def test_a_product_profile_passes():
    assert_product_controlled(PRODUCT_CONFIG, PRODUCT_CONFIG_TEXT, PRODUCT_ENV, None)
    # an explicitly empty chain and an API-key-only auth store are fine
    assert_product_controlled({**PRODUCT_CONFIG, "fallback_providers": []},
                              PRODUCT_CONFIG_TEXT + "fallback_providers: []\n", PRODUCT_ENV,
                              {"credential_pool": {"openrouter": [{"auth_type": "api_key", "access_token": "x"}],
                                                   "anthropic": [{"auth_type": "api_key", "access_token": "sk-ant-api"}]}})


@pytest.mark.parametrize("token", ["CLIProxy", "tail999258", "claude-max", "ncc1701d", "santacruz", ".ts.net:8317"])
def test_config_text_naming_a_proxy_route_is_refused(token):
    assert token.lower() in _refused(text=PRODUCT_CONFIG_TEXT + f"# routed via {token}\n").lower()


@pytest.mark.parametrize("key", ["LITCO_MODEL_BASE_URL", "OPENAI_BASE_URL", "ANTHROPIC_BASE_URL", "GEMINI_BASE_URL"])
def test_an_endpoint_variable_pointing_at_the_proxy_is_refused_without_quoting_it(key):
    url = "https://santacruz.tail999258.ts.net:8317/v1?key=sk-SECRET"
    message = _refused(env={**PRODUCT_ENV, key: url})
    assert key in message and url not in message and "sk-SECRET" not in message


def test_secret_values_are_never_inspected():
    # a key that happens to contain a token is a secret, not an endpoint
    assert_product_controlled(PRODUCT_CONFIG, PRODUCT_CONFIG_TEXT, {**PRODUCT_ENV, "OPENROUTER_API_KEY": "claude-max"})


@pytest.mark.parametrize("config, text", [
    ({**PRODUCT_CONFIG, "fallback_providers": [{"provider": "anthropic", "model": "claude-opus-5"}]}, PRODUCT_CONFIG_TEXT),
    ({**PRODUCT_CONFIG, "fallback_model": {"provider": "openrouter", "model": "x/y"}}, PRODUCT_CONFIG_TEXT),
    (PRODUCT_CONFIG, PRODUCT_CONFIG_TEXT + "fallback_providers:\n  - provider: openrouter\n    model: x/y\n"),
    (PRODUCT_CONFIG, PRODUCT_CONFIG_TEXT + "fallback_model: {provider: openrouter, model: x/y}\n"),
])
def test_a_profile_fallback_chain_is_refused(config, text):
    assert "fallback chain" in _refused(config=config, text=text)


def test_model_base_url_on_a_proxy_is_refused():
    config = {"model": {**PRODUCT_CONFIG["model"], "base_url": "http://ncc1701d.local:8317/v1"}}
    assert "model.base_url" in _refused(config=config, text="")


@pytest.mark.parametrize("auth", [
    {"providers": {"anthropic": {"tokens": {"access_token": "sk-ant-oat01-x"}}}},
    {"providers": {"openai-codex": {"tokens": {"access_token": "a", "refresh_token": "r"}}}},
    {"credential_pool": {"anthropic": [{"auth_type": "oauth", "access_token": "x"}]}},
    {"credential_pool": {"anthropic": [{"access_token": "sk-ant-oat01-abc"}]}},
    {"credential_pool": {"openai-codex": [{"access_token": "a"}]}},
])
def test_a_subscription_account_in_auth_json_is_refused(auth):
    assert "subscription account" in _refused(auth=auth)
