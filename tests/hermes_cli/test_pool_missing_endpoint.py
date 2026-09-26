"""Endpoint-less manual credentials resolve through the public runtime ladder."""
import pytest
import hermes_yaml as yaml

from hermes_cli.auth import PROVIDER_REGISTRY, write_credential_pool
from hermes_cli.runtime_provider import resolve_runtime_provider
from hermes_constants import get_hermes_home


KEY = "test-only-not-a-real-provider-key"


def _store_entry(provider, **fields):
    write_credential_pool(provider, [{
        "id": "test-no-endpoint", "source": "manual", "auth_type": "api_key",
        "access_token": KEY, "priority": 0, **fields,
    }])


@pytest.mark.parametrize("endpoint", [None, "", "default"])
@pytest.mark.parametrize("requested", ["opencode", "opencode-zen"])
def test_missing_pool_endpoint_uses_provider_default(endpoint, requested):
    default = PROVIDER_REGISTRY["opencode-zen"].inference_base_url
    fields = {} if endpoint is None else {"base_url": default if endpoint == "default" else endpoint}
    _store_entry("opencode-zen", **fields)
    runtime = resolve_runtime_provider(requested=requested, target_model="glm-5.3")
    assert runtime["base_url"] == default
    assert runtime["api_key"] == KEY
    assert runtime["source"] == "manual"
    assert runtime["api_mode"] == "chat_completions"


@pytest.mark.parametrize("entry_url", [None, "default", "https://entry.example/v1"])
@pytest.mark.parametrize("config_provider", ["deepseek", "openai"])
def test_pool_endpoint_preserves_entry_and_provider_config_boundaries(entry_url, config_provider):
    default = PROVIDER_REGISTRY["deepseek"].inference_base_url
    config_url = "https://config.example/v1"
    (get_hermes_home() / "config.yaml").write_text(yaml.safe_dump({
        "model": {"provider": config_provider, "base_url": config_url},
    }), encoding="utf-8")
    fields = {} if entry_url is None else {"base_url": default if entry_url == "default" else entry_url}
    _store_entry("deepseek", **fields)
    runtime = resolve_runtime_provider(requested="deepseek", target_model="deepseek-chat")
    expected = entry_url if entry_url and entry_url != "default" else (
        config_url if config_provider == "deepseek" else default
    )
    assert runtime["base_url"] == expected
    assert runtime["api_key"] == KEY
    assert runtime["source"] == "manual"


def test_missing_endpoint_still_finalizes_anthropic_wire():
    _store_entry("opencode-zen")
    runtime = resolve_runtime_provider(requested="opencode-zen", target_model="claude-sonnet-4-6")
    default = PROVIDER_REGISTRY["opencode-zen"].inference_base_url
    assert runtime["api_mode"] == "anthropic_messages"
    assert runtime["base_url"] == default.removesuffix("/v1")
    assert runtime["api_key"] == KEY
