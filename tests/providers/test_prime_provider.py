"""Prime Inference uses the shared provider discovery and runtime paths."""

import json
from urllib.error import URLError
from unittest.mock import MagicMock

import pytest

from providers import get_provider_profile


@pytest.mark.parametrize("name", ("prime", "prime-inference", "primeintellect", "pinference"))
def test_prime_aliases_resolve_credentials_and_slash_model_ids(name, monkeypatch):
    from hermes_cli.auth import resolve_api_key_provider_credentials, resolve_provider
    from hermes_cli.models import parse_model_input
    from hermes_cli.runtime_provider import resolve_runtime_provider

    monkeypatch.setenv("PRIME_API_KEY", "prime-test-key")
    profile = get_provider_profile(name)
    assert profile is get_provider_profile("prime")
    assert resolve_provider(name) == profile.name
    credentials = resolve_api_key_provider_credentials(name)
    assert credentials["api_key"] == "prime-test-key"
    assert credentials["base_url"] == profile.base_url

    model = "example/chat-model"
    assert parse_model_input(f"{name}:{model}", "openrouter") == (profile.name, model)
    runtime = resolve_runtime_provider(requested=name, target_model=model)
    assert runtime["provider"] == profile.name
    assert runtime["api_mode"] == profile.api_mode == "chat_completions"
    assert runtime["base_url"] == profile.base_url
    assert runtime["api_key"] == "prime-test-key"


def test_prime_appears_in_picker_and_setup_env_metadata():
    from hermes_cli.config import OPTIONAL_ENV_VARS
    from hermes_cli.models import list_available_providers

    profile = get_provider_profile("prime")
    entry = next(p for p in list_available_providers() if p["id"] == profile.name)
    assert entry["label"] == profile.display_name
    assert set(profile.aliases) <= set(entry["aliases"])
    assert all(env in OPTIONAL_ENV_VARS for env in profile.env_vars)
    assert profile.default_aux_model in profile.fallback_models
    assert profile.get_hostname() == "api.pinference.ai"


def test_prime_live_catalog_preserves_model_ids(monkeypatch):
    from hermes_cli.models import provider_model_ids

    profile = get_provider_profile("prime")
    modalities = {"input": ["text"], "output": ["text"]}
    excluded = ("example/no-tools", "example/decision", "example/unknown")
    response = MagicMock()
    response.__enter__.return_value.read.return_value = json.dumps(
        {"object": "list", "data": [
            {"id": "example/live-chat", "supported_parameters": ["tools"], "specs": {"modalities": modalities}},
            {"id": excluded[0], "supported_parameters": [], "specs": {"modalities": modalities}},
            {"id": excluded[1], "supported_parameters": ["tools"],
             "specs": {"modalities": {"input": ["text"], "output": ["decisions"]}}},
            {"id": excluded[2]},
        ]}
    ).encode()
    opener = MagicMock(return_value=response)
    monkeypatch.setattr("hermes_cli.urllib_security.open_credentialed_url", opener)

    assert profile.fetch_models() == ["example/live-chat"]
    request = opener.call_args.args[0]
    assert request.full_url == profile.base_url + "/models"
    assert request.get_header("Authorization") is None

    monkeypatch.setenv("PRIME_API_KEY", "prime-test-key")
    offered = provider_model_ids(profile.name)
    assert "example/live-chat" in offered
    assert not set(excluded).intersection(offered)


def test_prime_configured_proxy_preserves_generic_catalog(monkeypatch):
    profile = get_provider_profile("prime")
    response = MagicMock()
    response.__enter__.return_value.read.return_value = b'{"data": [{"id": "proxy/chat"}]}'
    opener = MagicMock(return_value=response)
    monkeypatch.setattr("hermes_cli.urllib_security.open_credentialed_url", opener)

    assert profile.fetch_models(api_key="proxy-key", base_url="https://proxy.example/v1/") == ["proxy/chat"]
    request = opener.call_args.args[0]
    assert request.full_url == "https://proxy.example/v1/models"
    assert request.get_header("Authorization") == "Bearer proxy-key"


def test_prime_catalog_outage_uses_profile_fallback(monkeypatch):
    from agent import models_dev
    from hermes_cli.models import provider_model_ids

    profile = get_provider_profile("prime")
    monkeypatch.setenv("PRIME_API_KEY", "prime-test-key")
    monkeypatch.setattr(models_dev, "fetch_models_dev", lambda **_: {})
    opener = MagicMock(side_effect=URLError("catalog unavailable"))
    monkeypatch.setattr("hermes_cli.urllib_security.open_credentialed_url", opener)
    assert profile.fallback_models
    assert provider_model_ids(profile.name) == list(profile.fallback_models)


def test_prime_uses_matching_models_dev_metadata(monkeypatch):
    from agent import models_dev
    from agent.model_metadata import _infer_provider_from_url

    profile = get_provider_profile("prime")
    model = "example/catalog-chat"
    catalog = {
        profile.name: {
            "models": {
                model: {
                    "name": "Catalog chat",
                    "tool_call": True,
                    "reasoning": True,
                    "limit": {"context": 123456, "output": 4096},
                    "cost": {"input": 0.5, "output": 1.5},
                },
            },
        },
    }
    monkeypatch.setattr(models_dev, "fetch_models_dev", lambda **kwargs: catalog)

    assert _infer_provider_from_url(profile.base_url) == profile.name
    assert models_dev.list_agentic_models(profile.name, allow_network=False) == [model]
    assert models_dev.lookup_models_dev_context(profile.name, model) == 123456
    info = models_dev.get_model_info(profile.name, model)
    assert info.provider_id == profile.name
    assert info.reasoning and info.tool_call
    assert info.context_window == 123456
    assert info.cost_input == 0.5 and info.cost_output == 1.5
