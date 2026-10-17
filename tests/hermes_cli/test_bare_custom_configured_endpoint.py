"""Regression for #140405: bare custom must retain its configured main route."""

import pytest

from hermes_cli import runtime_provider as rp


@pytest.fixture
def configured_route():
    return {
        "model": {"provider": "local-lane", "default": "local-model"},
        "providers": {
            "local-lane": {
                "base_url": "http://127.0.0.1:12345/v1",
                "default_model": "local-model",
                "api_key": "synthetic-local-key",
                "api_mode": "chat_completions",
            },
        },
    }


@pytest.mark.parametrize("cloud_key", [False, True])
def test_bare_custom_uses_named_main_endpoint_and_credential(configured_route, monkeypatch, cloud_key):
    if cloud_key:
        monkeypatch.setenv("OPENROUTER_API_KEY", "synthetic-cloud-key")
    with rp.frozen_runtime_config(configured_route):
        named = rp.resolve_runtime_provider(requested="local-lane", target_model="local-model")
        bare = rp.resolve_runtime_provider(requested="custom", target_model="local-model")
    assert bare["base_url"] == named["base_url"]
    assert bare["api_key"] == named["api_key"]
    assert bare["api_mode"] == named["api_mode"]
    assert bare["provider"] == "custom"


@pytest.mark.parametrize("override", ["explicit", "model", "env", "literal-custom", "foreign-model", "missing-model", "disabled", "cloud-provider"])
def test_bare_custom_recovery_preserves_existing_route_boundaries(configured_route, monkeypatch, override):
    alternate = "http://127.0.0.1:23456/v1"
    kwargs = {"requested": "custom", "target_model": "local-model"}
    expected = alternate
    monkeypatch.setenv("OPENROUTER_API_KEY", "synthetic-cloud-key")
    if override == "explicit":
        kwargs["explicit_base_url"] = alternate
    elif override == "model":
        configured_route["model"]["base_url"] = alternate
    elif override == "env":
        monkeypatch.setenv("CUSTOM_BASE_URL", alternate)
    elif override == "literal-custom":
        configured_route["providers"]["custom"] = {"base_url": alternate}
    elif override == "foreign-model":
        kwargs["target_model"] = "another-model"
        expected = "https://openrouter.ai/api/v1"
    elif override == "missing-model":
        kwargs.pop("target_model")
        expected = "https://openrouter.ai/api/v1"
    elif override == "disabled":
        configured_route["providers"]["local-lane"]["enabled"] = False
        expected = "https://openrouter.ai/api/v1"
    else:
        configured_route["model"]["provider"] = "openrouter"
        expected = "https://openrouter.ai/api/v1"
    with rp.frozen_runtime_config(configured_route):
        runtime = rp.resolve_runtime_provider(**kwargs)
    assert runtime["base_url"] == expected
    if expected != configured_route["providers"]["local-lane"]["base_url"]:
        assert runtime["api_key"] != "synthetic-local-key"
