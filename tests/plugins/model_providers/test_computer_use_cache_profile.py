"""Tests for the Computer-Use Cache provider profile.

The profile is intentionally small: Computer-Use Cache is an OpenAI-compatible
proxy, so Hermes should treat it like a normal chat-completions endpoint while
letting users override the local base URL from config.
"""

from __future__ import annotations


def test_computer_use_cache_profile_registered():
    import providers

    profile = providers.get_provider_profile("computer-use-cache")
    assert profile is not None
    assert profile.name == "computer-use-cache"
    assert profile.api_mode == "chat_completions"
    assert profile.display_name == "Computer-Use Cache"
    assert profile.base_url == "http://127.0.0.1:8000/v1"
    assert profile.env_vars == ("COMPUTER_USE_CACHE_API_KEY",)


def test_computer_use_cache_aliases_resolve():
    import providers

    assert providers.get_provider_profile("cuc").name == "computer-use-cache"
    assert providers.get_provider_profile("computer_use_cache").name == "computer-use-cache"
    assert providers.get_provider_profile("code-model-cache").name == "computer-use-cache"


def test_computer_use_cache_auto_registers_api_key_provider():
    from hermes_cli.auth import PROVIDER_REGISTRY

    config = PROVIDER_REGISTRY["computer-use-cache"]
    assert config.auth_type == "api_key"
    assert config.api_key_env_vars == ("COMPUTER_USE_CACHE_API_KEY",)
    assert config.base_url_env_var == ""
    assert config.inference_base_url == "http://127.0.0.1:8000/v1"
