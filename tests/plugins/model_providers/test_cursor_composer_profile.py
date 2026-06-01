"""Cursor Composer provider profile tests."""

from providers import get_provider_profile
from hermes_cli.auth import PROVIDER_REGISTRY, resolve_provider
from hermes_cli.models import CANONICAL_PROVIDERS, provider_model_ids


def test_cursor_composer_profile_forces_agent_mode_route():
    profile = get_provider_profile("cursor-composer")

    assert profile is not None
    assert profile.base_url == "https://cursor-api.standardagents.ai/opencode/v1"
    assert profile.api_mode == "chat_completions"
    assert profile.env_vars == ("CURSOR_API_KEY", "CURSOR_COMPOSER_BASE_URL")
    assert "composer-2.5" in profile.fallback_models


def test_cursor_composer_auto_registers_auth_and_aliases():
    config = PROVIDER_REGISTRY["cursor-composer"]

    assert config.auth_type == "api_key"
    assert config.api_key_env_vars == ("CURSOR_API_KEY",)
    assert config.base_url_env_var == "CURSOR_COMPOSER_BASE_URL"
    assert resolve_provider("cursor") == "cursor-composer"
    assert resolve_provider("api-for-cursor") == "cursor-composer"


def test_cursor_composer_is_available_in_picker_and_models():
    slugs = [entry.slug for entry in CANONICAL_PROVIDERS]

    assert "cursor-composer" in slugs
    assert provider_model_ids("cursor-composer") == [
        "composer-2.5",
        "composer-2.5-fast",
        "composer-2",
        "composer-latest",
    ]
