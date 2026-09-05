"""Contract tests for the OpenGateway provider profile."""

from __future__ import annotations


def _profile():
    import model_tools  # noqa: F401
    import providers

    profile = providers.get_provider_profile("opengateway")
    assert profile is not None
    return profile


def test_identity_endpoint_and_auth():
    profile = _profile()

    assert profile.name == "opengateway"
    assert profile.api_mode == "chat_completions"
    assert profile.auth_type == "api_key"
    assert profile.base_url == "https://apis.opengateway.ai/v1"
    assert profile.env_vars == ("OPENGATEWAY_API_KEY",)


def test_models_are_live_discovered_with_known_agentic_fallbacks():
    profile = _profile()

    assert profile.fallback_models
    assert "moonshotai/kimi-k3-ultrafast" in profile.fallback_models
    assert "z-ai/glm-5.2-ultrafast" in profile.fallback_models
