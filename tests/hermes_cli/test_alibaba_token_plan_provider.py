"""Regression tests for Alibaba Token Plan provider wiring."""

from hermes_cli.auth import PROVIDER_REGISTRY, resolve_provider
from hermes_cli.providers import HERMES_OVERLAYS, TRANSPORT_TO_API_MODE, get_label, get_provider
from hermes_cli.runtime_provider import _detect_api_mode_for_url


TOKEN_PLAN_ANTHROPIC_URL = "https://token-plan.ap-southeast-1.maas.aliyuncs.com/apps/anthropic"


def test_auth_registry_uses_token_plan_anthropic_endpoint():
    cfg = PROVIDER_REGISTRY["alibaba-token-plan"]

    assert cfg.inference_base_url == TOKEN_PLAN_ANTHROPIC_URL
    assert cfg.api_key_env_vars == ("ALIBABA_TOKEN_PLAN_API_KEY",)
    assert cfg.base_url_env_var == "ALIBABA_TOKEN_PLAN_BASE_URL"


def test_provider_overlay_uses_anthropic_messages_transport():
    overlay = HERMES_OVERLAYS["alibaba-token-plan"]

    assert overlay.transport == "anthropic_messages"
    assert TRANSPORT_TO_API_MODE[overlay.transport] == "anthropic_messages"
    assert overlay.base_url_override == TOKEN_PLAN_ANTHROPIC_URL
    assert overlay.extra_env_vars == ("ALIBABA_TOKEN_PLAN_API_KEY",)


def test_provider_definition_exposes_base_url_key_and_label():
    provider = get_provider("alibaba_token_plan")

    assert provider is not None
    assert provider.id == "alibaba-token-plan"
    assert provider.base_url == TOKEN_PLAN_ANTHROPIC_URL
    assert provider.api_key_env_vars == ("ALIBABA_TOKEN_PLAN_API_KEY",)
    assert get_label("alibaba_token") == "Alibaba Token Plan"


def test_token_plan_aliases_resolve_to_canonical_provider():
    assert resolve_provider("alibaba_token") == "alibaba-token-plan"
    assert resolve_provider("alibaba-token") == "alibaba-token-plan"
    assert resolve_provider("alibaba_token_plan") == "alibaba-token-plan"


def test_token_plan_anthropic_url_auto_detects_api_mode():
    assert _detect_api_mode_for_url(TOKEN_PLAN_ANTHROPIC_URL) == "anthropic_messages"
