"""Canonical provider identity contract."""

from dataclasses import FrozenInstanceError

import pytest

import providers
from providers import (
    ResolvedProvider,
    get_provider_label,
    is_aggregator,
    is_routing_aggregator,
    normalize_provider,
)
from providers.base import ProviderProfile


@pytest.fixture
def identity_profiles(monkeypatch):
    router = ProviderProfile(
        name="fixture-router",
        aliases=("fixture-route",),
        display_name="Fixture Router",
        is_aggregator=True,
    )
    reseller = ProviderProfile(
        name="fixture-reseller",
        aliases=("fixture-flat",),
        display_name="Fixture Reseller",
        is_aggregator=True,
        is_routing_aggregator=False,
    )
    direct = ProviderProfile(name="fixture-direct", display_name="Fixture Direct")
    by_name = {
        router.name: router,
        "fixture-route": router,
        reseller.name: reseller,
        "fixture-flat": reseller,
        direct.name: direct,
    }
    monkeypatch.setattr(providers, "get_provider_profile", by_name.get)
    return by_name


def test_identity_surface_is_exported_from_providers() -> None:
    assert providers.ResolvedProvider is ResolvedProvider
    assert providers.normalize_provider is normalize_provider
    assert providers.get_provider_label is get_provider_label
    assert providers.is_aggregator is is_aggregator
    assert providers.is_routing_aggregator is is_routing_aggregator


def test_normalize_provider_uses_registered_profile_aliases(identity_profiles) -> None:
    assert normalize_provider(" FIXTURE-ROUTE ") == "fixture-router"
    assert normalize_provider("unknown-provider") == "unknown-provider"
    assert normalize_provider("") == ""


def test_provider_label_comes_from_profile(identity_profiles) -> None:
    assert get_provider_label("fixture-route") == "Fixture Router"
    assert get_provider_label("unknown-provider") == "unknown-provider"


def test_aggregator_semantics_follow_profile_declaration(identity_profiles) -> None:
    assert is_aggregator("fixture-route") is True
    assert is_routing_aggregator("fixture-route") is True
    assert is_aggregator("fixture-flat") is True
    assert is_routing_aggregator("fixture-flat") is False
    assert is_aggregator("fixture-direct") is False
    assert is_routing_aggregator("fixture-direct") is False


def test_named_custom_routes_are_routing_aggregators(identity_profiles) -> None:
    assert normalize_provider("custom:fixture") == "custom:fixture"
    assert is_aggregator("custom:fixture") is True
    assert is_routing_aggregator("custom:fixture") is True


def test_bundled_profiles_own_representative_identity_declarations() -> None:
    openrouter = providers.get_provider_profile("openrouter")
    assert openrouter is not None
    assert openrouter.base_url_env_var == "OPENROUTER_BASE_URL"
    assert openrouter.is_aggregator is True
    assert is_routing_aggregator("openrouter") is True

    opencode = providers.get_provider_profile("opencode")
    assert opencode is not None
    assert opencode.name == "opencode-zen"
    assert opencode.is_aggregator is True
    assert opencode.is_routing_aggregator is False

    openai = providers.get_provider_profile("openai-api")
    assert openai is not None
    assert openai.display_name == "OpenAI API"
    assert openai.api_mode == "codex_responses"
    assert openai.base_url_env_var == "OPENAI_BASE_URL"

    xai_oauth = providers.get_provider_profile("grok-oauth")
    assert xai_oauth is not None
    assert xai_oauth.name == "xai-oauth"
    assert xai_oauth.auth_type == "oauth_external"

    moa = providers.get_provider_profile("moa")
    assert moa is not None
    assert moa.auth_type == "virtual"
    assert moa.base_url == "moa://local"

    tokenplan = providers.get_provider_profile("tencent-tokenplan")
    assert tokenplan is not None
    assert tokenplan.api_mode == "anthropic_messages"
    assert tokenplan.base_url_env_var == "TOKENPLAN_BASE_URL"


def test_resolved_provider_is_an_immutable_value_object() -> None:
    resolved = ResolvedProvider(
        id="fixture-router",
        display_name="Fixture Router",
        api_mode="chat_completions",
        auth_type="api_key",
        env_vars=("FIXTURE_API_KEY",),
        base_url="https://fixture.invalid/v1",
        base_url_env_var="FIXTURE_BASE_URL",
        is_aggregator=True,
        is_routing_aggregator=True,
        source="provider-profile",
    )

    assert resolved.id == "fixture-router"
    assert resolved.env_vars == ("FIXTURE_API_KEY",)
    with pytest.raises(FrozenInstanceError):
        resolved.id = "other"
