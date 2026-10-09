from providers.configured import (
    configured_custom_identity,
    expand_direct_api_alias,
    match_configured_provider,
    resolves_to_custom_provider,
)


def test_configured_provider_prefers_keyed_provider_over_legacy_entry():
    match = match_configured_provider(
        "custom:alpha",
        providers={
            "alpha": {
                "name": "Alpha",
                "api": "https://new.example/v1",
                "transport": "responses",
                "default_model": "new-model",
            }
        },
        custom_providers=[
            {
                "name": "Alpha",
                "base_url": "https://legacy.example/v1",
                "model": "old-model",
            }
        ],
    )

    assert match is not None
    assert match.source == "providers"
    assert match.identity == "custom:alpha"
    assert match.base_url == "https://new.example/v1"
    assert match.api_mode == "codex_responses"
    assert match.model == "new-model"


def test_disabled_keyed_provider_falls_through_to_legacy_entry():
    match = match_configured_provider(
        "alpha",
        providers={
            "alpha": {
                "enabled": False,
                "api": "https://disabled.example/v1",
            }
        },
        custom_providers=[
            {"name": "alpha", "base_url": "https://legacy.example/v1"}
        ],
    )

    assert match is not None
    assert match.source == "custom_providers"
    assert match.base_url == "https://legacy.example/v1"


def test_canonical_builtin_shadows_same_named_custom_entry():
    match = match_configured_provider(
        "nous",
        providers={"nous": {"api": "https://custom-nous.example/v1"}},
    )

    assert match is None


def test_builtin_alias_can_still_name_a_custom_entry():
    match = match_configured_provider(
        "kimi",
        providers={"kimi": {"api": "https://custom-kimi.example/v1"}},
    )

    assert match is not None
    assert match.identity == "custom:kimi"


def test_custom_alias_classification_is_provider_owned():
    assert resolves_to_custom_provider("ollama") is True
    assert resolves_to_custom_provider("vllm") is True
    assert resolves_to_custom_provider("nous") is False
    assert resolves_to_custom_provider("auto") is False


def test_direct_openai_alias_uses_caller_supplied_endpoint_facts():
    assert expand_direct_api_alias(
        "openai",
        None,
        preferred_base_url="https://proxy.example/v1/",
    ) == ("custom", "https://proxy.example/v1")
    assert expand_direct_api_alias("openai", None) == (
        "custom",
        "https://api.openai.com/v1",
    )


def test_configured_openai_provider_is_not_rewritten():
    assert expand_direct_api_alias(
        "openai",
        None,
        configured_provider=True,
        preferred_base_url="https://proxy.example/v1",
    ) == ("openai", None)


def test_configured_custom_identity_uses_lower_provider_facts():
    providers = {
        "alpha": {
            "name": "Alpha",
            "api": "https://alpha.example/v1",
            "default_model": "alpha-model",
        }
    }
    legacy = [
        {
            "name": "Beta",
            "base_url": "https://beta.example/v1",
            "models": ["beta-model"],
        }
    ]

    assert configured_custom_identity(
        base_url="https://alpha.example/v1/",
        providers=providers,
        custom_providers=legacy,
    ) == "custom:alpha"
    assert configured_custom_identity(
        model="beta-model",
        providers=providers,
        custom_providers=legacy,
    ) == "custom:beta"
    assert configured_custom_identity(
        config_provider="alpha",
        providers=providers,
        custom_providers=legacy,
    ) == "custom:alpha"
