import pytest

from hermes_cli.profile_provider_sharing import (
    ProfileProviderSharingConfig,
    ProfileProviderSharingError,
    builtin_provider_share_env_vars,
    custom_provider_share_env_vars,
    is_profile_provider_sharing_enabled,
    parse_profile_provider_sharing_config,
)


def test_missing_config_is_disabled():
    assert parse_profile_provider_sharing_config({}) == ProfileProviderSharingConfig()
    assert is_profile_provider_sharing_enabled({}) is False


def test_boolean_true_enables_default_capability_set():
    parsed = parse_profile_provider_sharing_config({
        "profiles": {"share_model_providers": True}
    })

    assert parsed.enabled is True
    assert parsed.source_profile == "default"
    assert parsed.capabilities == frozenset({
        "provider_env",
        "provider_base_urls",
        "providers",
        "excluded_providers",
    })


def test_mapping_accepts_source_profile_and_comma_capabilities():
    parsed = parse_profile_provider_sharing_config({
        "profiles": {
            "share_model_providers": {
                "enabled": True,
                "source_profile": "fleet-base",
                "capabilities": "provider_env, providers",
            }
        }
    })

    assert parsed == ProfileProviderSharingConfig(
        enabled=True,
        source_profile="fleet-base",
        capabilities=frozenset({"provider_env", "providers"}),
    )


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        (
            {"profiles": {"share_model_providers": "yes"}},
            "boolean or mapping",
        ),
        (
            {"profiles": {"share_model_providers": {"enabled": "yes"}}},
            "enabled must be true or false",
        ),
        (
            {
                "profiles": {
                    "share_model_providers": {
                        "enabled": True,
                        "source_profile": "../default",
                    }
                }
            },
            "source_profile contains invalid characters",
        ),
        (
            {
                "profiles": {
                    "share_model_providers": {
                        "enabled": True,
                        "capabilities": ["provider_env", "telegram_tokens"],
                    }
                }
            },
            "unknown values: telegram_tokens",
        ),
        (
            {
                "profiles": {
                    "share_model_providers": {
                        "enabled": True,
                        "capabilities": [],
                    }
                }
            },
            "at least one value",
        ),
    ],
)
def test_malformed_schema_raises_descriptive_error(payload, message):
    with pytest.raises(ProfileProviderSharingError, match=message):
        parse_profile_provider_sharing_config(payload)


def test_builtin_provider_env_scope_includes_keys_and_base_urls():
    env_vars = builtin_provider_share_env_vars(["deepseek", "alibaba", "unknown"])

    assert "DEEPSEEK_API_KEY" in env_vars
    assert "DEEPSEEK_BASE_URL" in env_vars
    assert "DASHSCOPE_API_KEY" in env_vars
    assert "DASHSCOPE_BASE_URL" in env_vars


def test_custom_provider_env_scope_includes_key_and_base_url_placeholders():
    env_vars = custom_provider_share_env_vars({
        "acme": {
            "key_env": "ACME_API_KEY",
            "base_url": "${ACME_BASE_URL}",
        },
        "literal": {
            "api_key_env": "LITERAL_API_KEY",
            "base_url": "https://api.example.com/v1",
        },
        "ignored": {"key_env": ""},
    })

    assert env_vars == {"ACME_API_KEY", "ACME_BASE_URL", "LITERAL_API_KEY"}
