"""Fallback string parsing and secret-safe malformed-entry warnings (#51560, #117806)."""

import logging

import pytest

from hermes_cli.fallback_config import _iter_fallback_entries, get_fallback_chain


def test_string_shorthand_is_shared_by_config_and_agent_parsers():
    raw = [
        "  nous : z-ai/glm-5.3-flash:free  ",
        {"provider": "openrouter", "model": "qwen/qwen3.6-plus"},
    ]
    expected = [
        {"provider": "nous", "model": "z-ai/glm-5.3-flash:free"},
        {"provider": "openrouter", "model": "qwen/qwen3.6-plus"},
    ]

    from agent.agent_init import _fallback_entries

    assert _iter_fallback_entries(raw) == expected
    assert _fallback_entries(raw) == expected
    assert get_fallback_chain({
        "fallback_providers": raw,
        "fallback_model": {"provider": "nous", "model": "z-ai/glm-5.3-flash:free"},
    }) == expected


def test_string_shorthand_supports_named_custom_providers_and_openrouter_presets():
    assert _iter_fallback_entries([
        "custom:local:qwen-2.5",
        "openrouter:@preset/team",
    ]) == [
        {"provider": "custom:local", "model": "qwen-2.5"},
        {"provider": "openrouter", "model": "@preset/team"},
    ]


def test_every_dropped_shape_warns_without_logging_values(caplog):
    secrets = (
        "FAKE_STRING_SECRET_SENTINEL_e0d4",
        "FAKE_API_KEY_SENTINEL_5d19",
        "FAKE_HEADER_SECRET_SENTINEL_74bc",
        "FAKE_URL_SECRET_SENTINEL_9f31",
    )
    raw = [
        f"malformed-{secrets[0]}",
        {
            "provider": "nous",
            "api_key": secrets[1],
            "extra_headers": {"Authorization": secrets[2]},
        },
        42,
        f"https://user:{secrets[3]}@host/v1",
    ]

    with caplog.at_level(logging.WARNING, logger="hermes_cli.fallback_config"):
        assert _iter_fallback_entries(raw) == []
        assert _iter_fallback_entries("openrouter:qwen/qwen3.6-plus") == [
            {"provider": "openrouter", "model": "qwen/qwen3.6-plus"}
        ]

    assert "entry[0] is a malformed string" in caplog.text
    assert "entry[1] (dict) missing 'model'" in caplog.text
    assert "entry[2] (int) is malformed" in caplog.text
    assert "entry[3] is a malformed string" in caplog.text
    assert "fallback configuration has a malformed root (str)" in caplog.text
    assert "effective fallback chain is EMPTY" in caplog.text
    assert not any(secret in caplog.text for secret in secrets)


@pytest.mark.parametrize("raw", [42, True, 3.5], ids=["int", "bool", "float"])
def test_malformed_scalar_root_warns_that_effective_chain_is_empty(raw, caplog):
    with caplog.at_level(logging.WARNING, logger="hermes_cli.fallback_config"):
        assert _iter_fallback_entries(raw) == []

    assert f"fallback configuration has a malformed root ({type(raw).__name__})" in caplog.text
    assert "effective fallback chain is EMPTY" in caplog.text


def test_combined_chain_warns_for_duplicates_without_leaking_values(caplog):
    secrets = ("FAKE_DUP_KEY_734b", "FAKE_DUP_HEADER_a981")
    duplicate = {
        "provider": "openrouter",
        "model": "qwen/qwen3.6-plus",
        "api_key": secrets[0],
        "extra_headers": {"Authorization": secrets[1]},
    }

    with caplog.at_level(logging.WARNING, logger="hermes_cli.fallback_config"):
        chain = get_fallback_chain({"fallback_providers": [duplicate, dict(duplicate)]})

    assert chain == [duplicate]
    assert "fallback_providers entry[1] is a duplicate" in caplog.text
    assert "entry dropped" in caplog.text
    assert not any(secret in caplog.text for secret in secrets)


@pytest.mark.parametrize(
    "malformed_primary",
    [
        42,
        {"provider": "openrouter", "api_key": "FAKE_MALFORMED_PRIMARY_SECRET_25ad"},
    ],
    ids=["scalar-root", "secret-bearing-entry"],
)
def test_valid_legacy_entry_prevents_false_empty_chain_warning(malformed_primary, caplog):
    secret = "FAKE_MALFORMED_PRIMARY_SECRET_25ad"
    config = {
        "fallback_providers": malformed_primary,
        "fallback_model": {"provider": "nous", "model": "backup-model"},
    }

    with caplog.at_level(logging.WARNING, logger="hermes_cli.fallback_config"):
        assert get_fallback_chain(config) == [
            {"provider": "nous", "model": "backup-model"}
        ]

    assert "effective fallback chain is EMPTY" not in caplog.text
    assert secret not in caplog.text


def test_malformed_entry_warnings_name_their_config_source(caplog):
    config = {
        "fallback_providers": ["missing-colon"],
        "fallback_model": {"provider": "nous"},
    }

    with caplog.at_level(logging.WARNING, logger="hermes_cli.fallback_config"):
        assert get_fallback_chain(config) == []

    assert "fallback_providers entry[0] is a malformed string" in caplog.text
    assert "fallback_model entry[0] (dict) missing 'model'" in caplog.text


def test_malformed_root_warnings_name_their_config_source(caplog):
    with caplog.at_level(logging.WARNING, logger="hermes_cli.fallback_config"):
        assert get_fallback_chain({
            "fallback_providers": 42,
            "fallback_model": True,
        }) == []

    assert "fallback_providers has a malformed root (int)" in caplog.text
    assert "fallback_model has a malformed root (bool)" in caplog.text
