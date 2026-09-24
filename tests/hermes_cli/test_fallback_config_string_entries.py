"""Fallback string parsing and secret-safe malformed-entry warnings (#51560, #117806)."""

import logging

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


def test_every_dropped_shape_warns_without_logging_values(caplog):
    secrets = (
        "FAKE_STRING_SECRET_SENTINEL_e0d4",
        "FAKE_API_KEY_SENTINEL_5d19",
        "FAKE_HEADER_SECRET_SENTINEL_74bc",
    )
    raw = [
        f"malformed-{secrets[0]}",
        {
            "provider": "nous",
            "api_key": secrets[1],
            "extra_headers": {"Authorization": secrets[2]},
        },
        42,
    ]

    with caplog.at_level(logging.WARNING, logger="hermes_cli.fallback_config"):
        assert _iter_fallback_entries(raw) == []
        assert _iter_fallback_entries(42) == []
        assert _iter_fallback_entries("openrouter:qwen/qwen3.6-plus") == [
            {"provider": "openrouter", "model": "qwen/qwen3.6-plus"}
        ]

    assert "entry[0] is a malformed string" in caplog.text
    assert "entry[1] (dict) missing 'model'" in caplog.text
    assert "entry[2] (int) is malformed" in caplog.text
    assert "Malformed fallback root (int)" in caplog.text
    assert "Malformed fallback root (str)" in caplog.text
    assert "effective fallback chain is EMPTY" in caplog.text
    assert not any(secret in caplog.text for secret in secrets)
