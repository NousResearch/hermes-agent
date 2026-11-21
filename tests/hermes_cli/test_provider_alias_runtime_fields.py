"""Provider aliases retain profile fields without weakening stale-field guards.

Regression coverage for #145475: ``model.provider: moonshot`` is normalized to
``kimi-coding`` when the runtime selects the provider, so the saved API mode and
relay endpoint must still belong to that route.
"""

from __future__ import annotations

import pytest


@pytest.mark.parametrize(
    ("provider", "configured_provider", "expected"),
    [
        ("kimi-coding", "moonshot", True),
        ("moonshot", "kimi-coding", True),
        ("kimi-coding", "kimi-coding", True),
        ("kimi-coding", "openai", False),
        ("not-a-provider", "also-not-one", False),
    ],
)
def test_provider_identity_alias_matrix(provider, configured_provider, expected):
    """Aliases and canonical names share identity; unrelated and unknown names do not."""
    from hermes_cli.runtime_provider import _same_registered_provider

    assert _same_registered_provider(provider, configured_provider) is expected


@pytest.mark.parametrize(
    ("configured_provider", "requested_provider", "keeps_fields"),
    [
        ("moonshot", "kimi-coding", True),
        ("kimi-coding", "kimi-coding", True),
        ("openai", "kimi-coding", False),
    ],
)
def test_runtime_resolver_preserves_only_same_provider_config_fields(
    monkeypatch, configured_provider, requested_provider, keeps_fields
):
    """The real config writer and resolver preserve both fields only for one route."""
    from hermes_cli.config import atomic_config_write, get_config_path
    from hermes_cli.runtime_provider import resolve_runtime_provider

    relay = "https://relay.example.invalid/v1"
    monkeypatch.setenv("KIMI_API_KEY", "kimi-fixture-token")
    atomic_config_write(
        get_config_path(),
        {
            "model": {
                "provider": configured_provider,
                "api_mode": "codex_responses",
                "base_url": relay,
                "default": "kimi-k2.5",
            }
        },
    )

    runtime = resolve_runtime_provider(requested=requested_provider)

    assert runtime["provider"] == "kimi-coding"
    if keeps_fields:
        assert runtime["api_mode"] == "codex_responses"
        assert runtime["base_url"] == relay
    else:
        assert runtime["api_mode"] != "codex_responses"
        assert runtime["base_url"] != relay
