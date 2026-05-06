"""Regression test for issue #20811.

When ``current_provider`` is the bare string ``"custom"`` and the resolved
slug for a user-defined custom provider is ``"custom:<name>"``, the
``/model`` picker still has to mark that row as the current one. The
match falls back to comparing ``current_base_url`` against the group's
endpoint URL.
"""

import hermes_cli.providers as providers_mod
from hermes_cli.model_switch import list_authenticated_providers


def test_bare_custom_provider_marked_current_via_base_url(monkeypatch):
    """``current_provider="custom"`` + matching ``current_base_url`` → is_current."""
    monkeypatch.setattr("agent.models_dev.fetch_models_dev", lambda: {})
    monkeypatch.setattr(providers_mod, "HERMES_OVERLAYS", {})

    providers = list_authenticated_providers(
        current_provider="custom",
        current_base_url="http://127.0.0.1:4141/v1",
        user_providers={},
        custom_providers=[
            {
                "name": "llama-swap",
                "base_url": "http://127.0.0.1:4141/v1",
                "model": "rotator-openrouter-coding",
            }
        ],
        max_models=50,
    )

    custom_rows = [p for p in providers if p["slug"].startswith("custom:")]
    assert custom_rows, "Expected at least one custom provider row"
    assert any(
        p["api_url"] == "http://127.0.0.1:4141/v1" and p["is_current"]
        for p in custom_rows
    ), "Custom provider matching current_base_url must have is_current=True"


def test_bare_custom_provider_url_match_is_case_insensitive(monkeypatch):
    """Trailing slash + uppercase host must still match."""
    monkeypatch.setattr("agent.models_dev.fetch_models_dev", lambda: {})
    monkeypatch.setattr(providers_mod, "HERMES_OVERLAYS", {})

    providers = list_authenticated_providers(
        current_provider="custom",
        current_base_url="http://LocalHost:4141/v1/",
        user_providers={},
        custom_providers=[
            {
                "name": "llama-swap",
                "base_url": "http://localhost:4141/v1",
                "model": "test-model",
            }
        ],
        max_models=50,
    )

    assert any(
        p["slug"].startswith("custom:") and p["is_current"]
        for p in providers
    )


def test_non_matching_base_url_does_not_set_is_current(monkeypatch):
    """If neither the slug nor the URL matches, is_current stays False."""
    monkeypatch.setattr("agent.models_dev.fetch_models_dev", lambda: {})
    monkeypatch.setattr(providers_mod, "HERMES_OVERLAYS", {})

    providers = list_authenticated_providers(
        current_provider="custom",
        current_base_url="http://127.0.0.1:9999/v1",
        user_providers={},
        custom_providers=[
            {
                "name": "llama-swap",
                "base_url": "http://127.0.0.1:4141/v1",
                "model": "test-model",
            }
        ],
        max_models=50,
    )

    custom_rows = [p for p in providers if p["slug"].startswith("custom:")]
    assert custom_rows, "Expected the row to be present"
    assert all(not p["is_current"] for p in custom_rows)
