"""Picker hide-log when a custom_providers URL collides with a built-in endpoint.

Section 4 of the model picker silently skips a custom row whose normalized URL
matches a built-in (including when that builtin URL comes from an env override
such as OPENAI_BASE_URL). Issue #107012: emit a discoverable warning naming
the hidden custom, the shadowing builtin slug, and the env var when applicable.
"""

import logging

import hermes_cli.providers as providers_mod
from hermes_cli.model_switch import list_authenticated_providers


def _isolate_picker(monkeypatch) -> None:
    monkeypatch.setattr("agent.models_dev.fetch_models_dev", lambda: {})
    monkeypatch.setattr(providers_mod, "HERMES_OVERLAYS", {})
    monkeypatch.setattr("hermes_cli.models.fetch_api_models", lambda *a, **k: [])
    monkeypatch.setattr("hermes_cli.models.cached_provider_model_ids", lambda *a, **k: [])
    monkeypatch.setattr("hermes_cli.models.provider_model_ids", lambda *a, **k: [])
    monkeypatch.setattr("hermes_cli.models_local.fetch_ollama_local_models", lambda *a, **k: None)


def _list_rows(custom_providers):
    return list_authenticated_providers(
        current_provider="openai-api",
        user_providers={},
        custom_providers=custom_providers,
        max_models=50,
        probe_custom_providers=False,
    )


def test_env_base_url_collision_logs_hidden_custom(monkeypatch, caplog):
    _isolate_picker(monkeypatch)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-openai")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://ai.example.com/v1")

    custom_providers = [
        {
            "name": "helunox",
            "base_url": "https://ai.example.com/v1",
            "api_key": "k",
            "models": ["m1"],
        }
    ]

    with caplog.at_level(logging.INFO, logger="hermes_cli.model_switch"):
        rows = _list_rows(custom_providers)

    assert any(r.get("slug") == "openai-api" for r in rows), (
        "openai-api builtin must be present so the env URL is recorded"
    )
    assert not any(
        str(r.get("slug", "")).startswith("custom:helunox") or r.get("name") == "helunox"
        for r in rows
    )
    assert any(
        "helunox" in m and ("OPENAI_BASE_URL" in m or "openai-api" in m)
        for m in caplog.messages
    )


def test_env_base_url_no_collision_shows_custom(monkeypatch, caplog):
    """CONTROL: a custom URL that does not match the env-overridden builtin stays visible."""
    _isolate_picker(monkeypatch)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-openai")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://ai.example.com/v1")

    custom_providers = [
        {
            "name": "other-lab",
            "base_url": "https://other.example.com/v1",
            "api_key": "k",
            "models": ["m2"],
        }
    ]

    with caplog.at_level(logging.INFO, logger="hermes_cli.model_switch"):
        rows = _list_rows(custom_providers)

    assert any(
        str(r.get("slug", "")).startswith("custom:other-lab") or r.get("name") == "other-lab"
        for r in rows
    )
    assert not any(
        "other-lab" in m and ("OPENAI_BASE_URL" in m or "openai-api" in m)
        for m in caplog.messages
    )
