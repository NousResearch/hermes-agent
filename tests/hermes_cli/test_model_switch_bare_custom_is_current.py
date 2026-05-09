"""Regression for #20810: bare ``custom`` current_provider must mark its
custom_providers row as ``is_current`` in the picker.

When config.yaml's active provider is bare ``"custom"`` (e.g. pointing at a
local Ollama / llama-swap instance), ``list_authenticated_providers`` resolves
its slug to ``custom:<name>``, but the ``is_current`` check at emit time used
``slug == current_provider`` — ``"custom:foo" == "custom"`` is ``False``, so
the actual active provider was rendered without the current marker and a
different provider (often the first built-in) carried the ``● current``
indicator instead.

Fix: track in the group whether its endpoint matched the active
``current_base_url`` and OR that flag into ``is_current`` at emit time.
"""

import hermes_cli.providers as providers_mod
from hermes_cli.model_switch import list_authenticated_providers


def test_bare_custom_current_provider_marks_matching_endpoint(monkeypatch):
    monkeypatch.setattr("agent.models_dev.fetch_models_dev", lambda: {})
    monkeypatch.setattr(providers_mod, "HERMES_OVERLAYS", {})

    providers = list_authenticated_providers(
        current_provider="custom",
        current_base_url="http://127.0.0.1:4141/v1",
        user_providers={},
        custom_providers=[
            {
                "name": "Local LLama-Swap",
                "base_url": "http://127.0.0.1:4141/v1",
                "model": "rotator-openrouter-coding",
            }
        ],
        max_models=50,
    )

    custom_rows = [p for p in providers if p.get("source") == "user-config"]
    assert custom_rows, "expected the custom_providers entry in the picker output"
    assert any(p["is_current"] for p in custom_rows), (
        "bare 'custom' current_provider must mark the matching custom row as current"
    )

    current_rows = [p for p in providers if p.get("is_current")]
    assert len(current_rows) == 1, (
        f"exactly one row should be marked current, got {[p['slug'] for p in current_rows]}"
    )
    assert current_rows[0]["api_url"] == "http://127.0.0.1:4141/v1"


def test_non_matching_endpoint_is_not_marked_current(monkeypatch):
    monkeypatch.setattr("agent.models_dev.fetch_models_dev", lambda: {})
    monkeypatch.setattr(providers_mod, "HERMES_OVERLAYS", {})

    providers = list_authenticated_providers(
        current_provider="custom",
        current_base_url="http://127.0.0.1:4141/v1",
        user_providers={},
        custom_providers=[
            {
                "name": "Other Endpoint",
                "base_url": "http://127.0.0.1:9999/v1",
                "model": "some-model",
            }
        ],
        max_models=50,
    )

    other = [p for p in providers if p.get("api_url") == "http://127.0.0.1:9999/v1"]
    assert other, "expected the non-matching custom row to be present"
    assert not other[0]["is_current"], (
        "non-matching endpoint must not be marked current"
    )
