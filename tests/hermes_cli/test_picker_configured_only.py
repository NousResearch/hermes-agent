"""Tests for ``picker_configured_only`` on ``list_authenticated_providers``.

When the flag is on, the /model picker should hide built-in / curated rows
even when their credentials are present, and skip the live ``/v1/models``
override that normally replaces a user-defined endpoint's configured
``models:`` list.  See #13796.
"""

from hermes_cli.model_switch import list_authenticated_providers


def _suppress_curated_rows(monkeypatch):
    """Common monkeypatching to neutralize the section 1/2/2b machinery so a
    test only has to assert what reaches results, not what was filtered out
    by credential checks. ``picker_configured_only`` should still work even
    without these patches; they keep tests fast and offline."""
    monkeypatch.setattr("agent.models_dev.fetch_models_dev", lambda: {})
    monkeypatch.setattr("hermes_cli.providers.HERMES_OVERLAYS", {})


def test_picker_configured_only_hides_credentialed_builtin_rows(monkeypatch):
    """A user with ANTHROPIC_API_KEY but no ``providers:``/``custom_providers:``
    should see an empty picker when picker_configured_only is on, because no
    row in the result is user-defined."""
    monkeypatch.setattr("agent.models_dev.fetch_models_dev", lambda: {"anthropic": {}})
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")

    without_filter = list_authenticated_providers(
        user_providers={},
        custom_providers=[],
    )
    with_filter = list_authenticated_providers(
        user_providers={},
        custom_providers=[],
        picker_configured_only=True,
    )

    assert any(p["slug"] == "anthropic" for p in without_filter)
    assert with_filter == []


def test_picker_configured_only_keeps_user_provider_rows(monkeypatch):
    """``providers:`` entries (section 3) must survive the filter."""
    _suppress_curated_rows(monkeypatch)

    user_providers = {
        "local-ollama": {
            "name": "Local Ollama",
            "api": "http://localhost:11434/v1",
            "default_model": "qwen3.5:cloud",
            "models": ["qwen3.5:cloud", "kimi-k2.5:cloud"],
        }
    }

    providers = list_authenticated_providers(
        user_providers=user_providers,
        custom_providers=[],
        picker_configured_only=True,
    )

    assert [p["slug"] for p in providers] == ["local-ollama"]
    assert providers[0]["models"] == ["qwen3.5:cloud", "kimi-k2.5:cloud"]


def test_picker_configured_only_keeps_custom_provider_rows(monkeypatch):
    """``custom_providers:`` entries (section 4) must survive the filter."""
    _suppress_curated_rows(monkeypatch)

    custom_providers = [
        {
            "name": "Remote Cloud",
            "base_url": "https://example.com/v1",
            "model": "gpt-5.4",
        }
    ]

    providers = list_authenticated_providers(
        user_providers={},
        custom_providers=custom_providers,
        picker_configured_only=True,
    )

    assert [p["slug"] for p in providers] == ["custom:remote-cloud"]
    assert providers[0]["models"] == ["gpt-5.4"]


def test_picker_configured_only_skips_live_models_override(monkeypatch):
    """Section 3 normally calls ``fetch_api_models`` and overwrites the
    configured ``models:`` list with the upstream catalog. Under the
    configured-only flag, that override is skipped so the picker shows
    exactly what the user wrote in config."""
    _suppress_curated_rows(monkeypatch)
    monkeypatch.setenv("CRS_TEST_KEY", "sk-test")

    fetch_calls: list[tuple] = []

    def fake_fetch_api_models(api_key, base_url):  # pragma: no cover - asserted via call list
        fetch_calls.append((api_key, base_url))
        return ["server-extra-1", "server-extra-2"]

    monkeypatch.setattr("hermes_cli.models.fetch_api_models", fake_fetch_api_models)

    user_providers = {
        "crs": {
            "name": "CRS",
            "base_url": "http://127.0.0.1:3000/api/v1",
            "key_env": "CRS_TEST_KEY",
            "model": "configured-a",
            "models": {
                "configured-a": {"context_length": 200000},
                "configured-b": {"context_length": 200000},
            },
        }
    }

    providers = list_authenticated_providers(
        user_providers=user_providers,
        custom_providers=[],
        picker_configured_only=True,
    )

    assert fetch_calls == []
    assert [p["slug"] for p in providers] == ["crs"]
    assert providers[0]["models"] == ["configured-a", "configured-b"]


def test_picker_configured_only_off_preserves_existing_behaviour(monkeypatch):
    """Default (flag off) keeps live override + built-in rows — regression
    guard so this PR doesn't accidentally invert defaults."""
    _suppress_curated_rows(monkeypatch)
    monkeypatch.setenv("CRS_TEST_KEY", "sk-test")
    monkeypatch.setattr(
        "hermes_cli.models.fetch_api_models",
        lambda api_key, base_url: ["live-only"],
    )

    user_providers = {
        "crs": {
            "base_url": "http://127.0.0.1:3000/api/v1",
            "key_env": "CRS_TEST_KEY",
            "model": "configured-a",
            "models": {"configured-a": {}},
        }
    }

    providers = list_authenticated_providers(
        user_providers=user_providers,
        custom_providers=[],
    )

    crs = next(p for p in providers if p["slug"] == "crs")
    assert crs["models"] == ["live-only"]
