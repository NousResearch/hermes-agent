"""Tests for ``list_picker_providers`` — the /model picker filter.

``list_picker_providers`` wraps ``list_authenticated_providers`` and
post-processes the result for interactive pickers (Telegram, Discord):

- OpenRouter's ``models`` are replaced with the live-filtered output of
  ``fetch_openrouter_models``, so IDs the live catalog no longer carries
  drop out.
- Provider rows with an empty ``models`` list are dropped, except custom
  endpoints (``is_user_defined=True`` with an ``api_url``) where the user
  may supply their own model set through config.

These tests exercise the filter in isolation by mocking
``list_authenticated_providers`` and ``fetch_openrouter_models`` so no
network or auth state is required.
"""

import pytest
from hermes_cli import model_switch


def _make_provider(slug, name=None, models=None, *, is_current=False,
                   is_user_defined=False, source="built-in", api_url=None):
    """Build a dict shaped like ``list_authenticated_providers`` output."""
    entry = {
        "slug": slug,
        "name": name or slug.title(),
        "is_current": is_current,
        "is_user_defined": is_user_defined,
        "models": list(models or []),
        "total_models": len(models or []),
        "source": source,
    }
    if api_url is not None:
        entry["api_url"] = api_url
    return entry


def test_openrouter_models_replaced_with_live_catalog(monkeypatch):
    """OpenRouter row's ``models`` should come from fetch_openrouter_models."""
    base = [
        _make_provider("openrouter", models=["openai/gpt-stale", "old/model"]),
    ]
    live = [("openai/gpt-5.4", "recommended"), ("moonshotai/kimi-k2.6", "")]

    monkeypatch.setattr(model_switch, "list_authenticated_providers",
                        lambda **kw: list(base))
    monkeypatch.setattr("hermes_cli.models.fetch_openrouter_models",
                        lambda *a, **kw: list(live))

    result = model_switch.list_picker_providers(max_models=50)

    assert len(result) == 1
    openrouter = result[0]
    assert openrouter["slug"] == "openrouter"
    assert openrouter["models"] == ["openai/gpt-5.4", "moonshotai/kimi-k2.6"]
    assert openrouter["total_models"] == 2


def test_openrouter_falls_back_to_base_models_on_fetch_failure(monkeypatch):
    """If the live catalog fetch raises, keep whatever base provided."""
    fallback_models = ["openai/gpt-5.4", "moonshotai/kimi-k2.6"]
    base = [_make_provider("openrouter", models=fallback_models)]

    def _raise(*_a, **_kw):
        raise RuntimeError("network down")

    monkeypatch.setattr(model_switch, "list_authenticated_providers",
                        lambda **kw: list(base))
    monkeypatch.setattr("hermes_cli.models.fetch_openrouter_models", _raise)

    result = model_switch.list_picker_providers(max_models=50)

    assert len(result) == 1
    assert result[0]["models"] == fallback_models


def test_openrouter_empty_live_catalog_drops_row(monkeypatch):
    """If the live catalog returns nothing for OpenRouter, drop the row."""
    base = [_make_provider("openrouter", models=["something/stale"])]

    monkeypatch.setattr(model_switch, "list_authenticated_providers",
                        lambda **kw: list(base))
    monkeypatch.setattr("hermes_cli.models.fetch_openrouter_models",
                        lambda *a, **kw: [])

    result = model_switch.list_picker_providers(max_models=50)

    assert result == []


def test_non_openrouter_rows_passed_through_unchanged(monkeypatch):
    """Non-OpenRouter providers keep their curated ``models`` as-is."""
    base = [
        _make_provider("anthropic", models=["claude-sonnet-4-6", "claude-opus-4-7"]),
        _make_provider("gemini", models=["gemini-3-flash-preview"]),
    ]

    monkeypatch.setattr(model_switch, "list_authenticated_providers",
                        lambda **kw: list(base))
    # fetch_openrouter_models must not be consulted when there's no openrouter row
    monkeypatch.setattr("hermes_cli.models.fetch_openrouter_models",
                        lambda *a, **kw: pytest.fail("should not be called"))

    result = model_switch.list_picker_providers(max_models=50)

    assert [p["slug"] for p in result] == ["anthropic", "gemini"]
    assert result[0]["models"] == ["claude-sonnet-4-6", "claude-opus-4-7"]
    assert result[1]["models"] == ["gemini-3-flash-preview"]


def test_empty_models_row_dropped(monkeypatch):
    """Built-in provider with an empty ``models`` list is dropped."""
    base = [
        _make_provider("anthropic", models=[]),  # drop
        _make_provider("openrouter", models=["anything"]),  # replaced by live
    ]

    monkeypatch.setattr(model_switch, "list_authenticated_providers",
                        lambda **kw: list(base))
    monkeypatch.setattr("hermes_cli.models.fetch_openrouter_models",
                        lambda *a, **kw: [("openai/gpt-5.4", "recommended")])

    result = model_switch.list_picker_providers(max_models=50)

    assert [p["slug"] for p in result] == ["openrouter"]


def test_custom_endpoint_with_api_url_kept_when_models_empty(monkeypatch):
    """User-defined endpoints with an ``api_url`` survive even if models empty.

    Rationale: custom endpoints may accept any model id the user types --
    the picker still shows the row so the user can enter one manually.
    """
    base = [
        _make_provider("local-ollama", is_user_defined=True,
                       api_url="http://localhost:11434/v1", models=[],
                       source="user-config"),
    ]

    monkeypatch.setattr(model_switch, "list_authenticated_providers",
                        lambda **kw: list(base))
    monkeypatch.setattr("hermes_cli.models.fetch_openrouter_models",
                        lambda *a, **kw: [])

    result = model_switch.list_picker_providers(max_models=50)

    assert len(result) == 1
    assert result[0]["slug"] == "local-ollama"
    assert result[0]["models"] == []


def test_user_defined_without_api_url_and_empty_models_dropped(monkeypatch):
    """An is_user_defined row WITHOUT api_url and no models is still dropped.

    The exemption is specifically for custom endpoints that can accept
    arbitrary model ids; without an api_url there's nothing to point at.
    """
    base = [
        _make_provider("orphan", is_user_defined=True, api_url=None, models=[]),
    ]

    monkeypatch.setattr(model_switch, "list_authenticated_providers",
                        lambda **kw: list(base))
    monkeypatch.setattr("hermes_cli.models.fetch_openrouter_models",
                        lambda *a, **kw: [])

    result = model_switch.list_picker_providers(max_models=50)

    assert result == []


def test_max_models_caps_openrouter_live_output(monkeypatch):
    """``max_models`` caps how many OpenRouter IDs land in the row."""
    live = [(f"vendor/model-{i}", "") for i in range(20)]
    base = [_make_provider("openrouter", models=["placeholder"])]

    monkeypatch.setattr(model_switch, "list_authenticated_providers",
                        lambda **kw: list(base))
    monkeypatch.setattr("hermes_cli.models.fetch_openrouter_models",
                        lambda *a, **kw: list(live))

    result = model_switch.list_picker_providers(max_models=5)

    assert len(result) == 1
    assert len(result[0]["models"]) == 5
    assert result[0]["models"] == [mid for mid, _ in live[:5]]
    # total_models reflects the full live catalog, not the capped slice.
    assert result[0]["total_models"] == 20


def test_passthrough_kwargs_to_base(monkeypatch):
    """All kwargs (current_provider, user_providers, custom_providers, max_models)
    must be forwarded to ``list_authenticated_providers`` unchanged.
    """
    captured = {}

    def _capture(**kwargs):
        captured.update(kwargs)
        return []

    monkeypatch.setattr(model_switch, "list_authenticated_providers", _capture)
    monkeypatch.setattr("hermes_cli.models.fetch_openrouter_models",
                        lambda *a, **kw: [])

    model_switch.list_picker_providers(
        current_provider="openrouter",
        user_providers={"foo": {"api": "http://x"}},
        custom_providers=[{"name": "bar", "base_url": "http://y"}],
        max_models=12,
    )

    assert captured["current_provider"] == "openrouter"
    assert captured["user_providers"] == {"foo": {"api": "http://x"}}
    assert captured["custom_providers"] == [{"name": "bar", "base_url": "http://y"}]
    assert captured["max_models"] == 12
