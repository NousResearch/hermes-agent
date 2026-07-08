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


def test_include_moa_adds_virtual_provider_with_named_presets(monkeypatch):
    """Gateway pickers opt into a virtual MoA provider so presets are tappable."""
    base = [_make_provider("minimax", models=["MiniMax-M3"])]
    moa_config = {
        "moa": {
            "default_preset": "battle",
            "presets": {
                "battle": {"enabled": True},
                "smart": {"enabled": True},
            },
        }
    }

    monkeypatch.setattr(model_switch, "list_authenticated_providers",
                        lambda **kw: list(base))
    monkeypatch.setattr("hermes_cli.config.load_config", lambda: moa_config)
    monkeypatch.setattr("hermes_cli.models.fetch_openrouter_models",
                        lambda *a, **kw: pytest.fail("should not be called"))

    result = model_switch.list_picker_providers(
        current_provider="moa",
        max_models=50,
        include_moa=True,
    )

    assert [p["slug"] for p in result] == ["moa", "minimax"]
    moa = result[0]
    assert moa["name"] == "Mixture of Agents"
    assert moa["is_current"] is True
    assert moa["source"] == "virtual"
    assert moa["models"] == ["battle", "smart"]
    assert moa["total_models"] == 2


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
    """All kwargs must be forwarded to ``list_authenticated_providers`` unchanged.

    The gateway /model picker passes ``current_base_url`` and ``current_model``
    so custom endpoint grouping can mark the current row. Dropping those kwargs
    regressed Telegram/Discord into the text-list fallback.
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
        current_base_url="http://x",
        current_model="openai/gpt-5.4",
        user_providers={"foo": {"api": "http://x"}},
        custom_providers=[{"name": "bar", "base_url": "http://y"}],
        max_models=12,
    )

    assert captured["current_provider"] == "openrouter"
    assert captured["current_base_url"] == "http://x"
    assert captured["current_model"] == "openai/gpt-5.4"
    assert captured["user_providers"] == {"foo": {"api": "http://x"}}
    assert captured["custom_providers"] == [{"name": "bar", "base_url": "http://y"}]
    assert captured["max_models"] == 12


def test_current_custom_endpoint_passthrough_marks_current_row(monkeypatch):
    """Interactive picker should preserve current custom endpoint semantics."""
    monkeypatch.setattr("agent.models_dev.fetch_models_dev", lambda: {})
    monkeypatch.setattr("agent.models_dev.PROVIDER_TO_MODELS_DEV", {})
    monkeypatch.setattr("hermes_cli.providers.HERMES_OVERLAYS", {})
    monkeypatch.setattr("hermes_cli.models.fetch_openrouter_models",
                        lambda *a, **kw: [])
    # Hermeticity: the custom-endpoint branch calls fetch_api_models() against
    # the base_url (localhost:11434). On a machine running a real Ollama that
    # returns live models and the assertion below (which expects the config's
    # models) fails. Force it to report no live catalog so the config models win.
    monkeypatch.setattr("hermes_cli.models.fetch_api_models",
                        lambda *a, **kw: [])

    result = model_switch.list_picker_providers(
        current_provider="custom:ollama",
        current_base_url="http://localhost:11434/v1",
        current_model="glm-5.1",
        user_providers={},
        custom_providers=[
            {
                "name": "Ollama — GLM 5.1",
                "base_url": "http://localhost:11434/v1",
                "api_key": "ollama",
                "model": "glm-5.1",
            },
            {
                "name": "Ollama — Qwen3",
                "base_url": "http://localhost:11434/v1",
                "api_key": "ollama",
                "model": "qwen3",
            },
        ],
        max_models=50,
    )

    custom_rows = [p for p in result if p.get("is_user_defined")]
    assert len(custom_rows) == 1
    row = custom_rows[0]
    assert row["slug"] == "custom:ollama"
    assert row["is_current"] is True
    assert row["models"] == ["glm-5.1", "qwen3"]


def test_current_model_not_duplicated_when_catalog_entry_is_namespaced(monkeypatch):
    """Regression: the current model must not appear twice in the picker.

    ``current_model`` is stored bare in config (e.g. ``claude-opus-4-8``) while
    a provider's curated catalog lists it namespaced (``claude-app/claude-opus-4-8``).
    The post-pass that injects the current model at the top of the current row
    used a plain ``current_model not in _models`` check, which never matched the
    namespaced entry — so the model was injected a SECOND time and the platform
    pickers (Discord/Telegram strip the prefix for display) showed it twice.

    Here the current custom row's catalog is the bare + namespaced pair; a bare
    ``current_model`` must be recognised as already present via the namespaced
    entry and NOT re-injected.
    """
    monkeypatch.setattr("agent.models_dev.fetch_models_dev", lambda: {})
    monkeypatch.setattr("agent.models_dev.PROVIDER_TO_MODELS_DEV", {})
    monkeypatch.setattr("hermes_cli.providers.HERMES_OVERLAYS", {})
    monkeypatch.setattr("hermes_cli.models.fetch_openrouter_models",
                        lambda *a, **kw: [])

    # The custom-endpoint branch seeds the row's models from ``current_model``.
    # Give the row a namespaced catalog entry via the config ``model`` list so
    # the injected bare id would collide on display but not on a naive ``in``.
    result = model_switch.list_authenticated_providers(
        current_provider="custom:relay",
        current_base_url="http://localhost:9099/v1",
        current_model="claude-opus-4-8",
        custom_providers=[
            {
                "name": "Relay",
                "base_url": "http://localhost:9099/v1",
                "api_key": "x",
                "models": ["relay/claude-opus-4-8", "relay/claude-sonnet-5"],
            },
        ],
    )

    current_rows = [p for p in result if p.get("is_current")]
    assert current_rows, "expected a current row"
    row = current_rows[0]
    display = [m.split("/")[-1] for m in row["models"]]
    # The bare current model resolves to the namespaced entry — no duplicate.
    assert display.count("claude-opus-4-8") == 1, row["models"]
    assert row["total_models"] == len(row["models"])


def test_current_model_injected_when_genuinely_absent(monkeypatch):
    """Guard the other side: an uncurated current model IS still injected.

    The namespace-aware check must not over-match — a model the catalog does
    not carry (in any namespaced form) must still be prepended so it stays
    selectable in the picker.
    """
    monkeypatch.setattr("agent.models_dev.fetch_models_dev", lambda: {})
    monkeypatch.setattr("agent.models_dev.PROVIDER_TO_MODELS_DEV", {})
    monkeypatch.setattr("hermes_cli.providers.HERMES_OVERLAYS", {})
    monkeypatch.setattr("hermes_cli.models.fetch_openrouter_models",
                        lambda *a, **kw: [])

    result = model_switch.list_authenticated_providers(
        current_provider="custom:relay",
        current_base_url="http://localhost:9099/v1",
        current_model="some-uncurated-model",
        custom_providers=[
            {
                "name": "Relay",
                "base_url": "http://localhost:9099/v1",
                "api_key": "x",
                "models": ["relay/claude-opus-4-8"],
            },
        ],
    )

    current_rows = [p for p in result if p.get("is_current")]
    assert current_rows, "expected a current row"
    row = current_rows[0]
    assert "some-uncurated-model" in row["models"], row["models"]


def test_numbered_failover_lanes_hidden_from_picker(monkeypatch):
    """claude-apx-N / claude-bpx-N failover lanes are hidden from the picker.

    They are internal auto-failover targets, not hand-selectable providers, and
    20+ of them crowd real providers out of the dropdown's 25-option cap. N
    INCLUDES 0 (claude-bpx-0 / claude-apx-0 are lanes too). They must NOT appear
    in the picker; the relay pools (claude-apr / claude-bpr) and everything else
    must survive.
    """
    base = [
        _make_provider("anthropic", models=["claude-opus-4-8"]),
        _make_provider("claude-apr", models=["claude-opus-4-8"]),
        _make_provider("claude-bpr", models=["claude-opus-4-8"]),
        _make_provider("claude-apx-0", models=["claude-opus-4-8"]),
        _make_provider("claude-apx-1", models=["claude-opus-4-8"]),
        _make_provider("claude-apx-10", models=["claude-opus-4-8"]),
        _make_provider("claude-bpx-0", models=["claude-opus-4-8"]),
        _make_provider("claude-bpx-5", models=["claude-opus-4-8"]),
        _make_provider("yunwu", models=["claude-opus-4-8"]),
    ]

    monkeypatch.setattr(model_switch, "list_authenticated_providers",
                        lambda **kw: list(base))
    monkeypatch.setattr("hermes_cli.models.fetch_openrouter_models",
                        lambda *a, **kw: pytest.fail("should not be called"))

    result = [p["slug"] for p in model_switch.list_picker_providers(max_models=50)]

    # Every numbered lane — INCLUDING -0 — is gone.
    for lane in ("claude-apx-0", "claude-apx-1", "claude-apx-10",
                 "claude-bpx-0", "claude-bpx-5"):
        assert lane not in result, f"{lane} should be hidden"
    # Relay pools + real providers survive.
    for keep in ("anthropic", "claude-apr", "claude-bpr", "yunwu"):
        assert keep in result, f"{keep} should remain visible"


def test_current_failover_lane_stays_visible(monkeypatch):
    """A numbered lane is kept ONLY when it's the currently-active provider.

    So a user who is actually running on claude-bpx-5 can still see it in
    the picker (to switch away), while the other lanes stay hidden.
    """
    base = [
        _make_provider("claude-bpx-1", models=["claude-opus-4-8"]),
        _make_provider("claude-bpx-5", models=["claude-opus-4-8"],
                       is_current=True),
        _make_provider("yunwu", models=["claude-opus-4-8"]),
    ]

    monkeypatch.setattr(model_switch, "list_authenticated_providers",
                        lambda **kw: list(base))
    monkeypatch.setattr("hermes_cli.models.fetch_openrouter_models",
                        lambda *a, **kw: pytest.fail("should not be called"))

    result = [p["slug"] for p in model_switch.list_picker_providers(
        current_provider="claude-bpx-5", max_models=50)]

    assert "claude-bpx-5" in result   # current lane visible
    assert "claude-bpx-1" not in result  # other lanes still hidden
    assert "yunwu" in result


def test_non_failover_claude_providers_never_hidden(monkeypatch):
    """The hide-rule must be surgical: only claude-{apx,bpx}-N (N any int) match.

    Relay pools and unrelated slugs that merely contain 'apx'/'bpx' text must
    not be swept up by the failover-lane regex.
    """
    base = [
        _make_provider("claude-apr"),        # relay pool, not a lane
        _make_provider("claude-bpr"),        # relay pool, not a lane
        _make_provider("claude-app"),        # legacy base, not a lane
        _make_provider("claude-apxtra"),     # 'apx' substring but not -N
    ]
    for p in base:
        p["models"] = ["m"]

    monkeypatch.setattr(model_switch, "list_authenticated_providers",
                        lambda **kw: list(base))
    monkeypatch.setattr("hermes_cli.models.fetch_openrouter_models",
                        lambda *a, **kw: pytest.fail("should not be called"))

    result = [p["slug"] for p in model_switch.list_picker_providers(max_models=50)]

    assert result == [
        "claude-apr", "claude-bpr", "claude-app", "claude-apxtra",
    ]
