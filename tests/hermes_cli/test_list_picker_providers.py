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


@pytest.fixture(autouse=True)
def _disable_live_custom_provider_model_probe(monkeypatch):
    """Keep custom-provider picker fixtures independent of local model servers."""
    monkeypatch.setattr("hermes_cli.models.fetch_api_models", lambda *_a, **_kw: None)


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


# ---------------------------------------------------------------------------
# list_authenticated_providers: alias/canonical de-dup for Kimi (#49439)
# ---------------------------------------------------------------------------
#
# A single Kimi credential used to surface TWO picker rows: the alias slug
# "kimi" (emitted by the PROVIDER_TO_MODELS_DEV pass) plus its canonical
# "kimi-coding" (re-emitted by the CANONICAL_PROVIDERS cross-check pass),
# both backed by the same kimi-for-coding models.dev provider. The picker
# must list each authenticated credential exactly once, under the CANONICAL
# slug ("kimi-coding") — matching list_authenticated_providers' other alias
# rows and the overlay slug-resolution contract (see
# test_overlay_slug_resolution.py).


def _stub_kimi_discovery(monkeypatch, *, canonical):
    """Isolate list_authenticated_providers to the Kimi alias family.

    Restricts the models.dev map / catalog / overlays / canonical list to
    just the Kimi entries and stubs the model-id fetch so discovery stays
    offline and deterministic. ``canonical`` is the CANONICAL_PROVIDERS list
    the 2b cross-check pass should iterate.
    """
    import agent.models_dev as md
    import hermes_cli.models as hm

    kimi_map = {
        "kimi": "kimi-for-coding",
        "kimi-coding": "kimi-for-coding",
        "moonshot": "kimi-for-coding",
        "kimi-coding-cn": "kimi-for-coding",
    }
    monkeypatch.setattr(md, "PROVIDER_TO_MODELS_DEV", kimi_map)
    monkeypatch.setattr(
        md, "fetch_models_dev",
        lambda *a, **k: {
            "kimi-for-coding": {"name": "Kimi For Coding", "env": ["KIMI_API_KEY"]},
        },
    )

    class _PInfo:
        name = "Kimi For Coding"

    monkeypatch.setattr(md, "get_provider_info", lambda _pid: _PInfo())
    monkeypatch.setattr("hermes_cli.providers.HERMES_OVERLAYS", {})
    monkeypatch.setattr(hm, "CANONICAL_PROVIDERS", canonical)
    monkeypatch.setattr(hm, "cached_provider_model_ids",
                        lambda *a, **k: ["kimi-k2.6", "kimi-k2.5"])
    monkeypatch.setattr(hm, "clear_provider_models_cache", lambda *a, **k: None)


def test_single_kimi_credential_yields_one_canonical_row(monkeypatch):
    """One Kimi key yields a single row under the canonical 'kimi-coding' slug."""
    import hermes_cli.models as hm

    _stub_kimi_discovery(
        monkeypatch,
        canonical=[hm.ProviderEntry("kimi-coding", "Kimi / Kimi Coding Plan", "desc")],
    )
    monkeypatch.setenv("KIMI_API_KEY", "sk-test-kimi")

    rows = model_switch.list_authenticated_providers(max_models=10)
    slugs = [r["slug"] for r in rows]

    # Exactly one Kimi / kimi-for-coding-backed row, under the canonical slug —
    # not both the alias ("kimi") and its canonical ("kimi-coding").
    kimi_rows = [s for s in slugs if s in {"kimi", "kimi-coding"}]
    assert kimi_rows == ["kimi-coding"], (
        f"expected a single canonical Kimi row, got: {slugs}"
    )
    assert slugs.count("kimi-coding") == 1
    assert "kimi" not in slugs


def test_distinct_kimi_china_credential_still_listed(monkeypatch):
    """A separate China (kimi-coding-cn) credential remains its own row.

    Negative-control guard: the de-dup must collapse only the alias/canonical
    pair that share a credential, not legitimately distinct providers.
    """
    import hermes_cli.models as hm

    _stub_kimi_discovery(
        monkeypatch,
        canonical=[
            hm.ProviderEntry("kimi-coding", "Kimi / Kimi Coding Plan", "desc"),
            hm.ProviderEntry("kimi-coding-cn", "Kimi / Moonshot (China)", "desc"),
        ],
    )
    monkeypatch.setenv("KIMI_API_KEY", "sk-test-kimi")
    monkeypatch.setenv("KIMI_CN_API_KEY", "sk-test-kimi-cn")

    rows = model_switch.list_authenticated_providers(max_models=10)
    slugs = [r["slug"] for r in rows]

    assert "kimi-coding" in slugs       # canonical global row
    assert slugs.count("kimi-coding") == 1
    assert "kimi" not in slugs          # alias collapsed into the canonical row
    assert "kimi-coding-cn" in slugs    # distinct China endpoint preserved
