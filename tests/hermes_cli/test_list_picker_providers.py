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
import hermes_cli.models as models_mod
import hermes_cli.model_switch_providers as hermes_cli_model_switch_providers
from hermes_cli import model_switch_providers


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
    monkeypatch.setattr(hermes_cli_model_switch_providers, "list_authenticated_providers", _capture)
    monkeypatch.setattr("hermes_cli.models.fetch_openrouter_models",
                        lambda *a, **kw: [])

    model_switch_providers.list_picker_providers(
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

    result = model_switch_providers.list_picker_providers(
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
                "discover_models": False,
            },
            {
                "name": "Ollama — Qwen3",
                "base_url": "http://localhost:11434/v1",
                "api_key": "ollama",
                "model": "qwen3",
                "discover_models": False,
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
    import hermes_cli.models_catalog_static as hermes_cli_models_catalog_static
    from hermes_cli import models_catalog_static

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
    monkeypatch.setattr(hermes_cli_models_catalog_static, "CANONICAL_PROVIDERS", canonical)
    monkeypatch.setattr(hm, "cached_provider_model_ids",
                        lambda *a, **k: ["kimi-k2.6", "kimi-k2.5"])
    monkeypatch.setattr(hm, "clear_provider_models_cache", lambda *a, **k: None)


def test_single_kimi_credential_yields_one_canonical_row(monkeypatch):
    """One Kimi key yields a single row under the canonical 'kimi-coding' slug."""
    import hermes_cli.models as hm
    from hermes_cli import models_catalog_static

    _stub_kimi_discovery(
        monkeypatch,
        canonical=[models_catalog_static.ProviderEntry("kimi-coding", "Kimi / Kimi Coding Plan", "desc")],
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
    from hermes_cli import models_catalog_static

    _stub_kimi_discovery(
        monkeypatch,
        canonical=[
            models_catalog_static.ProviderEntry("kimi-coding", "Kimi / Kimi Coding Plan", "desc"),
            models_catalog_static.ProviderEntry("kimi-coding-cn", "Kimi / Moonshot (China)", "desc"),
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


def test_non_blocking_listing_opens_no_socket(monkeypatch, tmp_path):
    """#74003: ``non_blocking_catalogs=True`` must not run a single live catalog probe in the calling
    thread — not the per-provider ``/models`` prefetch and not OpenRouter's curated-catalog GET —
    even with several credentialed providers and an empty on-disk cache."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(models_mod, "_openrouter_catalog_cache", None)
    for key in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "DEEPSEEK_API_KEY", "GROQ_API_KEY",
                "MISTRAL_API_KEY", "XAI_API_KEY", "OPENROUTER_API_KEY"):
        monkeypatch.setenv(key, "dummy")
    (tmp_path / "provider_models_cache.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr("agent.models_dev.fetch_models_dev", lambda *a, **k: {})

    live: list[str] = []

    def _live_probe(*args, **kwargs):
        live.append((args, kwargs))
        return []

    monkeypatch.setattr(models_mod, "provider_model_ids", _live_probe)
    monkeypatch.setattr(models_mod, "fetch_api_models", _live_probe)
    monkeypatch.setattr(models_mod, "_fetch_live_catalog_index", _live_probe)
    monkeypatch.setattr(models_mod, "_spawn_swr_refresh", lambda *a, **k: None)  # background warm is not "live"

    rows = model_switch_providers.list_picker_providers(
        max_models=50, include_moa=True, current_provider="openrouter", user_providers={},
        custom_providers=[], excluded_providers=[], non_blocking_catalogs=True)

    assert live == [], f"cache-only listing ran live probes in the request path: {live}"
    assert any(r.get("slug") == "openrouter" and r.get("models") for r in rows), "OpenRouter row lost its curated snapshot"


# ---------------------------------------------------------------------------
# providers.openrouter.models survives the OpenRouter row rebuild (#121903)
# ---------------------------------------------------------------------------


def _picker_with_openrouter(monkeypatch, user_providers, *, live, max_models=None):
    """Run ``list_picker_providers`` over one OpenRouter row plus one DeepSeek control row."""
    base = [
        _make_provider("openrouter", "OpenRouter", ["stale/row"]),
        _make_provider("deepseek", "DeepSeek", ["pinned/extra", "deepseek-chat"]),
    ]
    monkeypatch.setattr(hermes_cli_model_switch_providers, "list_authenticated_providers",
                        lambda **_kw: [dict(r) for r in base], raising=False)
    monkeypatch.setattr(model_switch, "list_authenticated_providers",
                        lambda **_kw: [dict(r) for r in base])
    monkeypatch.setattr("hermes_cli.models.fetch_openrouter_models",
                        lambda *a, **kw: [(mid, "") for mid in live])
    rows = model_switch_providers.list_picker_providers(user_providers=user_providers, max_models=max_models)
    return {r["slug"]: r for r in rows}


def test_configured_openrouter_models_lead_the_live_list(monkeypatch):
    rows = _picker_with_openrouter(
        monkeypatch, {"openrouter": {"models": ["stealth/pinned", "a/curated"]}},
        live=["a/curated", "b/curated"])
    assert rows["openrouter"]["models"] == ["stealth/pinned", "a/curated", "b/curated"]
    assert rows["openrouter"]["total_models"] == 3


def test_configured_openrouter_models_accept_every_declared_shape(monkeypatch):
    rows = _picker_with_openrouter(
        monkeypatch, {"openrouter": {"models": {"stealth/mapped": {"context_length": 1000}}}},
        live=["a/curated"])
    assert rows["openrouter"]["models"] == ["stealth/mapped", "a/curated"]


def test_configured_openrouter_models_preserve_the_uncapped_catalog(monkeypatch):
    rows = _picker_with_openrouter(
        monkeypatch, {"openrouter": {"models": ["stealth/pinned"]}},
        live=["a/curated", "b/curated", "c/curated"], max_models=2)
    assert rows["openrouter"]["models"] == ["stealth/pinned", "a/curated", "b/curated", "c/curated"]
    assert rows["openrouter"]["total_models"] == 4


def test_openrouter_row_without_configured_models_is_the_live_list(monkeypatch):
    rows = _picker_with_openrouter(monkeypatch, {"deepseek": {"models": ["pinned/extra"]}},
                                   live=["a/curated"])
    assert rows["openrouter"]["models"] == ["a/curated"]  # stale ids still drop out
    assert rows["deepseek"]["models"] == ["pinned/extra", "deepseek-chat"]  # other rows untouched


def test_configured_openrouter_models_survive_real_config_and_picker(tmp_path, monkeypatch):
    """Configured IDs stay profile-local through disk loading and real row construction."""
    import json
    from agent import secret_scope
    from hermes_cli.inventory import load_picker_context
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override

    homes = [tmp_path / "a", tmp_path / "b"]
    for home, model in zip(homes, ("stealth/profile-a", "stealth/profile-b")):
        home.mkdir()
        (home / "config.yaml").write_text(json.dumps({
            "model": {"provider": "openrouter"},
            "model_catalog": {"enabled": False},
            "providers": {"openrouter": {"models": [model, "catalog/shared"]}},
        }), encoding="utf-8")
        (home / ".env").write_text("OPENROUTER_API_KEY=dummy\n", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(homes[0]))
    monkeypatch.setattr("agent.models_dev.fetch_models_dev", lambda *a, **k: {})
    monkeypatch.setattr(models_mod, "cached_provider_model_ids", lambda *a, **k: ["catalog/shared"])
    monkeypatch.setattr(models_mod, "fetch_openrouter_models",
                        lambda *a, **k: [("catalog/shared", ""), ("catalog/other", "")])
    monkeypatch.setattr(models_mod, "_spawn_swr_refresh", lambda *a, **k: None)

    secret_scope.set_multiplex_active(True)
    try:
        for home, expected in ((homes[0], "stealth/profile-a"),
                               (homes[1], "stealth/profile-b"),
                               (homes[0], "stealth/profile-a")):
            home_token = set_hermes_home_override(home)
            secret_token = secret_scope.set_secret_scope(
                secret_scope.build_profile_secret_scope(home), profile_home=str(home))
            try:
                ctx = load_picker_context()
                rows = model_switch_providers.list_picker_providers(
                    current_provider=ctx.current_provider, user_providers=ctx.user_providers,
                    custom_providers=ctx.custom_providers, non_blocking_catalogs=True,
                    probe_custom_providers=False, max_models=2)
                row = next(r for r in rows if r["slug"] == "openrouter")
                assert row["models"] == [expected, "catalog/shared", "catalog/other"]
                assert row["total_models"] == 3
            finally:
                secret_scope.reset_secret_scope(secret_token)
                reset_hermes_home_override(home_token)
    finally:
        secret_scope.set_multiplex_active(False)


def test_curated_openrouter_row_keeps_free_tail_past_max_models(monkeypatch):
    """The OpenRouter row is already curated; its bottom "Free tier" block must survive the picker cap
    while an ordinary provider row stays capped."""
    curated = [(f"vendor/model-{i}", "") for i in range(55)] + [("stealth/free-model", "free, stealth model")]
    other = [f"other-{i}" for i in range(60)]
    monkeypatch.setattr(model_switch, "list_authenticated_providers", lambda **_: [
        _make_provider("openrouter", models=[mid for mid, _ in curated][:50]),
        _make_provider("deepseek", models=other[:50]) | {"total_models": len(other)},
    ])
    monkeypatch.setattr(models_mod, "fetch_openrouter_models", lambda **_: curated)

    rows = {r["slug"]: r for r in model_switch_providers.list_picker_providers(max_models=50)}

    assert rows["openrouter"]["models"] == [mid for mid, _ in curated]
    assert len(rows["deepseek"]["models"]) == 50
