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
