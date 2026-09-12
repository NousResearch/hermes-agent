"""Built-in/canonical picker rows honor a curated ``providers.<id>.models`` shortlist.

Custom-endpoint rows already narrow to the configured list when
``discover_models: false``; built-in (models.dev-mapped) and canonical rows
only *prepended* it, so a ~105-model live catalog swallowed a 6-model
shortlist (#107106). These tests pin the narrowed semantics for both laps
and the default extend behavior when discovery stays on.
"""

import agent.models_dev as md
import hermes_cli.models as hm
import hermes_cli.models_catalog_static as models_catalog_static
import pytest
from hermes_cli import model_switch
from hermes_cli import model_switch_providers as providers_mod


def _isolate_canonical(monkeypatch, *, slug, key_var, live_models):
    """Restrict every picker input to one canonical provider row.

    Empties the models.dev map / overlays so sections 1-2 stay silent, points
    the canonical list at *slug*, gates it on *key_var* and stubs the live
    catalog to *live_models* (what ``cached_provider_model_ids`` would return).
    """
    monkeypatch.setattr(md, "PROVIDER_TO_MODELS_DEV", {})
    monkeypatch.setattr(md, "fetch_models_dev", lambda *a, **k: {})
    monkeypatch.setattr("hermes_cli.providers.HERMES_OVERLAYS", {})
    canonical = [models_catalog_static.ProviderEntry(slug, slug.title(), "desc")]
    monkeypatch.setattr(hm, "CANONICAL_PROVIDERS", canonical)
    monkeypatch.setattr(models_catalog_static, "CANONICAL_PROVIDERS", canonical)
    monkeypatch.setattr(
        hm, "cached_provider_model_ids", lambda *a, **k: list(live_models)
    )
    monkeypatch.setattr(hm, "clear_provider_models_cache", lambda *a, **k: None)
    monkeypatch.setenv(key_var, "sk-test")


def _canonical_row(rows, slug):
    return next(r for r in rows if r["slug"] == slug and r["source"] == "canonical")


_LIVE = [f"live/model-{i}" for i in range(12)]
_SHORTLIST = ["keep/model-a", "keep/model-b", "keep/model-c"]


def test_canonical_row_narrows_to_curated_models_when_discovery_off(monkeypatch):
    _isolate_canonical(
        monkeypatch, slug="deepinfra", key_var="DEEPINFRA_API_KEY", live_models=_LIVE
    )

    rows = model_switch.list_authenticated_providers(
        current_provider="deepinfra",
        user_providers={
            "deepinfra": {"discover_models": False, "models": dict.fromkeys(_SHORTLIST)}
        },
        max_models=50,
    )
    row = _canonical_row(rows, "deepinfra")
    assert row["models"] == _SHORTLIST
    assert row["total_models"] == 3


def test_canonical_row_extends_catalog_by_default(monkeypatch):
    _isolate_canonical(
        monkeypatch, slug="deepinfra", key_var="DEEPINFRA_API_KEY", live_models=_LIVE
    )

    rows = model_switch.list_authenticated_providers(
        current_provider="deepinfra",
        user_providers={"deepinfra": {"models": dict.fromkeys(_SHORTLIST)}},
        max_models=50,
    )
    row = _canonical_row(rows, "deepinfra")
    assert row["models"][:3] == _SHORTLIST
    assert len(row["models"]) == len(_LIVE) + len(_SHORTLIST)


def test_canonical_dict_metadata_without_flag_does_not_narrow(monkeypatch):
    _isolate_canonical(
        monkeypatch, slug="deepinfra", key_var="DEEPINFRA_API_KEY", live_models=_LIVE
    )

    rows = model_switch.list_authenticated_providers(
        current_provider="deepinfra",
        user_providers={
            "deepinfra": {"models": {m: {"context_length": 8192} for m in _SHORTLIST}}
        },
        max_models=50,
    )
    row = _canonical_row(rows, "deepinfra")
    assert len(row["models"]) == len(_LIVE) + len(_SHORTLIST)


def test_builtin_row_narrows_to_curated_models_when_discovery_off(monkeypatch):
    """Section-1 (models.dev-mapped) rows share the same shortlist semantics."""
    monkeypatch.setattr(md, "PROVIDER_TO_MODELS_DEV", {"togetherai": "togetherai"})
    monkeypatch.setattr(
        md,
        "fetch_models_dev",
        lambda *a, **k: {
            "togetherai": {"name": "Together AI", "env": ["TOGETHERAI_API_KEY"]}
        },
    )

    class _PInfo:
        name = "Together AI"

    monkeypatch.setattr(md, "get_provider_info", lambda _pid: _PInfo())
    monkeypatch.setattr("hermes_cli.providers.HERMES_OVERLAYS", {})
    monkeypatch.setattr(hm, "CANONICAL_PROVIDERS", [])
    monkeypatch.setattr(models_catalog_static, "CANONICAL_PROVIDERS", [])
    monkeypatch.setattr(hm, "cached_provider_model_ids", lambda *a, **k: list(_LIVE))
    monkeypatch.setattr(hm, "clear_provider_models_cache", lambda *a, **k: None)
    monkeypatch.setenv("TOGETHERAI_API_KEY", "sk-test")

    rows = model_switch.list_authenticated_providers(
        current_provider="togetherai",
        user_providers={
            "togetherai": {
                "discover_models": False,
                "models": dict.fromkeys(_SHORTLIST),
            }
        },
        max_models=50,
    )
    row = next(r for r in rows if r["slug"] == "togetherai")
    assert row["models"] == _SHORTLIST
    assert row["total_models"] == 3
