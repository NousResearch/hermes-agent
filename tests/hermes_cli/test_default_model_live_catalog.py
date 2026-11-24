"""A provider's implicit default must not name a model the account no longer has.

``get_default_model_for_provider`` answers "which model does a run that never chose one send?" — a
non-interactive surface, not a picker. It answered straight from the curated static row, which is a
hand-written cold-start seed. For GitHub Copilot that row still led with ``gpt-5.4`` long after
GitHub retired it, so a ``provider: copilot`` home with no ``model.default`` resolved to an id the
API rejects with ``model_not_supported`` — a failure that reads like an auth or quota problem.

The curated row still supplies the *order* whenever its ids are live: that ordering is the
cost-safety intent (and why ``test_openai_cost_safe_defaults`` exists), so the fix intersects
instead of replacing.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

import hermes_cli.models as models_mod

PROVIDER = "acme-provider"


def _cache(models=None, *, fp="fp-live", fallback=False):
    row = {"fp": fp, "at": 0.0, "models": list(models or [])}
    if fallback:
        row["fallback"] = True
    return {PROVIDER: row}


@pytest.fixture
def curated(monkeypatch):
    """A curated row whose first entry was retired upstream."""
    monkeypatch.setattr(
        models_mod, "static_catalog",
        lambda provider: ["retired-flagship", "live-cheap", "live-big"],
    )
    monkeypatch.setattr(models_mod, "_credential_fingerprint", lambda provider: "fp-live")
    return monkeypatch


def _resolve(cache):
    with patch.object(models_mod, "_load_provider_models_cache", return_value=cache):
        return models_mod.get_default_model_for_provider(PROVIDER)


def test_retired_curated_id_is_never_the_default(curated):
    """The seed's [0] is gone from the account's catalog: resolve to the next intended pick."""
    assert _resolve(_cache(["live-cheap", "live-big"])) == "live-cheap"


def test_curated_order_beats_live_order(curated):
    """Cost intent lives in the curated order, not in the API's ordering."""
    assert _resolve(_cache(["live-big", "live-cheap"])) == "live-cheap"


def test_cold_cache_keeps_the_seed(curated):
    """Before any fetch (``hermes auth add``, first launch) the seed is all there is."""
    assert _resolve({}) == "retired-flagship"


def test_placeholder_row_is_not_a_catalog(curated):
    """A ``fallback: true`` row is an outage placeholder, not the account's catalog."""
    assert _resolve(_cache(["live-cheap"], fallback=True)) == "retired-flagship"


def test_row_for_a_previous_credential_is_ignored(curated):
    """Rows are keyed by credential fingerprint; a rotated key must not answer for this one."""
    assert _resolve(_cache(["live-cheap"], fp="fp-old")) == "retired-flagship"


def test_unreadable_cache_degrades_to_the_seed(curated):
    """A default resolver must not raise because the cache file is missing or corrupt."""
    with patch.object(models_mod, "_load_provider_models_cache", side_effect=OSError("boom")):
        assert models_mod.get_default_model_for_provider(PROVIDER) == "retired-flagship"


def test_no_curated_id_survives_uses_the_live_ordering(curated, monkeypatch):
    """When the seed is entirely dead, the account's own catalog decides."""
    monkeypatch.setattr(models_mod, "static_catalog", lambda provider: ["retired-a", "retired-b"])
    assert _resolve(_cache(["live-big", "live-cheap"])) == "live-big"


def test_resolution_never_fetches(curated):
    """The resolver is reached from offline paths: it must read the disk row, never go online."""
    with patch.object(models_mod, "provider_model_ids", side_effect=AssertionError("egress")):
        assert _resolve(_cache(["live-cheap"])) == "live-cheap"
