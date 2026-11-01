"""grok-4.7 (xAI, GA 2026-09-21): 500K context, reasoning floor, xAI picker headline.

Without its own catalog entry grok-4.7 falls through the generic ``grok-4`` catch-all
to 256,000 -- a half-size window that never errors, it just compacts ~2x too early.
A catalog row alone does not fix an already-cached window: the step-1 persistent
cache keeps returning 256,000 unless the key is also a pre-catalog stale key.
"""
from __future__ import annotations

import importlib


def test_grok_4_7_catalog_entry_is_500k():
    from agent.model_metadata import DEFAULT_CONTEXT_LENGTHS
    assert DEFAULT_CONTEXT_LENGTHS.get("grok-4.7") == 500_000
    # Same tier as its predecessor, strictly above the "grok-4" catch-all.
    assert DEFAULT_CONTEXT_LENGTHS["grok-4.7"] == DEFAULT_CONTEXT_LENGTHS["grok-4.6"]
    assert DEFAULT_CONTEXT_LENGTHS["grok-4.7"] > DEFAULT_CONTEXT_LENGTHS["grok-4"]


def test_stale_grok_4_7_detected_by_generic_guard():
    from agent.model_metadata import _stale_pre_catalog_cache_entry
    catch_all = 256_000  # the value older builds persisted via the "grok-4" entry
    for slug in ("grok-4.7", "xai/grok-4.7", "x-ai/grok-4.7"):
        assert _stale_pre_catalog_cache_entry(slug, catch_all), slug
    assert not _stale_pre_catalog_cache_entry("grok-4.7", 500_000)
    assert not _stale_pre_catalog_cache_entry("grok-4", catch_all)


def test_stale_grok_4_7_cache_dropped_and_reresolves_to_500k(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    import agent.model_metadata as mm
    importlib.reload(mm)
    try:
        base_url = "https://api.x.ai/v1"
        mm.save_context_length("grok-4.7", base_url, 256_000)
        ctx = mm.get_model_context_length("grok-4.7", base_url=base_url, provider="xai")
        assert ctx == 500_000
    finally:
        importlib.reload(mm)


def test_grok_4_7_gets_the_reasoning_stale_timeout_floor():
    from agent.reasoning_timeouts import get_reasoning_stale_timeout_floor
    assert get_reasoning_stale_timeout_floor("grok-4.7") == get_reasoning_stale_timeout_floor("grok-4.6")
    assert get_reasoning_stale_timeout_floor("grok-4.7")


def test_xai_offline_catalog_leads_with_grok_4_7(monkeypatch):
    from hermes_cli import models_catalog_static as mcs
    monkeypatch.setattr("agent.models_dev._load_disk_cache", lambda: {})
    ids = mcs._xai_curated_models()
    assert ids[0] == "grok-4.7"
    assert "grok-4.6" in ids and len(ids) == len(set(ids))
    # A stale models.dev cache that predates grok-4.7 still surfaces it first.
    monkeypatch.setattr(
        "agent.models_dev._load_disk_cache",
        lambda: {"xai": {"models": {"grok-4.6": {}, "grok-4.5": {}}}},
    )
    assert mcs._xai_curated_models()[0] == "grok-4.7"
