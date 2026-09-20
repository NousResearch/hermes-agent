"""Regression tests for the Anthropic model-picker dropping curated aliases.

Bug — newly-routed curated aliases vanished on a native Anthropic setup
    ``provider_model_ids("anthropic")`` returned the live ``/v1/models`` dump
    verbatim whenever Anthropic credentials were configured. Anthropic's API
    lags behind freshly-routed aliases (e.g. ``claude-fable-5``, which is
    reachable on Anthropic before the models endpoint enumerates it), so the
    curated entry disappeared from the picker. The picker now merges the
    curated ``_PROVIDER_MODELS["anthropic"]`` list with the live catalog —
    curated entries first, live-only models appended, deduped — mirroring the
    OpenAI curated-merge philosophy.
"""

from unittest.mock import patch

from hermes_cli import models as M


def test_anthropic_native_list_keeps_aggregator_flagships():
    """Native Anthropic must list the same current flagships OpenRouter/Nous already ship.

    Aggregator catalogs get the new aliases first; the native curated list is what
    `/model` falls back to when live `/v1/models` lags or 401s. Newest-first order
    is the contract that keeps Fable 5.1 / Opus 5 from hiding behind older 4.x ids.
    """
    or_ids = {mid for mid, _ in M.OPENROUTER_MODELS}
    native = M._PROVIDER_MODELS["anthropic"]
    for slug in ("claude-fable-5.1", "claude-opus-5"):
        assert f"anthropic/{slug}" in or_ids
        assert slug in native
    assert native.index("claude-fable-5.1") < native.index("claude-fable-5")
    assert native.index("claude-opus-5") < native.index("claude-opus-4-8")


def test_anthropic_curated_alias_survives_when_live_omits_it():
    """A curated alias missing from /v1/models still surfaces (first)."""
    curated = M._PROVIDER_MODELS["anthropic"]
    assert "claude-fable-5.1" in curated  # sanity: newest Fable alias is curated
    assert "claude-fable-5" in curated  # sanity: the alias is curated
    assert "claude-opus-5" in curated  # sanity: native flagship matches aggregators
    assert "claude-sonnet-5" in curated  # newest Sonnet alias is curated

    # Live catalog the API would actually return — no fable-5.1 / opus-5.
    live = ["claude-opus-4-8", "claude-sonnet-4-6", "claude-haiku-4-5-20251001"]
    with patch.object(M, "_fetch_anthropic_models", return_value=live):
        result = M.provider_model_ids("anthropic")

    assert "claude-fable-5.1" in result
    assert "claude-fable-5" in result
    assert "claude-opus-5" in result
    assert "claude-sonnet-5" in result
    # Curated order is preserved at the front.
    assert result[:len(curated)] == list(curated)


def test_anthropic_merge_dedupes_overlap_and_appends_live_only():
    """Models in both lists appear once; live-only models are appended."""
    live = [
        "claude-opus-4-8",          # overlaps curated
        "claude-sonnet-4-6",        # overlaps curated
        "claude-future-9-99",       # live-only, not curated
    ]
    with patch.object(M, "_fetch_anthropic_models", return_value=live):
        result = M.provider_model_ids("anthropic")

    # No duplicates introduced by the merge.
    assert result.count("claude-opus-4-8") == 1
    # Live-only entry is preserved (discovery still works for unknown models).
    assert "claude-future-9-99" in result
    # Curated entries lead, live-only trails.
    assert result.index("claude-fable-5.1") < result.index("claude-future-9-99")
    assert result.index("claude-opus-5") < result.index("claude-future-9-99")


def test_anthropic_falls_back_to_curated_when_live_unavailable():
    """No creds / live failure -> curated list verbatim (alias still present)."""
    with patch.object(M, "_fetch_anthropic_models", return_value=None):
        result = M.provider_model_ids("anthropic")

    assert result == list(M._PROVIDER_MODELS["anthropic"])
    assert "claude-fable-5.1" in result
    assert "claude-opus-5" in result
    assert "claude-fable-5" in result
