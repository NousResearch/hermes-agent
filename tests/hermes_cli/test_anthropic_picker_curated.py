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


def test_anthropic_curated_alias_survives_when_live_omits_it():
    """A curated alias missing from /v1/models still surfaces (first)."""
    curated = M._PROVIDER_MODELS["anthropic"]
    assert "claude-fable-5" in curated  # sanity: the alias is curated

    # Live catalog the API would actually return — no fable-5.
    live = ["claude-opus-4-8", "claude-sonnet-4-6", "claude-haiku-4-5-20251001"]
    with patch.object(M, "_fetch_anthropic_models", return_value=live):
        result = M.provider_model_ids("anthropic")

    assert "claude-fable-5" in result
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
    assert result.index("claude-fable-5") < result.index("claude-future-9-99")


def test_anthropic_falls_back_to_curated_when_live_unavailable():
    """No creds / live failure -> curated list verbatim (alias still present)."""
    with patch.object(M, "_fetch_anthropic_models", return_value=None):
        result = M.provider_model_ids("anthropic")

    assert result == list(M._PROVIDER_MODELS["anthropic"])
    assert "claude-fable-5" in result


def test_anthropic_curated_lists_opus_48_fast_sibling():
    """Opus 4.8's fast variant is a SEPARATE model id, so the desktop picker
    only exposes its Fast toggle when the ``-fast`` sibling is enumerated next
    to its base (collapseModelFamilies groups them, resolveFastControl lights
    the variant toggle). Regression for the missing desktop fast control.
    """
    curated = M._PROVIDER_MODELS["anthropic"]
    assert "claude-opus-4-8" in curated
    assert "claude-opus-4-8-fast" in curated

    # Fast is offered as a variant model, NOT the speed=fast request param —
    # so the base must not claim param-fast support (that path is Opus 4.6).
    assert M.model_supports_fast_mode("claude-opus-4-8") is False
    assert M.model_supports_fast_mode("claude-opus-4-6") is True


def test_anthropic_fast_sibling_survives_live_merge():
    """The curated fast variant surfaces even when /v1/models omits it."""
    live = ["claude-opus-4-8", "claude-sonnet-4-6"]
    with patch.object(M, "_fetch_anthropic_models", return_value=live):
        result = M.provider_model_ids("anthropic")

    assert "claude-opus-4-8-fast" in result
