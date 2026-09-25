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

Bug — one model listed twice under two spellings
    That merge dedupes on the id string, and Anthropic's ``/v1/models`` answers
    fully hyphenated (``claude-opus-5-5``) while marketing copy and OpenRouter
    use dots (``claude-opus-5.5``). A curated entry spelled with a dot therefore
    never matched its live twin: both survived, one near the top and one
    appended at the bottom, so the picker offered the same model twice. The
    curated Anthropic ids are now spelled the way the live endpoint spells them.
"""

import re
from unittest.mock import patch

from hermes_cli import models as M


def _same_model_key(model_id: str) -> str:
    """Fold the dot/hyphen version spellings of one Anthropic model together."""
    return re.sub(r"(\d)\.(\d)", r"\1-\2", str(model_id).strip().lower())


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
    curated = list(M._PROVIDER_MODELS["anthropic"])
    assert result[:len(curated)] == curated and result[-1] == "claude-future-9-99"


def test_anthropic_falls_back_to_curated_when_live_unavailable():
    """No creds / live failure -> curated list verbatim (alias still present)."""
    with patch.object(M, "_fetch_anthropic_models", return_value=None):
        result = M.provider_model_ids("anthropic")

    assert result == list(M._PROVIDER_MODELS["anthropic"])


def test_curated_anthropic_ids_use_the_live_endpoint_spelling():
    """Merging curated against its own hyphenated spelling must add nothing.

    Contract between the curated list and the merge, not a snapshot of either:
    ``/v1/models`` answers fully hyphenated, so any curated id carrying a dotted
    version (``claude-opus-5.5``) is a second spelling of a model the live dump
    already names, and the id-keyed merge cannot fold the two together.
    """
    curated = list(M._PROVIDER_MODELS["anthropic"])
    live_spelling = [_same_model_key(m) for m in curated]

    with patch.object(M, "_fetch_anthropic_models", return_value=live_spelling):
        result = M.provider_model_ids("anthropic")

    assert len(result) == len(curated), (
        "curated Anthropic ids that differ from the live /v1/models spelling: "
        f"{sorted(set(live_spelling) - {m.lower() for m in curated})}"
    )


def test_merged_anthropic_picker_never_offers_one_model_twice():
    """No two rows of the merged picker are spellings of the same model."""
    live = ["claude-opus-5-5", "claude-opus-5", "claude-fable-5-1", "claude-sonnet-5"]
    with patch.object(M, "_fetch_anthropic_models", return_value=live):
        result = M.provider_model_ids("anthropic")

    keys = [_same_model_key(m) for m in result]
    duplicated = sorted({k for k in keys if keys.count(k) > 1})
    assert not duplicated, f"picker lists these models under two spellings: {duplicated}"
