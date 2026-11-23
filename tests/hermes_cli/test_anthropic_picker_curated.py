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

import re
from unittest.mock import patch

from agent.anthropic_message_convert import normalize_model_name
from hermes_cli import models as M
from hermes_cli.models_catalog_static import _OPENROUTER_ONLY, OPENROUTER_MODELS


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


# Contracts on the curated native list itself. The native Anthropic API spells versions with
# hyphens (``claude-fable-5-1``); OpenRouter/Nous use dots (``anthropic/claude-fable-5.1``).

_FAMILY_RANK = {"fable": 0, "opus": 1, "sonnet": 2, "haiku": 3}


def _family_and_version(model_id: str) -> tuple[str, tuple[int, ...]]:
    family, *parts = model_id.removeprefix("claude-").split("-")
    return family, tuple(int(p) for p in parts if not re.fullmatch(r"\d{8}", p))


def test_anthropic_curated_ids_are_native_wire_ids():
    """A dotted id is rewritten on the wire, so the picker would list it beside the live
    ``/v1/models`` row for the same model: every curated id must already be its wire form."""
    curated = M._PROVIDER_MODELS["anthropic"]
    assert [m for m in curated if normalize_model_name(m) != m] == []

    live = [normalize_model_name(m) for m in curated]
    with patch.object(M, "_fetch_anthropic_models", return_value=live):
        assert M.provider_model_ids("anthropic") == list(curated)


def test_anthropic_curated_covers_aggregator_claude_models():
    """Every Claude model the OpenRouter/Nous lists offer (minus routing variants) is pickable
    on the native Anthropic provider, as its bare or dated wire id."""
    native = M._PROVIDER_MODELS["anthropic"]
    missing = [
        mid for mid, _ in OPENROUTER_MODELS
        if mid.startswith("anthropic/") and mid not in _OPENROUTER_ONLY
        and not any(re.fullmatch(re.escape(normalize_model_name(mid)) + r"(-\d{8})?", n) for n in native)
    ]
    assert missing == []


def test_anthropic_curated_is_grouped_by_tier_newest_first():
    """Same order as the OpenRouter Claude block: Fable, Opus, Sonnet, Haiku, each newest first."""
    keys = [_family_and_version(m) for m in M._PROVIDER_MODELS["anthropic"]]
    ranks = [_FAMILY_RANK[family] for family, _ in keys]
    assert ranks == sorted(ranks)
    out_of_order = [(a, b) for a, b in zip(keys, keys[1:]) if a[0] == b[0] and a[1] <= b[1]]
    assert out_of_order == []
