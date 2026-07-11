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


def _curated_fast_pairs(provider: str):
    """Return ``(base, fast)`` id pairs where both the base model and its
    ``<base>-fast`` variant are curated for ``provider``.

    Derived from the live curated list rather than hard-coded ids, so the
    contract survives catalog churn: whatever the current ``-fast`` variants
    are, each must ship alongside its base so the desktop picker's
    ``collapseModelFamilies`` can group them and ``resolveFastControl`` can
    light the variant toggle.
    """
    curated = list(M._PROVIDER_MODELS[provider])
    curated_set = set(curated)
    return [
        (m[: -len("-fast")], m)
        for m in curated
        if m.endswith("-fast") and m[: -len("-fast")] in curated_set
    ]


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


def test_anthropic_curated_fast_variants_ship_with_their_base():
    """Every curated ``-fast`` Anthropic variant ships alongside its base id.

    A fast variant is a SEPARATE model id, so the desktop picker only exposes
    its Fast toggle when the ``-fast`` sibling is enumerated next to its base
    (collapseModelFamilies groups them, resolveFastControl lights the variant
    toggle). This is the picker-control contract, expressed generically over
    whatever fast variants the catalog currently carries — not pinned to a
    single catalog snapshot.
    """
    pairs = _curated_fast_pairs("anthropic")
    # Regression guard: the Anthropic catalog must carry at least one variant
    # fast pair (the desktop Fast toggle vanished when it carried none).
    assert pairs, "no curated <base>/<base>-fast pair in the Anthropic catalog"

    for base, fast in pairs:
        # Both ids present (grouping precondition).
        assert base in M._PROVIDER_MODELS["anthropic"]
        assert fast in M._PROVIDER_MODELS["anthropic"]
        # Fast is offered as a variant model, NOT the speed=fast request param,
        # so the base must not also claim param-fast support (that would double
        # up the control). The param-fast path is a separate mechanism.
        assert M.model_supports_fast_mode(base) is False


def test_desktop_model_options_exposes_curated_fast_sibling():
    """The desktop ``model.options`` payload surfaces the curated fast sibling.

    The desktop picker builds its rows via ``build_models_payload`` (the same
    substrate the ``model.options`` gateway method calls), which populates a
    provider row's ``models`` from the curated catalog — NOT from
    ``provider_model_ids()``. This drives that path offline and asserts the
    Anthropic row exposes each curated base/-fast pair with capabilities the
    picker can gate the Fast control on.
    """
    from hermes_cli.inventory import build_models_payload, load_picker_context

    ctx = load_picker_context()
    payload = build_models_payload(
        ctx,
        include_unconfigured=True,  # surface Anthropic even without creds
        picker_hints=True,
        canonical_order=True,
        pricing=False,              # no network
        capabilities=True,
        probe_custom_providers=False,
        probe_current_custom_provider=False,
    )
    anthropic_rows = [
        r for r in payload.get("providers", [])
        if str(r.get("slug", "")).lower() == "anthropic"
    ]
    assert anthropic_rows, "Anthropic row absent from model.options payload"
    row = anthropic_rows[0]
    row_models = set(row.get("models", []))
    caps = row.get("capabilities", {})

    for base, fast in _curated_fast_pairs("anthropic"):
        assert base in row_models, f"{base} missing from desktop Anthropic row"
        assert fast in row_models, f"{fast} missing from desktop Anthropic row"
        # The picker gates the variant toggle on the base NOT advertising
        # param-fast, so the sibling is the only way to reach fast.
        assert caps.get(base, {}).get("fast") is False


def test_anthropic_fast_sibling_survives_live_merge():
    """The curated fast variant surfaces even when /v1/models omits it."""
    live = ["claude-opus-4-8", "claude-sonnet-4-6"]
    with patch.object(M, "_fetch_anthropic_models", return_value=live):
        result = M.provider_model_ids("anthropic")

    assert "claude-opus-4-8-fast" in result
