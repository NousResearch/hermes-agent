"""Regression tests for typed /model <name> case-insensitive canonicalization.

Repro: with the current provider set to minimax-oauth (catalog
['MiniMax-M3', 'MiniMax-M2.7', 'MiniMax-M2.7-highspeed']), typing
/model minimax-m3 (lowercased full id) returned success=True with
new_model='minimax-m3' literally. That string is not in the catalog
the provider serves, so the next API call hits api.minimax.io/anthropic
with an unknown model id and the provider either errors or quietly maps
the request to a different model -- both bad outcomes for a paying user.

The fix adds a canonical-slug normalization step in PATH B of
switch_model: before any provider-detect/validation/route logic, look
up new_model (case-insensitive equality) in the current provider catalog.
If exactly one entry matches, rewrite new_model to the catalog spelling.
If multiple entries match, fall through (downstream steps will see the
original name and can surface ambiguity). Aggregators are skipped -- their
step d already canonicalizes via slash/bare variants.

The fix uses strict case-insensitive equality rather than prefix / fuzzy
matching: a short input like m3 would match MiniMax-M3 AND any other
catalog entry whose lowercased name contains m3 as a substring on a
provider with broader catalogs (e.g. openrouter has 28 models). That is
a silent-pick hazard. Keeping the matcher strict means a short shorthand
still falls through and the existing detect_provider_for_model +
validation paths can produce a proper suggestion response.

Hermetic: every external lookup is mocked (no network), mirroring
tests/hermes_cli/test_model_switch_configured_provider_routing.py.
"""

from unittest.mock import patch

from hermes_cli.model_switch import switch_model

_ACCEPTED = {"accepted": True, "persist": True, "recognized": True, "message": None}


def _run_switch(
    *,
    raw_input,
    current_provider,
    current_model="old-model",
    current_base_url="",
    user_providers=None,
    custom_providers=None,
    current_provider_catalog=None,
    validation=_ACCEPTED,
):
    """Drive switch_model with the resolution chain mocked out.

    External lookups are patched so this test isolates the canonical-slug
    normalization step in PATH B. current_provider_catalog mimics the
    live list_provider_models(current_provider) return value for the
    current provider -- the static catalog the user is currently on.
    """
    catalog = current_provider_catalog or []
    with patch("hermes_cli.model_switch.resolve_alias", return_value=None), \
         patch(
             "hermes_cli.model_switch.list_provider_models",
             side_effect=lambda provider, *a, **kw: (
                 catalog if provider == current_provider else []
             ),
         ), \
         patch("hermes_cli.model_switch.normalize_model_for_provider", side_effect=lambda model, provider: model), \
         patch("hermes_cli.models.validate_requested_model", return_value=validation), \
         patch("hermes_cli.models.detect_provider_for_model", return_value=None), \
         patch("hermes_cli.model_switch.get_model_info", return_value=None), \
         patch("hermes_cli.model_switch.get_model_capabilities", return_value=None), \
         patch(
             "hermes_cli.runtime_provider.resolve_runtime_provider",
             return_value={
                 "api_key": "***",
                 "base_url": current_base_url or "http://resolved/v1",
                 "api_mode": "",
             },
         ):
        return switch_model(
            raw_input=raw_input,
            current_provider=current_provider,
            current_model=current_model,
            current_base_url=current_base_url,
            user_providers=user_providers or {},
            custom_providers=custom_providers or [],
        )


# -- Canonical-slug normalization on the current provider catalog ---------


def test_lowercase_full_id_canonicalizes_to_catalog_id():
    """minimax-m3 (lowercased full id) is mapped to MiniMax-M3 --
    preserves intent when the user forgets the camel case."""
    result = _run_switch(
        raw_input="minimax-m3",
        current_provider="minimax-oauth",
        current_provider_catalog=["MiniMax-M3", "MiniMax-M2.7", "MiniMax-M2.7-highspeed"],
    )
    assert result.success is True, result.error_message
    assert result.new_model == "MiniMax-M3"


def test_uppercase_full_id_canonicalizes_to_catalog_id():
    """MINIMAX-M3 (uppercased full id) maps the same way as lowercased:
    case-insensitive equality, not just lower-canonicalization."""
    result = _run_switch(
        raw_input="MINIMAX-M3",
        current_provider="minimax-oauth",
        current_provider_catalog=["MiniMax-M3", "MiniMax-M2.7", "MiniMax-M2.7-highspeed"],
    )
    assert result.success is True, result.error_message
    assert result.new_model == "MiniMax-M3"


def test_short_shorthand_is_not_silently_rewritten():
    """A short shorthand (m3) does NOT match any catalog entry under
    strict case-insensitive equality and therefore must fall through.
    Silent prefix or substring rewriting here would re-introduce the
    bug class we are fixing: ambiguous catalog hits on broad providers.
    Downstream steps see m3 and can produce a proper suggestion."""
    result = _run_switch(
        raw_input="m3",
        current_provider="minimax-oauth",
        current_provider_catalog=["MiniMax-M3", "MiniMax-M2.7", "MiniMax-M2.7-highspeed"],
    )
    assert result.success is True
    assert result.new_model == "m3"


def test_exact_match_is_not_rewritten():
    """When the user types the canonical id verbatim, the result must
    equal that id exactly -- no surprise rewrites that could mask
    intentional overrides (e.g. a fork on the provider side)."""
    result = _run_switch(
        raw_input="MiniMax-M3",
        current_provider="minimax-oauth",
        current_provider_catalog=["MiniMax-M3", "MiniMax-M2.7", "MiniMax-M2.7-highspeed"],
    )
    assert result.success is True
    assert result.new_model == "MiniMax-M3"


def test_unknown_name_does_not_rewrite():
    """A name that matches nothing in the current catalog falls through
    unchanged so downstream steps (provider detection, validation, error
    reporting) can still act on it. The bug was silent rewrites; this
    test guards against a silent-rewrite-to-wrong-catalog regression."""
    result = _run_switch(
        raw_input="totally-unknown-model-xyz",
        current_provider="minimax-oauth",
        current_provider_catalog=["MiniMax-M3", "MiniMax-M2.7", "MiniMax-M2.7-highspeed"],
    )
    assert result.success is True
    assert result.new_model == "totally-unknown-model-xyz"


def test_ambiguous_short_name_falls_through():
    """m2.7 could match MiniMax-M2.7 OR MiniMax-M2.7-highspeed; when
    both are in the catalog the canonicalizer must NOT pick one -- fall
    through and let later steps surface the ambiguity. Without this
    guard the bug class would reappear in a different shape (silent
    pick of the wrong sibling)."""
    result = _run_switch(
        raw_input="m2.7",
        current_provider="minimax-oauth",
        current_provider_catalog=["MiniMax-M3", "MiniMax-M2.7", "MiniMax-M2.7-highspeed"],
    )
    # No silent rewrite. m2.7 is not equal to any catalog id, so the
    # result preserves the input verbatim for the rest of the pipeline.
    assert result.new_model == "m2.7"


def test_canonicalization_is_a_noop_on_aggregators_with_exact_match():
    """Aggregator catalogs already canonicalize via step d; verify we
    do not double-process and corrupt the slug. openrouter is the
    canonical aggregator used in tests."""
    result = _run_switch(
        raw_input="anthropic/claude-sonnet-4-6",
        current_provider="openrouter",
        current_provider_catalog=["anthropic/claude-sonnet-4-6", "openai/gpt-5.5"],
    )
    assert result.success is True
    assert result.new_model == "anthropic/claude-sonnet-4-6"