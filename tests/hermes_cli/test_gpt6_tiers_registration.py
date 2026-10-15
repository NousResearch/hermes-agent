"""Behavior contracts for the GPT-6 Sol/Terra/Luna registration (the 5.6 tier successors).

Invariant tests only, no list snapshots. They pin what would silently regress:

1. `/model gpt` still lands on the flagship: Astra outranks Sol, Sol outranks
   Terra/Luna, and every GPT-6 tier outranks its 5.6 predecessor.
2. The Codex OAuth `-900k` opt-in machinery treats the gpt-6 tiers exactly like
   the 5.6 ones: picker synthesis, dated snapshots, wire stripping, the
   compaction auto-raise on the base slug (and not on the variant), and the
   gpt-5.6 effort ladder (``max`` allowed).
"""

from agent.auxiliary_client import _compression_threshold_for_model
from agent.model_metadata import (
    _verified_codex_ctx_for_slug,
    is_codex_900k_base,
    strip_codex_context_variant_suffix,
)
from agent.reasoning_effort import CODEX_GPT56_EFFORTS, codex_supported_efforts
from hermes_cli.codex_models import _finalize_codex_models
from hermes_cli.model_switch import _model_sort_key

GPT6_TIERS = ("gpt-6-sol", "gpt-6-luna")  # terra: never published by OpenAI, not on OpenRouter/Codex (2026-09-22)

def test_model_gpt_resolves_flagship_across_gpt6_tiers():
    models = ["gpt-6-luna", "gpt-5.6-sol", "gpt-6-sol", "gpt-6-astra"]
    models.sort(key=lambda m: _model_sort_key(m, "gpt"))
    assert models[:2] == ["gpt-6-astra", "gpt-6-sol"]
    assert models.index("gpt-6-luna") < models.index("gpt-5.6-sol")

def test_gpt6_tiers_share_the_codex_900k_contract_with_56():
    ids = _finalize_codex_models(["gpt-5.5"])  # forward-compat synthesizes the tiers from 5.5
    for base in GPT6_TIERS:
        assert ids.index(f"{base}-900k") == ids.index(base) + 1, base
        assert f"{base}-pro-900k" not in ids
        assert is_codex_900k_base(f"{base}-2026-09-22"), base  # dated snapshots inherit eligibility
        assert strip_codex_context_variant_suffix(f"openai/{base}-900k") == f"openai/{base}"
        assert _verified_codex_ctx_for_slug(f"{base}-900k") == _verified_codex_ctx_for_slug("gpt-5.6-sol-900k")
        assert _compression_threshold_for_model(base, provider="openai-codex") == \
            _compression_threshold_for_model("gpt-5.6-sol", provider="openai-codex")
        assert _compression_threshold_for_model(f"{base}-900k", provider="openai-codex") is None
        assert codex_supported_efforts(f"openai/{base}") == CODEX_GPT56_EFFORTS


def test_gpt61_sol_takes_astra_ladder_without_astra_gating():
    """``none`` 400s on gpt-6.1-sol (live 2026-09-29), but it is not account-gated like Astra."""
    from agent.reasoning_effort import CODEX_ASTRA_EFFORTS, is_astra_model

    for slug in ("gpt-6.1-sol", "openai/gpt-6.1-sol-pro", "gpt-6.1-sol-2026-09-29"):
        assert codex_supported_efforts(slug) == CODEX_ASTRA_EFFORTS, slug
        assert not is_astra_model(slug), slug
    assert "none" in codex_supported_efforts("gpt-6-sol")


def test_openrouter_omits_disable_the_openai_ladder_rejects(monkeypatch):
    """OpenRouter's catalog advertises ``none`` for both Sol generations; only 6.1 must omit the disable."""
    import hermes_cli.models_reasoning_caps as caps_mod
    from providers import get_provider_profile

    monkeypatch.setattr(caps_mod, "openrouter_model_reasoning_capabilities", lambda model: {
        "supports_reasoning": True, "mandatory": False,
        "supported_efforts": ["max", "xhigh", "high", "medium", "low", "none"]})
    p = get_provider_profile("openrouter")
    off = {"enabled": False}
    body, _ = p.build_api_kwargs_extras(reasoning_config=off, supports_reasoning=True, model="openai/gpt-6.1-sol")
    assert "reasoning" not in body
    body, _ = p.build_api_kwargs_extras(reasoning_config=off, supports_reasoning=True, model="openai/gpt-6-sol")
    assert body["reasoning"] == off


def test_gpt61_sol_resolves_context_and_pricing_like_its_tier():
    from agent.model_metadata import DEFAULT_CONTEXT_LENGTHS, _CODEX_OAUTH_CONTEXT_FALLBACK
    from agent.usage_pricing import _OFFICIAL_DOCS_PRICING

    assert DEFAULT_CONTEXT_LENGTHS["gpt-6.1-sol"] == DEFAULT_CONTEXT_LENGTHS["gpt-6-sol"]
    assert _CODEX_OAUTH_CONTEXT_FALLBACK["gpt-6.1-sol"] == _CODEX_OAUTH_CONTEXT_FALLBACK["gpt-6-sol"]
    base = _OFFICIAL_DOCS_PRICING[("openai", "gpt-6.1-sol")]
    assert _OFFICIAL_DOCS_PRICING[("openai", "gpt-6.1-sol-pro")] is base
    assert base.cache_read_cost_per_million == base.input_cost_per_million / 20  # 5%, not 6 Sol's 10%


def test_gpt61_sol_900k_is_opt_in_exact_and_billed_as_the_base():
    from agent.model_metadata import _CODEX_OAUTH_STALE_ADVERTISED_CTX, is_codex_context_variant
    from agent.usage_pricing import _OFFICIAL_DOCS_PRICING

    ids = _finalize_codex_models(["gpt-6.1-sol"])  # what live discovery hands the picker
    assert ids.index("gpt-6.1-sol-900k") == ids.index("gpt-6.1-sol") + 1
    assert not {"gpt-6.1-sol", "gpt-6.1-sol-900k"} & set(_finalize_codex_models(["gpt-5.5"]))  # no entitlement, no entry
    assert not is_codex_900k_base("gpt-6.1-sol-pro")  # exact slug: -pro is not routable on Codex
    assert is_codex_context_variant("openai/gpt-6.1-sol-900k")
    assert strip_codex_context_variant_suffix("gpt-6.1-sol-900k") == "gpt-6.1-sol"
    # The base keeps the advertised 272K; only the explicit variant opts into the bump.
    assert _verified_codex_ctx_for_slug("gpt-6.1-sol") is None
    assert _CODEX_OAUTH_STALE_ADVERTISED_CTX < _verified_codex_ctx_for_slug("gpt-6.1-sol-900k") < 922_000  # 1.05M context - 128K max output
    assert _OFFICIAL_DOCS_PRICING[("openai", "gpt-6.1-sol-900k")] is _OFFICIAL_DOCS_PRICING[("openai", "gpt-6.1-sol")]
    assert _compression_threshold_for_model("gpt-6.1-sol-900k", provider="openai-codex") is None
