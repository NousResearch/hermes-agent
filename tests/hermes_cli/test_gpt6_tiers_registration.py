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
