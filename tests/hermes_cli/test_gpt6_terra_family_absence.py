"""Unreleased GPT-6 Terra must not be advertised as a GPT-6 family member."""

import json
from pathlib import Path

from agent.auxiliary_client import _is_codex_gpt54_or_gpt55
from agent.model_metadata import (
    DEFAULT_CONTEXT_LENGTHS,
    _CODEX_900K_ELIGIBLE_BASES,
    _CODEX_900K_SNAPSHOT_BASES,
    _CODEX_OAUTH_CONTEXT_FALLBACK,
    _CODEX_OAUTH_VERIFIED_ABOVE_ADVERTISED_PREFIXES,
    is_codex_900k_base,
)
from agent.reasoning_effort import GPT6_TIER_PREFIXES
from hermes_cli.models import OPENROUTER_MODELS, _PROVIDER_MODELS


def test_unreleased_terra_not_in_curated_gpt6_family():
    aggregator = {model for model, _description in OPENROUTER_MODELS}
    website_catalog = json.loads(
        (Path(__file__).resolve().parents[2] / "website/static/api/model-catalog.json").read_text(encoding="utf-8")
    )
    website_ids = {row["id"] for provider in website_catalog["providers"].values() for row in provider["models"]}
    for model_ids in (
        aggregator,
        set(_PROVIDER_MODELS["nous"]),
        set(_PROVIDER_MODELS["openai-api"]),
        website_ids,
    ):
        assert not any("gpt-6-terra" in model for model in model_ids)
    assert "gpt-5.6-terra" in _PROVIDER_MODELS["openai-api"]
    assert "openai/gpt-6-sol" in aggregator
    assert "openai/gpt-6-luna" in aggregator


def test_unreleased_terra_has_no_family_or_codex_context_contract():
    assert "gpt-6-terra" not in DEFAULT_CONTEXT_LENGTHS
    assert "gpt-6-terra" not in _CODEX_OAUTH_CONTEXT_FALLBACK
    assert "gpt-6-terra" not in _CODEX_OAUTH_VERIFIED_ABOVE_ADVERTISED_PREFIXES
    assert "gpt-6-terra" not in _CODEX_900K_ELIGIBLE_BASES
    assert "gpt-6-terra" not in _CODEX_900K_SNAPSHOT_BASES
    assert "gpt-6-terra" not in GPT6_TIER_PREFIXES
    assert not is_codex_900k_base("gpt-6-terra")
    assert not is_codex_900k_base("gpt-6-terra-2026-09-22")
    assert not _is_codex_gpt54_or_gpt55("gpt-6-terra", "openai-codex")
    assert is_codex_900k_base("gpt-5.6-terra")
    assert is_codex_900k_base("gpt-6-sol")
    assert is_codex_900k_base("gpt-6-luna")
