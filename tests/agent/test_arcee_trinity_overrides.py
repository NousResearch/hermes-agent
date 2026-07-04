"""Tests for Arcee Trinity Large Thinking per-model overrides.

Arcee Trinity Large Thinking is a reasoning model that wants:
- Fixed temperature=0.5 (vs the global default)
- Compression threshold=0.75 (delay compression to preserve reasoning context)

The helpers must match the bare model name, including when it arrives via
OpenRouter as ``arcee-ai/trinity-large-thinking``, but must NOT hit sibling
Arcee models like trinity-large-preview or trinity-mini.
"""

from __future__ import annotations

import pytest

from agent.auxiliary_client import (
    OMIT_TEMPERATURE,
    _compression_threshold_for_model,
    _fixed_temperature_for_model,
    _is_arcee_trinity_thinking,
    _is_codex_gpt55,
    _is_kimi_model,
)


@pytest.mark.parametrize(
    "model",
    [
        "trinity-large-thinking",
        "arcee-ai/trinity-large-thinking",
        "Arcee-AI/Trinity-Large-Thinking",  # case-insensitive
        "  trinity-large-thinking  ",  # whitespace tolerant
    ],
)
def test_is_arcee_trinity_thinking_matches(model: str) -> None:
    assert _is_arcee_trinity_thinking(model) is True


@pytest.mark.parametrize(
    "model",
    [
        None,
        "",
        "trinity-large-preview",
        "arcee-ai/trinity-large-preview:free",
        "trinity-mini",
        "arcee-ai/trinity-mini",
        "trinity-large",  # prefix-only must not match
        "claude-sonnet-4.6",
        "gpt-5.4",
    ],
)
def test_is_arcee_trinity_thinking_rejects_non_matches(model) -> None:
    assert _is_arcee_trinity_thinking(model) is False


def test_fixed_temperature_for_trinity_thinking() -> None:
    assert _fixed_temperature_for_model("trinity-large-thinking") == 0.5
    assert _fixed_temperature_for_model("arcee-ai/trinity-large-thinking") == 0.5


def test_fixed_temperature_sibling_arcee_models_unaffected() -> None:
    # Preview and mini do not pin temperature — caller chooses its default.
    assert _fixed_temperature_for_model("trinity-large-preview") is None
    assert _fixed_temperature_for_model("trinity-mini") is None


def test_compression_threshold_for_trinity_thinking() -> None:
    assert _compression_threshold_for_model("trinity-large-thinking") == 0.75
    assert _compression_threshold_for_model("arcee-ai/trinity-large-thinking") == 0.75


def test_compression_threshold_default_none_for_other_models() -> None:
    # None means "leave the user's config value unchanged".
    assert _compression_threshold_for_model(None) is None
    assert _compression_threshold_for_model("") is None
    assert _compression_threshold_for_model("trinity-large-preview") is None
    assert _compression_threshold_for_model("claude-sonnet-4.6") is None
    assert _compression_threshold_for_model("kimi-k2") is None


# ---------------------------------------------------------------------------
# Codex gpt-5.5 compaction-threshold autoraise
#
# ChatGPT's Codex OAuth backend caps gpt-5.5 at a 272K window (verified live:
# ~330K-token request rejected with context_length_exceeded, ~250K accepted).
# The default 50% compaction trigger would fire at ~136K — half the usable
# window — so this route raises the trigger to 85%. Only the Codex OAuth route
# is affected; the same slug on OpenAI direct / OpenRouter / Copilot exposes a
# larger window and keeps the user's global threshold.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "model",
    [
        "gpt-5.5",
        "gpt-5.5-pro",
        "gpt-5.5-2026-04-23",  # dated snapshot
        "gpt-5.5-codex-mini",  # Codex variant of the 5.5 family (also 272K-capped)
        "openai/gpt-5.5",  # aggregator-prefixed (still on the codex route)
        "GPT-5.5",  # case-insensitive
        "  gpt-5.5  ",  # whitespace tolerant
    ],
)
def test_is_codex_gpt55_matches_on_codex_provider(model: str) -> None:
    assert _is_codex_gpt55(model, "openai-codex") is True


@pytest.mark.parametrize(
    "provider",
    ["openrouter", "openai", "copilot", "openai-api", "", None],
)
def test_is_codex_gpt55_rejects_non_codex_providers(provider) -> None:
    # gpt-5.5 on any non-Codex route keeps the larger window — no override.
    assert _is_codex_gpt55("gpt-5.5", provider) is False


@pytest.mark.parametrize(
    "model",
    ["gpt-5.4", "gpt-5", "gpt-5.55", "gpt-5.50", "", None],
)
def test_is_codex_gpt55_rejects_non_55_models(model) -> None:
    # gpt-5.55 / gpt-5.50 are different families and must NOT match — the
    # "gpt-5.5-" / "gpt-5.5." prefix guards require a separator after "5.5".
    assert _is_codex_gpt55(model, "openai-codex") is False


def test_compression_threshold_for_codex_gpt55() -> None:
    assert _compression_threshold_for_model("gpt-5.5", "openai-codex") == 0.85
    assert _compression_threshold_for_model("gpt-5.5-pro", "openai-codex") == 0.85
    assert _compression_threshold_for_model("openai/gpt-5.5", "openai-codex") == 0.85


def test_compression_threshold_codex_gpt55_other_routes_unaffected() -> None:
    # Same slug, different route → no override (keep the user's config value).
    assert _compression_threshold_for_model("gpt-5.5", "openrouter") is None
    assert _compression_threshold_for_model("gpt-5.5", "openai") is None
    assert _compression_threshold_for_model("gpt-5.5", "copilot") is None
    assert _compression_threshold_for_model("openai/gpt-5.5") is None  # no provider


def test_compression_threshold_codex_gpt55_opt_out() -> None:
    # allow_codex_gpt55_autoraise=False reverts to the global default (None).
    assert (
        _compression_threshold_for_model(
            "gpt-5.5", "openai-codex", allow_codex_gpt55_autoraise=False
        )
        is None
    )


def test_compression_threshold_opt_out_does_not_disable_trinity() -> None:
    # The opt-out flag is scoped to the Codex gpt-5.5 autoraise; the Arcee
    # Trinity override must still apply when the flag is False.
    assert (
        _compression_threshold_for_model(
            "trinity-large-thinking", "openrouter", allow_codex_gpt55_autoraise=False
        )
        == 0.75
    )


# ---------------------------------------------------------------------------
# Kimi Coding temperature (issue #53455)
#
# Regular Kimi models (e.g. kimi-k2, kimi-k2.5) manage temperature server-side
# and we omit it (OMIT_TEMPERATURE). But models on the Kimi Coding API
# (api.kimi.com/coding/v1) like kimi-k2.7-code strictly REQUIRE temperature=1
# and 400 on any other value or omission.  Detect the coding endpoint by
# its base_url and return 1.0 instead of OMIT_TEMPERATURE.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "model",
    [
        "kimi-k2",
        "kimi-k2.5",
        "kimi-k2.5-pro",
        "moonshot-v1-8k",  # starts with moonshot not kimi-, so NOT a match
        None,
        "",
    ],
)
def test_is_kimi_model_matches(model) -> None:
    if model and model.startswith("kimi-"):
        assert _is_kimi_model(model) is True
    else:
        assert _is_kimi_model(model) is False


def test_fixed_temperature_kimi_regular_omits() -> None:
    """Regular Kimi API (non-coding) models omit temperature."""
    from agent.auxiliary_client import _fixed_temperature_for_model, OMIT_TEMPERATURE

    # Regular kimi models: OMIT_TEMPERATURE
    assert _fixed_temperature_for_model("kimi-k2") is OMIT_TEMPERATURE
    assert _fixed_temperature_for_model("kimi-k2.5") is OMIT_TEMPERATURE
    # No base_url → treat as regular kimi
    assert _fixed_temperature_for_model("kimi-k2.7-code") is OMIT_TEMPERATURE


def test_fixed_temperature_kimi_coding_requires_1() -> None:
    """Kimi Coding API models (base_url contains /coding/) must use temperature=1."""
    from agent.auxiliary_client import _fixed_temperature_for_model

    # kimi-k2.7-code on the coding endpoint → must be 1.0
    assert (
        _fixed_temperature_for_model(
            "kimi-k2.7-code", base_url="https://api.kimi.com/coding/v1"
        )
        == 1.0
    )
    # Same model with no base_url → OMIT (treated as regular kimi)
    assert (
        _fixed_temperature_for_model("kimi-k2.7-code") is OMIT_TEMPERATURE
    )
    # kimi-k2.5 on coding endpoint → still OMIT (only coding models need T=1)
    assert (
        _fixed_temperature_for_model(
            "kimi-k2.5", base_url="https://api.kimi.com/coding/v1"
        )
        is OMIT_TEMPERATURE
    )


def test_fixed_temperature_kimi_coding_variants() -> None:
    """Various kimi models tested with coding/non-coding base URLs."""
    from agent.auxiliary_client import _fixed_temperature_for_model, OMIT_TEMPERATURE

    # Non-kimi model unaffected
    assert _fixed_temperature_for_model("claude-sonnet-4.6", "https://api.kimi.com/coding/v1") is None
    # kimi-k2 on coding — only coding-* models should get the override
    assert _fixed_temperature_for_model("kimi-k2", "https://api.kimi.com/coding/v1") is OMIT_TEMPERATURE
    # kimi-k2.7-code with /coding in path gets 1.0
    assert _fixed_temperature_for_model("kimi-k2.7-code", "https://api.kimi.com/coding/v1") == 1.0
