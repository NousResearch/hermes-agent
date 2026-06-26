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
    _compression_threshold_for_model,
    _fixed_temperature_for_model,
    _is_arcee_trinity_thinking,
    _is_codex_gpt54_or_gpt55,
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
# Codex gpt-5.4 / gpt-5.5 compaction-threshold autoraise
#
# ChatGPT's Codex OAuth backend caps both families at a 272K window (verified
# live via the Codex /models resolver and per-slug fallback table). The default
# 50% compaction trigger would fire at ~136K — half the usable window — so this
# route raises the trigger to 85%. Only the Codex OAuth route is affected; the
# same slugs on OpenAI direct / OpenRouter / Copilot expose a larger window and
# keep the user's global threshold.
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
        "gpt-5.4",  # base 5.4 (272K-capped)
        "gpt-5.4-pro",  # pro 5.4 variant (272K-capped)
        "gpt-5.4-2026-01-01",  # dated 5.4 snapshot
        "openai/gpt-5.4",  # aggregator-prefixed 5.4
    ],
)
def test_is_codex_gpt54_or_gpt55_matches_on_codex_provider(model: str) -> None:
    assert _is_codex_gpt54_or_gpt55(model, "openai-codex") is True


@pytest.mark.parametrize(
    "provider",
    ["openrouter", "openai", "copilot", "openai-api", "", None],
)
def test_is_codex_gpt54_or_gpt55_rejects_non_codex_providers(provider) -> None:
    # gpt-5.4 / gpt-5.5 on any non-Codex route keep the larger window.
    assert _is_codex_gpt54_or_gpt55("gpt-5.5", provider) is False
    assert _is_codex_gpt54_or_gpt55("gpt-5.4", provider) is False


@pytest.mark.parametrize(
    "model",
    ["gpt-5", "gpt-5.55", "gpt-5.50", "gpt-5.45", "gpt-5.40", "", None],
)
def test_is_codex_gpt54_or_gpt55_rejects_non_54_55_models(model) -> None:
    # Close numeric neighbours must NOT match — the prefix guards require a
    # separator after "5.4" / "5.5" so e.g. gpt-5.45 and gpt-5.55 stay out.
    assert _is_codex_gpt54_or_gpt55(model, "openai-codex") is False


def test_compression_threshold_for_codex_gpt55() -> None:
    assert _compression_threshold_for_model("gpt-5.4", "openai-codex") == 0.85
    assert _compression_threshold_for_model("gpt-5.4-pro", "openai-codex") == 0.85
    assert _compression_threshold_for_model("openai/gpt-5.4", "openai-codex") == 0.85
    assert _compression_threshold_for_model("gpt-5.5", "openai-codex") == 0.85
    assert _compression_threshold_for_model("gpt-5.5-pro", "openai-codex") == 0.85
    assert _compression_threshold_for_model("openai/gpt-5.5", "openai-codex") == 0.85


def test_compression_threshold_codex_gpt55_other_routes_unaffected() -> None:
    # Same slug, different route → no override (keep the user's config value).
    assert _compression_threshold_for_model("gpt-5.4", "openrouter") is None
    assert _compression_threshold_for_model("gpt-5.4", "openai") is None
    assert _compression_threshold_for_model("gpt-5.4", "copilot") is None
    assert _compression_threshold_for_model("gpt-5.5", "openrouter") is None
    assert _compression_threshold_for_model("gpt-5.5", "openai") is None
    assert _compression_threshold_for_model("gpt-5.5", "copilot") is None
    assert _compression_threshold_for_model("openai/gpt-5.4") is None  # no provider
    assert _compression_threshold_for_model("openai/gpt-5.5") is None  # no provider


def test_compression_threshold_codex_gpt55_opt_out() -> None:
    # Historical flag name still governs both Codex families.
    assert (
        _compression_threshold_for_model(
            "gpt-5.4", "openai-codex", allow_codex_gpt55_autoraise=False
        )
        is None
    )
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
