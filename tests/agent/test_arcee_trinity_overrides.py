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

from agent.agent_init import _resolve_compression_threshold
from agent.auxiliary_client import (
    _compression_threshold_for_model,
    _effective_compression_threshold_percent,
    _fixed_temperature_for_model,
    _is_arcee_trinity_thinking,
    _is_codex_gpt54_or_gpt55,
    _is_codex_spark,
    _update_compressor_model,
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




def test_fixed_temperature_for_trinity_thinking() -> None:
    assert _fixed_temperature_for_model("trinity-large-thinking") == 0.5
    assert _fixed_temperature_for_model("arcee-ai/trinity-large-thinking") == 0.5






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
        "gpt-5", "gpt-5.55", "gpt-5.50", "gpt-5.45", "gpt-5.40",
        "gpt-daybreak-blue-latest-mini", "", None,
    ],
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
    assert _is_codex_gpt54_or_gpt55("gpt-daybreak-blue-latest", "openai-codex") is True
    assert _compression_threshold_for_model("gpt-daybreak-blue-latest", "openai-codex") == 0.85


@pytest.mark.parametrize(
    "model",
    [
        "gpt-5.6-sol-900k",
        "gpt-5.6-terra-900k",
        "gpt-5.6-luna-900k",
        "gpt-5.4-900k",
        "gpt-daybreak-blue-latest-900k",
        "openai/gpt-5.6-sol-900k",
    ],
)
def test_900k_variants_keep_global_threshold(model) -> None:
    """The 85% autoraise compensates for the small 272K window; ``-900k``
    opt-in variants run at ~900K, so they keep the user's global
    ``compression.threshold`` (default 50%) — no override returned."""
    assert _is_codex_gpt54_or_gpt55(model, "openai-codex") is False
    assert _compression_threshold_for_model(model, "openai-codex") is None


def test_base_slugs_still_autoraised_alongside_900k_variants() -> None:
    """Sanity pair: the base slug autoraises while its variant does not."""
    assert _compression_threshold_for_model("gpt-5.6-sol", "openai-codex") == 0.85
    assert _compression_threshold_for_model("gpt-5.6-sol-900k", "openai-codex") is None








# ---------------------------------------------------------------------------
# Codex gpt-5.3-codex-spark compaction-threshold autoraise
#
# gpt-5.3-codex-spark is Codex-OAuth-only (ChatGPT Pro entitlement) with a
# native 128K context window.  The default 50% compaction trigger would fire
# at ~64K — wasting half the usable window, often before the session has
# accumulated enough turns to summarize meaningfully.  This route raises the
# trigger to 70% (~90K) to preserve more raw context while leaving ~38K
# headroom before the 128K hard limit.
# ---------------------------------------------------------------------------






@pytest.mark.parametrize(
    "model",
    [
        "gpt-5.5",  # different family
        "gpt-5.3-codex",  # sibling, not spark
        "gpt-5.3",  # bare 5.3, not spark
        "gpt-5.3-codex-spark-mini",  # hypothetical variant — not matched yet
        "", None,
    ],
)
def test_is_codex_spark_rejects_non_spark_models(model) -> None:
    assert _is_codex_spark(model, "openai-codex") is False








# ── _resolve_compression_threshold (init_agent application logic) ────────────
#
# The Codex overrides are *autoraises*: they raise the trigger (0.85 for the
# gpt-5.4/5.5 272K family, 0.70 for spark) but must never LOWER a higher
# user-configured global threshold.










def test_resolve_no_override_keeps_global() -> None:
    # No per-model override (model_cthresh is None) → global threshold, no notice.
    effective, notice = _resolve_compression_threshold(
        0.50, None, is_codex_autoraise=False
    )
    assert effective == 0.50
    assert notice is None


# ---------------------------------------------------------------------------
# _effective_compression_threshold_percent (external-engine forwarding)
#
# Host update_model() call sites forward the resolved threshold (autoraise
# included) to external context engines whose update_model accepts
# threshold_percent, so ri-context-governor triggers at 0.85 like the
# built-in compressor instead of silently keeping the 0.5 adapter default.
# ---------------------------------------------------------------------------


def test_effective_threshold_percent_codex_gpt56_autoraise() -> None:
    assert _effective_compression_threshold_percent("gpt-5.6-sol", "openai-codex") == 0.85
    assert _effective_compression_threshold_percent("gpt-5.6-terra", "openai-codex") == 0.85
    assert _effective_compression_threshold_percent("gpt-5.6-luna", "openai-codex") == 0.85
    assert _effective_compression_threshold_percent("gpt-5.4", "openai-codex") == 0.85


def test_effective_threshold_percent_falls_back_to_global() -> None:
    # No override for the route → the global compression.threshold (0.50).
    assert _effective_compression_threshold_percent("deepseek-v4-flash", "deepseek") == 0.5
    # Same slug on a non-Codex route keeps the global threshold.
    assert _effective_compression_threshold_percent("gpt-5.6-sol", "openai") == 0.5


def test_effective_threshold_percent_never_lowers_user_threshold() -> None:
    # Autoraise is a raise-only override: a higher user global wins.
    assert (
        _effective_compression_threshold_percent(
            "gpt-5.6-sol", "openai-codex", global_threshold=0.9
        )
        == 0.9
    )
    # A lower user global is raised to the autoraised value.
    assert (
        _effective_compression_threshold_percent(
            "gpt-5.6-sol", "openai-codex", global_threshold=0.3
        )
        == 0.85
    )


class _AcceptsThresholdEngine:
    """Engine with the ri-context-governor-style update_model signature."""

    def __init__(self) -> None:
        self.threshold_percent = None
        self.threshold_tokens = None

    def update_model(
        self,
        model,
        context_length,
        base_url="",
        api_key="",
        provider="",
        api_mode="",
        threshold_percent=None,
    ) -> None:
        self.model = model
        self.context_length = context_length
        if threshold_percent is not None:
            self.threshold_percent = threshold_percent
            self.threshold_tokens = int(context_length * threshold_percent)


class _BuiltinLikeEngine:
    """Engine with the built-in ContextCompressor signature (no threshold kwarg)."""

    def __init__(self) -> None:
        self.threshold_percent = 0.5
        self.threshold_tokens = None

    def update_model(
        self,
        model,
        context_length,
        base_url="",
        api_key="",
        provider="",
        api_mode="",
        max_tokens=None,
    ) -> None:
        self.model = model
        self.context_length = context_length
        self.threshold_tokens = int(context_length * self.threshold_percent)


def test_update_compressor_model_forwards_threshold_when_supported() -> None:
    engine = _AcceptsThresholdEngine()
    _update_compressor_model(
        engine,
        model="gpt-5.6-sol",
        context_length=272000,
        provider="openai-codex",
        api_mode="codex_app_server",
        threshold_percent=0.85,
    )
    assert engine.threshold_percent == 0.85
    assert engine.threshold_tokens == int(272000 * 0.85)


def test_update_compressor_model_skips_unsupported_engine() -> None:
    # The built-in ContextCompressor re-resolves internally; the guard must
    # not pass an unknown kwarg and must not clobber its own threshold.
    engine = _BuiltinLikeEngine()
    _update_compressor_model(
        engine,
        model="gpt-5.6-sol",
        context_length=272000,
        provider="openai-codex",
        api_mode="codex_app_server",
        threshold_percent=0.85,
    )
    assert engine.threshold_percent == 0.5
    assert engine.threshold_tokens == int(272000 * 0.5)


