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
    _fixed_temperature_for_model,
    _is_arcee_trinity_thinking,
    _is_codex_gpt54_or_gpt55,
    _is_codex_spark,
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


