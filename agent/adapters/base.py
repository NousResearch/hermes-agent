"""Base constants and model-limit tables for the Anthropic adapter."""

import logging
from typing import Dict

logger = logging.getLogger(__name__)

# ── Thinking budget ──────────────────────────────────────────────────────────

THINKING_BUDGET = {"xhigh": 32000, "high": 16000, "medium": 8000, "low": 4000}

ADAPTIVE_EFFORT_MAP = {
    "xhigh": "max",
    "high": "high",
    "medium": "medium",
    "low": "low",
    "minimal": "low",
}

# ── Max output token limits per Anthropic model ──────────────────────────────
# Source: Anthropic docs + Cline model catalog.  Anthropic's API requires
# max_tokens as a mandatory field.  Previously we hardcoded 16384, which
# starves thinking-enabled models (thinking tokens count toward the limit).
_ANTHROPIC_OUTPUT_LIMITS: Dict[str, int] = {
    # Claude 4.6
    "claude-opus-4-6":   128_000,
    "claude-sonnet-4-6":  64_000,
    # Claude 4.5
    "claude-opus-4-5":    64_000,
    "claude-sonnet-4-5":  64_000,
    "claude-haiku-4-5":   64_000,
    # Claude 4
    "claude-opus-4":      32_000,
    "claude-sonnet-4":    64_000,
    # Claude 3.7
    "claude-3-7-sonnet": 128_000,
    # Claude 3.5
    "claude-3-5-sonnet":   8_192,
    "claude-3-5-haiku":    8_192,
    # Claude 3
    "claude-3-opus":       4_096,
    "claude-3-sonnet":     4_096,
    "claude-3-haiku":      4_096,
    # Third-party Anthropic-compatible providers
    "minimax":            131_072,
}

# For any model not in the table, assume the highest current limit.
# Future Anthropic models are unlikely to have *less* output capacity.
_ANTHROPIC_DEFAULT_OUTPUT_LIMIT = 128_000


def _get_anthropic_max_output(model: str) -> int:
    """Look up the max output token limit for an Anthropic model.

    Uses substring matching against _ANTHROPIC_OUTPUT_LIMITS so date-stamped
    model IDs (claude-sonnet-4-5-20250929) and variant suffixes (:1m, :fast)
    resolve correctly.  Longest-prefix match wins to avoid e.g. "claude-3-5"
    matching before "claude-3-5-sonnet".

    Normalizes dots to hyphens so that model names like
    ``anthropic/claude-opus-4.6`` match the ``claude-opus-4-6`` table key.
    """
    m = model.lower().replace(".", "-")
    best_key = ""
    best_val = _ANTHROPIC_DEFAULT_OUTPUT_LIMIT
    for key, val in _ANTHROPIC_OUTPUT_LIMITS.items():
        if key in m and len(key) > len(best_key):
            best_key = key
            best_val = val
    return best_val


def _supports_adaptive_thinking(model: str) -> bool:
    """Return True for Claude 4.6 models that support adaptive thinking."""
    return any(v in model for v in ("4-6", "4.6"))
