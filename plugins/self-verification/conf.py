"""Self-verification confidence scoring configuration and threshold management.

Defines the 0-100 continuous confidence scoring system, threshold constants,
and config reading utilities. Single source of truth for confidence levels
and threshold logic used by both verifier.py and __init__.py.

References:
  - Claude Code code-review: 0-100 scoring with threshold 80
  - LLM-as-a-Verifier (arXiv 2607.05391): continuous scoring via logit expectation
"""

from __future__ import annotations

import logging
import os
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Confidence scoring levels (0-100 continuous, not binary pass/fail)
# ---------------------------------------------------------------------------

CONFIDENCE_LEVELS: dict[int, str] = {
    0: "不确定，可能是误报",
    25: "有点怀疑，需要进一步验证",
    50: "中等把握，可能是真的但影响可控",
    75: "高把握，真实且重要",
    100: "绝对确定，一定存在",
}

DEFAULT_THRESHOLD: int = 50  # Below this threshold, issues are not reported


def describe_confidence(score: int) -> str:
    """Return a human-readable description for a confidence score.

    Finds the closest defined level and returns its description.
    Falls back to the lowest level for scores below 0, and highest for > 100.
    """
    if score < 0:
        return CONFIDENCE_LEVELS[0]
    if score > 100:
        return CONFIDENCE_LEVELS[100]

    # Find the closest defined level
    levels = sorted(CONFIDENCE_LEVELS.keys())
    best = levels[0]
    for level in levels:
        if score >= level:
            best = level
        else:
            break
    return CONFIDENCE_LEVELS[best]


# ---------------------------------------------------------------------------
# Config reading
# ---------------------------------------------------------------------------


def _load_config() -> dict[str, Any]:
    """Load hermes config.yaml, returning {} on any failure."""
    try:
        cfg = __import__("hermes_cli.config", fromlist=["load_config"]).load_config()
        return cfg or {}
    except Exception:
        logger.debug("self-verification: failed to load config", exc_info=True)
        return {}


def get_plugin_config() -> dict[str, Any]:
    """Read self-verification plugin config from config.yaml.

    Returns the ``plugins.self-verification`` section, defaulting to an
    empty dict if not configured.
    """
    cfg = _load_config()
    plugin_cfg = (cfg.get("plugins") or {}).get("self-verification") or {}
    if isinstance(plugin_cfg, dict):
        return plugin_cfg
    return {}


def get_confidence_threshold() -> int:
    """Read the confidence threshold from config or env, defaulting to 50.

    Precedence: env var > config.yaml > DEFAULT_THRESHOLD.
    """
    env_threshold = os.environ.get("SELF_VERIFICATION_THRESHOLD")
    if env_threshold is not None:
        try:
            val = int(env_threshold)
            if 0 <= val <= 100:
                return val
        except ValueError:
            pass

    plugin_cfg = get_plugin_config()
    if "confidence_threshold" in plugin_cfg:
        try:
            val = int(plugin_cfg["confidence_threshold"])
            if 0 <= val <= 100:
                return val
        except (ValueError, TypeError):
            pass

    return DEFAULT_THRESHOLD


def get_max_retries() -> int:
    """Read max retries from config or env, defaulting to 3.

    Precedence: env var > config.yaml > default (3).
    """
    env_retries = os.environ.get("SELF_VERIFICATION_MAX_RETRIES")
    if env_retries is not None:
        try:
            val = int(env_retries)
            if val >= 0:
                return val
        except ValueError:
            pass

    plugin_cfg = get_plugin_config()
    if "max_retries" in plugin_cfg:
        try:
            val = int(plugin_cfg["max_retries"])
            if val >= 0:
                return val
        except (ValueError, TypeError):
            pass

    return 3


def get_language() -> str:
    """Read display language from config.yaml, defaulting to 'zh'.

    Used to select the correct i18n footnote text.
    """
    cfg = _load_config()
    lang = cfg.get("display", {}).get("language", "zh")
    if isinstance(lang, str) and lang:
        return lang
    return "zh"


def is_strict_mode() -> bool:
    """Check if strict (blocking) mode is enabled.

    Precedence: env var ``SELF_VERIFICATION_BLOCK=1`` > config.yaml
    ``plugins.self-verification.strict``.
    """
    env_block = os.environ.get("SELF_VERIFICATION_BLOCK", "").lower()
    if env_block in {"1", "true", "yes", "on"}:
        return True

    plugin_cfg = get_plugin_config()
    if plugin_cfg.get("strict", False):
        return True

    return False


def is_plugin_disabled() -> bool:
    """Check if the plugin is disabled via env var or config.

    Precedence: env var ``SELF_VERIFICATION_DISABLE=1`` > config.yaml
    ``plugins.self-verification.enabled`` (default: True).
    """
    env_disable = os.environ.get("SELF_VERIFICATION_DISABLE", "").lower()
    if env_disable in {"1", "true", "yes", "on"}:
        return True

    plugin_cfg = get_plugin_config()
    if "enabled" in plugin_cfg:
        return not bool(plugin_cfg["enabled"])

    return False  # Default: enabled
