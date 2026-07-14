"""Deterministic, conservative per-turn model tier selection.

The router deliberately returns only a model ID: callers keep their already
resolved provider credentials and runtime settings.  This prevents a routing
policy from silently switching providers or bypassing shared credential and
fallback handling.
"""

from __future__ import annotations

import re
from typing import Any

_HARD_PATTERN = re.compile(
    r"\b("
    r"code|coding|debug|bug|error|exception|traceback|stack trace|"
    r"test|pytest|unit test|implement|implementation|build|refactor|"
    r"script|python|javascript|typescript|sql|regex|api|endpoint|"
    r"deploy|deployment|docker|kubernetes|terraform|infra|infrastructure|"
    r"terminal|shell|command|git|repository|repo|database|migration"
    r")\b",
    re.IGNORECASE,
)
_BALANCED_PATTERN = re.compile(
    r"\b("
    r"explain|compare|comparison|summari[sz]e|rewrite|draft|"
    r"plan|design|analy[sz]e|analysis|recommend|recommendation|"
    r"brainstorm|review|pros and cons|difference"
    r")\b",
    re.IGNORECASE,
)
_URL_PATTERN = re.compile(r"https?://", re.IGNORECASE)


def _tier_entry(config: dict[str, Any], key: str) -> dict[str, Any]:
    value = config.get(key)
    return value if isinstance(value, dict) else {}


def _tier_model(entry: dict[str, Any], runtime: dict[str, Any]) -> str:
    """Return an eligible same-provider model ID, or an empty string."""
    model = entry.get("model")
    if not isinstance(model, str) or not model.strip():
        return ""
    provider = entry.get("provider")
    if provider and provider != runtime.get("provider"):
        return ""
    return model.strip()


def _positive_limit(config: dict[str, Any], key: str, default: int) -> int:
    try:
        value = int(config.get(key, default))
    except (TypeError, ValueError):
        return default
    return value if value > 0 else default


def _is_hard(message: str) -> bool:
    return bool(
        "\n" in message
        or "```" in message
        or _URL_PATTERN.search(message)
        or _HARD_PATTERN.search(message)
    )


def resolve_smart_model_route(
    message: str,
    *,
    model: str,
    runtime: dict[str, Any],
    config: dict[str, Any] | None,
    platform: str,
) -> dict[str, str]:
    """Choose an eligible configured model tier, otherwise retain ``model``.

    ``platforms`` is an allowlist.  A configured tier may optionally specify a
    provider, but that provider must exactly match the caller's already-resolved
    runtime; this helper never chooses credentials or a base URL.
    """
    primary = {"model": model, "label": "primary"}
    config = config if isinstance(config, dict) else {}
    if not config.get("enabled"):
        return primary

    platforms = config.get("platforms") or []
    if isinstance(platforms, str):
        platforms = [platforms]
    if platform not in platforms:
        return primary

    text = (message or "").strip()
    if not text or _is_hard(text):
        return primary

    words = len(text.split())
    max_simple_chars = _positive_limit(config, "max_simple_chars", 160)
    max_simple_words = _positive_limit(config, "max_simple_words", 28)
    max_balanced_chars = _positive_limit(config, "max_balanced_chars", 512)
    max_balanced_words = _positive_limit(config, "max_balanced_words", 96)

    wants_balanced = (
        bool(_BALANCED_PATTERN.search(text))
        or len(text) > max_simple_chars
        or words > max_simple_words
    )
    if wants_balanced:
        if len(text) > max_balanced_chars or words > max_balanced_words:
            return primary
        balanced = _tier_model(_tier_entry(config, "balanced_model"), runtime)
        return {"model": balanced, "label": "balanced"} if balanced else primary

    cheap = _tier_model(_tier_entry(config, "cheap_model"), runtime)
    return {"model": cheap, "label": "cheap"} if cheap else primary
