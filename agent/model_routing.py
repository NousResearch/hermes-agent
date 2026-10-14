"""Cheap-model routing for obvious, standalone low-complexity turns."""

from __future__ import annotations

import re
from typing import Iterable, Mapping, Any

_SIMPLE = re.compile(
    r"^\s*(?:hello|hi|hey|thanks?|thank you|define\b|what (?:is|are)\b|"
    r"explain\b|rewrite\b|fix (?:the )?grammar\b|summarize\b)",
    re.I,
)
_ACTION = re.compile(
    r"\b(deploy|install|configure|build|implement|debug|run|execute|delete|"
    r"send|email|schedule|book|cancel|commit|push|merge|publish|control|turn on|turn off)\b",
    re.I,
)


def select_turn_model(
    user_message: str,
    primary_model: str,
    config: Mapping[str, Any] | None,
    *,
    history: Iterable[dict] | None = None,
    enabled: bool = True,
) -> str:
    """Choose a cheap model only for a new, simple, non-action session."""
    if not enabled or not isinstance(config, Mapping):
        return primary_model
    cheap_model = str(config.get("model") or "").strip()
    if not cheap_model:
        return primary_model
    if history:
        return primary_model
    text = str(user_message or "").strip()
    try:
        max_chars = int(config.get("max_chars", 240))
    except (TypeError, ValueError):
        max_chars = 240
    if not text or len(text) > max(1, max_chars):
        return primary_model
    if _ACTION.search(text):
        return primary_model
    return cheap_model if _SIMPLE.search(text) else primary_model
