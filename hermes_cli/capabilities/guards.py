"""Buidl Agent Harness guard evaluators.

Guards are safe pure-Python checks. They do not install hooks or execute shell.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

GuardClassification = Literal["GREEN", "RED", "NOISE"]


@dataclass(frozen=True)
class GuardResult:
    allowed: bool
    classification: GuardClassification
    reasons: tuple[str, ...]


_BLOCK_PATTERNS: tuple[tuple[str, str], ...] = (
    ("niko-terminal", r"\b(niko|you)\b.{0,40}\b(run|execute|type)\b.{0,40}\b(terminal|command|shell|sudo|docker|ghcr)\b"),
    ("credentials", r"\b(send|give|provide|paste|edit|install)\b.{0,50}\b(credentials?|provider keys?|api keys?|github token|secret|env file)\b"),
    ("blind-prompt", r"\b(live blind prompt|known live prompt|blind prompt)\b.{0,60}\b(store|print|fixture|hardcode|prepare|save)?\b"),
    ("provider-call", r"\b(call|run|execute)\b.{0,40}\b(provider|anthropic|openai|claude|model api)\b.{0,40}\b(without approval)?\b"),
    ("image-generation", r"\b(generate|enable|run)\b.{0,40}\b(image generation|image gen|image|higgsfield|fal)\b"),
    ("production", r"\b(production deploy|deploy production|production)\b"),
    ("dns", r"\b(dns|domain records?)\b"),
    ("database", r"\b(db migration|database migration|run migrations?|drop tables?)\b"),
    ("money", r"\b(billing|credits?|payments?|payment method)\b"),
    ("caddy-hash", r"\b(caddy hash|basicauth hash|basic auth hash)\b"),
    ("env-value", r"\b(print|show|paste|log)\b.{0,40}\b(env value|\.env|environment value)\b"),
)

_NOISE_PATTERNS = ("harmless wording difference", "non-blocking", "noise")


def evaluate_buidl_guardrails(text: str) -> GuardResult:
    raw = str(text or "")
    lower = raw.lower()
    reasons = [name for name, pattern in _BLOCK_PATTERNS if re.search(pattern, lower, re.I | re.S)]
    if reasons:
        return GuardResult(False, "RED", tuple(reasons))
    if any(pattern in lower for pattern in _NOISE_PATTERNS):
        return GuardResult(True, "NOISE", ("not blocking",))
    return GuardResult(True, "GREEN", ())
