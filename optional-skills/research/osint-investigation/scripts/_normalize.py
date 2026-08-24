"""Shared entity-name normalization helpers (stdlib-only).

Used by entity_resolution.py and timing_analysis.py.
"""
from __future__ import annotations

import re

# Legal suffixes / corporate boilerplate to strip during normalization.
_SUFFIX_TOKENS = {
    "INC", "INCORPORATED", "LLC", "LLP", "LP", "LTD", "LIMITED",
    "CORP", "CORPORATION", "CO", "COMPANY",
    "GROUP", "GRP", "HOLDINGS", "HOLDING",
    "PARTNERS", "ASSOCIATES",
    "INTERNATIONAL", "INTL",
    "ENTERPRISES", "ENTERPRISE",
    "SERVICES", "SERVICE", "SVCS",
    "SOLUTIONS", "MANAGEMENT", "MGMT", "CONSULTING",
    "TECHNOLOGY", "TECHNOLOGIES", "TECH",
    "INDUSTRIES", "INDUSTRY",
    "AMERICA", "AMERICAN",
    "USA", "US",
    "PLLC", "PC",
    "TRUST", "FOUNDATION",
}

_PUNCT_RE = re.compile(r"[^\w\s]")
_WS_RE = re.compile(r"\s+")


def normalize_name(name: str | None) -> str:
    """Standard normalization: uppercase, strip suffixes, drop punctuation."""
    if not name:
        return ""
    s = _PUNCT_RE.sub(" ", name.upper())
    s = _WS_RE.sub(" ", s).strip()
    tokens = [t for t in s.split() if t and t not in _SUFFIX_TOKENS]
    return " ".join(tokens)


def normalize_aggressive(name: str | None) -> str:
    """Aggressive normalization: sorted unique tokens (word-bag)."""
    base = normalize_name(name)
    if not base:
        return ""
    return " ".join(sorted(set(base.split())))


def name_tokens(name: str | None, min_len: int = 4) -> set[str]:
    """Token set used for overlap matching."""
    base = normalize_name(name)
    if not base:
        return set()
    return {t for t in base.split() if len(t) >= min_len}


def token_overlap_ratio(left: str | None, right: str | None) -> tuple[float, int]:
    """Return (jaccard-like ratio, shared token count) over min-len tokens."""
    a = name_tokens(left)
    b = name_tokens(right)
    if not a or not b:
        return 0.0, 0
    shared = a & b
    if not shared:
        return 0.0, 0
    union = a | b
    return len(shared) / len(union), len(shared)
