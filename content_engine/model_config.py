"""Model roles for the image pipeline (P1).

Single source of truth for which FAL model plays which role, their cost, and
which need the async queue (slow) vs sync. gpt-image-2 is intentionally absent:
it times out on every call from this box (verified 2026-06-08).

Track A (illustrated) uses these. Track B (infographic) is code-rendered and
never touches a paid model.
"""
from __future__ import annotations

import os

# role -> model key (overridable via env for tuning without code change)
BASE_MODEL = os.getenv("CONTENT_BASE_MODEL", "krea_medium").strip()   # default illustrated
HERO_MODEL = os.getenv("CONTENT_HERO_MODEL", "nano_banana").strip()   # budget-gated hero
# degrade order if BASE fails (cheap, reliable models only)
DEGRADE_CHAIN = ["krea_medium", "flux_klein", "z_image"]

# Models that are slow enough to need the async queue (never sync).
SLOW_MODELS = {"nano_banana", "krea_large", "flux_pro", "ideogram"}

# Any spend above this per-image requires a budget check before running.
HERO_COST_THRESHOLD_GBP = 0.05


def degrade_from(primary: str) -> list[str]:
    """Ordered model attempts: primary first, then the reliable degrade chain."""
    return [primary] + [m for m in DEGRADE_CHAIN if m != primary]
