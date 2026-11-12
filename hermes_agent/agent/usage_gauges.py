"""Usage-window allowances as the server sends them: one list, in display order.

The Portal account (``usage_window.gauges``) and the API (``x-nous-credits-window-gauges``, also on the 402)
send the same shape, so Hermes draws whatever arrives and a new allowance needs no Hermes release.
``scope`` says which models a full gauge stops: ``all`` = every paid model, ``top_tier`` = the models the
gateway marks top tier. Free models draw on nothing.

Hermes keeps only its own words for the ids it knows (they are translated and fitted to each surface);
any other id shows the server's ``label``.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any, Iterable, Optional

BP_FULL = 10_000
SCOPE_ALL = "all"
SCOPE_TOP_TIER = "top_tier"
_SCOPES = frozenset({SCOPE_ALL, SCOPE_TOP_TIER})
# The pause reasons that mean the subscription payment failed. Any other reason (including "unknown",
# sent for a malformed one) is still a pause, with copy that does not claim a payment failed.
PAYMENT_PAUSE_REASONS = frozenset({"payment_past_due"})


def paused_for_payment(reason: Any) -> bool:
    return reason in PAYMENT_PAUSE_REASONS


@dataclass(frozen=True)
class WindowGauge:
    id: str
    scope: str            # SCOPE_ALL | SCOPE_TOP_TIER
    label: str            # the server's display fallback
    used_bp: int          # 0..10000
    resets_at_ms: Optional[int] = None

    @property
    def full(self) -> bool:
        return self.used_bp >= BP_FULL

    @property
    def used_fraction(self) -> float:
        return min(1.0, max(0.0, self.used_bp / BP_FULL))


def parse_gauges(value: Any) -> Optional[tuple[WindowGauge, ...]]:
    """The wire list (a JSON array, or its string form from a header) → gauges in order. None when absent,
    unreadable or empty. An entry with an unknown scope, a repeated id or a bad percent is dropped: an
    older Hermes must never misread a kind of limit it does not know."""
    if isinstance(value, tuple) and all(isinstance(g, WindowGauge) for g in value):
        return value or None
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except ValueError:
            return None
    if not isinstance(value, list):
        return None
    out: list[WindowGauge] = []
    seen: set[str] = set()
    for item in value:
        if not isinstance(item, dict):
            continue
        gid, scope, bp = item.get("id"), item.get("scope"), item.get("used_bp")
        if not isinstance(gid, str) or not gid or gid in seen or scope not in _SCOPES:
            continue
        if isinstance(bp, bool) or not isinstance(bp, int) or not 0 <= bp <= BP_FULL:
            continue
        reset = _plausible_reset_ms(item.get("resets_at_ms"))
        label = item.get("label")
        seen.add(gid)
        out.append(WindowGauge(gid, scope, label.strip() if isinstance(label, str) and label.strip() else gid,
                               bp, reset))
    return tuple(out) or None


# A reset more than a year ahead is a bad stamp (the longest allowance is a month): shown as unknown, never
# as a date or a duration decades away. A past stamp needs no bound: it already reads "shortly".
_RESET_MAX_AHEAD_MS = 366 * 86_400_000


def _plausible_reset_ms(value: Any) -> Optional[int]:
    if not isinstance(value, int) or isinstance(value, bool):
        return None
    return value if 0 < value <= time.time() * 1000 + _RESET_MAX_AHEAD_MS else None


def all_scope(gauges: Iterable[WindowGauge]) -> tuple[WindowGauge, ...]:
    return tuple(g for g in gauges if g.scope == SCOPE_ALL)


def top_tier_gauge(gauges: Iterable[WindowGauge]) -> Optional[WindowGauge]:
    return next((g for g in gauges if g.scope == SCOPE_TOP_TIER), None)


def binding_gauge(gauges: Iterable[WindowGauge]) -> Optional[WindowGauge]:
    """The all-scope gauge that binds: of the full ones, the one that frees up LAST (with the day and the
    week both full, the day's reset alone lets nothing through); else the fullest (ties → first listed)."""
    alls = all_scope(gauges)
    if not alls:
        return None
    full = [g for g in alls if g.full]
    if full:
        return max(full, key=lambda g: g.resets_at_ms or 0)
    return max(alls, key=lambda g: g.used_bp)


# Hermes's words for the ids it knows: (row title, "for <phrase>", "your <name> limit", picker short name).
_WORDS: dict[str, tuple[str, str, str, str]] = {
    "day": ("Today", "today", "daily", "today"),
    "week": ("This week", "this week", "weekly", "week"),
    "month": ("This month", "this month", "monthly", "month"),
    "top_tier": ("Large models (monthly)", "your monthly large-model usage", "monthly large-model", "large models (monthly)"),
}


def _words(g: WindowGauge) -> Optional[tuple[str, str, str, str]]:
    return _WORDS.get(g.id)


def gauge_title(g: WindowGauge) -> str:
    """Row title in ``/usage``: ``Today``; an unknown id shows the server's label."""
    return w[0] if (w := _words(g)) else g.label


def gauge_limit_name(g: WindowGauge) -> str:
    """``You've hit your <name> limit``: ``daily``; an unknown id shows the server's label."""
    return w[2] if (w := _words(g)) else g.label


def gauge_short(g: WindowGauge) -> str:
    """Picker summary row name: ``today``; an unknown id shows the server's label."""
    return w[3] if (w := _words(g)) else g.label
