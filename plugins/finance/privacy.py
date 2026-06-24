"""Privacy formatting for finance tool output.

Two modes, selected by ``finance.privacy_mode`` in config.yaml:

* ``full`` — exact figures (e.g. ``$2,431.18``).
* ``summarized`` — values are bucketed into ranges (e.g. ``$2k-$5k``) *before*
  they ever enter a prompt or tool result, so the model can reason about
  magnitude and trends without ingesting exact balances.

Access tokens and account/routing numbers are never formatted here because
they are never loaded into tool output in the first place.
"""

from __future__ import annotations

from typing import Optional

FULL = "full"
SUMMARIZED = "summarized"

# (upper_bound_exclusive, label) for the absolute magnitude of a value.
_BUCKETS = [
    (100, "<{sym}100"),
    (500, "{sym}100-{sym}500"),
    (1_000, "{sym}500-{sym}1k"),
    (2_000, "{sym}1k-{sym}2k"),
    (5_000, "{sym}2k-{sym}5k"),
    (10_000, "{sym}5k-{sym}10k"),
    (25_000, "{sym}10k-{sym}25k"),
    (50_000, "{sym}25k-{sym}50k"),
    (100_000, "{sym}50k-{sym}100k"),
    (250_000, "{sym}100k-{sym}250k"),
    (500_000, "{sym}250k-{sym}500k"),
    (1_000_000, "{sym}500k-{sym}1M"),
]
_OVER_CAP = "{sym}1M+"


def normalize_mode(mode: Optional[str]) -> str:
    """Coerce a config value to a known privacy mode (defaults to ``full``)."""
    value = (mode or "").strip().lower()
    return SUMMARIZED if value in {SUMMARIZED, "summary", "bucketed", "redacted"} else FULL


def _symbol(currency: Optional[str]) -> str:
    code = (currency or "USD").upper()
    return "$" if code == "USD" else f"{code} "


def bucket_amount(value: Optional[float], currency: Optional[str] = None) -> str:
    """Return a coarse range label for *value* (sign preserved)."""
    if value is None:
        return "n/a"
    sym = _symbol(currency)
    magnitude = abs(float(value))
    sign = "-" if value < 0 else ""
    if magnitude == 0:
        return f"{sym}0"
    for upper, label in _BUCKETS:
        if magnitude < upper:
            return sign + label.format(sym=sym)
    return sign + _OVER_CAP.format(sym=sym)


def format_money(value: Optional[float], currency: Optional[str] = None, *, mode: str = FULL) -> str:
    """Format a monetary *value* per the active privacy *mode*."""
    if value is None:
        return "n/a"
    if normalize_mode(mode) == SUMMARIZED:
        return bucket_amount(value, currency)
    sym = _symbol(currency)
    return f"{sym}{float(value):,.2f}"


def is_summarized(mode: Optional[str]) -> bool:
    return normalize_mode(mode) == SUMMARIZED
