"""
utils/helpers.py
----------------
Shared, side-effect-free utility functions for the nft-analytics skill.

Imported by analyzer modules and the formatter to eliminate duplicated code.
No external dependencies — standard library only.

Public API
----------
ts_to_dt(ts)              Unix timestamp (int/float) → UTC datetime | None
months_between(a, b)      Whole months between two datetimes (int, >= 0)
shorten_address(addr)     "D1f3...9a7" style truncation for terminal display
roi_percent(buy, sell)    ROI as a float percentage
average(values)           Arithmetic mean; returns 0.0 for an empty list
clamp(value, lo, hi)      Clamp a number to the closed interval [lo, hi]
yes_no(value)             bool → "YES" | "NO"
roi_str(value)            float → "+12%" | "-3%"
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import List, Optional, Union


# ── Timestamp helpers ─────────────────────────────────────────────────────────

def ts_to_dt(ts: Optional[Union[int, float]]) -> Optional[datetime]:
    """
    Convert a Unix timestamp in seconds to a timezone-aware UTC datetime.

    Returns ``None`` if *ts* is falsy or falls outside the valid OS range.

    Examples
    --------
    >>> ts_to_dt(1700000000)
    datetime.datetime(2023, 11, 14, 22, 13, 20, tzinfo=datetime.timezone.utc)
    >>> ts_to_dt(None) is None
    True
    """
    if ts is None:
        return None
    try:
        return datetime.fromtimestamp(int(ts), tz=timezone.utc)
    except (ValueError, OSError, OverflowError):
        return None


def months_between(earlier: datetime, later: datetime) -> int:
    """
    Return the number of whole calendar months between two datetimes.

    Uses a 30-day month approximation.  Always returns a non-negative integer;
    returns 0 when *later* <= *earlier*.

    Examples
    --------
    >>> from datetime import datetime, timezone
    >>> months_between(datetime(2024,1,1,tzinfo=timezone.utc),
    ...                datetime(2024,7,20,tzinfo=timezone.utc))
    6
    """
    delta = later - earlier
    return max(0, int(delta.days / 30))


# ── Address helpers ───────────────────────────────────────────────────────────

def shorten_address(address: str, prefix: int = 4, suffix: int = 3) -> str:
    """
    Shorten a Solana base-58 address for terminal display.

    Returns the original string unchanged if it is already short enough.

    Examples
    --------
    >>> shorten_address("D1f3abc9a7AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA")
    'D1f3...AAA'
    >>> shorten_address("short")
    'short'
    """
    if not address:
        return ""
    if len(address) <= prefix + suffix + 3:  # 3 = len("...")
        return address
    return f"{address[:prefix]}...{address[-suffix:]}"


# ── Math helpers ──────────────────────────────────────────────────────────────

def roi_percent(buy_price: float, sell_price: float) -> float:
    """
    Compute return on investment as a percentage.

    ROI = (sell - buy) / buy × 100.
    Returns 0.0 when *buy_price* is zero to avoid division errors.

    Examples
    --------
    >>> round(roi_percent(1.0, 1.2), 4)
    20.0
    >>> roi_percent(2.0, 1.5)
    -25.0
    """
    if buy_price == 0:
        return 0.0
    return (sell_price - buy_price) / buy_price * 100.0


def average(values: List[float]) -> float:
    """
    Arithmetic mean of a list of floats.

    Returns 0.0 for an empty list instead of raising ``ZeroDivisionError``.

    Examples
    --------
    >>> average([10.0, 20.0, 30.0])
    20.0
    >>> average([])
    0.0
    """
    return sum(values) / len(values) if values else 0.0


def clamp(value: float, lo: float, hi: float) -> float:
    """
    Clamp *value* to the closed interval [*lo*, *hi*].

    Examples
    --------
    >>> clamp(5.0, 0.0, 3.0)
    3.0
    >>> clamp(-1.0, 0.0, 10.0)
    0.0
    """
    return max(lo, min(hi, value))


# ── Formatting helpers ────────────────────────────────────────────────────────

def yes_no(value: bool) -> str:
    """
    Convert a boolean to the strings ``"YES"`` or ``"NO"``.

    Examples
    --------
    >>> yes_no(True)
    'YES'
    >>> yes_no(False)
    'NO'
    """
    return "YES" if value else "NO"


def roi_str(value: float) -> str:
    """
    Format a ROI float as a sign-prefixed, zero-decimal percentage string.

    Examples
    --------
    >>> roi_str(12.0)
    '+12%'
    >>> roi_str(-5.7)
    '-6%'
    >>> roi_str(0.0)
    '+0%'
    """
    sign = "+" if value >= 0 else ""
    return f"{sign}{value:.0f}%"


# ── Standalone self-test ──────────────────────────────────────────────────────

if __name__ == "__main__":
    import datetime as _dt
    from datetime import timezone as _tz

    print("── utils/helpers.py self-test ──\n")

    dt = ts_to_dt(1700000000)
    assert dt is not None and dt.tzinfo is not None
    assert ts_to_dt(None) is None
    print(f"  ts_to_dt(1700000000)     = {dt}  ✓")

    a = _dt.datetime(2024, 1, 1, tzinfo=_tz.utc)
    b = _dt.datetime(2024, 7, 20, tzinfo=_tz.utc)
    assert months_between(a, b) == 6
    assert months_between(b, a) == 0
    print(f"  months_between(Jan, Jul) = {months_between(a, b)}  ✓")

    addr  = "D1f3abc9a7AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
    short = shorten_address(addr)
    assert short == "D1f3...AAA", repr(short)
    assert shorten_address("short") == "short"
    print(f"  shorten_address(long)    = {short!r}  ✓")

    assert round(roi_percent(1.0, 1.2), 6) == 20.0
    assert round(roi_percent(2.0, 1.5), 6) == -25.0
    assert roi_percent(0.0, 1.0) == 0.0
    print(f"  roi_percent(1.0, 1.2)    = {roi_percent(1.0, 1.2):.4f}  ✓")

    assert average([10.0, 20.0, 30.0]) == 20.0
    assert average([]) == 0.0
    print(f"  average([10,20,30])      = {average([10.0, 20.0, 30.0])}  ✓")

    assert clamp(5.0, 0.0, 3.0) == 3.0
    assert clamp(-1.0, 0.0, 10.0) == 0.0
    assert clamp(2.5, 0.0, 10.0) == 2.5
    print(f"  clamp(5.0, 0, 3)         = {clamp(5.0, 0.0, 3.0)}  ✓")

    assert yes_no(True) == "YES" and yes_no(False) == "NO"
    print(f"  yes_no(True/False)       = YES / NO  ✓")

    assert roi_str(12.0) == "+12%"
    assert roi_str(-5.7) == "-6%"
    assert roi_str(0.0)  == "+0%"
    print(f"  roi_str(12.0)            = {roi_str(12.0)!r}  ✓")

    print("\nAll helpers: PASS ✓")
