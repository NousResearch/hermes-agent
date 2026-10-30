"""MiniMax token-plan allowance: the 5-hour window and the weekly one.

``GET {origin}/v1/token_plan/remains`` reports how much of the current plan is left.
The wire is undocumented, so four observed properties drive this module and each is
easy to get backwards:

* the endpoint always answers HTTP 200 — a rejected key comes back as
  ``base_resp.status_code`` 2049 inside a JSON body — so the envelope decides success;
* the percentages are what REMAINS, while Hermes' account-usage contract counts what
  was SPENT, so they are inverted here;
* timestamps are epoch MILLISECONDS;
* the text/coding pool is the ``general`` bucket; the ``video`` bucket has no 5-hour
  window at all (MiniMax meters video on the weekly window alone).

The window NAMES come from MiniMax's published plan rules (text models: a 5-hour window
and a weekly window), not from the row's own bounds: ``start_time``/``end_time`` do not
reliably span the plan window (observed at both 5h and 4h for the same bucket), so a
label derived from them would name a window the plan does not have. Only the RESET
instant comes from the wire, and it agrees with ``remains_time``.

Everything in this module is provider-specific; the generic contract it fills
(``AccountUsageWindow``) lives in ``agent.account_usage``.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, Optional
from urllib.parse import urlparse

_QUOTA_PATH = "/v1/token_plan/remains"
_TIMEOUT_S = 8.0
# The text/coding pool — the one a coding session spends against.
_TEXT_FAMILY = "general"
# The video-generation pool, which MiniMax meters on the weekly window alone.
_VIDEO_FAMILY = "video"

# MiniMax's plan rules for text/image/audio models: a 5-hour window and a weekly
# window, each starting with the account's first use in that window.
_INTERVAL_LABEL = "5h limit"
_INTERVAL_SHORT = "5h"
_WEEKLY_LABEL = "weekly limit"
_WEEKLY_SHORT = "wk"


def quota_origin(base_url: str) -> str:
    """``https://api.minimax.io/anthropic`` → ``https://api.minimax.io``.

    The plan API answers at the API root, not under the Anthropic-compatible route the
    profile's base URL points at. Each profile keeps its own origin, so the China plan
    is read from the China host and the global plan from the global one.
    """
    parsed = urlparse(str(base_url or "").strip())
    if not parsed.scheme or not parsed.netloc:
        return ""
    return f"{parsed.scheme}://{parsed.netloc}"


def _epoch_ms(value: Any) -> Optional[int]:
    """Epoch milliseconds → int; None for anything that is not a positive number."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return int(value) if value > 0 else None


def _percent(value: Any) -> Optional[float]:
    """A remaining percentage in 0..100, clamped; None when the wire omits it."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return max(0.0, min(100.0, float(value)))


def _window(label: str, short_label: str, remaining: float, reset_ms: Optional[int]):
    """One usage window, or None when there is nothing to report.

    ``used_percent`` is what Hermes counts (spent); the MiniMax wire reports what is
    left, so it is inverted here and every surface downstream keeps one convention.
    """
    from agent.account_usage import AccountUsageWindow

    reset_at = datetime.fromtimestamp(reset_ms / 1000, tz=UTC) if reset_ms else None
    return AccountUsageWindow(label=label, short_label=short_label,
                              used_percent=max(0.0, min(100.0, 100.0 - remaining)),
                              reset_at=reset_at)


def _text_row(rows: list[dict]) -> Optional[dict]:
    """The bucket a coding session spends against: ``general``, else the first that is
    not the video pool. None when only video is present (video has no 5-hour window)."""
    for row in rows:
        if row.get("model_name") == _TEXT_FAMILY:
            return row
    for row in rows:
        if row.get("model_name") != _VIDEO_FAMILY:
            return row
    return None


def parse_quota(payload: Any) -> list:
    """The plan's usage windows from a token-plan payload; [] when it carries none.

    The 5-hour window comes from the text pool and the weekly window from the same
    row (the two windows reset independently, so they are two readings of one plan).
    """
    if not isinstance(payload, dict):
        return []
    envelope = payload.get("base_resp")
    if isinstance(envelope, dict) and envelope.get("status_code") not in (0, None):
        return []  # the endpoint answers 200 even for a refused key
    rows = [row for row in (payload.get("model_remains") or []) if isinstance(row, dict)]
    if not rows:
        return []

    windows = []
    row = _text_row(rows)
    if row is not None:
        interval = _percent(row.get("current_interval_remaining_percent"))
        if interval is not None:
            windows.append(_window(_INTERVAL_LABEL, _INTERVAL_SHORT, interval,
                                   _epoch_ms(row.get("end_time"))))

    weekly_row = row if row is not None else rows[0]
    weekly = _percent(weekly_row.get("current_weekly_remaining_percent"))
    if weekly is not None:
        windows.append(_window(_WEEKLY_LABEL, _WEEKLY_SHORT, weekly,
                               _epoch_ms(weekly_row.get("weekly_end_time"))))
    return windows


def fetch_quota(api_key: str, origin: str) -> list:
    """The plan's usage windows from MiniMax; [] when the plan cannot be read.

    Fail-open by design: an unreachable or refused plan API leaves the session without
    an allowance readout rather than interrupting it.
    """
    token = str(api_key or "").strip()
    if not token or not origin:
        return []
    import httpx

    url = origin.rstrip("/") + _QUOTA_PATH
    try:
        with httpx.Client(timeout=_TIMEOUT_S) as client:
            response = client.get(url, headers={"Authorization": f"Bearer {token}",
                                                "Accept": "application/json"})
            response.raise_for_status()
            payload = response.json()
    except (httpx.HTTPError, ValueError, OSError):
        return []
    return parse_quota(payload)
