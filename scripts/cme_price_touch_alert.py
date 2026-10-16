"""Emit one Telegram-ready alert when CME trades at or below $270 in RTH."""

from __future__ import annotations

import json
import math
import os
import sys
import tempfile
from datetime import datetime, time, timedelta, timezone
from pathlib import Path
from urllib.error import URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen
from zoneinfo import ZoneInfo


NY = ZoneInfo("America/New_York")
PRICE_LIMIT = 270.0
CHART_URL = "https://query1.finance.yahoo.com/v8/finance/chart/CME?range=1d&interval=1m&includePrePost=false"
SOURCE_PAGE = "https://finance.yahoo.com/quote/CME/"
_TIMEOUT = 15
_MAX_BYTES = 1_000_000
_CLOSE_GRACE = timedelta(seconds=90)
_MAX_BAR_AGE = timedelta(minutes=20)


def _state_path() -> Path:
    home = Path(os.environ.get("HERMES_HOME", Path.home() / ".hermes"))
    return home / "state" / "cme-price-touch.json"


def _load_state(path: Path) -> dict:
    if not path.exists():
        return {"schema_version": 1, "session_date": None, "last_bar": None, "alerted_date": None}
    state = json.loads(path.read_text(encoding="utf-8"))
    if (
        not isinstance(state, dict)
        or state.get("schema_version") != 1
        or (state.get("last_bar") is not None and not isinstance(state["last_bar"], int))
        or (state.get("session_end") is not None and not isinstance(state["session_end"], int))
    ):
        raise ValueError(f"invalid CME alert state: {path}")
    return state


def _save_state(path: Path, state: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as tmp:
        json.dump(state, tmp, sort_keys=True)
        tmp.write("\n")
        tmp_path = Path(tmp.name)
    os.replace(tmp_path, path)


def _fetch_chart() -> dict:
    request = Request(CHART_URL, headers={"User-Agent": "Hermes-CME-Price-Alert/1.0"})
    with urlopen(request, timeout=_TIMEOUT) as response:
        if urlparse(response.geturl()).hostname != "query1.finance.yahoo.com":
            raise ValueError("Yahoo Finance chart request redirected to an unexpected host")
        body = response.read(_MAX_BYTES + 1)
    if len(body) > _MAX_BYTES:
        raise ValueError("Yahoo Finance chart response exceeded the size limit")
    payload = json.loads(body)
    chart = payload.get("chart", {})
    if chart.get("error"):
        raise ValueError(f"Yahoo Finance chart error: {chart['error']}")
    result = (chart.get("result") or [None])[0]
    if not isinstance(result, dict):
        raise ValueError("Yahoo Finance returned no CME chart")
    meta = result.get("meta", {})
    if (
        meta.get("symbol") != "CME"
        or meta.get("currency") != "USD"
        or meta.get("exchangeTimezoneName") != "America/New_York"
    ):
        raise ValueError("Yahoo Finance returned unexpected CME identity, currency, or timezone")
    return result


def _regular_session(result: dict) -> tuple[datetime, datetime] | None:
    period = result.get("meta", {}).get("currentTradingPeriod", {}).get("regular", {})
    start, end = period.get("start"), period.get("end")
    if not isinstance(start, int) or not isinstance(end, int) or end <= start:
        raise ValueError("Yahoo Finance returned invalid regular-session boundaries")
    return datetime.fromtimestamp(start, NY), datetime.fromtimestamp(end, NY)


def _closed_bars(result: dict, start: datetime, end: datetime, now: datetime) -> list[tuple[int, float]]:
    timestamps = result.get("timestamp") or []
    quotes = result.get("indicators", {}).get("quote") or []
    if not quotes or not isinstance(quotes[0].get("low"), list):
        raise ValueError("Yahoo Finance did not return regular-session interval lows")
    lows = quotes[0]["low"]
    if len(timestamps) != len(lows):
        raise ValueError("Yahoo Finance timestamps and interval lows have different lengths")

    now_ts = now.timestamp()
    bars = []
    for stamp, raw_low in zip(timestamps, lows):
        if not isinstance(stamp, int) or not (start.timestamp() <= stamp < end.timestamp()):
            continue
        if stamp + 60 > now_ts or raw_low is None:
            continue
        low = float(raw_low)
        if not math.isfinite(low) or low <= 0:
            raise ValueError("Yahoo Finance returned an invalid interval low")
        bars.append((stamp, low))

    if bars and now.timestamp() - (max(stamp for stamp, _ in bars) + 60) > _MAX_BAR_AGE.total_seconds():
        raise ValueError("Yahoo Finance regular-session bars are more than 20 minutes stale")
    if not bars and now - start > _MAX_BAR_AGE:
        raise ValueError("Yahoo Finance returned no fresh regular-session bars")
    return bars


def _alert(price: float, bar_start: datetime) -> str:
    local_start = datetime.fromtimestamp(bar_start.timestamp(), NY)
    local_end = local_start + timedelta(minutes=1)
    return (
        "CME (NASDAQ:CME) touched US$270 or below.\n"
        f"Regular-session 1-minute bar low: US${price:.2f}.\n"
        f"Bar time: {local_start:%Y-%m-%d %H:%M}–{local_end:%H:%M} America/New_York.\n"
        f"Source: Yahoo Finance Chart API — {SOURCE_PAGE}"
    )


def run(now: datetime | None = None) -> str | None:
    now = (now or datetime.now(NY)).astimezone(NY)
    state_path = _state_path()
    state = _load_state(state_path)
    if not state_path.exists():
        _save_state(state_path, state)

    # The ticker uses a quarter-hour cron in Hong Kong; no price request is made outside NYSE/Nasdaq hours.
    if now.weekday() >= 5 or not (time(9, 30) <= now.time() <= time(16, 0)):
        return None
    today = now.date().isoformat()
    if (
        state.get("session_date") == today
        and isinstance(state.get("session_end"), int)
        and now.timestamp() > state["session_end"] + _CLOSE_GRACE.total_seconds()
    ):
        return None

    result = _fetch_chart()
    session = _regular_session(result)
    if session is None:
        return None
    start, end = session
    if start.date() != now.date() or start.time() != time(9, 30):
        return None  # Holiday, closed day, or source has not published today's session yet.
    if now < start or now > end + _CLOSE_GRACE:
        return None  # Yahoo's session metadata supplies the early-close boundary.

    bars = _closed_bars(result, start, end, now)
    cursor = state.get("last_bar") if state.get("session_date") == start.date().isoformat() else None
    fresh = [(stamp, low) for stamp, low in bars if cursor is None or stamp > cursor]
    latest = max((stamp for stamp, _ in bars), default=cursor)
    state.update({
        "session_date": today,
        "session_end": int(end.timestamp()),
        "last_bar": latest,
    })

    message = None
    if state.get("alerted_date") != start.date().isoformat():
        touched = [(stamp, low) for stamp, low in fresh if low <= PRICE_LIMIT]
        if touched:
            stamp, low = min(touched, key=lambda item: item[1])
            state["alerted_date"] = start.date().isoformat()
            message = _alert(low, datetime.fromtimestamp(stamp, NY))
    _save_state(state_path, state)
    return message


def main() -> int:
    try:
        message = run()
    except (OSError, URLError, json.JSONDecodeError, ValueError, KeyError) as exc:
        print(f"CME price monitor failed closed: {exc}", file=sys.stderr)
        return 1
    if message:
        print(message)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
