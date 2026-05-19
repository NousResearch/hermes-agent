from __future__ import annotations

from typing import Any


def _as_float(value: Any) -> float | None:
    if value is None or value == "" or value == "--":
        return None
    try:
        return float(str(value).replace(",", ""))
    except (TypeError, ValueError):
        return None


def _grade(*, option_markets: int, crossed_markets: int, missing_bid_ask: int, wide_markets: int = 0, missing_iv: int = 0) -> str:
    if option_markets <= 0:
        return "red"
    anomaly_count = crossed_markets + missing_bid_ask + wide_markets + missing_iv
    anomaly_ratio = anomaly_count / option_markets
    if crossed_markets > 5 or anomaly_ratio > 0.75:
        return "red"
    if anomaly_count > 0:
        return "yellow"
    return "green"


def _plural(count: int, singular: str, plural: str | None = None) -> str:
    return f"{count} {singular if count == 1 else (plural or singular + 's')}"


def diagnose_nasdaq_ibit_chain(chain: dict[str, Any], *, wide_pct_threshold: float = 1.00) -> dict[str, Any]:
    option_markets = 0
    valid_bid_ask = 0
    missing_bid_ask = 0
    crossed_markets = 0
    wide_markets = 0

    for row in chain.get("table", {}).get("rows", []):
        if row.get("expirygroup") or not row.get("strike"):
            continue
        for prefix in ("c", "p"):
            option_markets += 1
            bid = _as_float(row.get(f"{prefix}_Bid"))
            ask = _as_float(row.get(f"{prefix}_Ask"))
            if bid is None or ask is None:
                missing_bid_ask += 1
                continue
            valid_bid_ask += 1
            if ask < bid:
                crossed_markets += 1
                continue
            mid = (bid + ask) / 2
            if mid > 0 and (ask - bid) / mid > wide_pct_threshold:
                wide_markets += 1

    notes: list[str] = []
    if crossed_markets:
        notes.append(_plural(crossed_markets, "crossed market"))
    if wide_markets:
        notes.append(_plural(wide_markets, "wide market"))
    if missing_bid_ask:
        notes.append(_plural(missing_bid_ask, "missing bid/ask", "missing bid/ask"))
    if not notes:
        notes.append("No bid/ask anomalies detected")

    return {
        "source": "IBIT Nasdaq option-chain",
        "option_markets": option_markets,
        "valid_bid_ask": valid_bid_ask,
        "missing_bid_ask": missing_bid_ask,
        "crossed_markets": crossed_markets,
        "wide_markets": wide_markets,
        "missing_iv": 0,
        "grade": _grade(
            option_markets=option_markets,
            crossed_markets=crossed_markets,
            missing_bid_ask=missing_bid_ask,
            wide_markets=wide_markets,
        ),
        "notes": notes,
    }


def diagnose_deribit_options(rows: list[dict[str, Any]]) -> dict[str, Any]:
    option_markets = len(rows)
    valid_bid_ask = 0
    missing_bid_ask = 0
    crossed_markets = 0
    missing_iv = 0

    for row in rows:
        mark_iv = _as_float(row.get("mark_iv"))
        if mark_iv is None or mark_iv <= 0:
            missing_iv += 1
        bid = _as_float(row.get("bid_price"))
        ask = _as_float(row.get("ask_price"))
        if bid is None or ask is None:
            missing_bid_ask += 1
            continue
        valid_bid_ask += 1
        if ask < bid:
            crossed_markets += 1

    notes: list[str] = []
    if crossed_markets:
        notes.append(_plural(crossed_markets, "crossed market"))
    if missing_bid_ask:
        notes.append(_plural(missing_bid_ask, "missing bid/ask", "missing bid/ask"))
    if missing_iv:
        notes.append(_plural(missing_iv, "missing IV", "missing IV"))
    if not notes:
        notes.append("No bid/ask or IV anomalies detected")

    return {
        "source": "Deribit BTC options",
        "option_markets": option_markets,
        "valid_bid_ask": valid_bid_ask,
        "missing_bid_ask": missing_bid_ask,
        "crossed_markets": crossed_markets,
        "wide_markets": 0,
        "missing_iv": missing_iv,
        "grade": _grade(
            option_markets=option_markets,
            crossed_markets=crossed_markets,
            missing_bid_ask=missing_bid_ask,
            missing_iv=missing_iv,
        ),
        "notes": notes,
    }
