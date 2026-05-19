from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from datetime import date, datetime
from io import StringIO
from typing import Any

_DERIBIT_MONTHS = {
    "JAN": 1,
    "FEB": 2,
    "MAR": 3,
    "APR": 4,
    "MAY": 5,
    "JUN": 6,
    "JUL": 7,
    "AUG": 8,
    "SEP": 9,
    "OCT": 10,
    "NOV": 11,
    "DEC": 12,
}


@dataclass(frozen=True)
class BtcShareResult:
    btc_held: float
    shares_outstanding: float
    btc_per_share: float
    shares_per_btc: float
    raw_btc_row_label: str


@dataclass(frozen=True)
class NormalizedOptionRow:
    native_symbol: str
    expiry: date
    option_type: str
    strike_native: float
    iv_mark: float | None
    price_bid: float | None
    price_ask: float | None
    spot_native: float
    moneyness_spot: float
    open_interest: float | None
    volume: float | None
    source_confidence: str
    execution_confidence: str
    timestamp_source: str
    quality_flags: list[str]
    contract_multiplier: float | None = None
    btc_equivalent_per_contract: float | None = None
    notional_btc: float | None = None
    strike_btc_equivalent: float | None = None


def _parse_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip().replace(",", "").replace("$", "")
    if not text or text in {"-", "—", "N/A", "n/a"}:
        return None
    try:
        if text.endswith("%"):
            return float(text[:-1]) / 100.0
        return float(text)
    except ValueError:
        return None


def compute_btc_per_share(csv_text: str) -> BtcShareResult:
    parsed_rows = list(csv.reader(StringIO(csv_text.lstrip("\ufeff"))))
    btc_held = None
    btc_label = ""
    shares_outstanding = None

    for raw_row in parsed_rows:
        row_text = " ".join(raw_row).upper()
        if "SHARES OUTSTANDING" in row_text:
            for value in raw_row:
                parsed = _parse_float(value)
                if parsed is not None:
                    shares_outstanding = parsed

    header_index = None
    for idx, raw_row in enumerate(parsed_rows):
        normalized = [cell.strip().lower() for cell in raw_row]
        if "ticker" in normalized and "name" in normalized:
            header_index = idx
            break
    if header_index is not None:
        header = parsed_rows[header_index]
        rows = [dict(zip(header, raw_row)) for raw_row in parsed_rows[header_index + 1 :] if len(raw_row) >= 2]
    else:
        rows = list(csv.DictReader(StringIO(csv_text.lstrip("\ufeff"))))

    for row in rows:
        label = " ".join(str(row.get(key, "")) for key in ("Ticker", "Name", "Asset Class", "Holding Ticker", "Market Currency")).upper()
        if btc_held is None and ("BTC" in label or "BITCOIN" in label):
            for field in ("Shares", "Quantity", "Units", "Notional", "Amount"):
                parsed = _parse_float(row.get(field))
                if parsed is not None:
                    btc_held = parsed
                    btc_label = label.strip()
                    break

    if btc_held is None:
        raise ValueError("Could not find BTC/Bitcoin holdings row")
    if shares_outstanding is None:
        raise ValueError("Could not find shares outstanding row")
    if btc_held <= 0 or shares_outstanding <= 0:
        raise ValueError("BTC held and shares outstanding must be positive")

    btc_per_share = btc_held / shares_outstanding
    return BtcShareResult(
        btc_held=btc_held,
        shares_outstanding=shares_outstanding,
        btc_per_share=btc_per_share,
        shares_per_btc=1.0 / btc_per_share,
        raw_btc_row_label=btc_label,
    )


def _parse_deribit_expiry(token: str) -> date:
    day_digits = ""
    idx = 0
    while idx < len(token) and token[idx].isdigit():
        day_digits += token[idx]
        idx += 1
    if not day_digits:
        raise ValueError(f"Deribit expiry missing day: {token}")
    month_token = token[idx : idx + 3].upper()
    year_token = token[idx + 3 :]
    day = int(day_digits)
    month = _DERIBIT_MONTHS[month_token]
    year = 2000 + int(year_token)
    return date(year, month, day)


def normalize_deribit_instrument(instrument: dict[str, Any], as_of: str) -> NormalizedOptionRow:
    symbol = str(instrument["instrument_name"])
    parts = symbol.split("-")
    if len(parts) != 4 or parts[0] != "BTC":
        raise ValueError(f"Unsupported Deribit BTC option symbol: {symbol}")
    expiry = _parse_deribit_expiry(parts[1])
    strike = float(parts[2])
    option_type = {"C": "call", "P": "put"}[parts[3].upper()]
    spot = _parse_float(instrument.get("underlying_price"))
    if spot is None or spot <= 0:
        raise ValueError("Deribit instrument missing positive underlying_price")

    mark_iv = _parse_float(instrument.get("mark_iv"))
    if mark_iv is not None and mark_iv > 2.0:
        mark_iv = mark_iv / 100.0

    price_bid = _parse_float(instrument.get("bid_price"))
    price_ask = _parse_float(instrument.get("ask_price"))
    row_dict = {
        "price_bid": price_bid,
        "price_ask": price_ask,
        "iv_mark": mark_iv,
        "dte": max((expiry - datetime.fromisoformat(as_of).date()).days, 0),
        "timestamp_source": as_of,
    }
    flags = quality_check_option_row(row_dict, as_of=as_of)
    if not flags:
        flags = ["ok"]

    return NormalizedOptionRow(
        native_symbol=symbol,
        expiry=expiry,
        option_type=option_type,
        strike_native=strike,
        iv_mark=mark_iv,
        price_bid=price_bid,
        price_ask=price_ask,
        spot_native=spot,
        moneyness_spot=strike / spot,
        open_interest=_parse_float(instrument.get("open_interest")),
        volume=_parse_float(instrument.get("volume")),
        source_confidence="official_exchange_api",
        execution_confidence="screen_only",
        timestamp_source=as_of,
        quality_flags=flags,
    )


def estimate_black_scholes_iv(
    *,
    option_type: str,
    spot: float,
    strike: float,
    years_to_expiry: float,
    price: float,
    rate: float = 0.045,
) -> float | None:
    if spot <= 0 or strike <= 0 or years_to_expiry <= 0 or price <= 0:
        return None

    def norm_cdf(x: float) -> float:
        return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

    def bs_price(vol: float) -> float:
        if vol <= 0:
            intrinsic = max(0.0, spot - strike) if option_type == "call" else max(0.0, strike - spot)
            return intrinsic
        d1 = (math.log(spot / strike) + (rate + 0.5 * vol * vol) * years_to_expiry) / (vol * math.sqrt(years_to_expiry))
        d2 = d1 - vol * math.sqrt(years_to_expiry)
        discounted_strike = strike * math.exp(-rate * years_to_expiry)
        if option_type == "call":
            return spot * norm_cdf(d1) - discounted_strike * norm_cdf(d2)
        return discounted_strike * norm_cdf(-d2) - spot * norm_cdf(-d1)

    low, high = 0.0001, 5.0
    if price < bs_price(low) - 1e-9 or price > bs_price(high) + 1e-9:
        return None
    for _ in range(80):
        mid = (low + high) / 2.0
        if bs_price(mid) < price:
            low = mid
        else:
            high = mid
    return (low + high) / 2.0


def normalize_ibit_option(option: dict[str, Any], btc_per_share: float, as_of: str) -> NormalizedOptionRow:
    expiry = datetime.fromisoformat(str(option["expiry"])).date()
    strike = _parse_float(option.get("strike"))
    underlying = _parse_float(option.get("underlying_price"))
    if strike is None or strike <= 0:
        raise ValueError("IBIT option missing positive strike")
    if underlying is None or underlying <= 0:
        raise ValueError("IBIT option missing positive underlying_price")
    option_type_raw = str(option.get("type") or option.get("option_type") or "").lower()
    if option_type_raw in {"c", "call"}:
        option_type = "call"
        suffix = "C"
    elif option_type_raw in {"p", "put"}:
        option_type = "put"
        suffix = "P"
    else:
        raise ValueError(f"Unsupported IBIT option type: {option_type_raw}")
    iv_mark = _parse_float(option.get("iv") or option.get("implied_volatility"))
    price_bid = _parse_float(option.get("bid"))
    price_ask = _parse_float(option.get("ask"))
    timestamp = str(option.get("timestamp") or as_of)
    row_dict = {
        "price_bid": price_bid,
        "price_ask": price_ask,
        "iv_mark": iv_mark,
        "dte": max((expiry - datetime.fromisoformat(as_of).date()).days, 0),
        "timestamp_source": timestamp,
    }
    flags = quality_check_option_row(row_dict, as_of=as_of)
    if not flags:
        flags = ["ok"]
    multiplier = 100.0
    btc_equiv = multiplier * btc_per_share
    return NormalizedOptionRow(
        native_symbol=f"IBIT-{expiry.isoformat()}-{strike}-{suffix}",
        expiry=expiry,
        option_type=option_type,
        strike_native=strike,
        iv_mark=iv_mark,
        price_bid=price_bid,
        price_ask=price_ask,
        spot_native=underlying,
        moneyness_spot=strike / underlying,
        open_interest=_parse_float(option.get("open_interest") or option.get("openInterest")),
        volume=_parse_float(option.get("volume")),
        source_confidence="semi_official_public_json",
        execution_confidence="screen_only",
        timestamp_source=timestamp,
        quality_flags=flags,
        contract_multiplier=multiplier,
        btc_equivalent_per_contract=btc_equiv,
        notional_btc=btc_equiv,
        strike_btc_equivalent=strike / btc_per_share,
    )


def quality_check_option_row(row: dict[str, Any], as_of: str, stale_minutes: int = 60) -> list[str]:
    flags: list[str] = []
    bid = _parse_float(row.get("price_bid"))
    ask = _parse_float(row.get("price_ask"))
    iv_mark = _parse_float(row.get("iv_mark"))
    dte = _parse_float(row.get("dte"))

    if bid is not None and ask is not None and bid > ask:
        flags.append("crossed_market")
    if bid is not None and bid < 0 or ask is not None and ask < 0:
        flags.append("negative_price")
    if iv_mark is not None and not (0.001 <= iv_mark <= 5.0):
        flags.append("implausible_iv")
    if dte is not None and dte <= 0:
        flags.append("expired_or_zero_dte")

    timestamp = row.get("timestamp_source")
    if timestamp:
        age = datetime.fromisoformat(as_of) - datetime.fromisoformat(str(timestamp))
        if age.total_seconds() > stale_minutes * 60:
            flags.append("stale_source")
    return flags


def select_atm_by_moneyness(rows: list[dict[str, Any]]) -> dict[str, Any]:
    liquid = []
    for row in rows:
        bid = _parse_float(row.get("price_bid"))
        ask = _parse_float(row.get("price_ask"))
        oi = _parse_float(row.get("open_interest")) or 0.0
        moneyness = _parse_float(row.get("moneyness_spot"))
        if moneyness is None:
            continue
        if bid is not None and ask is not None and bid > 0 and ask >= bid and oi > 0:
            liquid.append(row)
    if not liquid:
        raise ValueError("No liquid option rows available for ATM selection")
    return min(liquid, key=lambda row: abs(float(row["moneyness_spot"]) - 1.0))


def _format_atm_table(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return "_No rows available._"
    lines = ["| Expiry | DTE | Symbol | IV mark |", "|---|---:|---|---:|"]
    for row in rows:
        iv = row.get("iv_mark")
        iv_text = "n/a" if iv is None else f"{float(iv) * 100:.2f}%"
        lines.append(f"| {row.get('expiry')} | {row.get('dte')} | {row.get('native_symbol', '')} | {iv_text} |")
    return "\n".join(lines)


def _format_cme_table(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return "_CME source missing or no rows available._"
    lines = ["| Expiry | DTE | Symbol | Bid | Ask | Confidence |", "|---|---:|---|---:|---:|---|"]
    for row in rows[:20]:
        bid = row.get("price_bid")
        ask = row.get("price_ask")
        bid_text = "n/a" if bid is None else f"{float(bid):,.2f}"
        ask_text = "n/a" if ask is None else f"{float(ask):,.2f}"
        lines.append(
            f"| {row.get('expiry')} | {row.get('dte')} | {row.get('native_symbol', '')} | {bid_text} | {ask_text} | {row.get('execution_confidence', 'screen_only_not_executable')} |"
        )
    return "\n".join(lines)


def generate_monitor_report(
    *,
    run_id: str,
    as_of_cst: str,
    btc_spot: float | None,
    btc_per_share: float | None,
    deribit_atm_rows: list[dict[str, Any]],
    ibit_atm_rows: list[dict[str, Any]],
    dislocations: list[dict[str, Any]],
    quality_warnings: list[str],
    cme_rows: list[dict[str, Any]] | None = None,
) -> str:
    lines = [
        f"# BTC Vol Desk Monitor — {as_of_cst}",
        "",
        f"**Run ID:** `{run_id}`",
        "",
        "**Evidence standard:** all public/API screen outputs are `screen-only` unless linked to quote or trade records.",
        "",
        "## BTC / ETF Reference",
        "",
        f"- BTC spot/reference: {btc_spot:,.2f}" if btc_spot is not None else "- BTC spot/reference: missing",
        f"- BTC per ETF share: {btc_per_share:.12f}" if btc_per_share is not None else "- BTC per ETF share: missing",
        "",
        "## Deribit ATM IV Term Structure",
        "",
        _format_atm_table(deribit_atm_rows),
        "",
        "## IBIT / ETF ATM IV Term Structure",
        "",
        _format_atm_table(ibit_atm_rows),
        "",
        "## CME / Databento BTC Options Status",
        "",
        "**Evidence label:** Databento CME rows are `screen_only_not_executable`; they are licensed vendor marks, not executable quotes.",
        "",
        _format_cme_table(cme_rows or []),
        "",
        "## Dislocation Board",
        "",
    ]
    if dislocations:
        lines.extend(["| Candidate | Gross IV diff | Confidence | Next action |", "|---|---:|---|---|"])
        for row in dislocations:
            lines.append(
                f"| {row.get('candidate')} | {float(row.get('gross_iv_diff_vol_pts', 0)):.2f} vol pts | {row.get('confidence')} | {row.get('next_action')} |"
            )
    else:
        lines.append("_No dislocation rows generated._")
    lines.extend(["", "## Quality Warnings", ""])
    if quality_warnings:
        lines.extend(f"- {warning}" for warning in quality_warnings)
    else:
        lines.append("- None")
    return "\n".join(lines) + "\n"
