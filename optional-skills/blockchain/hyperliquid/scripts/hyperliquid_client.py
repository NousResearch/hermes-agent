#!/usr/bin/env python3
"""
Hyperliquid CLI Tool for Hermes Agent
-------------------------------------
Queries the Hyperliquid info endpoint for market and account data.
Uses only Python standard library - no external packages required.

Usage:
  python3 hyperliquid_client.py dexs
  python3 hyperliquid_client.py markets [--dex DEX] [--limit N]
  python3 hyperliquid_client.py spots [--limit N]
  python3 hyperliquid_client.py candles <coin> [--interval 1h] [--hours 24]
  python3 hyperliquid_client.py funding <coin> [--hours 72]
  python3 hyperliquid_client.py l2 <coin> [--levels 10]
  python3 hyperliquid_client.py state [address] [--dex DEX]
  python3 hyperliquid_client.py spot-balances [address]
  python3 hyperliquid_client.py fills [address] [--hours N] [--limit N]
  python3 hyperliquid_client.py orders [address] [--limit N]
  python3 hyperliquid_client.py review [address] [--coin COIN] [--hours N]
  python3 hyperliquid_client.py export <coin> [--interval 1h] [--hours N]

Environment:
  HYPERLIQUID_API_URL  Override API base URL
                       (default: https://api.hyperliquid.xyz)
  HYPERLIQUID_USER_ADDRESS  Default address for state/fills/orders/review commands
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
import time
import urllib.error
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


USER_AGENT = "HermesAgent/1.0"
DEFAULT_USER_ENV = "HYPERLIQUID_USER_ADDRESS"
DEFAULT_API_BASE = "https://api.hyperliquid.xyz"


def _hermes_home() -> Path:
    return Path(os.environ.get("HERMES_HOME", "~/.hermes")).expanduser()


def _dotenv_paths() -> List[Path]:
    paths: List[Path] = []
    project_env = Path.cwd() / ".env"
    if project_env.exists():
        paths.append(project_env)

    user_env = _hermes_home() / ".env"
    if user_env.exists():
        paths.append(user_env)

    return paths


def _load_dotenv_values() -> Dict[str, str]:
    values: Dict[str, str] = {}
    for env_path in _dotenv_paths():
        try:
            lines = env_path.read_text(encoding="utf-8").splitlines()
        except UnicodeDecodeError:
            lines = env_path.read_text(encoding="latin-1").splitlines()

        for raw_line in lines:
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = raw_line.partition("=")
            key = key.strip()
            value = value.strip()
            if value.startswith('"') and value.endswith('"') and len(value) >= 2:
                value = value[1:-1].replace('\\"', '"').replace('\\\\', '\\')
            values[key] = value
    return values


def _env_lookup(key: str, default: str = "") -> str:
    value = os.environ.get(key, "").strip()
    if value:
        return value
    dotenv_value = _load_dotenv_values().get(key, "").strip()
    if dotenv_value:
        return dotenv_value
    return default


def _api_base() -> str:
    return _env_lookup("HYPERLIQUID_API_URL", DEFAULT_API_BASE).rstrip("/")


def _info_url() -> str:
    api_base = _api_base()
    if api_base.endswith("/info"):
        return api_base
    return f"{api_base}/info"


def _resolve_user(user: Optional[str]) -> str:
    candidate = (user or "").strip()
    if candidate:
        return candidate

    env_value = _env_lookup(DEFAULT_USER_ENV, "")
    if env_value:
        return env_value

    sys.exit(
        "Missing Hyperliquid address. Pass <address> explicitly or set "
        f"{DEFAULT_USER_ENV} in your environment or ~/.hermes/.env."
    )


def _post_info(payload: Dict[str, Any], timeout: int = 20, retries: int = 2) -> Any:
    data = json.dumps(payload).encode("utf-8")
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
        "User-Agent": USER_AGENT,
    }

    for attempt in range(retries + 1):
        request = urllib.request.Request(_info_url(), data=data, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                body = json.load(response)
            return body
        except urllib.error.HTTPError as exc:
            if exc.code == 429 and attempt < retries:
                time.sleep(1.5 * (attempt + 1))
                continue
            sys.exit(f"Hyperliquid HTTP error: {exc}")
        except urllib.error.URLError as exc:
            sys.exit(f"Hyperliquid connection error: {exc}")
        except json.JSONDecodeError as exc:
            sys.exit(f"Hyperliquid response was not valid JSON: {exc}")

    return None


def _safe_float(value: Any) -> Optional[float]:
    try:
        if value is None or value == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _limit_items(items: List[Dict[str, Any]], limit: int) -> List[Dict[str, Any]]:
    if limit <= 0:
        return items
    return items[:limit]


def _hours_ago_ms(hours: float, now_ms: Optional[int] = None) -> int:
    end_ms = now_ms if now_ms is not None else int(time.time() * 1000)
    return end_ms - int(hours * 60 * 60 * 1000)


def _format_timestamp_ms(value: Any) -> str:
    try:
        ts_ms = int(value)
    except (TypeError, ValueError):
        return "-"
    return dt.datetime.utcfromtimestamp(ts_ms / 1000).strftime("%Y-%m-%d %H:%M:%S UTC")


def _compact_number(value: Any, decimals: int = 2) -> str:
    number = _safe_float(value)
    if number is None:
        return "-"
    sign = "-" if number < 0 else ""
    number = abs(number)
    if number >= 1_000_000_000:
        return f"{sign}{number / 1_000_000_000:.{decimals}f}B"
    if number >= 1_000_000:
        return f"{sign}{number / 1_000_000:.{decimals}f}M"
    if number >= 1_000:
        return f"{sign}{number / 1_000:.{decimals}f}K"
    if number >= 100:
        return f"{sign}{number:.2f}"
    if number >= 1:
        return f"{sign}{number:.4f}".rstrip("0").rstrip(".")
    return f"{sign}{number:.6f}".rstrip("0").rstrip(".")


def _format_price(value: Any) -> str:
    number = _safe_float(value)
    if number is None:
        return "-"
    if abs(number) >= 1000:
        return f"{number:,.2f}"
    if abs(number) >= 1:
        return f"{number:,.4f}".rstrip("0").rstrip(".")
    return f"{number:,.6f}".rstrip("0").rstrip(".")


def _format_percent(value: Any, decimals: int = 2) -> str:
    number = _safe_float(value)
    if number is None:
        return "-"
    return f"{number:+.{decimals}f}%"


def _format_fraction_percent(value: Any, decimals: int = 4) -> str:
    number = _safe_float(value)
    if number is None:
        return "-"
    return f"{number * 100:+.{decimals}f}%"


def _percent_change(current: Any, previous: Any) -> Optional[float]:
    curr = _safe_float(current)
    prev = _safe_float(previous)
    if curr is None or prev is None or prev == 0:
        return None
    return ((curr - prev) / prev) * 100


def _short_address(address: Any) -> str:
    if not isinstance(address, str) or len(address) < 12:
        return str(address)
    return f"{address[:6]}...{address[-4:]}"


def _render_table(headers: List[tuple[str, str]], rows: List[Dict[str, Any]]) -> str:
    if not rows:
        return "(no data)"

    prepared_rows: List[List[str]] = []
    widths = [len(label) for label, _ in headers]

    for row in rows:
        rendered = []
        for index, (_label, key) in enumerate(headers):
            value = row.get(key, "")
            text = str(value)
            rendered.append(text)
            if len(text) > widths[index]:
                widths[index] = len(text)
        prepared_rows.append(rendered)

    lines = []
    header_line = "  ".join(label.ljust(widths[idx]) for idx, (label, _key) in enumerate(headers))
    separator = "  ".join("-" * widths[idx] for idx in range(len(headers)))
    lines.extend([header_line, separator])

    for rendered in prepared_rows:
        lines.append("  ".join(rendered[idx].ljust(widths[idx]) for idx in range(len(rendered))))
    return "\n".join(lines)


def _normalize_dexs(payload: Any) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not isinstance(payload, list):
        return rows

    for index, item in enumerate(payload):
        if item is None:
            rows.append(
                {
                    "index": index,
                    "name": "",
                    "label": "first-perp-dex",
                    "full_name": "First perp dex",
                    "deployer": "-",
                    "asset_caps": 0,
                }
            )
            continue

        if not isinstance(item, dict):
            continue

        caps = item.get("assetToStreamingOiCap") or []
        rows.append(
            {
                "index": index,
                "name": item.get("name", ""),
                "label": item.get("name") or "first-perp-dex",
                "full_name": item.get("fullName") or "-",
                "deployer": item.get("deployer") or "-",
                "asset_caps": len(caps) if isinstance(caps, list) else 0,
            }
        )
    return rows


def _normalize_perp_markets(payload: Any) -> List[Dict[str, Any]]:
    if not isinstance(payload, list) or len(payload) < 2:
        return []

    meta = payload[0] if isinstance(payload[0], dict) else {}
    ctxs = payload[1] if isinstance(payload[1], list) else []
    universe = meta.get("universe") if isinstance(meta, dict) else []
    if not isinstance(universe, list):
        return []

    rows: List[Dict[str, Any]] = []
    for index, spec in enumerate(universe):
        if not isinstance(spec, dict):
            continue
        ctx = ctxs[index] if index < len(ctxs) and isinstance(ctxs[index], dict) else {}
        mark_px = ctx.get("markPx") or ctx.get("midPx") or ctx.get("oraclePx")
        row = {
            "coin": spec.get("name", f"asset-{index}"),
            "mark_px": mark_px,
            "mid_px": ctx.get("midPx"),
            "oracle_px": ctx.get("oraclePx"),
            "prev_day_px": ctx.get("prevDayPx"),
            "change_pct": _percent_change(mark_px, ctx.get("prevDayPx")),
            "funding": ctx.get("funding"),
            "premium": ctx.get("premium"),
            "open_interest": ctx.get("openInterest"),
            "day_ntl_vlm": ctx.get("dayNtlVlm"),
            "day_base_vlm": ctx.get("dayBaseVlm"),
            "max_leverage": spec.get("maxLeverage"),
            "sz_decimals": spec.get("szDecimals"),
            "is_delisted": bool(spec.get("isDelisted")),
            "only_isolated": bool(spec.get("onlyIsolated")),
            "margin_mode": spec.get("marginMode") or "-",
        }
        rows.append(row)
    return rows


def _normalize_spot_markets(payload: Any) -> List[Dict[str, Any]]:
    if not isinstance(payload, list) or len(payload) < 2:
        return []

    meta = payload[0] if isinstance(payload[0], dict) else {}
    ctxs = payload[1] if isinstance(payload[1], list) else []
    pairs = meta.get("universe") if isinstance(meta, dict) else []
    tokens = meta.get("tokens") if isinstance(meta, dict) else []
    token_lookup = {}
    if isinstance(tokens, list):
        for token in tokens:
            if isinstance(token, dict) and "index" in token:
                token_lookup[token["index"]] = token.get("name", str(token["index"]))

    rows: List[Dict[str, Any]] = []
    if not isinstance(pairs, list):
        return rows

    for index, pair in enumerate(pairs):
        if not isinstance(pair, dict):
            continue
        ctx = ctxs[index] if index < len(ctxs) and isinstance(ctxs[index], dict) else {}
        raw_name = pair.get("name", f"@{index}")
        tokens_for_pair = pair.get("tokens") if isinstance(pair.get("tokens"), list) else []
        display_name = raw_name
        if "/" not in raw_name and len(tokens_for_pair) == 2:
            base = token_lookup.get(tokens_for_pair[0], str(tokens_for_pair[0]))
            quote = token_lookup.get(tokens_for_pair[1], str(tokens_for_pair[1]))
            display_name = f"{base}/{quote} ({raw_name})"

        mark_px = ctx.get("markPx") or ctx.get("midPx")
        rows.append(
            {
                "pair": raw_name,
                "display_name": display_name,
                "mark_px": mark_px,
                "mid_px": ctx.get("midPx"),
                "prev_day_px": ctx.get("prevDayPx"),
                "change_pct": _percent_change(mark_px, ctx.get("prevDayPx")),
                "day_ntl_vlm": ctx.get("dayNtlVlm"),
            }
        )
    return rows


def _normalize_candles(payload: Any) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not isinstance(payload, list):
        return rows

    for candle in payload:
        if not isinstance(candle, dict):
            continue
        rows.append(
            {
                "time": candle.get("t") or candle.get("time"),
                "open": candle.get("o"),
                "high": candle.get("h"),
                "low": candle.get("l"),
                "close": candle.get("c"),
                "volume": candle.get("v"),
                "trades": candle.get("n"),
            }
        )

    rows.sort(key=lambda item: int(item.get("time") or 0))
    return rows


def _normalize_funding_history(payload: Any) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not isinstance(payload, list):
        return rows

    for item in payload:
        if not isinstance(item, dict):
            continue
        rows.append(
            {
                "coin": item.get("coin", "-"),
                "funding_rate": item.get("fundingRate"),
                "premium": item.get("premium"),
                "time": item.get("time"),
            }
        )

    rows.sort(key=lambda item: int(item.get("time") or 0))
    return rows


def _normalize_book_levels(payload: Any) -> Dict[str, List[Dict[str, Any]]]:
    if not isinstance(payload, dict):
        return {"bids": [], "asks": []}

    levels = payload.get("levels")
    if not isinstance(levels, list) or len(levels) < 2:
        return {"bids": [], "asks": []}

    def convert(side: Iterable[Any]) -> List[Dict[str, Any]]:
        converted = []
        for entry in side:
            if isinstance(entry, dict):
                converted.append(
                    {
                        "px": entry.get("px"),
                        "sz": entry.get("sz"),
                        "orders": entry.get("n"),
                    }
                )
            elif isinstance(entry, (list, tuple)) and len(entry) >= 2:
                converted.append(
                    {
                        "px": entry[0],
                        "sz": entry[1],
                        "orders": entry[2] if len(entry) > 2 else None,
                    }
                )
        return converted

    return {"bids": convert(levels[0]), "asks": convert(levels[1])}


def _normalize_positions(payload: Any) -> Dict[str, Any]:
    if not isinstance(payload, dict):
        return {"summary": {}, "positions": []}

    positions: List[Dict[str, Any]] = []
    for item in payload.get("assetPositions", []):
        if not isinstance(item, dict):
            continue
        position = item.get("position") if isinstance(item.get("position"), dict) else item
        if not isinstance(position, dict):
            continue
        leverage = position.get("leverage") if isinstance(position.get("leverage"), dict) else {}
        positions.append(
            {
                "coin": position.get("coin", "-"),
                "size": position.get("szi"),
                "entry_px": position.get("entryPx"),
                "position_value": position.get("positionValue"),
                "unrealized_pnl": position.get("unrealizedPnl"),
                "return_on_equity": position.get("returnOnEquity"),
                "liquidation_px": position.get("liquidationPx"),
                "margin_used": position.get("marginUsed"),
                "leverage": leverage.get("value"),
                "leverage_type": leverage.get("type"),
            }
        )

    positions.sort(
        key=lambda item: abs(_safe_float(item.get("position_value")) or 0.0),
        reverse=True,
    )

    summary = payload.get("marginSummary") if isinstance(payload.get("marginSummary"), dict) else {}
    cross_summary = (
        payload.get("crossMarginSummary") if isinstance(payload.get("crossMarginSummary"), dict) else {}
    )

    return {
        "summary": {
            "account_value": summary.get("accountValue"),
            "total_ntl_pos": summary.get("totalNtlPos"),
            "total_raw_usd": summary.get("totalRawUsd"),
            "withdrawable": payload.get("withdrawable"),
            "cross_account_value": cross_summary.get("accountValue"),
        },
        "positions": positions,
    }


def _normalize_spot_balances(payload: Any) -> List[Dict[str, Any]]:
    if not isinstance(payload, dict):
        return []

    rows: List[Dict[str, Any]] = []
    for item in payload.get("balances", []):
        if not isinstance(item, dict):
            continue
        rows.append(
            {
                "coin": item.get("coin", item.get("token", "-")),
                "total": item.get("total"),
                "hold": item.get("hold"),
                "entry_ntl": item.get("entryNtl"),
            }
        )

    rows.sort(key=lambda item: abs(_safe_float(item.get("entry_ntl")) or 0.0), reverse=True)
    return rows


def _normalize_fills(payload: Any) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not isinstance(payload, list):
        return rows

    for item in payload:
        if not isinstance(item, dict):
            continue
        fill = item.get("fill") if isinstance(item.get("fill"), dict) else item
        rows.append(
            {
                "coin": fill.get("coin", "-"),
                "dir": fill.get("dir") or fill.get("side") or "-",
                "px": fill.get("px"),
                "sz": fill.get("sz"),
                "closed_pnl": fill.get("closedPnl"),
                "fee": fill.get("fee"),
                "fee_token": fill.get("feeToken"),
                "start_position": fill.get("startPosition"),
                "time": fill.get("time"),
                "hash": fill.get("hash"),
                "oid": fill.get("oid"),
                "twap_id": item.get("twapId"),
            }
        )

    rows.sort(key=lambda item: int(item.get("time") or 0), reverse=True)
    return rows


def _normalize_orders(payload: Any) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not isinstance(payload, list):
        return rows

    for item in payload:
        if not isinstance(item, dict):
            continue
        order = item.get("order") if isinstance(item.get("order"), dict) else item
        rows.append(
            {
                "coin": order.get("coin", "-"),
                "side": order.get("side", "-"),
                "limit_px": order.get("limitPx") or order.get("px"),
                "size": order.get("sz") or order.get("origSz"),
                "timestamp": item.get("statusTimestamp")
                or order.get("timestamp")
                or order.get("time"),
                "status": item.get("status") or order.get("status") or "-",
                "oid": order.get("oid"),
                "order_type": order.get("orderType") or "-",
            }
        )

    rows.sort(key=lambda item: int(item.get("timestamp") or 0), reverse=True)
    return rows


def _direction_bucket(direction: Any) -> str:
    text = str(direction or "").strip().lower()
    if "open" in text and "long" in text:
        return "open_long"
    if "close" in text and "long" in text:
        return "close_long"
    if "open" in text and "short" in text:
        return "open_short"
    if "close" in text and "short" in text:
        return "close_short"
    if text in {"b", "buy"}:
        return "buy"
    if text in {"s", "sell"}:
        return "sell"
    return "other"


def _average(values: Iterable[Optional[float]]) -> Optional[float]:
    clean_values = [value for value in values if value is not None]
    if not clean_values:
        return None
    return round(sum(clean_values) / len(clean_values), 12)


def _is_spot_coin(coin: str) -> bool:
    return "/" in coin or coin.startswith("@")


def _safe_info_query(payload: Dict[str, Any]) -> Any:
    try:
        return _post_info(payload)
    except SystemExit:
        return None


def _market_context_for_coin(coin: str, interval: str, start_ms: int, end_ms: int) -> Dict[str, Any]:
    candles = _normalize_candles(
        _safe_info_query(
            {
                "type": "candleSnapshot",
                "req": {
                    "coin": coin,
                    "interval": interval,
                    "startTime": start_ms,
                    "endTime": end_ms,
                },
            }
        )
    )
    funding_history: List[Dict[str, Any]] = []
    if not _is_spot_coin(coin):
        funding_history = _normalize_funding_history(
            _safe_info_query(
                {
                    "type": "fundingHistory",
                    "coin": coin,
                    "startTime": start_ms,
                    "endTime": end_ms,
                }
            )
        )

    candle_change = None
    if candles:
        candle_change = _percent_change(candles[-1].get("close"), candles[0].get("open"))

    funding_average = _average(_safe_float(item.get("funding_rate")) for item in funding_history)
    return {
        "coin": coin,
        "interval": interval,
        "candle_count": len(candles),
        "price_change_pct": candle_change,
        "window_open": candles[0].get("open") if candles else None,
        "window_close": candles[-1].get("close") if candles else None,
        "average_funding_rate": funding_average,
        "funding_samples": len(funding_history),
    }


def _build_coin_review(coin: str, fills: List[Dict[str, Any]], interval: str, start_ms: int, end_ms: int) -> Dict[str, Any]:
    pnl_values = [_safe_float(fill.get("closed_pnl")) for fill in fills]
    fee_values = [_safe_float(fill.get("fee")) for fill in fills]
    scored = [value for value in pnl_values if value is not None]
    wins = [value for value in scored if value > 0]
    losses = [value for value in scored if value < 0]
    breakeven = [value for value in scored if value == 0]

    direction_counts = Counter(_direction_bucket(fill.get("dir")) for fill in fills)
    market_context = _market_context_for_coin(coin, interval, start_ms, end_ms)
    total_pnl = sum(value for value in pnl_values if value is not None)
    total_fees = sum(value for value in fee_values if value is not None)
    net_after_fees = total_pnl - total_fees

    if direction_counts["open_long"] > direction_counts["open_short"]:
        open_bias = "long"
    elif direction_counts["open_short"] > direction_counts["open_long"]:
        open_bias = "short"
    elif direction_counts["open_long"] or direction_counts["open_short"]:
        open_bias = "mixed"
    else:
        open_bias = "none"

    return {
        "coin": coin,
        "fill_count": len(fills),
        "realized_pnl": total_pnl,
        "total_fees": total_fees,
        "net_after_fees": net_after_fees,
        "wins": len(wins),
        "losses": len(losses),
        "breakeven": len(breakeven),
        "win_rate_pct": (len(wins) / (len(wins) + len(losses)) * 100) if (len(wins) + len(losses)) else None,
        "open_long_count": direction_counts["open_long"],
        "open_short_count": direction_counts["open_short"],
        "close_long_count": direction_counts["close_long"],
        "close_short_count": direction_counts["close_short"],
        "open_bias": open_bias,
        "market_context": market_context,
    }


def _review_findings(summary: Dict[str, Any], coin_reviews: List[Dict[str, Any]]) -> List[str]:
    findings: List[str] = []

    if summary["fill_count"] == 0:
        return ["No fills were found in the requested review window."]

    if summary["outcome_fill_count"] == 0:
        findings.append("Most fills in this window look like opens or adjustments, so realized-outcome review is limited until positions close.")

    if summary["net_after_fees"] < 0:
        findings.append(
            f"Net realized PnL after fees was negative ({_compact_number(summary['net_after_fees'])} USDC-equivalent units in reported fill terms)."
        )
    elif summary["net_after_fees"] > 0:
        findings.append(
            f"Net realized PnL after fees was positive ({_compact_number(summary['net_after_fees'])} USDC-equivalent units in reported fill terms)."
        )

    realized_abs = abs(summary["realized_pnl"])
    if summary["total_fees"] > 0:
        if realized_abs == 0:
            findings.append("Fees were non-trivial while realized PnL stayed flat, which usually means churn without enough edge.")
        elif summary["total_fees"] / realized_abs >= 0.25:
            ratio_pct = (summary["total_fees"] / realized_abs) * 100
            findings.append(f"Fees consumed about {ratio_pct:.1f}% of absolute realized PnL, so execution efficiency is materially affecting results.")

    if summary["fill_count"] >= 20 and summary["net_after_fees"] < 0:
        win_rate = summary.get("win_rate_pct")
        if win_rate is None or win_rate < 45:
            findings.append("Activity was high relative to results, which suggests overtrading in this review window.")

    if coin_reviews:
        worst_coin = min(coin_reviews, key=lambda item: item["net_after_fees"])
        best_coin = max(coin_reviews, key=lambda item: item["net_after_fees"])
        if worst_coin["net_after_fees"] < 0:
            findings.append(
                f"The weakest coin was {worst_coin['coin']} with net after fees of {_compact_number(worst_coin['net_after_fees'])}."
            )
        if best_coin["net_after_fees"] > 0 and best_coin["coin"] != worst_coin["coin"]:
            findings.append(
                f"The strongest coin was {best_coin['coin']} with net after fees of {_compact_number(best_coin['net_after_fees'])}."
            )

    for item in coin_reviews:
        market_change = item["market_context"].get("price_change_pct")
        if item["net_after_fees"] >= 0 or market_change is None:
            continue
        if market_change > 2 and item["open_short_count"] > item["open_long_count"]:
            findings.append(f"{item['coin']}: losses came while leaning short into a rising market window.")
        elif market_change < -2 and item["open_long_count"] > item["open_short_count"]:
            findings.append(f"{item['coin']}: losses came while leaning long into a falling market window.")

    deduped: List[str] = []
    for finding in findings:
        if finding not in deduped:
            deduped.append(finding)
    return deduped[:6]


def _recent_fill_rows(fills: List[Dict[str, Any]], limit: int) -> List[Dict[str, Any]]:
    rows = []
    for fill in _limit_items(fills, limit):
        rows.append(
            {
                "time": fill.get("time"),
                "coin": fill.get("coin"),
                "dir": fill.get("dir"),
                "px": fill.get("px"),
                "sz": fill.get("sz"),
                "closed_pnl": fill.get("closed_pnl"),
                "fee": fill.get("fee"),
                "fee_token": fill.get("fee_token"),
            }
        )
    return rows


def _coin_slug(coin: str) -> str:
    slug = str(coin or "market").strip().lower()
    for old, new in (("/", "-"), (":", "-"), ("@", "spot-"), (" ", "-")):
        slug = slug.replace(old, new)
    return slug or "market"


def _default_export_path(coin: str, interval: str, hours: float) -> Path:
    hour_label = str(int(hours)) if float(hours).is_integer() else str(hours).replace(".", "p")
    filename = f"hyperliquid-{_coin_slug(coin)}-{interval}-{hour_label}h.json"
    return Path.cwd() / filename


def _write_json_file(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _export_summary(candles: List[Dict[str, Any]], funding_history: List[Dict[str, Any]]) -> Dict[str, Any]:
    candle_change = None
    if candles:
        candle_change = _percent_change(candles[-1].get("close"), candles[0].get("open"))
    return {
        "candle_count": len(candles),
        "funding_count": len(funding_history),
        "window_open": candles[0].get("open") if candles else None,
        "window_close": candles[-1].get("close") if candles else None,
        "price_change_pct": candle_change,
        "average_funding_rate": _average(_safe_float(item.get("funding_rate")) for item in funding_history),
    }


def run_dexs(_args: argparse.Namespace) -> Dict[str, Any]:
    payload = _post_info({"type": "perpDexs"})
    rows = _normalize_dexs(payload)
    return {"api_url": _info_url(), "count": len(rows), "dexs": rows}


def run_markets(args: argparse.Namespace) -> Dict[str, Any]:
    payload: Dict[str, Any] = {"type": "metaAndAssetCtxs"}
    if args.dex:
        payload["dex"] = args.dex
    rows = _normalize_perp_markets(_post_info(payload))

    if args.sort == "name":
        rows.sort(key=lambda item: item["coin"])
    elif args.sort == "oi":
        rows.sort(key=lambda item: _safe_float(item.get("open_interest")) or 0.0, reverse=True)
    elif args.sort == "funding_abs":
        rows.sort(key=lambda item: abs(_safe_float(item.get("funding")) or 0.0), reverse=True)
    elif args.sort == "change_abs":
        rows.sort(key=lambda item: abs(_safe_float(item.get("change_pct")) or 0.0), reverse=True)
    else:
        rows.sort(key=lambda item: _safe_float(item.get("day_ntl_vlm")) or 0.0, reverse=True)

    return {
        "dex": args.dex or "",
        "count": len(rows),
        "sort": args.sort,
        "markets": _limit_items(rows, args.limit),
    }


def run_spots(args: argparse.Namespace) -> Dict[str, Any]:
    rows = _normalize_spot_markets(_post_info({"type": "spotMetaAndAssetCtxs"}))

    if args.sort == "name":
        rows.sort(key=lambda item: item["display_name"])
    elif args.sort == "change_abs":
        rows.sort(key=lambda item: abs(_safe_float(item.get("change_pct")) or 0.0), reverse=True)
    else:
        rows.sort(key=lambda item: _safe_float(item.get("day_ntl_vlm")) or 0.0, reverse=True)

    return {"count": len(rows), "sort": args.sort, "pairs": _limit_items(rows, args.limit)}


def run_candles(args: argparse.Namespace) -> Dict[str, Any]:
    end_ms = int(time.time() * 1000)
    start_ms = _hours_ago_ms(args.hours, end_ms)
    payload = {
        "type": "candleSnapshot",
        "req": {
            "coin": args.coin,
            "interval": args.interval,
            "startTime": start_ms,
            "endTime": end_ms,
        },
    }
    candles = _normalize_candles(_post_info(payload))
    summary = {}
    if candles:
        highs = [_safe_float(item.get("high")) for item in candles]
        lows = [_safe_float(item.get("low")) for item in candles]
        clean_highs = [value for value in highs if value is not None]
        clean_lows = [value for value in lows if value is not None]
        summary = {
            "first_time": candles[0]["time"],
            "last_time": candles[-1]["time"],
            "open": candles[0]["open"],
            "close": candles[-1]["close"],
            "high": max(clean_highs) if clean_highs else None,
            "low": min(clean_lows) if clean_lows else None,
            "change_pct": _percent_change(candles[-1]["close"], candles[0]["open"]),
        }
    return {
        "coin": args.coin,
        "interval": args.interval,
        "hours": args.hours,
        "count": len(candles),
        "summary": summary,
        "candles": _limit_items(candles, args.limit),
    }


def run_funding(args: argparse.Namespace) -> Dict[str, Any]:
    end_ms = int(time.time() * 1000)
    start_ms = _hours_ago_ms(args.hours, end_ms)
    payload = {"type": "fundingHistory", "coin": args.coin, "startTime": start_ms, "endTime": end_ms}
    rows = _normalize_funding_history(_post_info(payload))
    avg_rate = None
    if rows:
        values = [_safe_float(item.get("funding_rate")) for item in rows]
        clean_values = [value for value in values if value is not None]
        if clean_values:
            avg_rate = sum(clean_values) / len(clean_values)
    return {
        "coin": args.coin,
        "hours": args.hours,
        "count": len(rows),
        "average_funding_rate": avg_rate,
        "history": _limit_items(list(reversed(rows)), args.limit),
    }


def run_l2(args: argparse.Namespace) -> Dict[str, Any]:
    payload: Dict[str, Any] = {"type": "l2Book", "coin": args.coin}
    if args.n_sig_figs is not None:
        payload["nSigFigs"] = args.n_sig_figs
    if args.mantissa is not None:
        payload["mantissa"] = args.mantissa
    raw = _post_info(payload)
    levels = _normalize_book_levels(raw)
    return {
        "coin": args.coin,
        "time": raw.get("time") if isinstance(raw, dict) else None,
        "bids": _limit_items(levels["bids"], args.levels),
        "asks": _limit_items(levels["asks"], args.levels),
    }


def run_state(args: argparse.Namespace) -> Dict[str, Any]:
    user = _resolve_user(args.user)
    payload: Dict[str, Any] = {"type": "clearinghouseState", "user": user}
    if args.dex:
        payload["dex"] = args.dex
    normalized = _normalize_positions(_post_info(payload))
    return {
        "user": user,
        "dex": args.dex or "",
        "summary": normalized["summary"],
        "positions": normalized["positions"],
    }


def run_spot_balances(args: argparse.Namespace) -> Dict[str, Any]:
    user = _resolve_user(args.user)
    payload = {"type": "spotClearinghouseState", "user": user}
    rows = _normalize_spot_balances(_post_info(payload))
    return {"user": user, "count": len(rows), "balances": _limit_items(rows, args.limit)}


def run_fills(args: argparse.Namespace) -> Dict[str, Any]:
    user = _resolve_user(args.user)
    payload: Dict[str, Any] = {"user": user}
    if args.hours is not None:
        payload["type"] = "userFillsByTime"
        payload["startTime"] = _hours_ago_ms(args.hours)
    else:
        payload["type"] = "userFills"
    if args.aggregate_by_time:
        payload["aggregateByTime"] = True
    rows = _normalize_fills(_post_info(payload))
    return {
        "user": user,
        "hours": args.hours,
        "aggregate_by_time": args.aggregate_by_time,
        "count": len(rows),
        "fills": _limit_items(rows, args.limit),
    }


def run_orders(args: argparse.Namespace) -> Dict[str, Any]:
    user = _resolve_user(args.user)
    payload = {"type": "historicalOrders", "user": user}
    rows = _normalize_orders(_post_info(payload))
    return {"user": user, "count": len(rows), "orders": _limit_items(rows, args.limit)}


def run_review(args: argparse.Namespace) -> Dict[str, Any]:
    user = _resolve_user(args.user)
    end_ms = int(time.time() * 1000)
    start_ms = _hours_ago_ms(args.hours, end_ms)
    payload: Dict[str, Any] = {"type": "userFillsByTime", "user": user, "startTime": start_ms}
    if args.aggregate_by_time:
        payload["aggregateByTime"] = True

    fills = _normalize_fills(_post_info(payload))
    if args.coin:
        target = args.coin.lower()
        fills = [fill for fill in fills if str(fill.get("coin", "")).lower() == target]
    fills = _limit_items(fills, args.fills)

    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for fill in fills:
        grouped.setdefault(fill.get("coin", "-"), []).append(fill)

    coin_reviews = [
        _build_coin_review(coin, coin_fills, args.interval, start_ms, end_ms)
        for coin, coin_fills in sorted(grouped.items(), key=lambda item: len(item[1]), reverse=True)
    ]

    pnl_values = [_safe_float(fill.get("closed_pnl")) for fill in fills]
    fee_values = [_safe_float(fill.get("fee")) for fill in fills]
    scored = [value for value in pnl_values if value is not None]
    wins = [value for value in scored if value > 0]
    losses = [value for value in scored if value < 0]
    direction_counts = Counter(_direction_bucket(fill.get("dir")) for fill in fills)
    total_pnl = sum(value for value in pnl_values if value is not None)
    total_fees = sum(value for value in fee_values if value is not None)

    summary = {
        "fill_count": len(fills),
        "scored_fill_count": len(scored),
        "outcome_fill_count": len(wins) + len(losses),
        "unique_coins": len(grouped),
        "realized_pnl": total_pnl,
        "total_fees": total_fees,
        "net_after_fees": total_pnl - total_fees,
        "wins": len(wins),
        "losses": len(losses),
        "breakeven": len([value for value in scored if value == 0]),
        "win_rate_pct": (len(wins) / (len(wins) + len(losses)) * 100) if (len(wins) + len(losses)) else None,
        "open_long_count": direction_counts["open_long"],
        "open_short_count": direction_counts["open_short"],
        "close_long_count": direction_counts["close_long"],
        "close_short_count": direction_counts["close_short"],
    }

    return {
        "user": user,
        "coin_filter": args.coin,
        "hours": args.hours,
        "interval": args.interval,
        "fills_requested": args.fills,
        "summary": summary,
        "findings": _review_findings(summary, coin_reviews),
        "coin_reviews": coin_reviews,
        "recent_fills": _recent_fill_rows(fills, args.recent),
    }


def run_export(args: argparse.Namespace) -> Dict[str, Any]:
    end_ms = args.end_time_ms if args.end_time_ms is not None else int(time.time() * 1000)
    start_ms = _hours_ago_ms(args.hours, end_ms)

    candle_payload = {
        "type": "candleSnapshot",
        "req": {
            "coin": args.coin,
            "interval": args.interval,
            "startTime": start_ms,
            "endTime": end_ms,
        },
    }
    candles = _normalize_candles(_post_info(candle_payload))

    funding_history: List[Dict[str, Any]] = []
    if not _is_spot_coin(args.coin):
        funding_history = _normalize_funding_history(
            _safe_info_query(
                {
                    "type": "fundingHistory",
                    "coin": args.coin,
                    "startTime": start_ms,
                    "endTime": end_ms,
                }
            )
        )

    output_path = Path(args.output) if args.output else _default_export_path(args.coin, args.interval, args.hours)
    payload = {
        "schema_version": "hyperliquid-market-export-v1",
        "source": {
            "api_url": _info_url(),
            "interval": args.interval,
            "coin": args.coin,
            "market_type": "spot" if _is_spot_coin(args.coin) else "perp",
        },
        "window": {
            "start_time_ms": start_ms,
            "end_time_ms": end_ms,
            "hours": args.hours,
        },
        "summary": _export_summary(candles, funding_history),
        "candles": candles,
        "funding_history": funding_history,
    }
    _write_json_file(output_path, payload)
    return {
        "coin": args.coin,
        "interval": args.interval,
        "hours": args.hours,
        "output_path": str(output_path),
        "summary": payload["summary"],
        "schema_version": payload["schema_version"],
    }


def render_dexs(data: Dict[str, Any]) -> str:
    rows = [
        {
            "label": item["label"],
            "full_name": item["full_name"],
            "deployer": _short_address(item["deployer"]),
            "asset_caps": item["asset_caps"],
        }
        for item in data["dexs"]
    ]
    return "\n".join(
        [
            f"API: {data['api_url']}",
            f"Perp dexs: {data['count']}",
            "",
            _render_table(
                [
                    ("Dex", "label"),
                    ("Full Name", "full_name"),
                    ("Deployer", "deployer"),
                    ("Asset Caps", "asset_caps"),
                ],
                rows,
            ),
        ]
    )


def render_markets(data: Dict[str, Any]) -> str:
    rows = [
        {
            "coin": item["coin"],
            "mark_px": _format_price(item["mark_px"]),
            "change_pct": _format_percent(item["change_pct"]),
            "funding": _format_fraction_percent(item["funding"]),
            "open_interest": _compact_number(item["open_interest"]),
            "day_ntl_vlm": _compact_number(item["day_ntl_vlm"]),
        }
        for item in data["markets"]
    ]
    lines = [
        f"Dex: {data['dex'] or 'first-perp-dex'}",
        f"Markets returned: {len(data['markets'])} of {data['count']}",
        "",
        _render_table(
            [
                ("Coin", "coin"),
                ("Mark", "mark_px"),
                ("Chg", "change_pct"),
                ("Funding", "funding"),
                ("OI", "open_interest"),
                ("24h Vol", "day_ntl_vlm"),
            ],
            rows,
        ),
    ]
    return "\n".join(lines)


def render_spots(data: Dict[str, Any]) -> str:
    rows = [
        {
            "pair": item["display_name"],
            "mark_px": _format_price(item["mark_px"]),
            "change_pct": _format_percent(item["change_pct"]),
            "day_ntl_vlm": _compact_number(item["day_ntl_vlm"]),
        }
        for item in data["pairs"]
    ]
    return "\n".join(
        [
            f"Spot pairs returned: {len(data['pairs'])} of {data['count']}",
            "",
            _render_table(
                [
                    ("Pair", "pair"),
                    ("Mark", "mark_px"),
                    ("Chg", "change_pct"),
                    ("24h Vol", "day_ntl_vlm"),
                ],
                rows,
            ),
        ]
    )


def render_candles(data: Dict[str, Any]) -> str:
    rows = [
        {
            "time": _format_timestamp_ms(item["time"]),
            "open": _format_price(item["open"]),
            "high": _format_price(item["high"]),
            "low": _format_price(item["low"]),
            "close": _format_price(item["close"]),
            "volume": _compact_number(item["volume"]),
        }
        for item in data["candles"]
    ]
    summary = data.get("summary") or {}
    lines = [
        f"Coin: {data['coin']}",
        f"Interval: {data['interval']}",
        f"Hours: {data['hours']}",
        f"Candles returned: {len(data['candles'])} of {data['count']}",
    ]
    if summary:
        lines.extend(
            [
                f"Open -> Close: {_format_price(summary.get('open'))} -> {_format_price(summary.get('close'))}",
                f"Range: {_format_price(summary.get('low'))} to {_format_price(summary.get('high'))}",
                f"Change: {_format_percent(summary.get('change_pct'))}",
            ]
        )
    lines.extend(
        [
            "",
            _render_table(
                [
                    ("Time", "time"),
                    ("Open", "open"),
                    ("High", "high"),
                    ("Low", "low"),
                    ("Close", "close"),
                    ("Volume", "volume"),
                ],
                rows,
            ),
        ]
    )
    return "\n".join(lines)


def render_funding(data: Dict[str, Any]) -> str:
    rows = [
        {
            "time": _format_timestamp_ms(item["time"]),
            "coin": item["coin"],
            "funding": _format_fraction_percent(item["funding_rate"]),
            "premium": _format_fraction_percent(item["premium"]),
        }
        for item in data["history"]
    ]
    lines = [
        f"Coin: {data['coin']}",
        f"Hours: {data['hours']}",
        f"Entries returned: {len(data['history'])} of {data['count']}",
        f"Average funding: {_format_fraction_percent(data['average_funding_rate'])}",
        "",
        _render_table(
            [
                ("Time", "time"),
                ("Coin", "coin"),
                ("Funding", "funding"),
                ("Premium", "premium"),
            ],
            rows,
        ),
    ]
    return "\n".join(lines)


def render_l2(data: Dict[str, Any]) -> str:
    bid_rows = [
        {"px": _format_price(item["px"]), "sz": _compact_number(item["sz"]), "orders": item["orders"] or "-"}
        for item in data["bids"]
    ]
    ask_rows = [
        {"px": _format_price(item["px"]), "sz": _compact_number(item["sz"]), "orders": item["orders"] or "-"}
        for item in data["asks"]
    ]
    lines = [
        f"Coin: {data['coin']}",
        f"Book time: {_format_timestamp_ms(data['time'])}",
        "",
        "Bids",
        _render_table([("Price", "px"), ("Size", "sz"), ("Orders", "orders")], bid_rows),
        "",
        "Asks",
        _render_table([("Price", "px"), ("Size", "sz"), ("Orders", "orders")], ask_rows),
    ]
    return "\n".join(lines)


def render_state(data: Dict[str, Any]) -> str:
    summary = data["summary"]
    position_rows = [
        {
            "coin": item["coin"],
            "size": item["size"],
            "entry_px": _format_price(item["entry_px"]),
            "position_value": _compact_number(item["position_value"]),
            "unrealized_pnl": _compact_number(item["unrealized_pnl"]),
            "roe": _format_fraction_percent(item["return_on_equity"], 2),
            "liq": _format_price(item["liquidation_px"]),
            "lev": f"{item['leverage'] or '-'}x",
        }
        for item in data["positions"]
    ]

    lines = [
        f"User: {data['user']}",
        f"Dex: {data['dex'] or 'first-perp-dex'}",
        f"Account value: {summary.get('account_value') or '-'}",
        f"Total notional position: {summary.get('total_ntl_pos') or '-'}",
        f"Withdrawable: {summary.get('withdrawable') or '-'}",
        f"Positions: {len(data['positions'])}",
    ]
    if position_rows:
        lines.extend(
            [
                "",
                _render_table(
                    [
                        ("Coin", "coin"),
                        ("Size", "size"),
                        ("Entry", "entry_px"),
                        ("Pos Val", "position_value"),
                        ("uPnL", "unrealized_pnl"),
                        ("ROE", "roe"),
                        ("Liq", "liq"),
                        ("Lev", "lev"),
                    ],
                    position_rows,
                ),
            ]
        )
    return "\n".join(lines)


def render_spot_balances(data: Dict[str, Any]) -> str:
    rows = [
        {
            "coin": item["coin"],
            "total": _compact_number(item["total"]),
            "hold": _compact_number(item["hold"]),
            "entry_ntl": _compact_number(item["entry_ntl"]),
        }
        for item in data["balances"]
    ]
    return "\n".join(
        [
            f"User: {data['user']}",
            f"Balances returned: {len(data['balances'])} of {data['count']}",
            "",
            _render_table(
                [
                    ("Coin", "coin"),
                    ("Total", "total"),
                    ("Hold", "hold"),
                    ("Entry Ntl", "entry_ntl"),
                ],
                rows,
            ),
        ]
    )


def render_fills(data: Dict[str, Any]) -> str:
    rows = [
        {
            "time": _format_timestamp_ms(item["time"]),
            "coin": item["coin"],
            "dir": item["dir"],
            "px": _format_price(item["px"]),
            "sz": _compact_number(item["sz"]),
            "closed_pnl": _compact_number(item["closed_pnl"]),
            "fee": f"{_compact_number(item['fee'])} {item['fee_token'] or ''}".strip(),
        }
        for item in data["fills"]
    ]
    lines = [
        f"User: {data['user']}",
        f"Aggregate by time: {data['aggregate_by_time']}",
        f"Fills returned: {len(data['fills'])} of {data['count']}",
        "",
        _render_table(
            [
                ("Time", "time"),
                ("Coin", "coin"),
                ("Dir", "dir"),
                ("Px", "px"),
                ("Sz", "sz"),
                ("Closed PnL", "closed_pnl"),
                ("Fee", "fee"),
            ],
            rows,
        ),
    ]
    return "\n".join(lines)


def render_orders(data: Dict[str, Any]) -> str:
    rows = [
        {
            "time": _format_timestamp_ms(item["timestamp"]),
            "coin": item["coin"],
            "side": item["side"],
            "limit_px": _format_price(item["limit_px"]),
            "size": _compact_number(item["size"]),
            "status": item["status"],
            "oid": item["oid"] or "-",
        }
        for item in data["orders"]
    ]
    return "\n".join(
        [
            f"User: {data['user']}",
            f"Orders returned: {len(data['orders'])} of {data['count']}",
            "",
            _render_table(
                [
                    ("Time", "time"),
                    ("Coin", "coin"),
                    ("Side", "side"),
                    ("Px", "limit_px"),
                    ("Sz", "size"),
                    ("Status", "status"),
                    ("OID", "oid"),
                ],
                rows,
            ),
        ]
    )


def render_review(data: Dict[str, Any]) -> str:
    summary = data["summary"]
    coin_rows = [
        {
            "coin": item["coin"],
            "fills": item["fill_count"],
            "net": _compact_number(item["net_after_fees"]),
            "win_rate": _format_percent(item["win_rate_pct"]),
            "trend": _format_percent(item["market_context"].get("price_change_pct")),
            "funding": _format_fraction_percent(item["market_context"].get("average_funding_rate")),
            "bias": item["open_bias"],
        }
        for item in data["coin_reviews"]
    ]
    recent_rows = [
        {
            "time": _format_timestamp_ms(item["time"]),
            "coin": item["coin"],
            "dir": item["dir"],
            "px": _format_price(item["px"]),
            "sz": _compact_number(item["sz"]),
            "closed_pnl": _compact_number(item["closed_pnl"]),
            "fee": f"{_compact_number(item['fee'])} {item['fee_token'] or ''}".strip(),
        }
        for item in data["recent_fills"]
    ]

    lines = [
        f"User: {data['user']}",
        f"Review window: {data['hours']} hours",
        f"Coin filter: {data['coin_filter'] or 'all traded coins'}",
        f"Fills analyzed: {summary['fill_count']}",
        f"Unique coins: {summary['unique_coins']}",
        f"Realized PnL: {_compact_number(summary['realized_pnl'])}",
        f"Fees: {_compact_number(summary['total_fees'])}",
        f"Net after fees: {_compact_number(summary['net_after_fees'])}",
        f"Win rate: {_format_percent(summary['win_rate_pct'])}",
    ]

    if data["findings"]:
        lines.extend(["", "Findings"])
        for finding in data["findings"]:
            lines.append(f"- {finding}")

    if coin_rows:
        lines.extend(
            [
                "",
                "Coin Breakdown",
                _render_table(
                    [
                        ("Coin", "coin"),
                        ("Fills", "fills"),
                        ("Net", "net"),
                        ("Win Rate", "win_rate"),
                        ("Trend", "trend"),
                        ("Funding", "funding"),
                        ("Bias", "bias"),
                    ],
                    coin_rows,
                ),
            ]
        )

    if recent_rows:
        lines.extend(
            [
                "",
                "Recent Fills",
                _render_table(
                    [
                        ("Time", "time"),
                        ("Coin", "coin"),
                        ("Dir", "dir"),
                        ("Px", "px"),
                        ("Sz", "sz"),
                        ("Closed PnL", "closed_pnl"),
                        ("Fee", "fee"),
                    ],
                    recent_rows,
                ),
            ]
        )

    return "\n".join(lines)


def render_export(data: Dict[str, Any]) -> str:
    summary = data["summary"]
    return "\n".join(
        [
            f"Coin: {data['coin']}",
            f"Interval: {data['interval']}",
            f"Hours: {data['hours']}",
            f"Schema: {data['schema_version']}",
            f"Output: {data['output_path']}",
            f"Candles: {summary['candle_count']}",
            f"Funding samples: {summary['funding_count']}",
            f"Window open -> close: {_format_price(summary.get('window_open'))} -> {_format_price(summary.get('window_close'))}",
            f"Price change: {_format_percent(summary.get('price_change_pct'))}",
            f"Average funding: {_format_fraction_percent(summary.get('average_funding_rate'))}",
        ]
    )


def _add_json_flag(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--json", action="store_true", help="Print raw JSON output")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Hyperliquid CLI Tool for Hermes Agent")
    subparsers = parser.add_subparsers(dest="command", required=True)

    dexs = subparsers.add_parser("dexs", help="List available perpetual dexs")
    _add_json_flag(dexs)
    dexs.set_defaults(func=run_dexs, renderer=render_dexs)

    markets = subparsers.add_parser("markets", help="List perpetual market contexts")
    markets.add_argument("--dex", default="", help="Perp dex name; empty means first perp dex")
    markets.add_argument("--limit", type=int, default=20, help="Rows to display; 0 means all")
    markets.add_argument(
        "--sort",
        choices=["volume", "oi", "funding_abs", "change_abs", "name"],
        default="volume",
        help="Sort mode",
    )
    _add_json_flag(markets)
    markets.set_defaults(func=run_markets, renderer=render_markets)

    spots = subparsers.add_parser("spots", help="List spot market contexts")
    spots.add_argument("--limit", type=int, default=20, help="Rows to display; 0 means all")
    spots.add_argument(
        "--sort",
        choices=["volume", "change_abs", "name"],
        default="volume",
        help="Sort mode",
    )
    _add_json_flag(spots)
    spots.set_defaults(func=run_spots, renderer=render_spots)

    candles = subparsers.add_parser("candles", help="Fetch candle history for a market")
    candles.add_argument("coin", help='Coin name, e.g. "BTC" or "PURR/USDC" or "mydex:BTC"')
    candles.add_argument("--interval", default="1h", help="Candle interval, e.g. 1m, 15m, 1h, 4h, 1d")
    candles.add_argument("--hours", type=float, default=24.0, help="Lookback window in hours")
    candles.add_argument("--limit", type=int, default=20, help="Rows to display; 0 means all")
    _add_json_flag(candles)
    candles.set_defaults(func=run_candles, renderer=render_candles)

    funding = subparsers.add_parser("funding", help="Fetch funding history for a perp market")
    funding.add_argument("coin", help='Coin name, e.g. "BTC" or "mydex:COIN"')
    funding.add_argument("--hours", type=float, default=72.0, help="Lookback window in hours")
    funding.add_argument("--limit", type=int, default=20, help="Rows to display; 0 means all")
    _add_json_flag(funding)
    funding.set_defaults(func=run_funding, renderer=render_funding)

    l2 = subparsers.add_parser("l2", help="Inspect the current L2 book for a market")
    l2.add_argument("coin", help='Coin name, e.g. "BTC" or "PURR/USDC"')
    l2.add_argument("--levels", type=int, default=10, help="Levels per side to display")
    l2.add_argument("--n-sig-figs", type=int, default=None, help="Optional server-side book aggregation")
    l2.add_argument("--mantissa", type=int, default=None, help="Optional mantissa when using nSigFigs")
    _add_json_flag(l2)
    l2.set_defaults(func=run_l2, renderer=render_l2)

    state = subparsers.add_parser("state", help="Inspect a user's perp account state")
    state.add_argument("user", nargs="?", default="", help=f"Optional address; falls back to ${DEFAULT_USER_ENV}")
    state.add_argument("--dex", default="", help="Perp dex name; empty means first perp dex")
    _add_json_flag(state)
    state.set_defaults(func=run_state, renderer=render_state)

    spot_balances = subparsers.add_parser("spot-balances", help="Inspect a user's spot token balances")
    spot_balances.add_argument("user", nargs="?", default="", help=f"Optional address; falls back to ${DEFAULT_USER_ENV}")
    spot_balances.add_argument("--limit", type=int, default=20, help="Rows to display; 0 means all")
    _add_json_flag(spot_balances)
    spot_balances.set_defaults(func=run_spot_balances, renderer=render_spot_balances)

    fills = subparsers.add_parser("fills", help="Inspect a user's recent fills")
    fills.add_argument("user", nargs="?", default="", help=f"Optional address; falls back to ${DEFAULT_USER_ENV}")
    fills.add_argument("--hours", type=float, default=None, help="Optional time window; uses userFillsByTime")
    fills.add_argument("--limit", type=int, default=20, help="Rows to display; 0 means all")
    fills.add_argument(
        "--aggregate-by-time",
        action="store_true",
        help="Aggregate partial fills when the API supports it",
    )
    _add_json_flag(fills)
    fills.set_defaults(func=run_fills, renderer=render_fills)

    orders = subparsers.add_parser("orders", help="Inspect a user's historical orders")
    orders.add_argument("user", nargs="?", default="", help=f"Optional address; falls back to ${DEFAULT_USER_ENV}")
    orders.add_argument("--limit", type=int, default=20, help="Rows to display; 0 means all")
    _add_json_flag(orders)
    orders.set_defaults(func=run_orders, renderer=render_orders)

    review = subparsers.add_parser("review", help="Generate a lightweight post-trade review from recent fills")
    review.add_argument("user", nargs="?", default="", help=f"Optional address; falls back to ${DEFAULT_USER_ENV}")
    review.add_argument("--coin", default="", help="Optional exact coin filter, e.g. BTC or PURR/USDC")
    review.add_argument("--hours", type=float, default=72.0, help="Lookback window in hours")
    review.add_argument("--fills", type=int, default=50, help="Maximum fills to analyze")
    review.add_argument("--recent", type=int, default=10, help="Recent fills to display in the review")
    review.add_argument("--interval", default="1h", help="Candle interval for market context")
    review.add_argument(
        "--aggregate-by-time",
        action="store_true",
        help="Aggregate partial fills when the API supports it",
    )
    _add_json_flag(review)
    review.set_defaults(func=run_review, renderer=render_review)

    export = subparsers.add_parser("export", help="Export normalized candles and funding history to a JSON file")
    export.add_argument("coin", help='Coin name, e.g. "BTC" or "PURR/USDC" or "mydex:BTC"')
    export.add_argument("--interval", default="1h", help="Candle interval for the exported dataset")
    export.add_argument("--hours", type=float, default=168.0, help="Lookback window in hours")
    export.add_argument("--end-time-ms", type=int, default=None, help="Optional fixed end time for reproducible exports")
    export.add_argument("--output", default="", help="Path to the JSON export file")
    _add_json_flag(export)
    export.set_defaults(func=run_export, renderer=render_export)

    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    payload = args.func(args)
    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        print(args.renderer(payload))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
