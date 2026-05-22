#!/usr/bin/env python3
"""
stocks_client.py - Stock market data CLI tool for the Hermes Agent project.
Zero external dependencies - Python stdlib only.
"""

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from http.cookiejar import CookieJar

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

USER_AGENT = "Mozilla/5.0 (compatible; HermesAgent/1.0)"
YF_BASE = "https://query1.finance.yahoo.com"
YF_BASE2 = "https://query2.finance.yahoo.com"
AV_BASE = "https://www.alphavantage.co/query"

MAX_RETRIES = 3
BACKOFF_BASE = 1.5  # seconds

# Global cookie jar + opener (handles Yahoo Finance session cookies)
_cookie_jar = CookieJar()
_opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(_cookie_jar))
_crumb: str | None = None

# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def print_json(data: dict | list) -> None:
    print(json.dumps(data, indent=2, ensure_ascii=False))


def fmt_price(value) -> str | None:
    if value is None:
        return None
    try:
        return f"{float(value):.2f}"
    except (TypeError, ValueError):
        return None


def fmt_large(value) -> str | None:
    """Format large numbers with B/T suffix."""
    if value is None:
        return None
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    if abs(v) >= 1e12:
        return f"{v / 1e12:.2f}T"
    if abs(v) >= 1e9:
        return f"{v / 1e9:.2f}B"
    if abs(v) >= 1e6:
        return f"{v / 1e6:.2f}M"
    return str(int(v))


def fmt_pct(value) -> str | None:
    if value is None:
        return None
    try:
        return f"{float(value):.2f}%"
    except (TypeError, ValueError):
        return None


def safe_get(d: dict, *keys, default=None):
    """Safely traverse nested dict."""
    cur = d
    for k in keys:
        if not isinstance(cur, dict):
            return default
        cur = cur.get(k, default)
        if cur is None:
            return default
    return cur


def ts_to_date(ts) -> str | None:
    """Convert Unix timestamp to ISO date string."""
    if ts is None:
        return None
    try:
        return datetime.fromtimestamp(int(ts), tz=timezone.utc).strftime("%Y-%m-%d")
    except (OSError, ValueError, TypeError):
        return None


# ---------------------------------------------------------------------------
# HTTP layer with retry + exponential backoff
# ---------------------------------------------------------------------------


def _build_request(url: str, headers: dict | None = None) -> urllib.request.Request:
    req = urllib.request.Request(url)
    req.add_header("User-Agent", USER_AGENT)
    req.add_header("Accept", "application/json, */*")
    req.add_header("Accept-Language", "en-US,en;q=0.9")
    if headers:
        for k, v in headers.items():
            req.add_header(k, v)
    return req


def fetch_url(url: str, headers: dict | None = None, retries: int = MAX_RETRIES) -> dict | list | None:
    """Fetch a URL, parse JSON, retry on transient errors."""
    last_err = None
    for attempt in range(retries):
        try:
            req = _build_request(url, headers)
            with _opener.open(req, timeout=15) as resp:
                raw = resp.read()
                return json.loads(raw.decode("utf-8", errors="replace"))
        except urllib.error.HTTPError as e:
            last_err = e
            if e.code in (404, 400):
                break  # no point retrying
            wait = BACKOFF_BASE ** attempt
            time.sleep(wait)
        except urllib.error.URLError as e:
            last_err = e
            wait = BACKOFF_BASE ** attempt
            time.sleep(wait)
        except json.JSONDecodeError as e:
            last_err = e
            break
    return None


# ---------------------------------------------------------------------------
# Yahoo Finance crumb / cookie management
# ---------------------------------------------------------------------------


def _fetch_crumb() -> str | None:
    """
    Yahoo Finance v8 requires a crumb + consent cookie.
    We hit the consent page once to grab cookies, then fetch the crumb.
    """
    global _crumb
    if _crumb is not None:
        return _crumb

    # Step 1: touch Yahoo Finance to get cookies
    try:
        req = _build_request("https://finance.yahoo.com/")
        with _opener.open(req, timeout=10) as resp:
            resp.read()
    except Exception:
        pass

    # Step 2: fetch crumb
    crumb_url = f"{YF_BASE}/v1/test/getcrumb"
    try:
        req = _build_request(crumb_url)
        with _opener.open(req, timeout=10) as resp:
            crumb_raw = resp.read().decode("utf-8").strip()
            if crumb_raw and crumb_raw != "":
                _crumb = crumb_raw
                return _crumb
    except Exception:
        pass

    return None


def yf_url(path: str, params: dict | None = None) -> str:
    """Build a Yahoo Finance URL, injecting crumb if available."""
    crumb = _fetch_crumb()
    if params is None:
        params = {}
    if crumb:
        params["crumb"] = crumb
    qs = urllib.parse.urlencode(params)
    base = f"{YF_BASE}{path}"
    return f"{base}?{qs}" if qs else base


# ---------------------------------------------------------------------------
# Yahoo Finance API calls
# ---------------------------------------------------------------------------


def yf_chart(symbol: str, interval: str = "1d", range_: str = "1d") -> dict | None:
    params = {"interval": interval, "range": range_}
    crumb = _fetch_crumb()
    if crumb:
        params["crumb"] = crumb
    qs = urllib.parse.urlencode(params)
    url = f"{YF_BASE}/v8/finance/chart/{urllib.parse.quote(symbol)}?{qs}"
    data = fetch_url(url)
    if data is None:
        # fallback to query2
        url2 = f"{YF_BASE2}/v8/finance/chart/{urllib.parse.quote(symbol)}?{qs}"
        data = fetch_url(url2)
    return data


def yf_search(query: str, count: int = 5) -> dict | None:
    params = {"q": query, "quotesCount": count, "newsCount": 0}
    crumb = _fetch_crumb()
    if crumb:
        params["crumb"] = crumb
    qs = urllib.parse.urlencode(params)
    url = f"{YF_BASE}/v1/finance/search?{qs}"
    data = fetch_url(url)
    if data is None:
        url2 = f"{YF_BASE2}/v1/finance/search?{qs}"
        data = fetch_url(url2)
    return data


def yf_quote_summary(symbol: str) -> dict | None:
    """Fetch detailed quote summary (quoteSummary) for PE, market cap, etc."""
    modules = "summaryDetail,defaultKeyStatistics,price"
    params = {"modules": modules}
    crumb = _fetch_crumb()
    if crumb:
        params["crumb"] = crumb
    qs = urllib.parse.urlencode(params)
    url = f"{YF_BASE}/v11/finance/quoteSummary/{urllib.parse.quote(symbol)}?{qs}"
    data = fetch_url(url)
    if data is None:
        url2 = f"{YF_BASE2}/v11/finance/quoteSummary/{urllib.parse.quote(symbol)}?{qs}"
        data = fetch_url(url2)
    return data


# ---------------------------------------------------------------------------
# Alpha Vantage (optional, requires API key)
# ---------------------------------------------------------------------------


def av_overview(symbol: str) -> dict | None:
    key = os.environ.get("ALPHA_VANTAGE_KEY")
    if not key:
        return None
    params = {"function": "OVERVIEW", "symbol": symbol, "apikey": key}
    qs = urllib.parse.urlencode(params)
    url = f"{AV_BASE}?{qs}"
    data = fetch_url(url)
    if isinstance(data, dict) and data.get("Symbol"):
        return data
    return None


# ---------------------------------------------------------------------------
# Data extraction helpers
# ---------------------------------------------------------------------------


def extract_quote_from_chart(symbol: str, chart_data: dict) -> dict:
    """Extract current quote info from v8 chart response."""
    result = {
        "symbol": symbol.upper(),
        "price": None,
        "change": None,
        "change_pct": None,
        "volume": None,
        "market_cap": None,
        "pe_ratio": None,
        "52w_high": None,
        "52w_low": None,
        "currency": None,
        "exchange": None,
        "short_name": None,
    }

    chart = safe_get(chart_data, "chart", "result")
    if not chart or not isinstance(chart, list) or len(chart) == 0:
        return result

    r = chart[0]
    meta = r.get("meta", {})

    result["currency"] = meta.get("currency")
    result["exchange"] = meta.get("exchangeName")
    result["short_name"] = meta.get("shortName") or meta.get("longName")

    # Price
    price = meta.get("regularMarketPrice") or meta.get("chartPreviousClose")
    result["price"] = fmt_price(price)

    # Change
    prev_close = meta.get("previousClose") or meta.get("chartPreviousClose")
    if price and prev_close:
        chg = float(price) - float(prev_close)
        chg_pct = (chg / float(prev_close)) * 100
        result["change"] = fmt_price(chg)
        result["change_pct"] = fmt_pct(chg_pct)

    result["volume"] = meta.get("regularMarketVolume")
    result["52w_high"] = fmt_price(meta.get("fiftyTwoWeekHigh"))
    result["52w_low"] = fmt_price(meta.get("fiftyTwoWeekLow"))

    return result


def extract_quote_summary_fields(qs_data: dict) -> dict:
    """Extract PE, market cap, etc. from quoteSummary response."""
    out = {
        "market_cap": None,
        "pe_ratio": None,
        "52w_high": None,
        "52w_low": None,
        "volume": None,
        "short_name": None,
    }

    result = safe_get(qs_data, "quoteSummary", "result")
    if not result or not isinstance(result, list) or len(result) == 0:
        return out

    r = result[0]

    # price module
    price_mod = r.get("price", {})
    out["market_cap"] = fmt_large(safe_get(price_mod, "marketCap", "raw"))
    out["short_name"] = price_mod.get("shortName") or price_mod.get("longName")

    # summaryDetail
    sd = r.get("summaryDetail", {})
    pe_raw = safe_get(sd, "trailingPE", "raw")
    out["pe_ratio"] = fmt_price(pe_raw) if pe_raw else None
    out["52w_high"] = fmt_price(safe_get(sd, "fiftyTwoWeekHigh", "raw"))
    out["52w_low"] = fmt_price(safe_get(sd, "fiftyTwoWeekLow", "raw"))
    out["volume"] = safe_get(sd, "volume", "raw") or safe_get(sd, "regularMarketVolume", "raw")

    # defaultKeyStatistics
    ks = r.get("defaultKeyStatistics", {})
    if out["pe_ratio"] is None:
        pe_raw = safe_get(ks, "trailingEps", "raw")
        # can't compute PE from EPS alone without price, skip

    return out


# ---------------------------------------------------------------------------
# Command: quote
# ---------------------------------------------------------------------------


def cmd_quote(symbols: list[str]) -> None:
    results = []

    for sym in symbols:
        sym = sym.upper().strip()
        entry = {"symbol": sym, "data_source": "Yahoo Finance"}

        # Fetch chart for price data
        chart_data = yf_chart(sym, interval="1d", range_="1d")
        if chart_data:
            q = extract_quote_from_chart(sym, chart_data)
            entry.update(q)

        # Fetch quoteSummary for enriched data
        qs_data = yf_quote_summary(sym)
        if qs_data:
            qs_fields = extract_quote_summary_fields(qs_data)
            # Prefer quoteSummary values if chart didn't have them
            for field in ("market_cap", "pe_ratio", "52w_high", "52w_low", "volume", "short_name"):
                if entry.get(field) is None and qs_fields.get(field) is not None:
                    entry[field] = qs_fields[field]
                elif field == "market_cap" and qs_fields.get(field) is not None:
                    # Always prefer formatted market cap from quoteSummary
                    entry[field] = qs_fields[field]

        # Optionally enrich with Alpha Vantage
        av_key = os.environ.get("ALPHA_VANTAGE_KEY")
        if av_key:
            av_data = av_overview(sym)
            if av_data:
                entry["data_source"] = "Yahoo Finance + Alpha Vantage"
                if entry.get("pe_ratio") is None:
                    pe = av_data.get("PERatio")
                    entry["pe_ratio"] = pe if pe and pe != "None" and pe != "-" else None
                if entry.get("market_cap") is None:
                    mc = av_data.get("MarketCapitalization")
                    entry["market_cap"] = fmt_large(mc)
                if entry.get("52w_high") is None:
                    entry["52w_high"] = av_data.get("52WeekHigh")
                if entry.get("52w_low") is None:
                    entry["52w_low"] = av_data.get("52WeekLow")

        results.append(entry)

    if len(results) == 1:
        print_json(results[0])
    else:
        print_json(results)


# ---------------------------------------------------------------------------
# Command: search
# ---------------------------------------------------------------------------


def cmd_search(query: str) -> None:
    data = yf_search(query, count=5)
    if not data:
        print_json({"error": "Search failed or no results", "query": query, "data_source": "Yahoo Finance"})
        return

    quotes = data.get("quotes") or []
    if not quotes:
        print_json({"error": "No matches found", "query": query, "data_source": "Yahoo Finance"})
        return

    results = []
    for q in quotes[:5]:
        results.append({
            "symbol": q.get("symbol"),
            "name": q.get("longname") or q.get("shortname"),
            "exchange": q.get("exchange") or q.get("exchDisp"),
            "type": q.get("quoteType"),
            "sector": q.get("sector"),
        })

    output = {
        "query": query,
        "matches": results,
        "data_source": "Yahoo Finance",
    }
    print_json(output)


# ---------------------------------------------------------------------------
# Command: history
# ---------------------------------------------------------------------------


def cmd_history(symbol: str, range_: str = "1mo") -> None:
    valid_ranges = ("1mo", "3mo", "6mo", "1y", "5y")
    if range_ not in valid_ranges:
        print_json({"error": f"Invalid range '{range_}'. Valid: {', '.join(valid_ranges)}"})
        return

    sym = symbol.upper().strip()
    chart_data = yf_chart(sym, interval="1d", range_=range_)

    if not chart_data:
        print_json({"error": f"Failed to fetch history for {sym}", "data_source": "Yahoo Finance"})
        return

    chart = safe_get(chart_data, "chart", "result")
    if not chart or not isinstance(chart, list) or len(chart) == 0:
        err = safe_get(chart_data, "chart", "error", "description") or "Unknown error"
        print_json({"error": err, "symbol": sym, "data_source": "Yahoo Finance"})
        return

    r = chart[0]
    timestamps = r.get("timestamp") or []
    indicators = r.get("indicators", {})
    quote_list = indicators.get("quote") or [{}]
    ohlcv = quote_list[0] if quote_list else {}

    opens = ohlcv.get("open") or []
    closes = ohlcv.get("close") or []
    highs = ohlcv.get("high") or []
    lows = ohlcv.get("low") or []
    volumes = ohlcv.get("volume") or []

    history = []
    for i, ts in enumerate(timestamps):
        def _v(lst, idx):
            try:
                val = lst[idx]
                return round(val, 2) if val is not None else None
            except IndexError:
                return None

        entry = {
            "date": ts_to_date(ts),
            "open": _v(opens, i),
            "close": _v(closes, i),
            "high": _v(highs, i),
            "low": _v(lows, i),
            "volume": _v(volumes, i),
        }
        history.append(entry)

    # Stats
    valid_closes = [c["close"] for c in history if c["close"] is not None]
    stats = {}
    if valid_closes:
        stats["min"] = fmt_price(min(valid_closes))
        stats["max"] = fmt_price(max(valid_closes))
        stats["avg"] = fmt_price(sum(valid_closes) / len(valid_closes))
        if len(valid_closes) >= 2:
            total_return = ((valid_closes[-1] - valid_closes[0]) / valid_closes[0]) * 100
            stats["total_return_pct"] = fmt_pct(total_return)
        else:
            stats["total_return_pct"] = None

    meta = r.get("meta", {})
    output = {
        "symbol": sym,
        "range": range_,
        "currency": meta.get("currency"),
        "exchange": meta.get("exchangeName"),
        "data_points": len(history),
        "stats": stats,
        "history": history,
        "data_source": "Yahoo Finance",
    }
    print_json(output)


# ---------------------------------------------------------------------------
# Command: compare
# ---------------------------------------------------------------------------


def cmd_compare(symbols: list[str]) -> None:
    if len(symbols) < 2:
        print_json({"error": "compare requires at least 2 symbols"})
        return

    comparisons = []

    for sym in symbols:
        sym = sym.upper().strip()
        entry = {
            "symbol": sym,
            "name": None,
            "price": None,
            "change_pct": None,
            "market_cap": None,
            "pe_ratio": None,
            "52w_high": None,
            "52w_low": None,
            "52w_performance_pct": None,
        }

        # Chart data
        chart_data = yf_chart(sym, interval="1d", range_="1d")
        if chart_data:
            q = extract_quote_from_chart(sym, chart_data)
            entry["name"] = q.get("short_name")
            entry["price"] = q.get("price")
            entry["change_pct"] = q.get("change_pct")
            entry["52w_high"] = q.get("52w_high")
            entry["52w_low"] = q.get("52w_low")

        # quoteSummary for enrichment
        qs_data = yf_quote_summary(sym)
        if qs_data:
            qs = extract_quote_summary_fields(qs_data)
            if qs.get("market_cap"):
                entry["market_cap"] = qs["market_cap"]
            if qs.get("pe_ratio"):
                entry["pe_ratio"] = qs["pe_ratio"]
            if entry["52w_high"] is None and qs.get("52w_high"):
                entry["52w_high"] = qs["52w_high"]
            if entry["52w_low"] is None and qs.get("52w_low"):
                entry["52w_low"] = qs["52w_low"]
            if entry["name"] is None and qs.get("short_name"):
                entry["name"] = qs["short_name"]

        # 52w performance: (current - 52w_low) / (52w_high - 52w_low)
        try:
            price_f = float(entry["price"]) if entry["price"] else None
            high_f = float(entry["52w_high"]) if entry["52w_high"] else None
            low_f = float(entry["52w_low"]) if entry["52w_low"] else None
            if price_f and low_f and price_f > 0 and low_f > 0:
                perf = ((price_f - low_f) / low_f) * 100
                entry["52w_performance_pct"] = fmt_pct(perf)
        except (ValueError, TypeError, ZeroDivisionError):
            pass

        comparisons.append(entry)

    output = {
        "comparison": comparisons,
        "symbols": [s.upper() for s in symbols],
        "data_source": "Yahoo Finance",
    }
    print_json(output)


# ---------------------------------------------------------------------------
# Command: crypto
# ---------------------------------------------------------------------------


def cmd_crypto(symbol: str, vs: str = "USD") -> None:
    sym = symbol.upper().strip()
    vs = vs.upper().strip()

    # If user already passed BTC-USD, keep as-is; otherwise append
    if "-" not in sym:
        ticker = f"{sym}-{vs}"
    else:
        ticker = sym

    chart_data = yf_chart(ticker, interval="1d", range_="1d")

    if not chart_data:
        print_json({
            "error": f"Failed to fetch crypto data for {ticker}",
            "symbol": ticker,
            "data_source": "Yahoo Finance",
        })
        return

    chart = safe_get(chart_data, "chart", "result")
    if not chart or not isinstance(chart, list) or len(chart) == 0:
        err = safe_get(chart_data, "chart", "error", "description") or "Symbol not found"
        print_json({"error": err, "symbol": ticker, "data_source": "Yahoo Finance"})
        return

    r = chart[0]
    meta = r.get("meta", {})

    price = meta.get("regularMarketPrice") or meta.get("chartPreviousClose")
    prev_close = meta.get("previousClose") or meta.get("chartPreviousClose")

    change = None
    change_pct = None
    if price and prev_close:
        try:
            chg = float(price) - float(prev_close)
            chg_pct = (chg / float(prev_close)) * 100
            change = fmt_price(chg)
            change_pct = fmt_pct(chg_pct)
        except (TypeError, ValueError, ZeroDivisionError):
            pass

    # 24h stats from indicators
    indicators = r.get("indicators", {})
    quote_list = indicators.get("quote") or [{}]
    ohlcv = quote_list[0] if quote_list else {}
    highs = [h for h in (ohlcv.get("high") or []) if h is not None]
    lows = [l for l in (ohlcv.get("low") or []) if l is not None]
    volumes = [v for v in (ohlcv.get("volume") or []) if v is not None]

    output = {
        "symbol": ticker,
        "base": sym if "-" not in sym else sym.split("-")[0],
        "quote_currency": vs,
        "price": fmt_price(price),
        "change": change,
        "change_pct": change_pct,
        "day_high": fmt_price(max(highs)) if highs else None,
        "day_low": fmt_price(min(lows)) if lows else None,
        "volume": fmt_large(sum(volumes)) if volumes else None,
        "52w_high": fmt_price(meta.get("fiftyTwoWeekHigh")),
        "52w_low": fmt_price(meta.get("fiftyTwoWeekLow")),
        "exchange": meta.get("exchangeName"),
        "short_name": meta.get("shortName") or meta.get("longName"),
        "data_source": "Yahoo Finance",
    }
    print_json(output)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="stocks_client",
        description="Stock & crypto market data CLI — Hermes Agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  stocks_client.py quote AAPL MSFT GOOGL
  stocks_client.py search "Tesla"
  stocks_client.py history AAPL --range 3mo
  stocks_client.py compare AAPL MSFT GOOGL AMZN
  stocks_client.py crypto BTC
  stocks_client.py crypto ETH --vs EUR
  ALPHA_VANTAGE_KEY=yourkey stocks_client.py quote AAPL
        """,
    )

    sub = parser.add_subparsers(dest="command", required=True)

    # quote
    p_quote = sub.add_parser("quote", help="Get current quote for one or more symbols")
    p_quote.add_argument("symbols", nargs="+", metavar="SYMBOL", help="Stock ticker symbol(s)")

    # search
    p_search = sub.add_parser("search", help="Search for stocks by name or symbol")
    p_search.add_argument("query", help="Search query (company name or partial symbol)")

    # history
    p_history = sub.add_parser("history", help="Price history for a symbol")
    p_history.add_argument("symbol", metavar="SYMBOL", help="Stock ticker symbol")
    p_history.add_argument(
        "--range",
        dest="range_",
        default="1mo",
        choices=["1mo", "3mo", "6mo", "1y", "5y"],
        help="Date range (default: 1mo)",
    )

    # compare
    p_compare = sub.add_parser("compare", help="Compare multiple stocks side by side")
    p_compare.add_argument("symbols", nargs="+", metavar="SYMBOL", help="At least 2 stock symbols")

    # crypto
    p_crypto = sub.add_parser("crypto", help="Crypto price (BTC, ETH, SOL, etc.)")
    p_crypto.add_argument("symbol", metavar="SYMBOL", help="Crypto symbol (e.g. BTC, ETH, SOL)")
    p_crypto.add_argument(
        "--vs",
        default="USD",
        metavar="CURRENCY",
        help="Quote currency (default: USD)",
    )

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    try:
        if args.command == "quote":
            cmd_quote(args.symbols)
        elif args.command == "search":
            cmd_search(args.query)
        elif args.command == "history":
            cmd_history(args.symbol, range_=args.range_)
        elif args.command == "compare":
            cmd_compare(args.symbols)
        elif args.command == "crypto":
            cmd_crypto(args.symbol, vs=args.vs)
        else:
            parser.print_help()
            sys.exit(1)
    except KeyboardInterrupt:
        print_json({"error": "Interrupted by user"})
        sys.exit(130)
    except Exception as e:
        print_json({"error": f"Unexpected error: {e}", "type": type(e).__name__})
        sys.exit(1)


if __name__ == "__main__":
    main()
