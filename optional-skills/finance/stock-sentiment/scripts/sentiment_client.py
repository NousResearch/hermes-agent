#!/usr/bin/env python3
"""Minimal, read-only stdlib client for the SentiSense API.

No third-party packages. Reads SENTISENSE_API_KEY from the environment, injects
the X-SentiSense-API-Key header, prepends the base URL, and normalizes the two
response envelopes so callers reason over clean values. Every endpoint is a GET;
nothing here can trade, move money, or modify account state.

Usage:
  python sentiment_client.py sentiment NVDA
  python sentiment_client.py mood
  python sentiment_client.py cluster-buys --days 7
"""
import argparse
import json
import os
import sys
import urllib.error
import urllib.parse
import urllib.request

BASE = "https://app.sentisense.ai"


def get(path, **params):
    """GET {BASE}{path} with the auth header; returns parsed JSON or exits non-zero."""
    params = {k: v for k, v in params.items() if v is not None}
    url = BASE + path + ("?" + urllib.parse.urlencode(params) if params else "")
    key = os.environ.get("SENTISENSE_API_KEY")
    if not key:
        sys.exit("SENTISENSE_API_KEY is not set "
                 "(free key: https://app.sentisense.ai/settings/developer)")
    req = urllib.request.Request(url, headers={"X-SentiSense-API-Key": key})
    try:
        with urllib.request.urlopen(req, timeout=20) as r:
            return json.load(r)
    except urllib.error.HTTPError as e:
        retry = e.headers.get("Retry-After")
        hint = f" (Retry-After: {retry}s)" if retry else ""
        sys.exit(f"HTTP {e.code} on {path}{hint}")
    except urllib.error.URLError as e:
        sys.exit(f"network error on {path}: {e.reason}")


def rows(raw):
    """Wrap-vs-flat: bare arrays and flat dicts pass through; {..., data} unwraps."""
    if isinstance(raw, list):
        return raw
    if isinstance(raw, dict) and "data" in raw:
        return raw["data"]
    return raw


def sentiment_scalar(series):
    """Latest polarity in [-1, 1]; the float is nested at metricValue.value.value."""
    if not series:
        return None
    return float(series[-1]["metricValue"]["value"]["value"])


def out(obj):
    print(json.dumps(obj, indent=2, default=str))


def metric(ticker, slug, start=None, end=None):
    return get(f"/api/v2/metrics/entity/{ticker.upper()}/metric/{slug}",
               startTime=start, endTime=end)


def cmd_series(a, slug):
    series = metric(a.ticker, slug, getattr(a, "start", None), getattr(a, "end", None))
    out({"metric": slug, "ticker": a.ticker.upper(),
         "points": len(series) if isinstance(series, list) else None,
         "latest": sentiment_scalar(series) if slug in ("sentiment", "sentisense") else
                   (series[-1] if isinstance(series, list) and series else None)})


def cmd_mood(_a):
    m = get("/api/v2/market-mood")
    market = m.get("market", {})
    sectors = m.get("sectors", {})
    # sectors is a dict with overlapping GICS labels; keep the first of each score.
    ranked = sorted(sectors.items(), key=lambda kv: kv[1].get("currentScore", 0), reverse=True)
    out({"score": market.get("currentScore"), "phase": market.get("phase"),
         "weeklyChange": market.get("weeklyChange"),
         "signals": market.get("signals", []),
         "topSectors": [k for k, _ in ranked[:3]],
         "bottomSectors": [k for k, _ in ranked[-3:]]})


def cmd_holders(a):
    quarters = get("/api/v1/institutional/quarters")  # bare array, [0] is latest
    if not quarters:
        sys.exit("no institutional quarters available")
    q = quarters[0].get("reportDate")
    out(rows(get(f"/api/v1/institutional/holders/{a.ticker.upper()}", reportDate=q)))


def main():
    p = argparse.ArgumentParser(description="Read-only SentiSense API client.")
    sub = p.add_subparsers(dest="cmd", required=True)

    def simple(name):
        return sub.add_parser(name)

    def ticker_cmd(name, *opts):
        sp = sub.add_parser(name)
        sp.add_argument("ticker")
        for o, d in opts:
            sp.add_argument(o, default=d)
        return sp

    # sentiment / score / mentions / social dominance time series
    s = ticker_cmd("sentiment"); s.add_argument("--start"); s.add_argument("--end")
    ticker_cmd("score"); ticker_cmd("mentions"); ticker_cmd("dominance")
    simple("mood")
    # AI insights
    ticker_cmd("insights"); simple("market-insights"); ticker_cmd("insight-types")
    # smart money
    cb = simple("cluster-buys"); cb.add_argument("--days", default="7")
    ticker_cmd("insider", ("--days", "90"))
    cg = simple("congress"); cg.add_argument("--days", default="7")
    ticker_cmd("filings", ("--days", "90"))
    ticker_cmd("holders"); ticker_cmd("consensus")
    ticker_cmd("actions", ("--days", "90")); ticker_cmd("estimates")
    aa = simple("analyst-activity"); aa.add_argument("--days", default="7")
    # calendar / news / stories / quotes
    ticker_cmd("earnings")
    ticker_cmd("news", ("--limit", "8"))
    st = simple("stories"); st.add_argument("--ticker"); st.add_argument("--limit", default="10")
    ticker_cmd("price")
    ch = ticker_cmd("chart"); ch.add_argument("--timeframe", default="1M")
    simple("popular")

    a = p.parse_args()
    t = getattr(a, "ticker", None)
    T = t.upper() if t else None
    days = getattr(a, "days", None)

    if a.cmd == "sentiment":
        cmd_series(a, "sentiment")
    elif a.cmd == "score":
        cmd_series(a, "sentisense")
    elif a.cmd == "mentions":
        cmd_series(a, "mentions")
    elif a.cmd == "dominance":
        cmd_series(a, "social_dominance")
    elif a.cmd == "mood":
        cmd_mood(a)
    elif a.cmd == "insights":
        out(rows(get(f"/api/v1/insights/stock/{T}")))
    elif a.cmd == "market-insights":
        out(rows(get("/api/v1/insights/market")))
    elif a.cmd == "insight-types":
        out(get(f"/api/v1/insights/stock/{T}/types"))
    elif a.cmd == "cluster-buys":
        out(rows(get("/api/v1/insider/cluster-buys", lookbackDays=days)))
    elif a.cmd == "insider":
        out(rows(get(f"/api/v1/insider/trades/{T}", lookbackDays=days)))
    elif a.cmd == "congress":
        out(rows(get("/api/v1/politicians/activity", lookbackDays=days)))
    elif a.cmd == "filings":
        out(rows(get(f"/api/v1/politicians/filings/{T}", lookbackDays=days)))
    elif a.cmd == "holders":
        cmd_holders(a)
    elif a.cmd == "consensus":
        out(rows(get(f"/api/v1/analyst/{T}/consensus")))
    elif a.cmd == "actions":
        out(rows(get(f"/api/v1/analyst/{T}/actions", lookbackDays=days)))
    elif a.cmd == "estimates":
        out(rows(get(f"/api/v1/analyst/{T}/estimates")))
    elif a.cmd == "analyst-activity":
        out(rows(get("/api/v1/analyst/activity", lookbackDays=days)))
    elif a.cmd == "earnings":
        out(rows(get("/api/v1/calendar/earnings", ticker=T)))
    elif a.cmd == "news":
        raw = get(f"/api/v1/documents/ticker/{T}", limit=a.limit)
        out({"totalCount": raw.get("totalCount"), "documents": raw.get("documents", [])})
    elif a.cmd == "stories":
        path = (f"/api/v1/documents/stories/ticker/{a.ticker.upper()}"
                if a.ticker else "/api/v1/documents/stories")
        out(rows(get(path, limit=a.limit)))
    elif a.cmd == "price":
        out(get("/api/v1/stocks/price", ticker=T))
    elif a.cmd == "chart":
        out(get("/api/v1/stocks/chart", ticker=T, timeframe=a.timeframe))
    elif a.cmd == "popular":
        out(get("/api/v1/stocks/popular"))


if __name__ == "__main__":
    main()
