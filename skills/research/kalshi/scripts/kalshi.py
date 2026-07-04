#!/usr/bin/env python3
"""Kalshi CLI helper — query prediction market data.

Read-only access to Kalshi's public REST API. No auth required for any
endpoint used here. Python stdlib only.

Usage:
    python3 kalshi.py status
    python3 kalshi.py events [--status open] [--series-ticker TICKER] [--limit N] [--cursor C]
    python3 kalshi.py event <event_ticker> [--with-markets]
    python3 kalshi.py markets [--event-ticker TICKER] [--status open] [--limit N] [--cursor C]
    python3 kalshi.py market <market_ticker>
    python3 kalshi.py orderbook <market_ticker> [--depth N]
    python3 kalshi.py recent-trades <market_ticker> [--limit N] [--cursor C]
    python3 kalshi.py candles <series_ticker> <market_ticker> [--hours N] [--interval 60|1440]
    python3 kalshi.py series <series_ticker>
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

BASE = "https://api.elections.kalshi.com/trade-api/v2"
USER_AGENT = "hermes-skill-kalshi (+https://github.com/NousResearch/hermes-agent)"
TIMEOUT = 15


class KalshiError(RuntimeError):
    """Raised when a Kalshi API call fails after all retries."""


def _get(path: str, params: dict | None = None) -> dict:
    """GET a Kalshi endpoint, return parsed JSON.

    Retries up to three times on 429 / 5xx with exponential backoff.
    Raises ``KalshiError`` on terminal failure so callers (and tests) can
    handle it without ``sys.exit`` side-effects.
    """
    url = f"{BASE}{path}"
    if params:
        clean = {k: v for k, v in params.items() if v is not None}
        if clean:
            url = f"{url}?{urllib.parse.urlencode(clean)}"

    last_err: Exception | None = None
    for attempt in range(3):
        req = urllib.request.Request(
            url,
            headers={"User-Agent": USER_AGENT, "Accept": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            last_err = exc
            if exc.code in (429, 500, 502, 503, 504) and attempt < 2:
                time.sleep(2 ** attempt)
                continue
            try:
                body = exc.read().decode("utf-8", errors="replace")
            except Exception:
                body = ""
            raise KalshiError(f"HTTP {exc.code} {exc.reason} for {url}\n{body}") from exc
        except urllib.error.URLError as exc:
            last_err = exc
            if attempt < 2:
                time.sleep(2 ** attempt)
                continue
            raise KalshiError(f"Network error for {url}: {exc}") from exc
    raise KalshiError(f"unreachable: {last_err}")


def _print(data: dict) -> None:
    """Pretty-print JSON to stdout.

    Lets ``BrokenPipeError`` propagate so the caller (``main``) can map it to
    a clean exit code. Keeping process termination centralized in ``main``
    makes this helper safe to reuse from tests and other entry points.
    """
    json.dump(data, sys.stdout, indent=2, sort_keys=False)
    sys.stdout.write("\n")
    sys.stdout.flush()


# ---------- Commands ----------

def cmd_status(_: argparse.Namespace) -> None:
    _print(_get("/exchange/status"))


def cmd_events(args: argparse.Namespace) -> None:
    _print(_get("/events", {
        "limit": args.limit,
        "status": args.status,
        "series_ticker": args.series_ticker,
        "cursor": args.cursor,
        "with_nested_markets": "true" if args.with_markets else None,
    }))


def cmd_event(args: argparse.Namespace) -> None:
    _print(_get(f"/events/{args.event_ticker}", {
        "with_nested_markets": "true" if args.with_markets else None,
    }))


def cmd_markets(args: argparse.Namespace) -> None:
    _print(_get("/markets", {
        "limit": args.limit,
        "status": args.status,
        "event_ticker": args.event_ticker,
        "series_ticker": args.series_ticker,
        "tickers": args.tickers,
        "cursor": args.cursor,
    }))


def cmd_market(args: argparse.Namespace) -> None:
    _print(_get(f"/markets/{args.market_ticker}"))


def cmd_orderbook(args: argparse.Namespace) -> None:
    _print(_get(f"/markets/{args.market_ticker}/orderbook", {
        "depth": args.depth,
    }))


def cmd_trades(args: argparse.Namespace) -> None:
    _print(_get("/markets/trades", {
        "ticker": args.market_ticker,
        "limit": args.limit,
        "cursor": args.cursor,
    }))


def cmd_candles(args: argparse.Namespace) -> None:
    end_ts = int(time.time())
    # Lookback must be at least one candle interval, otherwise the API returns
    # an empty/invalid window. interval is in minutes; convert to seconds.
    interval_seconds = args.interval * 60
    start_ts = end_ts - max(interval_seconds, args.hours * 3600)
    _print(_get(
        f"/series/{args.series_ticker}/markets/{args.market_ticker}/candlesticks",
        {
            "start_ts": start_ts,
            "end_ts": end_ts,
            "period_interval": args.interval,
        },
    ))


def cmd_series(args: argparse.Namespace) -> None:
    _print(_get(f"/series/{args.series_ticker}"))


# ---------- Argparse wiring ----------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="kalshi",
        description="Read-only Kalshi prediction market client (public API, no auth).",
    )
    sub = p.add_subparsers(dest="command", required=True)

    sp = sub.add_parser("status", help="Exchange health check")
    sp.set_defaults(func=cmd_status)

    sp = sub.add_parser("events", help="List events")
    sp.add_argument("--status", default="open",
                    choices=["open", "closed", "settled", "unopened"])
    sp.add_argument("--series-ticker", dest="series_ticker", default=None)
    sp.add_argument("--limit", type=int, default=None,
                    help="Per-page result cap (omit to use API default of 100, max 200)")
    sp.add_argument("--cursor", default=None,
                    help="Pagination cursor from a previous response")
    sp.add_argument("--with-markets", action="store_true",
                    help="Inline nested markets in each event")
    sp.set_defaults(func=cmd_events)

    sp = sub.add_parser("event", help="Get one event")
    sp.add_argument("event_ticker")
    sp.add_argument("--with-markets", action="store_true")
    sp.set_defaults(func=cmd_event)

    sp = sub.add_parser("markets", help="List markets")
    sp.add_argument("--status", default="open",
                    choices=["open", "closed", "settled", "unopened"])
    sp.add_argument("--event-ticker", dest="event_ticker", default=None)
    sp.add_argument("--series-ticker", dest="series_ticker", default=None)
    sp.add_argument("--tickers", default=None,
                    help="Comma-separated explicit market tickers")
    sp.add_argument("--limit", type=int, default=None,
                    help="Per-page result cap (omit to use API default of 100, max 200)")
    sp.add_argument("--cursor", default=None,
                    help="Pagination cursor from a previous response")
    sp.set_defaults(func=cmd_markets)

    sp = sub.add_parser("market", help="Get one market")
    sp.add_argument("market_ticker")
    sp.set_defaults(func=cmd_market)

    sp = sub.add_parser("orderbook", help="Get a market's orderbook")
    sp.add_argument("market_ticker")
    sp.add_argument("--depth", type=int, default=None,
                    help="Levels per side (omit for full book)")
    sp.set_defaults(func=cmd_orderbook)

    sp = sub.add_parser(
        "recent-trades",
        help="Recent public trades for a market (not user trades)",
    )
    sp.add_argument("market_ticker")
    sp.add_argument("--limit", type=int, default=None,
                    help="Per-page result cap (omit to use API default of 100)")
    sp.add_argument("--cursor", default=None,
                    help="Pagination cursor from a previous response")
    sp.set_defaults(func=cmd_trades)

    sp = sub.add_parser("candles", help="OHLC candlesticks for a market")
    sp.add_argument("series_ticker", help="Series prefix of the market ticker")
    sp.add_argument("market_ticker")
    sp.add_argument("--hours", type=int, default=1,
                    help="Lookback window in hours (default 1)")
    sp.add_argument("--interval", type=int, default=60, choices=[1, 60, 1440],
                    help="Period in minutes: 1 (undocumented but works), "
                         "60 (hourly), or 1440 (daily).")
    sp.set_defaults(func=cmd_candles)

    sp = sub.add_parser("series", help="Get one series")
    sp.add_argument("series_ticker")
    sp.set_defaults(func=cmd_series)

    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        args.func(args)
    except KalshiError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    except BrokenPipeError:
        # Downstream consumer (e.g. ``| head``) closed the pipe before we
        # finished writing. Suppress the noisy traceback Python prints at
        # interpreter shutdown by redirecting stdout to /dev/null.
        try:
            devnull = os.open(os.devnull, os.O_WRONLY)
            os.dup2(devnull, sys.stdout.fileno())
        except Exception:
            pass
        return 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
