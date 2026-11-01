#!/usr/bin/env python3
"""Reverse engineering a trader: pull one wallet's positions, profile its style, and score how
much of its book a candidate strategy reproduces.

    trader.py fetch 0xADDRESS --start 2025-06-01 --end 2026-07-01 [--venue hyperliquid]
              [--data DIR] [--out trader.json]
        Every fill that changed the wallet's positions (GET .../positions/changes), paged
        and split into 30-day requests, written as one file the mirror strategy reads.
        Needs $OXARCHIVE_API_KEY. Billed like trades: one credit per 1,000 rows.
        --data marks which markets the backtest dataset has (in_dataset).

    trader.py profile trader.json
        The style, from the fills alone: markets, round trips, holding times, long and
        short time, size, maker share, when it opens (UTC hour and weekday), realized PnL.

    trader.py replicate trader.json CANDIDATE.py --start DATE --end DATE [--data DIR]
              [-p NAME=VALUE ...] [-c NAME=VALUE ...] [--clock 1h]
        Backtests the trader's book (templates/mirror.py) and the candidate over the same
        window, then compares them hour by hour: how many of the trader's held hours the
        candidate is on the same side, on the other side, or flat, and how much of the
        trader's daily returns (in the engine, after costs) it tracks. Needs the 0xarchive-backtest
        wheel. Both runs go into an active study's ledger as trials.

fetch and profile use only the standard library.
"""

import argparse
import datetime as dt
import json
import os
import pathlib
import statistics
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

API = "https://api.0xarchive.io"
VERSION = "2026-10-01"
HERE = pathlib.Path(__file__).resolve().parent
# The mirror strategy: beside this script when both were copied into a study, else the
# skill's templates/.
MIRROR = next((p for p in (HERE / "mirror.py", HERE.parent / "templates" / "mirror.py")
               if p.is_file()), HERE.parent / "templates" / "mirror.py")
DAY = 86_400_000
HOUR = 3_600_000
PREFIX = {
    "hyperliquid": "/v1/hyperliquid/wallets/{a}/positions/changes",
    "hip3": "/v1/hyperliquid/hip3/wallets/{a}/positions/changes",
    "lighter": "/v1/lighter/accounts/{a}/positions/changes",
}


def ms(text):
    """Unix ms from ms digits, a date or an RFC 3339 time (UTC)."""
    if isinstance(text, (int, float)):
        return int(text)
    if text.isdigit():
        return int(text)
    t = dt.datetime.fromisoformat(text)
    if t.tzinfo is None:
        t = t.replace(tzinfo=dt.UTC)
    return int(t.timestamp() * 1000)


def iso(t):
    return dt.datetime.fromtimestamp(t / 1000, dt.UTC).strftime("%Y-%m-%d %H:%M")


def num(v):
    return None if v is None else float(v)


def get(path, params, key):
    url = API + path + "?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={"x-api-key": key, "0xArchive-Version": VERSION})
    for attempt in range(5):
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                return json.load(r)
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", "replace")
            if key:
                body = body.replace(key, "[redacted]")
            try:
                err = json.loads(body)
            except json.JSONDecodeError:
                err = {"error": body[:300]}
            code = err.get("error_code")
            if e.code in (429, 500, 503) and code != "insufficient_credits" and attempt < 4:
                time.sleep(float(e.headers.get("Retry-After") or 2 ** attempt))
                continue
            raise SystemExit(f"{e.code} {code}: {err.get('error')} (request_id {err.get('request_id')})")
        except urllib.error.URLError as e:
            detail = str(e.reason)
            raise SystemExit(detail.replace(key, "[redacted]") if key else detail) from None
    raise SystemExit("gave up after retries")


def cmd_fetch(args):
    key = os.environ.get("OXARCHIVE_API_KEY")
    if not key:
        raise SystemExit("set OXARCHIVE_API_KEY (https://www.0xarchive.io/signup)")
    path = PREFIX[args.venue].format(a=args.address)
    t0, t1 = ms(args.start), ms(args.end)
    rows, notices = [], set()
    # One request window per 30 days: the Free plan's per-request span.
    for w0 in range(t0, t1, 30 * DAY):
        w1 = min(w0 + 30 * DAY, t1)
        cursor = None
        while True:
            params = {"start": w0, "end": w1, "limit": 5000}
            if cursor:
                params["cursor"] = cursor
            page = get(path, params, key)
            rows.extend(page["data"])
            meta = page.get("meta", {})
            if meta.get("notice"):
                notices.add(meta["notice"])
            if not meta.get("has_more"):
                break
            cursor = meta["next_cursor"]
        print(f"{iso(w0)[:10]} to {iso(w1)[:10]}: {len(rows)} fills so far", file=sys.stderr)
    changes, gross, held, mismatched = [], 0.0, {}, 0
    for r in sorted(rows, key=lambda r: (ms(r["timestamp"]), r["symbol"])):
        end_pos = num(r.get("end_position"))
        if end_pos is None:
            continue
        d = r.get("direction") or ""
        # end_position is signed (short negative); the venue's direction text says the same.
        if end_pos > 0 and d.endswith("Short") or end_pos < 0 and d.endswith("Long"):
            mismatched += 1
        c = {"ts": ms(r["timestamp"]), "symbol": r["symbol"], "end_position": end_pos,
             "start_position": num(r.get("start_position")), "price": num(r.get("price")),
             "size": num(r.get("size")), "side": r.get("side"), "direction": d or None,
             "closed_pnl": num(r.get("closed_pnl")), "fee": num(r.get("fee")),
             "maker": (not r["crossed"]) if r.get("crossed") is not None else r.get("is_maker")}
        changes.append(c)
        if c["price"]:
            held[c["symbol"]] = abs(end_pos) * c["price"]
            g = 0.0
            for v in held.values():
                g += v
            gross = max(gross, g)
    changes.sort(key=lambda c: (c["ts"], c["symbol"]))
    book = {"address": args.address, "venue": args.venue, "start": t0, "end": t1,
            "fetched": dt.datetime.now(dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "max_gross_usd": gross, "changes": changes, "notices": sorted(notices)}
    if args.data:
        have = {p.name for p in pathlib.Path(args.data).iterdir() if p.is_dir()}
        book["in_dataset"] = sorted({c["symbol"] for c in changes} & have)
    pathlib.Path(args.out).write_text(json.dumps(book) + "\n", encoding="utf-8")
    print(f"{len(changes)} position changes in {len({c['symbol'] for c in changes})} markets "
          f"written to {args.out}; largest gross position ${gross:,.0f}")
    for n in sorted(notices):
        print(f"notice: {n}")
    if mismatched:
        print(f"warning: {mismatched} row(s) whose sign disagrees with their direction text")
    return 0


def profile(book):
    """The style of a book, from its position changes."""
    by = {}
    for c in book["changes"]:
        by.setdefault(c["symbol"], []).append(c)
    hours, weekdays = [0] * 24, [0] * 7
    markets, holds, long_ms, short_ms = {}, [], 0, 0
    pnl = fees = 0.0
    maker = known = 0
    for m, cs in by.items():
        trips, opened, prev_pos, prev_ts = 0, None, 0.0, None
        top = 0.0
        for c in cs:
            if prev_ts is not None:
                if prev_pos > 0:
                    long_ms += c["ts"] - prev_ts
                elif prev_pos < 0:
                    short_ms += c["ts"] - prev_ts
            pos = c["end_position"]
            if prev_pos == 0 and pos != 0 or prev_pos * pos < 0:
                when = dt.datetime.fromtimestamp(c["ts"] / 1000, dt.UTC)
                hours[when.hour] += 1
                weekdays[when.weekday()] += 1
                if prev_pos * pos < 0 and opened is not None:
                    holds.append(c["ts"] - opened)
                    trips += 1
                opened = c["ts"]
            elif prev_pos != 0 and pos == 0 and opened is not None:
                holds.append(c["ts"] - opened)
                trips += 1
                opened = None
            if c["price"]:
                top = max(top, abs(pos) * c["price"])
            pnl += c["closed_pnl"] or 0.0
            fees += c["fee"] or 0.0
            if c["maker"] is not None:
                known += 1
                maker += 1 if c["maker"] else 0
            prev_pos, prev_ts = pos, c["ts"]
        markets[m] = {"fills": len(cs), "round_trips": trips, "max_position_usd": round(top, 2),
                      "open_at_end": prev_pos != 0}
    held = long_ms + short_ms
    return {
        "address": book["address"],
        "window": [iso(book["start"]), iso(book["end"])],
        "fills": len(book["changes"]),
        "markets": dict(sorted(markets.items(), key=lambda kv: -kv[1]["fills"])),
        "not_in_dataset": sorted(set(markets) - set(book.get("in_dataset", markets))),
        "round_trips": len(holds),
        "hold_hours_median": round(statistics.median(holds) / HOUR, 2) if holds else None,
        "hold_hours_p90": (round(sorted(holds)[int(0.9 * (len(holds) - 1))] / HOUR, 2)
                           if holds else None),
        "long_share_of_held_time": round(long_ms / held, 3) if held else None,
        "max_gross_usd": round(book["max_gross_usd"], 2),
        "maker_share": round(maker / known, 3) if known else None,
        "opens_by_utc_hour": {h: n for h, n in enumerate(hours) if n},
        "opens_by_weekday": {d: n for d, n in zip(("mon", "tue", "wed", "thu", "fri", "sat", "sun"),
                                                  weekdays) if n},
        "realized_pnl": round(pnl, 2),
        "fees": round(fees, 2),
    }


def cmd_profile(args):
    book = json.loads(pathlib.Path(args.book).read_text(encoding="utf-8"))
    print(json.dumps(profile(book), indent=1))
    return 0


def _positions(fills):
    """{market: ([ts], [position after])} from a report's fills, oldest first."""
    out = {}
    for f in sorted(fills, key=lambda f: f.ts):
        ts, pos = out.setdefault(f.market, ([], []))
        ts.append(f.ts)
        pos.append((pos[-1] if pos else 0.0) + f.size)
    return out


def _at(path, t):
    import bisect

    ts, pos = path
    i = bisect.bisect_right(ts, t) - 1
    return pos[i] if i >= 0 else 0.0


def _corr(xs, ys):
    n = len(xs)
    mx = my = 0.0
    for a, b in zip(xs, ys):
        mx += a
        my += b
    mx, my = mx / n, my / n
    sxy = sxx = syy = 0.0
    for a, b in zip(xs, ys):
        sxy += (a - mx) * (b - my)
        sxx += (a - mx) * (a - mx)
        syy += (b - my) * (b - my)
    return sxy / (sxx * syy) ** 0.5 if sxx > 0 and syy > 0 else None


def compare(mirror, candidate, t0, t1):
    """Hour by hour over [t0, t1): the candidate against the trader's book, both as the
    engine traded them."""
    a, b = _positions(mirror.fills), _positions(candidate.fills)
    eps = 1e-12
    held = same = other = flat = extra = 0
    for m in sorted(set(a) | set(b)):
        for t in range(t0 - t0 % HOUR + HOUR, t1, HOUR):
            pa = _at(a[m], t) if m in a else 0.0
            pb = _at(b[m], t) if m in b else 0.0
            if abs(pa) > eps:
                held += 1
                if abs(pb) <= eps:
                    flat += 1
                elif (pa > 0) == (pb > 0):
                    same += 1
                else:
                    other += 1
            elif abs(pb) > eps:
                extra += 1
    share = lambda n, d: round(n / d, 3) if d else None
    ra, rb = mirror.daily_returns, candidate.daily_returns
    k = min(len(ra), len(rb))
    corr = _corr(ra[:k], rb[:k]) if k > 2 else None
    pa = {m: v["pnl"] for m, v in (mirror.attribution or {}).items()}
    pb = {m: v["pnl"] for m, v in (candidate.attribution or {}).items()}
    total_a = total_b = 0.0
    for v in pa.values():
        total_a += v
    for v in pb.values():
        total_b += v
    return {
        "trader_held_market_hours": held,
        "same_side": share(same, held),
        "other_side": share(other, held),
        "flat_while_trader_held": share(flat, held),
        "candidate_only_market_hours": extra,
        "trader_pnl": round(total_a, 2),
        "candidate_pnl": round(total_b, 2),
        # The two books run at different sizes, so compare their returns, not their PnL.
        "daily_return_correlation": None if corr is None else round(corr, 3),
        "markets_both_profitable": share(sum(1 for m in pa if pa[m] > 0 and pb.get(m, 0.0) > 0),
                                         sum(1 for m in pa if pa[m] > 0)),
        "trader_sharpe": (mirror.stats or {}).get("sharpe"),
        "candidate_sharpe": (candidate.stats or {}).get("sharpe"),
        "per_market_pnl": {m: {"trader": round(pa.get(m, 0.0), 2), "candidate": round(pb.get(m, 0.0), 2)}
                           for m in sorted(set(pa) | set(pb))},
    }


def _pairs(items):
    out = {}
    for item in items or ():
        name, _, value = item.partition("=")
        try:
            out[name] = json.loads(value)
        except json.JSONDecodeError:
            out[name] = value
    return out


def cmd_replicate(args):
    try:
        import oxarchive_backtest as oxb
    except ImportError:
        raise SystemExit('replicate needs the engine: pip install "0xarchive-backtest[curated]==0.3.1"') from None
    from oxarchive_backtest import study

    config = oxb.Config(**_pairs(args.config))
    t0, t1 = ms(args.start), ms(args.end)

    def run(path, params):
        text = pathlib.Path(path).read_text(encoding="utf-8")
        module = oxb.load_strategy(path, name=f"oxs_{pathlib.Path(path).stem}")
        with study.running(path, text):
            return oxb.backtest(oxb.strategy_of(module, params), t0, t1, args.data, config)

    try:
        mirror = run(args.mirror or MIRROR, {"path": str(pathlib.Path(args.book).resolve()),
                                             "clock": args.clock})
        candidate = run(args.candidate, _pairs(args.param))
    except (ValueError, oxb.Sealed, oxb.MissingData) as e:
        raise SystemExit(str(e)) from None
    out = compare(mirror, candidate, t0, t1)
    out["trader_result_hash"], out["candidate_result_hash"] = mirror.result_hash, candidate.result_hash
    print(json.dumps(out, indent=1))
    return 0


def main(argv=None):
    p = argparse.ArgumentParser(prog="trader.py", description=__doc__.split("\n\n")[0])
    sub = p.add_subparsers(dest="command", required=True)
    f = sub.add_parser("fetch", help="pull a wallet's position changes")
    f.add_argument("address")
    f.add_argument("--start", required=True)
    f.add_argument("--end", required=True)
    f.add_argument("--venue", default="hyperliquid", choices=sorted(PREFIX))
    f.add_argument("--data", help="the backtest dataset, to mark which markets it has")
    f.add_argument("--out", default="trader.json")
    f.set_defaults(run=cmd_fetch)
    f = sub.add_parser("profile", help="the trader's style from the fills")
    f.add_argument("book")
    f.set_defaults(run=cmd_profile)
    f = sub.add_parser("replicate", help="score a candidate against the trader's book")
    f.add_argument("book")
    f.add_argument("candidate")
    f.add_argument("--start", required=True)
    f.add_argument("--end", required=True)
    f.add_argument("--data", default="data/hyperliquid")
    f.add_argument("--clock", default="1h", help="the mirror's clock (default 1h)")
    f.add_argument("--mirror", help="the mirror strategy file (default: mirror.py beside this script)")
    f.add_argument("-p", "--param", action="append", metavar="NAME=VALUE")
    f.add_argument("-c", "--config", action="append", metavar="NAME=VALUE")
    f.set_defaults(run=cmd_replicate)
    args = p.parse_args(argv)
    return args.run(args)


if __name__ == "__main__":
    sys.exit(main())
