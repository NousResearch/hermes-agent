#!/usr/bin/env python3
"""Periodic usage tracker for the hermes gateway bot.

Designed to run on the gateway LXC itself (as a systemd timer) and
log per-bot usage to a CSV. The data source is the bot's own session
files at ~/.hermes/sessions/session_*.json — every Anthropic API call
the bot makes appends an entry to ``usage_history`` with token counts
and a timestamp, so we can sum across windows.

We don't poll Anthropic's /api/oauth/usage endpoint here because the
LXC authenticates with a setup-token (long-lived OAuth token from
``claude setup-token``), and Anthropic 403s setup-tokens against the
usage endpoint. Bot-side aggregation gives us the answer to the only
question that matters anyway: ``how much is the bot consuming?``

Modes:
  append   — read sessions, write one CSV row to ~/.hermes/usage.csv
  summary  — print latest snapshot + 1h/24h deltas
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

HERMES_HOME = Path(os.environ.get("HERMES_HOME") or Path.home() / ".hermes")
SESSIONS_DIR = HERMES_HOME / "sessions"
CSV_PATH = HERMES_HOME / "usage.csv"

# Opus 4.7 list pricing per Mtok. Cache write 1.25x input, cache read 0.10x.
PRICE_INPUT  = 15.00
PRICE_OUTPUT = 75.00
PRICE_CW     = 18.75
PRICE_CR     = 1.50


CSV_FIELDS = [
    "ts_utc",
    # 5h rolling window — matches Anthropic's session reset cadence.
    "turns_5h", "input_5h", "cw_5h", "cr_5h", "output_5h", "cost_5h",
    # 24h rolling — daily-ish view for spotting baseline drift.
    "turns_24h", "input_24h", "cw_24h", "cr_24h", "output_24h", "cost_24h",
    # All-time across whatever session files survive on disk.
    "turns_total", "input_total", "cw_total", "cr_total", "output_total", "cost_total",
]


def _est_cost(input_tok: int, cw: int, cr: int, output: int) -> float:
    return (
        input_tok * PRICE_INPUT  / 1_000_000
        + cw      * PRICE_CW     / 1_000_000
        + cr      * PRICE_CR     / 1_000_000
        + output  * PRICE_OUTPUT / 1_000_000
    )


def _aggregate(cutoff: datetime | None) -> dict:
    """Sum usage_history entries newer than ``cutoff`` (None = all-time)."""
    ti = to = cr = cw = 0
    turns = 0
    for path in glob.glob(str(SESSIONS_DIR / "session_*.json")):
        try:
            d = json.load(open(path))
        except Exception:
            continue
        for u in d.get("usage_history") or []:
            if cutoff is not None:
                try:
                    ts = datetime.fromisoformat(u.get("ts", ""))
                except Exception:
                    continue
                if ts.tzinfo is None:
                    ts = ts.replace(tzinfo=timezone.utc)
                if ts < cutoff:
                    continue
            ti    += u.get("input", 0)        or 0
            to    += u.get("output", 0)       or 0
            cr    += u.get("cache_read", 0)   or 0
            cw    += u.get("cache_write", 0)  or 0
            turns += 1
    return {
        "turns":  turns,
        "input":  ti,
        "output": to,
        "cw":     cw,
        "cr":     cr,
        "cost":   _est_cost(ti, cw, cr, to),
    }


def _row_for_now() -> dict:
    now = datetime.now(timezone.utc)
    five_h = _aggregate(now - timedelta(hours=5))
    one_d  = _aggregate(now - timedelta(hours=24))
    total  = _aggregate(None)
    return {
        "ts_utc":       now.isoformat(timespec="seconds"),
        "turns_5h":     five_h["turns"],
        "input_5h":     five_h["input"],
        "cw_5h":        five_h["cw"],
        "cr_5h":        five_h["cr"],
        "output_5h":    five_h["output"],
        "cost_5h":      f"{five_h['cost']:.4f}",
        "turns_24h":    one_d["turns"],
        "input_24h":    one_d["input"],
        "cw_24h":       one_d["cw"],
        "cr_24h":       one_d["cr"],
        "output_24h":   one_d["output"],
        "cost_24h":     f"{one_d['cost']:.4f}",
        "turns_total":  total["turns"],
        "input_total":  total["input"],
        "cw_total":     total["cw"],
        "cr_total":     total["cr"],
        "output_total": total["output"],
        "cost_total":   f"{total['cost']:.4f}",
    }


def _append(row: dict) -> None:
    CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    new_file = not CSV_PATH.exists()
    with open(CSV_PATH, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if new_file:
            w.writeheader()
        w.writerow(row)


def _read_csv() -> list[dict]:
    if not CSV_PATH.exists():
        return []
    with open(CSV_PATH, newline="") as f:
        return list(csv.DictReader(f))


def _delta(rows: list[dict], window: timedelta) -> dict | None:
    if not rows:
        return None
    latest = rows[-1]
    try:
        now = datetime.fromisoformat(latest["ts_utc"])
    except Exception:
        return None
    target = now - window
    earlier = None
    for r in reversed(rows[:-1]):
        try:
            ts = datetime.fromisoformat(r["ts_utc"])
        except Exception:
            continue
        if ts <= target:
            earlier = r
            break
    if earlier is None:
        return None

    def _f(d, k):
        try: return float(d.get(k) or 0)
        except: return 0.0
    def _i(d, k):
        try: return int(d.get(k) or 0)
        except: return 0

    return {
        "since":       earlier["ts_utc"],
        "turns":       _i(latest, "turns_total") - _i(earlier, "turns_total"),
        "cost":        _f(latest, "cost_total") - _f(earlier, "cost_total"),
        "input":       _i(latest, "input_total") - _i(earlier, "input_total"),
        "output":      _i(latest, "output_total") - _i(earlier, "output_total"),
        "cw":          _i(latest, "cw_total") - _i(earlier, "cw_total"),
        "cr":          _i(latest, "cr_total") - _i(earlier, "cr_total"),
    }


def cmd_summary() -> int:
    rows = _read_csv()
    if not rows:
        print(f"No data yet at {CSV_PATH}. Run with `append` first.")
        return 0
    latest = rows[-1]
    print(f"Latest snapshot ({latest['ts_utc']}):")
    print(f"  Last 5h: {latest['turns_5h']:>4} turns, "
          f"in={latest['input_5h']} cw={latest['cw_5h']} "
          f"cr={latest['cr_5h']} out={latest['output_5h']}  "
          f"cost ~${latest['cost_5h']}")
    print(f"  Last 24h:{latest['turns_24h']:>4} turns, "
          f"in={latest['input_24h']} cw={latest['cw_24h']} "
          f"cr={latest['cr_24h']} out={latest['output_24h']}  "
          f"cost ~${latest['cost_24h']}")
    print(f"  Total:   {latest['turns_total']:>4} turns, "
          f"in={latest['input_total']} cw={latest['cw_total']} "
          f"cr={latest['cr_total']} out={latest['output_total']}  "
          f"cost ~${latest['cost_total']}")
    for label, window in [("1h", timedelta(hours=1)), ("24h", timedelta(hours=24))]:
        d = _delta(rows, window)
        if not d:
            print(f"  {label} delta: (insufficient history)")
            continue
        print(f"  {label} delta (since {d['since']}): "
              f"+{d['turns']} turns, +${d['cost']:.4f} "
              f"(in=+{d['input']} cw=+{d['cw']} cr=+{d['cr']} out=+{d['output']})")
    return 0


def cmd_append() -> int:
    row = _row_for_now()
    _append(row)
    print(f"Appended {row['ts_utc']}: 5h={row['turns_5h']}t/${row['cost_5h']} "
          f"24h={row['turns_24h']}t/${row['cost_24h']} "
          f"total={row['turns_total']}t/${row['cost_total']}")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("append", help="Read sessions, append one CSV row")
    sub.add_parser("summary", help="Print latest snapshot + 1h/24h deltas")
    args = p.parse_args()
    if args.cmd == "append":
        return cmd_append()
    if args.cmd == "summary":
        return cmd_summary()
    p.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
