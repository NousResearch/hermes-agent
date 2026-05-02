#!/usr/bin/env python3
"""Spend-cap policy check.

Usage:
  python3 cap_check.py --price-eur 14.20 [--quantity 1] \
    [--cap-per-purchase 50] [--cap-per-day 100] \
    [--ledger /tmp/personal-shopper-ledger.json] [--commit] [--json]

The ledger is a tiny JSON file the agent writes to track committed
purchases for the day. With `--commit` the script appends an entry.
Without it, it just simulates and returns whether the purchase WOULD
be allowed.

Outputs (JSON mode):
  {
    "ok": true,
    "reason": null,
    "engaged_today_eur": 23.20,
    "remaining_today_eur": 76.80,
    "cap_per_purchase_eur": 50,
    "cap_per_day_eur": 100
  }

The agent should run this twice per Buy: once before showing the user
a confirmation (with the search-snippet price) and once after the live
page price is scraped (build_cart.py returns it). On --commit, only
call the second time with the verified price.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path


DEFAULT_LEDGER = Path("/tmp/personal-shopper-ledger.json")


def _load_ledger(path: Path) -> dict[str, list[dict[str, object]]]:
    if not path.exists():
        return {"entries": []}
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return {"entries": []}


def _save_ledger(path: Path, ledger: dict[str, list[dict[str, object]]]) -> None:
    path.write_text(json.dumps(ledger, ensure_ascii=False, indent=2))


def _spent_today(ledger: dict[str, list[dict[str, object]]]) -> float:
    midnight = (int(time.time()) // 86400) * 86400
    return sum(
        float(e.get("price_eur", 0)) * int(e.get("quantity", 1))
        for e in ledger.get("entries", [])
        if int(e.get("created_at", 0)) >= midnight
    )


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--price-eur", type=float, required=True)
    p.add_argument("--quantity", type=int, default=1)
    p.add_argument(
        "--cap-per-purchase",
        type=float,
        default=float(os.environ.get("SHOPPER_SPEND_CAP_EUR_PER_PURCHASE", "50")),
    )
    p.add_argument(
        "--cap-per-day",
        type=float,
        default=float(os.environ.get("SHOPPER_SPEND_CAP_EUR_PER_DAY", "100")),
    )
    p.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER)
    p.add_argument("--commit", action="store_true", help="Persist this entry to the ledger")
    p.add_argument("--json", action="store_true")
    args = p.parse_args()

    ledger = _load_ledger(args.ledger)
    spent = _spent_today(ledger)
    line_total = args.price_eur * args.quantity

    decision: dict[str, object] = {
        "ok": True,
        "reason": None,
        "engaged_today_eur": round(spent, 2),
        "remaining_today_eur": round(max(0.0, args.cap_per_day - spent), 2),
        "cap_per_purchase_eur": args.cap_per_purchase,
        "cap_per_day_eur": args.cap_per_day,
    }

    if line_total <= 0:
        decision["ok"] = False
        decision["reason"] = "price must be > 0"
    elif line_total > args.cap_per_purchase:
        decision["ok"] = False
        decision["reason"] = (
            f"per-purchase cap exceeded: {line_total:.2f} EUR > {args.cap_per_purchase:.2f} EUR"
        )
    elif spent + line_total > args.cap_per_day:
        decision["ok"] = False
        decision["reason"] = (
            f"per-day cap exceeded: {spent:.2f} engaged + {line_total:.2f} this purchase "
            f"> {args.cap_per_day:.2f} EUR cap"
        )

    if decision["ok"] and args.commit:
        ledger.setdefault("entries", []).append(
            {
                "price_eur": args.price_eur,
                "quantity": args.quantity,
                "created_at": int(time.time()),
            }
        )
        _save_ledger(args.ledger, ledger)
        decision["committed"] = True

    if args.json:
        sys.stdout.write(json.dumps(decision, ensure_ascii=False, indent=2) + "\n")
    else:
        if decision["ok"]:
            sys.stdout.write(
                f"OK -- {line_total:.2f} EUR, restant aujourd'hui "
                f"{decision['remaining_today_eur']:.2f} EUR\n"
            )
        else:
            sys.stdout.write(f"REFUS -- {decision['reason']}\n")
    return 0 if decision["ok"] else 3


if __name__ == "__main__":
    sys.exit(main())
