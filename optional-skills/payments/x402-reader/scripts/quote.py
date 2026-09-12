#!/usr/bin/env python3
"""Parse an x402 Payment Required JSON body. No network. No keys."""

from __future__ import annotations

import json
import sys

USDC_DECIMALS = 6


def load_body(raw: str) -> dict:
    data = json.loads(raw)
    if not isinstance(data, dict):
        raise ValueError("402 body must be a JSON object")
    return data


def quote_rows(body: dict) -> list[dict]:
    accepts = body.get("accepts") or []
    if not isinstance(accepts, list):
        raise ValueError("accepts must be a list")
    rows = []
    for item in accepts:
        if not isinstance(item, dict):
            continue
        amount = item.get("amount") if item.get("amount") is not None else item.get("maxAmountRequired")
        if amount is None or not str(amount).isdigit():
            continue
        micro = int(str(amount))
        network = str(item.get("network") or "")
        rows.append(
            {
                "network": network,
                "payTo": item.get("payTo") or item.get("pay_to"),
                "asset": item.get("asset"),
                "amount_usdc": micro / 10**USDC_DECIMALS,
                "preferred": network == "solana" or network.startswith("solana"),
            }
        )
    return rows


def pick_row(rows: list[dict]) -> dict | None:
    for row in rows:
        if row.get("preferred"):
            return row
    return rows[0] if rows else None


def main(argv: list[str] | None = None) -> int:
    _ = argv
    raw = sys.stdin.read()
    if not raw.strip():
        print("usage: quote.py < 402.json", file=sys.stderr)
        return 2
    try:
        rows = quote_rows(load_body(raw))
    except (json.JSONDecodeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    chosen = pick_row(rows)
    json.dump({"ok": bool(chosen), "quote": chosen, "accepts": rows}, sys.stdout)
    sys.stdout.write("\n")
    return 0 if chosen else 1


if __name__ == "__main__":
    raise SystemExit(main())
