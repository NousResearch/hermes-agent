#!/usr/bin/env python3
"""Compare nickname-only card reward assumptions from CSV."""

from __future__ import annotations

import argparse
import csv
import json
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from pathlib import Path

CENT = Decimal("0.01")


def money(value: Decimal) -> str:
    return str(value.quantize(CENT, rounding=ROUND_HALF_UP))


def parse_nonnegative(row: dict[str, str], field: str, nickname: str) -> Decimal:
    try:
        value = Decimal(row[field])
    except (KeyError, InvalidOperation) as exc:
        raise ValueError(f"{nickname}: invalid {field}") from exc
    if not value.is_finite() or value < 0:
        raise ValueError(f"{nickname}: {field} must be a finite nonnegative number")
    return value


def compare(path: Path, amount: Decimal) -> list[dict[str, str]]:
    if not amount.is_finite() or amount < 0:
        raise ValueError("amount must be a finite nonnegative number")
    results = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            nickname = (row.get("nickname") or "").strip()
            if not nickname:
                raise ValueError("every row needs a nickname")
            rate = parse_nonnegative(row, "reward_rate_percent", nickname)
            cap = parse_nonnegative(row, "cap_remaining", nickname)
            credit = parse_nonnegative(row, "credit_remaining", nickname)
            eligible = min(amount, cap)
            rewards = eligible * rate / Decimal("100")
            value = rewards + min(amount, credit)
            results.append(
                {
                    "nickname": nickname,
                    "purchase_amount": money(amount),
                    "eligible_spend": money(eligible),
                    "reward_rate_percent": str(rate),
                    "estimated_rewards": money(rewards),
                    "applied_credit": money(min(amount, credit)),
                    "estimated_value": money(value),
                }
            )
    return sorted(results, key=lambda item: Decimal(item["estimated_value"]), reverse=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cards", type=Path, required=True)
    parser.add_argument("--amount", type=Decimal, required=True)
    args = parser.parse_args()
    try:
        result = compare(args.cards, args.amount)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
