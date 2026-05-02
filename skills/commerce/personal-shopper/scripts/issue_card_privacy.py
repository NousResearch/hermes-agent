#!/usr/bin/env python3
"""Issue a single-use virtual card via the Privacy.com personal API.

Usage:
  PRIVACY_API_KEY=sk_live_... python3 issue_card_privacy.py \
    --amount-eur 19.90 --merchant 'Terres de Cafe' \
    --product 'Foo grain 1kg' [--json]

Outputs (JSON mode) on success:
  {
    "pan": "4111...1234",   # full PAN -- pass to user, do not store
    "cvv": "123",
    "exp_month": 5,
    "exp_year": 2031,
    "last4": "1234",
    "spend_limit_eur": 19.90,
    "provider_card_id": "card_token",
    "memo": "shopper:Terres de Cafe:Foo grain 1kg"
  }

The card is created with type=SINGLE_USE -- it self-revokes after the
first transaction. Spend limit is set in USD cents (Privacy.com is
USD-only) using a +10% FX buffer over an approximate 1.10 USD/EUR
exchange rate.

Privacy.com requires:
  - a US bank account to fund the issuing balance
  - a personal API key (Account -> Developers -> API Keys)
  - free tier: 12 cards/month
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import httpx


PRIVACY_BASE_URL = "https://api.privacy.com/v1"


def issue(
    *,
    api_key: str,
    amount_eur: float,
    merchant: str,
    product: str,
) -> dict[str, object]:
    # USD limit in cents: amount_eur * 1.10 buffer * ~1.10 USD/EUR -> cents
    usd_limit_cents = int(round(amount_eur * 1.10 * 110))
    body = {
        "type": "SINGLE_USE",
        "spend_limit": usd_limit_cents,
        "spend_limit_duration": "TRANSACTION",
        "memo": f"shopper:{merchant[:32]}:{product[:32]}",
    }
    headers = {"Authorization": f"api-key {api_key}"}
    with httpx.Client(timeout=httpx.Timeout(20.0)) as client:
        r = client.post(f"{PRIVACY_BASE_URL}/cards", json=body, headers=headers)
        r.raise_for_status()
        data = r.json()

    pan = str(data["pan"])
    return {
        "pan": pan,
        "cvv": str(data["cvv"]),
        "exp_month": int(data["exp_month"]),
        "exp_year": int(data["exp_year"]),
        "last4": pan[-4:] if len(pan) >= 4 else pan,
        "spend_limit_eur": amount_eur,
        "provider_card_id": str(data.get("token") or data.get("id") or ""),
        "memo": body["memo"],
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--amount-eur", type=float, required=True)
    p.add_argument("--merchant", required=True)
    p.add_argument("--product", required=True)
    p.add_argument("--json", action="store_true")
    args = p.parse_args()

    api_key = os.environ.get("PRIVACY_API_KEY")
    if not api_key:
        sys.stdout.write(
            json.dumps({"error": "PRIVACY_API_KEY env var not set"}, ensure_ascii=False) + "\n"
        )
        return 2

    try:
        out = issue(
            api_key=api_key,
            amount_eur=args.amount_eur,
            merchant=args.merchant,
            product=args.product,
        )
    except httpx.HTTPStatusError as exc:
        body_text = exc.response.text[:300] if exc.response is not None else ""
        sys.stdout.write(
            json.dumps(
                {
                    "error": f"privacy.com api error: {exc}",
                    "response_body": body_text,
                },
                ensure_ascii=False,
            )
            + "\n"
        )
        return 2
    except httpx.HTTPError as exc:
        sys.stdout.write(
            json.dumps({"error": f"privacy.com network error: {exc}"}, ensure_ascii=False) + "\n"
        )
        return 2

    if args.json:
        sys.stdout.write(json.dumps(out, ensure_ascii=False, indent=2) + "\n")
    else:
        sys.stdout.write(
            f"PAN  {out['pan']}\nExp  {out['exp_month']:02d}/{str(out['exp_year'])[-2:]}\n"
            f"CVV  {out['cvv']}\nLast4 {out['last4']}\nLimit {out['spend_limit_eur']} EUR\n"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
