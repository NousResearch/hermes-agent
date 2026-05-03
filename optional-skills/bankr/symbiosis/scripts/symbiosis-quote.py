#!/usr/bin/env python3
"""Symbiosis cross-chain quote (no execution).

Usage: ./symbiosis-quote.sh <src_chain> <src_token> <src_decimals> <amount> <dst_chain> <dst_token> <dst_decimals>

Example: ./symbiosis-quote.sh 8453 0x833589fCD6eDb6E08f4c7C32D4f71b54bdA02913 6 100 137 0x3c499c542cEF5E3811e1192ce70d8cC03d5c3359 6
"""

import json
import sys
import urllib.request

SYMBIOSIS_API = "https://api-v2.symbiosis.finance/crosschain/v1/swap"
PARTNER_ID = "bankr"
FAKE_ADDR = "0x1111111111111111111111111111111111111111"


def to_smallest_units(amount: str, decimals: int) -> str:
    parts = amount.split(".")
    integer = parts[0]
    frac = parts[1] if len(parts) > 1 else ""
    frac = (frac + "0" * decimals)[:decimals]
    return str(int(integer + frac))


def format_units(amount_raw: str, decimals: int) -> str:
    s = amount_raw.zfill(decimals + 1)
    int_part = s[: len(s) - decimals] or "0"
    frac_part = s[len(s) - decimals :].rstrip("0")
    return f"{int_part}.{frac_part}" if frac_part else int_part


def api_post(url: str, payload: dict) -> dict:
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={
            "Content-Type": "application/json",
            "User-Agent": "symbiosis-bankr-skill/1.0",
        },
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read())


def main():
    if len(sys.argv) < 8:
        print(__doc__.strip())
        sys.exit(1)

    src_chain = int(sys.argv[1])
    src_token = sys.argv[2]
    src_dec = int(sys.argv[3])
    amount = sys.argv[4]
    dst_chain = int(sys.argv[5])
    dst_token = sys.argv[6]
    dst_dec = int(sys.argv[7])

    result = api_post(SYMBIOSIS_API, {
        "tokenAmountIn": {
            "chainId": src_chain,
            "address": src_token,
            "decimals": src_dec,
            "amount": to_smallest_units(amount, src_dec),
        },
        "tokenOut": {
            "chainId": dst_chain,
            "address": dst_token,
            "decimals": dst_dec,
        },
        "from": FAKE_ADDR,
        "to": FAKE_ADDR,
        "slippage": 200,
        "partnerId": PARTNER_ID,
    })

    if "tx" not in result:
        msg = result.get("message", result.get("error", json.dumps(result)))
        print(f"ERROR: {msg}", file=sys.stderr)
        sys.exit(1)

    out = result.get("tokenAmountOut", {})
    out_human = format_units(out.get("amount", "0"), out.get("decimals", dst_dec))

    fee = result.get("fee", {})
    fee_human = float(format_units(fee.get("amount", "0"), fee.get("decimals", 6)))
    fee_usd = fee_human * fee.get("priceUsd", 1)

    est = result.get("estimatedTime", "?")

    print(f"{amount} -> {out_human} (chain {src_chain} -> {dst_chain})")
    print(f"Fee: ~${fee_usd:.4f}")
    print(f"Estimated time: {est}s")


if __name__ == "__main__":
    main()
