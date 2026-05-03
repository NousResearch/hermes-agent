#!/usr/bin/env python3
"""Symbiosis cross-chain swap via Bankr Submit API.

Usage: ./symbiosis-swap.sh <src_chain> <src_token> <src_decimals> <amount> <dst_chain> <dst_token> <dst_decimals> [slippage]

Example: ./symbiosis-swap.sh 8453 0x833589fCD6eDb6E08f4c7C32D4f71b54bdA02913 6 2 137 0x3c499c542cEF5E3811e1192ce70d8cC03d5c3359 6
         (2 USDC from Base to Polygon)
"""

import json
import os
import sys
import urllib.request

SYMBIOSIS_API = "https://api-v2.symbiosis.finance/crosschain/v1/swap"
BANKR_API = "https://api.bankr.bot"
PARTNER_ID = "bankr"
ZERO_ADDR = "0x0000000000000000000000000000000000000000"
MAX_UINT256 = "f" * 64


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


def api_post(url: str, payload: dict, headers: dict | None = None) -> dict:
    hdrs = {"Content-Type": "application/json", "User-Agent": "symbiosis-bankr-skill/1.0"}
    if headers:
        hdrs.update(headers)
    req = urllib.request.Request(url, data=json.dumps(payload).encode(), headers=hdrs)
    with urllib.request.urlopen(req, timeout=60) as resp:
        return json.loads(resp.read())


def api_get(url: str, headers: dict) -> dict:
    hdrs = {"User-Agent": "symbiosis-bankr-skill/1.0"}
    hdrs.update(headers)
    req = urllib.request.Request(url, headers=hdrs)
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read())


def load_bankr_key() -> str:
    config_path = os.environ.get("BANKR_CONFIG", os.path.expanduser("~/.bankr/config.json"))
    if not os.path.exists(config_path):
        print(f"ERROR: Bankr config not found at {config_path}", file=sys.stderr)
        sys.exit(1)
    with open(config_path) as f:
        return json.load(f)["apiKey"]


def get_wallet(bankr_key: str) -> str:
    result = api_get(
        f"{BANKR_API}/agent/balances?chains=base",
        {"X-API-Key": bankr_key},
    )
    return result["evmAddress"]


def bankr_submit(bankr_key: str, tx: dict, description: str) -> dict:
    return api_post(
        f"{BANKR_API}/agent/submit",
        {
            "transaction": tx,
            "description": description,
            "waitForConfirmation": True,
        },
        {"X-API-Key": bankr_key},
    )


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
    slippage = int(sys.argv[8]) if len(sys.argv) > 8 else 200

    # --- Setup ---
    bankr_key = load_bankr_key()
    wallet = get_wallet(bankr_key)
    amount_wei = to_smallest_units(amount, src_dec)
    print(f"Wallet: {wallet}")
    print(f"Amount in smallest units: {amount_wei}")

    # --- Step 1: Get quote + calldata from Symbiosis ---
    print("\n=== Getting Symbiosis quote ===")
    result = api_post(SYMBIOSIS_API, {
        "tokenAmountIn": {
            "chainId": src_chain,
            "address": src_token,
            "decimals": src_dec,
            "amount": amount_wei,
        },
        "tokenOut": {
            "chainId": dst_chain,
            "address": dst_token,
            "decimals": dst_dec,
        },
        "from": wallet,
        "to": wallet,
        "slippage": slippage,
        "partnerId": PARTNER_ID,
    })

    if "tx" not in result:
        msg = result.get("message", result.get("error", json.dumps(result)))
        print(f"ERROR from Symbiosis API: {msg}", file=sys.stderr)
        sys.exit(1)

    tx = result["tx"]
    approve_to = result.get("approveTo", "")
    out = result.get("tokenAmountOut", {})
    fee = result.get("fee", {})

    out_human = format_units(out.get("amount", "0"), out.get("decimals", dst_dec))
    fee_human = float(format_units(fee.get("amount", "0"), fee.get("decimals", 6)))
    fee_usd = fee_human * fee.get("priceUsd", 1)
    est_time = result.get("estimatedTime", "?")

    print(f"Quote: {amount} -> {out_human}")
    print(f"Fee: ~${fee_usd:.4f}")
    print(f"Estimated time: {est_time}s")
    print(f"Symbiosis router: {tx['to']}")

    # --- Step 2: Approve (if needed) ---
    if approve_to and src_token.lower() != ZERO_ADDR:
        print("\n=== Approving token for Symbiosis ===")
        padded = approve_to[2:].lower().zfill(64)
        approve_data = f"0x095ea7b3{padded}{MAX_UINT256}"

        approve_result = bankr_submit(bankr_key, {
            "to": src_token,
            "chainId": src_chain,
            "value": "0",
            "data": approve_data,
        }, "Approve token for Symbiosis cross-chain swap")

        if not approve_result.get("success"):
            print(f"ERROR: Approve failed: {json.dumps(approve_result)}", file=sys.stderr)
            sys.exit(1)
        print(f"Approve tx: {approve_result['transactionHash']}")

    # --- Step 3: Submit swap ---
    print("\n=== Submitting Symbiosis swap ===")
    swap_result = bankr_submit(bankr_key, {
        "to": tx["to"],
        "chainId": src_chain,
        "value": tx.get("value", "0"),
        "data": tx["data"],
    }, "Symbiosis cross-chain swap")

    if not swap_result.get("success"):
        print(f"ERROR: Swap failed: {json.dumps(swap_result)}", file=sys.stderr)
        sys.exit(1)

    swap_hash = swap_result["transactionHash"]
    print(f"Swap tx: {swap_hash}")
    print(f"Status: {swap_result.get('status', 'unknown')}")
    print(f"\n=== SUCCESS ===")
    print(f"Swapped {amount} on chain {src_chain} -> ~{out_human} on chain {dst_chain}")
    print(f"Fee: ~${fee_usd:.4f} | Estimated arrival: {est_time}s")
    print(f"Track: https://explorer.symbiosis.finance/transactions/{src_chain}/{swap_hash}")


if __name__ == "__main__":
    main()
