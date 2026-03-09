#!/usr/bin/env python3
"""
Solana Blockchain CLI Tool for Hermes Agent
--------------------------------------------
Queries the Solana JSON-RPC API using only Python standard library.
No external packages required.

Usage:
  python3 solana_client.py stats
  python3 solana_client.py wallet   <address>
  python3 solana_client.py tx       <signature>
  python3 solana_client.py token    <mint_address>
  python3 solana_client.py activity <address> [--limit N]
  python3 solana_client.py nft      <address>
  python3 solana_client.py whales   [--min-sol N]

Environment:
  SOLANA_RPC_URL  Override the default RPC endpoint (default: mainnet-beta public)
"""

import argparse
import json
import os
import sys
import urllib.request
import urllib.error
from typing import Any

RPC_URL = os.environ.get(
    "SOLANA_RPC_URL",
    "https://api.mainnet-beta.solana.com"
)

LAMPORTS_PER_SOL = 1_000_000_000


# ---------------------------------------------------------------------------
# RPC helpers
# ---------------------------------------------------------------------------

def rpc(method: str, params: list = None) -> Any:
    """Send a JSON-RPC request and return the result field."""
    payload = json.dumps({
        "jsonrpc": "2.0",
        "id": 1,
        "method": method,
        "params": params or [],
    }).encode()

    req = urllib.request.Request(
        RPC_URL,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            body = json.load(resp)
    except urllib.error.URLError as exc:
        sys.exit(f"RPC connection error: {exc}")

    if "error" in body:
        sys.exit(f"RPC error: {body['error']}")
    return body.get("result")


def rpc_batch(calls: list) -> list:
    """Send a batch of JSON-RPC requests."""
    payload = json.dumps([
        {"jsonrpc": "2.0", "id": i, "method": c["method"], "params": c.get("params", [])}
        for i, c in enumerate(calls)
    ]).encode()
    req = urllib.request.Request(
        RPC_URL,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            return json.load(resp)
    except urllib.error.URLError as exc:
        sys.exit(f"RPC batch error: {exc}")


def lamports_to_sol(lamports: int) -> float:
    return lamports / LAMPORTS_PER_SOL


def print_json(obj: Any) -> None:
    print(json.dumps(obj, indent=2))


# ---------------------------------------------------------------------------
# 1. Network Stats
# ---------------------------------------------------------------------------

def cmd_stats(_args):
    """Live Solana network: slot, epoch, TPS, supply, version."""
    results = rpc_batch([
        {"method": "getSlot"},
        {"method": "getEpochInfo"},
        {"method": "getRecentPerformanceSamples", "params": [1]},
        {"method": "getSupply"},
        {"method": "getVersion"},
    ])

    by_id = {r["id"]: r.get("result") for r in results}

    slot         = by_id[0]
    epoch_info   = by_id[1]
    perf_samples = by_id[2]
    supply       = by_id[3]
    version      = by_id[4]

    tps = None
    if perf_samples:
        s = perf_samples[0]
        tps = round(s["numTransactions"] / s["samplePeriodSecs"], 1)

    total_supply = lamports_to_sol(supply["value"]["total"])      if supply else None
    circ_supply  = lamports_to_sol(supply["value"]["circulating"]) if supply else None

    print_json({
        "slot":                     slot,
        "epoch":                    epoch_info.get("epoch")     if epoch_info else None,
        "slot_in_epoch":            epoch_info.get("slotIndex") if epoch_info else None,
        "tps":                      tps,
        "total_supply_SOL":         round(total_supply, 2) if total_supply else None,
        "circulating_supply_SOL":   round(circ_supply, 2)  if circ_supply  else None,
        "validator_version":        version.get("solana-core") if version else None,
    })


# ---------------------------------------------------------------------------
# 2. Wallet Info
# ---------------------------------------------------------------------------

def cmd_wallet(args):
    """SOL balance + SPL token accounts for an address."""
    address = args.address

    balance_result = rpc("getBalance", [address])
    sol_balance = lamports_to_sol(balance_result["value"])

    token_result = rpc("getTokenAccountsByOwner", [
        address,
        {"programId": "TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA"},
        {"encoding": "jsonParsed"},
    ])

    tokens = []
    for acct in (token_result.get("value") or []):
        info = acct["account"]["data"]["parsed"]["info"]
        token_amount = info["tokenAmount"]
        amount = float(token_amount["uiAmountString"] or 0)
        if amount > 0:
            tokens.append({
                "mint":     info["mint"],
                "amount":   amount,
                "decimals": token_amount["decimals"],
            })

    print_json({
        "address":     address,
        "balance_SOL": round(sol_balance, 9),
        "spl_tokens":  tokens,
    })


# ---------------------------------------------------------------------------
# 3. Transaction Details
# ---------------------------------------------------------------------------

def cmd_tx(args):
    """Full transaction details by signature."""
    result = rpc("getTransaction", [
        args.signature,
        {"encoding": "jsonParsed", "maxSupportedTransactionVersion": 0},
    ])

    if result is None:
        sys.exit("Transaction not found (may be too old for public RPC history).")

    meta         = result.get("meta", {}) or {}
    msg          = result.get("transaction", {}).get("message", {})
    account_keys = msg.get("accountKeys", [])

    pre  = meta.get("preBalances",  [])
    post = meta.get("postBalances", [])

    balance_changes = []
    for i, key in enumerate(account_keys):
        acct_key = key["pubkey"] if isinstance(key, dict) else key
        if i < len(pre) and i < len(post):
            change = lamports_to_sol(post[i] - pre[i])
            if change != 0:
                balance_changes.append({"account": acct_key, "change_SOL": round(change, 9)})

    programs = []
    for ix in msg.get("instructions", []):
        prog = ix.get("programId")
        if prog is None and "programIdIndex" in ix:
            k = account_keys[ix["programIdIndex"]]
            prog = k["pubkey"] if isinstance(k, dict) else k
        if prog:
            programs.append(prog)

    print_json({
        "signature":        args.signature,
        "slot":             result.get("slot"),
        "block_time":       result.get("blockTime"),
        "fee_SOL":          lamports_to_sol(meta.get("fee", 0)),
        "status":           "success" if meta.get("err") is None else "failed",
        "balance_changes":  balance_changes,
        "programs_invoked": list(dict.fromkeys(programs)),
    })


# ---------------------------------------------------------------------------
# 4. Token Info
# ---------------------------------------------------------------------------

def cmd_token(args):
    """SPL token metadata, supply, decimals, top holders."""
    mint = args.mint

    mint_info = rpc("getAccountInfo", [mint, {"encoding": "jsonParsed"}])
    if mint_info is None or mint_info.get("value") is None:
        sys.exit("Mint account not found.")

    parsed          = mint_info["value"]["data"]["parsed"]["info"]
    decimals        = parsed.get("decimals", 0)
    supply_raw      = int(parsed.get("supply", 0))
    supply_human    = supply_raw / (10 ** decimals)
    mint_authority  = parsed.get("mintAuthority")
    freeze_authority = parsed.get("freezeAuthority")

    largest = rpc("getTokenLargestAccounts", [mint])
    holders = []
    for acct in (largest.get("value") or [])[:5]:
        amount = float(acct.get("uiAmountString") or 0)
        pct = round((amount / supply_human * 100), 4) if supply_human > 0 else 0
        holders.append({
            "account": acct["address"],
            "amount":  amount,
            "percent": pct,
        })

    print_json({
        "mint":             mint,
        "decimals":         decimals,
        "supply":           round(supply_human, decimals),
        "mint_authority":   mint_authority,
        "freeze_authority": freeze_authority,
        "top_5_holders":    holders,
    })


# ---------------------------------------------------------------------------
# 5. Recent Activity
# ---------------------------------------------------------------------------

def cmd_activity(args):
    """Recent transaction signatures for an address."""
    limit  = min(args.limit, 25)
    result = rpc("getSignaturesForAddress", [args.address, {"limit": limit}])

    txs = [
        {
            "signature": item["signature"],
            "slot":       item.get("slot"),
            "block_time": item.get("blockTime"),
            "err":        item.get("err"),
        }
        for item in (result or [])
    ]

    print_json({"address": args.address, "transactions": txs})


# ---------------------------------------------------------------------------
# 6. NFT Portfolio
# ---------------------------------------------------------------------------

def cmd_nft(args):
    """NFTs owned by a wallet (amount=1 && decimals=0 heuristic)."""
    result = rpc("getTokenAccountsByOwner", [
        args.address,
        {"programId": "TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA"},
        {"encoding": "jsonParsed"},
    ])

    nfts = [
        acct["account"]["data"]["parsed"]["info"]["mint"]
        for acct in (result.get("value") or [])
        if acct["account"]["data"]["parsed"]["info"]["tokenAmount"]["decimals"] == 0
        and int(acct["account"]["data"]["parsed"]["info"]["tokenAmount"]["amount"]) == 1
    ]

    print_json({
        "address":   args.address,
        "nft_count": len(nfts),
        "nfts":      nfts,
        "note":      "Heuristic only. Compressed NFTs (cNFTs) are not detected.",
    })


# ---------------------------------------------------------------------------
# 7. Whale Detector
# ---------------------------------------------------------------------------

def cmd_whales(args):
    """Scan the latest block for large SOL transfers."""
    min_lamports = int(args.min_sol * LAMPORTS_PER_SOL)

    slot  = rpc("getSlot")
    block = rpc("getBlock", [
        slot,
        {
            "encoding": "jsonParsed",
            "transactionDetails": "full",
            "maxSupportedTransactionVersion": 0,
            "rewards": False,
        },
    ])

    if block is None:
        sys.exit("Could not retrieve latest block.")

    whales = []
    for tx in (block.get("transactions") or []):
        meta = tx.get("meta", {}) or {}
        if meta.get("err") is not None:
            continue

        msg          = tx["transaction"].get("message", {})
        account_keys = msg.get("accountKeys", [])
        pre          = meta.get("preBalances",  [])
        post         = meta.get("postBalances", [])

        for i in range(len(pre)):
            change = post[i] - pre[i]
            if change >= min_lamports:
                k        = account_keys[i]
                receiver = k["pubkey"] if isinstance(k, dict) else k
                sender   = None
                for j in range(len(pre)):
                    if pre[j] - post[j] >= min_lamports:
                        sk     = account_keys[j]
                        sender = sk["pubkey"] if isinstance(sk, dict) else sk
                        break
                whales.append({
                    "sender":     sender,
                    "receiver":   receiver,
                    "amount_SOL": round(lamports_to_sol(change), 4),
                })

    print_json({
        "slot":              slot,
        "min_threshold_SOL": args.min_sol,
        "large_transfers":   whales,
    })


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        prog="solana_client.py",
        description="Solana blockchain query tool for Hermes Agent",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("stats", help="Network stats: slot, epoch, TPS, supply, version")

    p_wallet = sub.add_parser("wallet", help="SOL balance + SPL tokens for an address")
    p_wallet.add_argument("address")

    p_tx = sub.add_parser("tx", help="Transaction details by signature")
    p_tx.add_argument("signature")

    p_token = sub.add_parser("token", help="SPL token metadata and top holders")
    p_token.add_argument("mint")

    p_activity = sub.add_parser("activity", help="Recent transactions for an address")
    p_activity.add_argument("address")
    p_activity.add_argument("--limit", type=int, default=10,
                            help="Number of transactions (max 25, default 10)")

    p_nft = sub.add_parser("nft", help="NFT portfolio for a wallet")
    p_nft.add_argument("address")

    p_whales = sub.add_parser("whales", help="Large SOL transfers in the latest block")
    p_whales.add_argument("--min-sol", type=float, default=1000.0,
                          help="Minimum SOL transfer size (default: 1000)")

    args = parser.parse_args()

    dispatch = {
        "stats":    cmd_stats,
        "wallet":   cmd_wallet,
        "tx":       cmd_tx,
        "token":    cmd_token,
        "activity": cmd_activity,
        "nft":      cmd_nft,
        "whales":   cmd_whales,
    }
    dispatch[args.command](args)


if __name__ == "__main__":
    main()
