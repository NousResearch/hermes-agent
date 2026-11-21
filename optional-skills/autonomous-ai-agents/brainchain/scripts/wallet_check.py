#!/usr/bin/env python3
"""Check that a wallet can pay for BrainChain tasks before the agent posts one.

Reads public chain state only: the token balance, and for USDG on Robinhood
Chain also the ETH for the one-time approval and the Permit2 allowance. Takes
the wallet's address, never its key.

    python wallet_check.py 0xADDRESS [--network mainnet|testnet] [--pay-on robinhood|base] [--tasks N]

Exits 0 when the wallet can pay for N tasks (default 1), 1 when it cannot,
2 on bad arguments or an unreachable RPC. Prints one JSON object.
"""
import argparse
import json
import re
import sys
import urllib.request

PERMIT2 = "0x000000000022D473030F116dDEE9F6B43aC78BA3"
PRICE_UNITS = 1_000_000  # 1 USDG or 1 USDC, 6 decimals

RAILS = {
    ("mainnet", "robinhood"): {"chain": "Robinhood Chain", "token": "USDG", "rpc": "https://rpc.mainnet.chain.robinhood.com",
                               "asset": "0x5fc5360D0400a0Fd4f2af552ADD042D716F1d168", "permit2": True},
    ("mainnet", "base"): {"chain": "Base", "token": "USDC", "rpc": "https://mainnet.base.org",
                          "asset": "0x833589fCD6eDb6E08f4c7C32D4f71b54bdA02913", "permit2": False},
    ("testnet", "robinhood"): {"chain": "Robinhood Chain testnet", "token": "USDG", "rpc": "https://rpc.testnet.chain.robinhood.com",
                               "asset": "0x1eC185467cA2D5d1c24f0EdCB91301Bca4d0a787", "permit2": True},
    ("testnet", "base"): {"chain": "Base Sepolia", "token": "USDC", "rpc": "https://sepolia.base.org",
                          "asset": "0x036CbD53842c5426634e7929541eC2318f3dCF7e", "permit2": False},
}

BALANCE_OF = "0x70a08231"
ALLOWANCE = "0xdd62ed3e"


def _word(address: str) -> str:
    return address.lower().removeprefix("0x").rjust(64, "0")


def rpc(url: str, method: str, params: list, timeout: float = 15.0):
    body = json.dumps({"jsonrpc": "2.0", "id": 1, "method": method, "params": params}).encode()
    # Some public RPCs refuse urllib's default user agent.
    request = urllib.request.Request(url, data=body, headers={"content-type": "application/json", "user-agent": "hermes-brainchain-skill/0.1"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        reply = json.load(response)
    if "error" in reply:
        raise RuntimeError(reply["error"].get("message", "RPC error"))
    return reply["result"]


def check(address: str, network: str, pay_on: str, tasks: int, call=None) -> dict:
    call = call or rpc
    rail = RAILS[(network, pay_on)]
    need = tasks * PRICE_UNITS
    balance = int(call(rail["rpc"], "eth_call", [{"to": rail["asset"], "data": BALANCE_OF + _word(address)}, "latest"]), 16)
    report = {"address": address, "chain": rail["chain"], "token": rail["token"],
              "balance": balance / 1e6, "needed": need / 1e6, "problems": []}
    if balance < need:
        report["problems"].append(f"Holds {balance / 1e6:g} {rail['token']} on {rail['chain']}; {tasks} task(s) need {need / 1e6:g}.")
    if rail["permit2"]:
        allowance = int(call(rail["rpc"], "eth_call", [{"to": rail["asset"], "data": ALLOWANCE + _word(address) + _word(PERMIT2)}, "latest"]), 16)
        report["permit2_allowance"] = "unlimited" if allowance >= 2**255 else allowance / 1e6
        if allowance < need:
            eth = int(call(rail["rpc"], "eth_getBalance", [address, "latest"]), 16)
            report["eth"] = eth / 1e18
            report["problems"].append("Permit2 may not spend enough USDG yet: run the approve command once (see Prerequisites).")
            if eth == 0:
                report["problems"].append(f"Has no ETH on {rail['chain']} to pay for that approval.")
    report["ready"] = not report["problems"]
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Check that a wallet can pay for BrainChain tasks.")
    parser.add_argument("address")
    parser.add_argument("--network", choices=["mainnet", "testnet"], default="mainnet")
    parser.add_argument("--pay-on", choices=["robinhood", "base"], default="robinhood")
    parser.add_argument("--tasks", type=int, default=1)
    args = parser.parse_args(argv)
    if not re.fullmatch(r"0x[0-9a-fA-F]{40}", args.address):
        print(json.dumps({"error": "Pass the wallet's address (0x and 40 hex characters), never its key."}))
        return 2
    if args.tasks < 1:
        print(json.dumps({"error": "--tasks must be 1 or more."}))
        return 2
    try:
        report = check(args.address, args.network, args.pay_on, args.tasks)
    except (OSError, RuntimeError, ValueError) as error:
        print(json.dumps({"error": f"Could not read the chain: {error}"}))
        return 2
    print(json.dumps(report))
    return 0 if report["ready"] else 1


if __name__ == "__main__":
    sys.exit(main())
