"""Minimal MineBean client helper.

Ships at agent.minebean.com/.well-known/skills/mine-bean/scripts/minebean_client.py
as a fallback path for Hermes agents that prefer a single-file install over the
full pip package. The pip package (hermes-mine-bean) is the recommended path
because it ships the full 7-tool plugin, MCP server, and console script.

This file is intentionally tiny. It just shows how to read the current MineBean
round state directly from Base mainnet without any dependencies beyond stdlib
+ requests (which Hermes already ships with).

Usage:
    python minebean_client.py
"""
from __future__ import annotations

import json
import sys
import urllib.request


BASE_RPC_URL = "https://mainnet.base.org"
GRIDMINING_ADDRESS = "0x9632495bDb93FD6B0740Ab69cc6c71C9c01da4f0"

# Function selector for `currentRoundId()`, computed once and hardcoded so we
# don't need eth-abi/eth-hash to derive it at runtime.
CURRENT_ROUND_ID_SELECTOR = "0x4cf088d9"


def _rpc(method: str, params: list, *, timeout: int = 30) -> dict:
    """Bare stdlib JSON-RPC call. User-Agent header is required by Base's RPC."""
    body = json.dumps({"jsonrpc": "2.0", "method": method, "params": params, "id": 1}).encode()
    req = urllib.request.Request(
        BASE_RPC_URL,
        data=body,
        headers={
            "Content-Type": "application/json",
            "User-Agent": "minebean-client/0.2.0",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode())


def current_round_id() -> int:
    """Read the current round id from GridMining."""
    result = _rpc(
        "eth_call",
        [{"to": GRIDMINING_ADDRESS, "data": CURRENT_ROUND_ID_SELECTOR}, "latest"],
    )
    if "error" in result:
        raise RuntimeError(f"RPC error: {result['error']}")
    return int(result["result"], 16)


def main() -> int:
    try:
        round_id = current_round_id()
        print(json.dumps({"ok": True, "current_round_id": round_id}))
        return 0
    except Exception as exc:
        print(json.dumps({"ok": False, "error": f"{type(exc).__name__}: {exc}"}))
        return 1


if __name__ == "__main__":
    sys.exit(main())
