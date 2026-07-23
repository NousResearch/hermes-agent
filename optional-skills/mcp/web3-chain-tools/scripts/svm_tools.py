"""Solana JSON-RPC helpers (read + pre-signed send)."""

from __future__ import annotations

from typing import Any, Dict

from solana.rpc.api import Client
from solana.rpc.types import TxOpts
from solders.pubkey import Pubkey


def make_solana(rpc_url: str) -> Client:
    if not rpc_url:
        raise ValueError("SOLANA_RPC_URL is not set")
    return Client(rpc_url)


def svm_query_balance(client: Client, address: str) -> Dict[str, Any]:
    pk = Pubkey.from_string(address)
    resp = client.get_balance(pk)
    return {"address": address, "lamports": int(resp.value)}


def svm_send_raw(client: Client, raw_b64: str) -> Dict[str, Any]:
    import base64

    raw = base64.b64decode(raw_b64)
    opts = TxOpts(skip_preflight=False, preflight_commitment="processed")
    resp = client.send_raw_transaction(raw, opts=opts)
    sig = resp.value
    if sig is None:
        raise RuntimeError("send_raw_transaction returned no signature")
    return {"signature": str(sig)}


def svm_simulate(client: Client, raw_b64: str) -> Dict[str, Any]:
    import base64

    from solders.transaction import VersionedTransaction

    raw = base64.b64decode(raw_b64)
    vtx = VersionedTransaction.from_bytes(raw)
    resp = client.simulate_transaction(vtx)
    val = resp.value
    return {"err": getattr(val, "err", None), "logs": getattr(val, "logs", None) or []}
