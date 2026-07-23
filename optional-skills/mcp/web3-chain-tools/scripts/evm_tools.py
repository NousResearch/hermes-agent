"""EVM helpers (HTTP JSON-RPC) — errors must not echo secrets."""

from __future__ import annotations

import json
from typing import Any, Dict, Optional

from web3 import Web3
from web3.types import TxParams


def make_web3(rpc_url: str) -> Web3:
    if not rpc_url:
        raise ValueError("ETHEREUM_RPC_URL is not set")
    w3 = Web3(Web3.HTTPProvider(rpc_url, request_kwargs={"timeout": 30}))
    if not w3.is_connected():
        raise ConnectionError("EVM RPC not reachable")
    return w3


def evm_query_balance(w3: Web3, address: str) -> Dict[str, Any]:
    if not Web3.is_address(address):
        raise ValueError("invalid EVM address")
    checksum = Web3.to_checksum_address(address)
    wei = w3.eth.get_balance(checksum)
    return {"address": checksum, "wei": str(wei), "ether": str(Web3.from_wei(wei, "ether"))}


def evm_estimate_gas(
    w3: Web3,
    *,
    from_addr: str,
    to_addr: Optional[str] = None,
    data: str = "0x",
    value_wei: str = "0",
) -> Dict[str, Any]:
    tx: TxParams = {
        "from": Web3.to_checksum_address(from_addr),
        "data": data if data.startswith("0x") else "0x" + data,
        "value": int(value_wei),
    }
    if to_addr:
        tx["to"] = Web3.to_checksum_address(to_addr)
    gas = w3.eth.estimate_gas(tx)
    gp = w3.eth.gas_price
    return {"gas": int(gas), "gasPrice": int(gp), "feeWeiApprox": int(gas) * int(gp)}


def evm_call(
    w3: Web3,
    *,
    to_addr: str,
    data: str,
    from_addr: Optional[str] = None,
) -> Dict[str, Any]:
    call_obj: Dict[str, Any] = {
        "to": Web3.to_checksum_address(to_addr),
        "data": data if data.startswith("0x") else "0x" + data,
    }
    if from_addr:
        call_obj["from"] = Web3.to_checksum_address(from_addr)
    raw = w3.eth.call(call_obj)
    return {"result": raw.hex()}


def evm_send_raw(w3: Web3, raw_hex: str) -> Dict[str, Any]:
    if not raw_hex.startswith("0x"):
        raw_hex = "0x" + raw_hex
    h = w3.eth.send_raw_transaction(bytes.fromhex(raw_hex[2:]))
    return {"txHash": h.hex()}


def evm_get_logs(
    w3: Web3,
    *,
    address: Optional[str],
    from_block: str = "latest",
    to_block: str = "latest",
    topics: Optional[list] = None,
) -> list:
    filt: Dict[str, Any] = {"fromBlock": from_block, "toBlock": to_block}
    if address:
        filt["address"] = Web3.to_checksum_address(address)
    if topics is not None:
        filt["topics"] = topics
    logs = w3.eth.get_logs(filt)
    return json.loads(Web3.to_json(logs))
