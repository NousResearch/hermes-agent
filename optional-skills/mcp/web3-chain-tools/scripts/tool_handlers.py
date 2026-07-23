"""Tool dispatch and shared runtime for ``web3_mcp_server``."""

from __future__ import annotations

import json
import os
from typing import Any, Dict

from mcp import types

from approval_hooks import approval_gateway_allow, sync_preview_redact
from event_queue import EventQueue
from rate_limit import ToolRateLimiter


def reject_secrets_in_args(arguments: Dict[str, Any]) -> None:
    bad = {"private_key", "privkey", "secret_key", "mnemonic", "seed"}
    for k in arguments:
        if k.lower() in bad:
            raise ValueError("Private signing material must not be passed as tool arguments")


class Runtime:
    def __init__(self) -> None:
        self._limiter = ToolRateLimiter()
        self._queue = EventQueue()
        self._w3 = None
        self._sol = None

    def w3(self):
        if self._w3 is None:
            from evm_tools import make_web3

            self._w3 = make_web3(os.environ.get("ETHEREUM_RPC_URL", "").strip())
        return self._w3

    def sol(self):
        if self._sol is None:
            from svm_tools import make_solana

            self._sol = make_solana(os.environ.get("SOLANA_RPC_URL", "").strip())
        return self._sol

    @property
    def queue(self) -> EventQueue:
        return self._queue


RT = Runtime()


async def handle_tool(tool_name: str, arguments: Dict[str, Any]) -> Any:
    reject_secrets_in_args(arguments)
    if not RT._limiter.allow(tool_name):
        return [types.TextContent(type="text", text=json.dumps({"error": "rate_limited"}))]
    if tool_name == "query_balance":
        chain = arguments["chain"]
        addr = arguments["address"]
        if chain == "evm":
            from evm_tools import evm_query_balance

            out = evm_query_balance(RT.w3(), addr)
        else:
            from svm_tools import svm_query_balance

            out = svm_query_balance(RT.sol(), addr)
        return [types.TextContent(type="text", text=json.dumps(out))]
    if tool_name == "evm_estimate_gas":
        from evm_tools import evm_estimate_gas

        out = evm_estimate_gas(
            RT.w3(),
            from_addr=arguments["from"],
            to_addr=arguments.get("to"),
            data=arguments.get("data") or "0x",
            value_wei=str(arguments.get("value_wei") or "0"),
        )
        return [types.TextContent(type="text", text=json.dumps(out))]
    if tool_name == "evm_call":
        from evm_tools import evm_call

        out = evm_call(
            RT.w3(),
            to_addr=arguments["to"],
            data=arguments["data"],
            from_addr=arguments.get("from"),
        )
        return [types.TextContent(type="text", text=json.dumps(out))]
    if tool_name == "send_raw_transaction":
        preview = sync_preview_redact({"chain": arguments["chain"], "raw": arguments.get("raw", "")})
        if not await approval_gateway_allow(tool_name="send_raw_transaction", preview=preview):
            return [types.TextContent(type="text", text=json.dumps({"error": "approval_denied"}))]
        if arguments["chain"] == "evm":
            from evm_tools import evm_send_raw

            out = evm_send_raw(RT.w3(), arguments["raw"])
        else:
            from svm_tools import svm_send_raw

            out = svm_send_raw(RT.sol(), arguments["raw"])
        return [types.TextContent(type="text", text=json.dumps(out))]
    if tool_name == "monitor_event":
        from evm_tools import evm_get_logs

        logs = evm_get_logs(
            RT.w3(),
            address=arguments.get("address"),
            from_block=arguments.get("from_block") or "latest",
            to_block=arguments.get("to_block") or "latest",
            topics=arguments.get("topics"),
        )
        eid = RT.queue.enqueue("evm", {"logs": logs, "count": len(logs)})
        return [types.TextContent(type="text", text=json.dumps({"enqueued_id": eid, "count": len(logs)}))]
    if tool_name == "dequeue_events":
        lim = int(arguments.get("limit") or 20)
        rows = RT.queue.dequeue(lim)
        return [types.TextContent(type="text", text=json.dumps({"events": rows}))]
    if tool_name == "evm_sign_and_send":
        if not await approval_gateway_allow(
            tool_name="evm_sign_and_send",
            preview=sync_preview_redact({"tx_keys": list(arguments.get("tx", {}).keys())}),
        ):
            return [types.TextContent(type="text", text=json.dumps({"error": "approval_denied"}))]
        env_name = os.environ.get("WEB3_EVM_PRIVATE_KEY_ENV", "WEB3_EVM_PRIVATE_KEY").strip()
        from signer_base import EnvPrivateKeySignerEVM
        from evm_tools import evm_send_raw

        signer = EnvPrivateKeySignerEVM(env_name)
        raw = signer.sign_evm_transaction(arguments["tx"])
        out = evm_send_raw(RT.w3(), raw)
        return [types.TextContent(type="text", text=json.dumps(out))]
    return [types.TextContent(type="text", text=json.dumps({"error": "unknown_tool"}))]
