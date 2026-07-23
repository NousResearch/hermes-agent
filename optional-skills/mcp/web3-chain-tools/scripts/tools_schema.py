"""MCP ``Tool`` metadata for the Web3 chain stdio server."""

from __future__ import annotations

from typing import List

from mcp import types


def build_tool_list() -> List[types.Tool]:
    return [
        types.Tool(
            name="query_balance",
            description="Native balance for an EVM checksum address or Solana pubkey.",
            inputSchema={
                "type": "object",
                "required": ["chain", "address"],
                "properties": {
                    "chain": {"type": "string", "enum": ["evm", "solana"]},
                    "address": {"type": "string"},
                },
            },
        ),
        types.Tool(
            name="evm_estimate_gas",
            description="Estimate gas for a partial tx (HTTP JSON-RPC).",
            inputSchema={
                "type": "object",
                "required": ["from"],
                "properties": {
                    "from": {"type": "string"},
                    "to": {"type": "string"},
                    "data": {"type": "string", "default": "0x"},
                    "value_wei": {"type": "string", "default": "0"},
                },
            },
        ),
        types.Tool(
            name="evm_call",
            description="Read-only eth_call.",
            inputSchema={
                "type": "object",
                "required": ["to", "data"],
                "properties": {
                    "to": {"type": "string"},
                    "data": {"type": "string"},
                    "from": {"type": "string"},
                },
            },
        ),
        types.Tool(
            name="send_raw_transaction",
            description="Broadcast a pre-signed EVM hex raw tx or Solana base64 wire tx.",
            inputSchema={
                "type": "object",
                "required": ["chain", "raw"],
                "properties": {
                    "chain": {"type": "string", "enum": ["evm", "solana"]},
                    "raw": {"type": "string"},
                },
            },
        ),
        types.Tool(
            name="monitor_event",
            description="Run eth_getLogs once and enqueue results for dequeue_events.",
            inputSchema={
                "type": "object",
                "properties": {
                    "address": {"type": "string"},
                    "from_block": {"type": "string", "default": "latest"},
                    "to_block": {"type": "string", "default": "latest"},
                    "topics": {"type": "array"},
                },
            },
        ),
        types.Tool(
            name="dequeue_events",
            description="Fetch pending queued monitor rows (SQLite under ~/.hermes/web3-mcp/).",
            inputSchema={
                "type": "object",
                "properties": {"limit": {"type": "integer", "default": 20}},
            },
        ),
        types.Tool(
            name="evm_sign_and_send",
            description="DEV ONLY: sign an unsigned tx dict via env key (WEB3_ALLOW_INSECURE_ENV_SIGNER=1).",
            inputSchema={
                "type": "object",
                "required": ["tx"],
                "properties": {"tx": {"type": "object"}},
            },
        ),
    ]
