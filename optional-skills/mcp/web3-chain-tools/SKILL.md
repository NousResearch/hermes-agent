---
name: web3-chain-tools
description: Run the bundled MCP stdio server for Ethereum (web3.py) and Solana (solana-py) — balances, eth_call, gas estimates, pre-signed broadcasts, log snapshots into a SQLite queue, optional WS listener. Use when wiring Hermes to chain RPC automation, DeFi monitoring, or wallet-adjacent tool flows via MCP.
version: 1.0.0
author: Hermes Agent
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [MCP, Web3, Ethereum, Solana, RPC, DeFi]
    related_skills: [fastmcp, native-mcp]
prerequisites:
  commands: [python3]
---

# Web3 chain MCP tools

Optional skill that ships a **stdio MCP server** under `scripts/web3_mcp_server.py`. Hermes discovers it like any other MCP process via `mcp_servers` in `~/.hermes/config.yaml`.

## Security defaults

- Never pass **private keys** or mnemonics as tool arguments — calls are rejected.
- `evm_sign_and_send` is **dev-only**: requires `WEB3_ALLOW_INSECURE_ENV_SIGNER=1` and a hex key in the env var named by `WEB3_EVM_PRIVATE_KEY_ENV` (default `WEB3_EVM_PRIVATE_KEY`).
- Optional HTTP approval: set `WEB3_APPROVAL_GATEWAY_URL` to POST `{tool, preview}` before any `send_raw_transaction` / `evm_sign_and_send`. Set `WEB3_APPROVAL_DENY_ON_ERROR=0` to allow when the gateway is unreachable (default denies).

## Install

```bash
pip install "hermes-agent[web3-mcp]"
```

Or from a Hermes git checkout root:

```bash
uv sync --extra web3-mcp
```

## Environment

| Variable | Purpose |
|----------|---------|
| `ETHEREUM_RPC_URL` | HTTP(S) JSON-RPC for EVM tools |
| `SOLANA_RPC_URL` | Solana JSON-RPC |
| `WEB3_MCP_QUEUE_DB` | Override SQLite path for `monitor_event` / `dequeue_events` |
| `WEB3_WS_URL` | Used only by `scripts/ws_listener.py` (optional long-running subscriber) |
| `WEB3_LOG_ADDRESS` | Contract address filter for the WS listener |

## Hermes config

See `references/config-snippet.yaml` for a ready `mcp_servers` block.

## Hardware / enclave signing

`HardwareWalletSigner` in `scripts/signer_base.py` is a **stub** — extend with your vendor SDK or a remote signer; keep keys off the agent host.

## Subagent automation

This bundle **does not** invoke Hermes subagents. Use `dequeue_events` from the agent, gateway cron, or an external worker after `monitor_event` / `ws_listener.py` enqueue rows.
