---
name: hermes-agentic-trader
description: Paper-safe Base market analysis plus direct on-chain Uniswap quoting and unsigned EIP-5792 call planning.
version: 0.7.0-repair.3
author: ivan09069
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [Trading, DeFi, Base, EVM, MCP, EIP-5792]
    related_skills: [evm]
---

# Hermes Agentic Trader — Repair Track

This repair track separates market discovery, quoting, and wallet execution so
Hermes never needs a raw private key.

## Default surface

The curated defi-trading MCP entry is market-data-only and uses CoinGecko tools.
The exact pinned 2.1.3 package's aggregator-backed quote, portfolio, gasless, and
write tools are excluded from the default tool surface.

Enable the safety plugin:

    hermes plugins enable hermes-agentic-trader

Keep trader.mode set to paper unless a separately reviewed live policy is being
tested. The legacy execute_swap and submit_gasless_swap MCP tools remain blocked
by the plugin even if they are manually enabled.

## Base market scan

For Base trending pools use get_trending_pools_by_network:

- network: base
- include: base_token,quote_token,dex
- duration: 24h

Do not use a pool contract address as the token-to-buy address. Resolve
relationships.base_token and relationships.quote_token to their token resources.

## Direct Uniswap quote

Use trader_uniswap_quote with:

- token_in: ERC-20 contract on Base
- token_out: a different ERC-20 contract on Base
- amount_in: integer base units as a decimal string
- fee: one of 100, 500, 3000, 10000

The tool first verifies that the configured EVM RPC reports chainId 8453, then
uses eth_call against Uniswap QuoterV2. The deployment addresses are pinned to
Uniswap/contracts commit e34ba78b05663c8c342cce69b0039067ace75691.

The returned quote_id is short-lived and binds the unsigned plan to the exact
token pair, amount, fee, and quoted output.

## Build unsigned wallet calls

Use trader_uniswap_build_calls with the fresh quote_id, wallet owner address,
and slippage_bps.

The result contains:

- wallet_getCapabilities preflight for Base;
- wallet_sendCalls version 2.0.0;
- chainId 0x2105;
- atomicRequired true;
- an exact-amount ERC-20 approve call;
- an exactInputSingle call to the pinned Base SwapRouter02;
- amountOutMinimum computed locally from the cached quote.

Hermes does not send wallet_sendCalls. It does not sign. It does not accept a
private key. A wallet that cannot satisfy atomicRequired must reject the bundle.

Always inspect the wallet simulation before signing.

## Security boundary

The pinned defi-trading-mcp@2.1.3 aggregator uses plaintext HTTP and is not a
trusted execution backend for this repair. Do not enable its quote, portfolio,
gasless, or write tools for live funds.

The direct planner supports ERC-20 to ERC-20 single-pool Uniswap V3 swaps only.
Native ETH wrapping, multi-hop routing, arbitrary routers, Permit2 signatures,
and automatic transaction submission are intentionally out of scope.

## Output

For market analysis end with:

PAPER MODE — no transaction submitted.

For a generated wallet request state explicitly:

UNSIGNED PLAN — wallet simulation and user approval required.
