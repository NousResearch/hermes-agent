---
name: graph-advocate
description: Route onchain data questions to the right Graph subgraph.
version: 2.7.0
author: Paul Barba (PaulieB14), Graph Advocate
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [Blockchain, TheGraph, Subgraph, DeFi, OnchainData, x402, A2A, ERC-8004, Polymarket, Limitless, Hyperliquid, Aave, Uniswap]
    related_skills: [hyperliquid, evm]
---

# Graph Advocate Skill

Routing agent for The Graph Protocol. Send a plain-English question about any
blockchain and get back the right service plus a ready-to-run query, with no
manual subgraph hunting, no MCP install, and no API key. Free routing over
HTTPS; paid analytics endpoints settle in USDC on Base via x402. This skill is
instruction-only: it calls a hosted HTTP API and runs no code on your machine.

Live service: `https://graphadvocate.com`. ERC-8004 agent #734 (Arbitrum),
#41034 (Base). ENS `graphadvocate.eth`. Source:
`https://github.com/PaulieB14/graph-advocate`.

## When to Use

- User asks "which subgraph for protocol X on chain Y" or wants the right
  GraphQL query against a Graph subgraph
- User wants live token balances / swaps / holders / NFTs across EVM, Solana,
  or TON (routes to The Graph's Token API)
- User wants Polymarket or Hyperliquid trader intelligence (skill score, PnL,
  ghost-fill risk, vault evaluator)
- User wants cross-venue prediction-market spread (Polymarket vs Limitless,
  or Polymarket vs Kalshi) with arbitrage direction
- User wants natural-language Q&A over the x402 Base settlements warehouse
- User wants to discover ERC-8004 agents on Base by capability

## Prerequisites

Stdlib only — no external packages, no API key needed for free-tier routing.
Optional `curl` (any modern HTTP client works). Outbound HTTPS to
`graphadvocate.com` required.

For paid endpoints, the calling agent needs an x402-compatible HTTP client
with a funded wallet on Base (USDC). Per-call settlement; no subscription.

## How to Run

Invoke through the `terminal` tool.

### Free routing (3 queries/sender/day, then $0.01 USDC via x402)

```bash
curl -sX POST https://graphadvocate.com/route \
  -H 'Content-Type: application/json' \
  -d '{"request":"USDC holders top 20 on Ethereum","sender":"0xYourAgentAddress"}'
```

Returns:

```json
{
  "recommendation": "token-api",
  "reason": "Token API exposes ERC-20 holder data directly...",
  "confidence": "high",
  "query_ready": { "tool": "...", "args": {} },
  "execution_result": { "source": "...", "data": {} },
  "cache_for_seconds": 300
}
```

### Plain-English chat (free for handshakes/intros)

```bash
curl -sX POST https://graphadvocate.com/chat \
  -H 'Content-Type: application/json' \
  -d '{"message":"Find me Uniswap V3 subgraphs across all chains"}'
```

### Quota check (no charge)

```bash
curl -s "https://graphadvocate.com/quota?sender=0xYourAgentAddress"
```

### Paid endpoints (x402, USDC on Base)

Every paid endpoint returns a 402 challenge with an `output_example` field in
the body so you can preview the payload shape before signing.

| Endpoint | Price | Returns |
|---|---:|---|
| `POST /polymarket/pnl-quick` | $0.02 | Skill score + classification for a Polymarket wallet |
| `POST /polymarket/pnl` | $0.05 | Full PnL: scores + per-position records |
| `POST /polymarket/screen` | $0.05 | Top wagerers on a market with ghost-fill risk |
| `POST /polymarket/risk` | $0.02 | Wallet-type detection + ghost-fill risk classification |
| `POST /hyperliquid/score` | $0.02 | Hyperliquid perps trader skill score |
| `POST /hyperliquid/pnl` | $0.05 | Per-coin PnL + open positions + recent activity |
| `POST /hyperliquid/screen` | $0.05 | Top N traders of a coin (N capped at 10) |
| `POST /hyperliquid/vault` | $0.10 | Vault evaluator: leader skill + depositor concentration |
| `POST /hyperliquid/risk` | $0.02 | Liquidation rate + funding burn + outflow flag |
| `POST /kalshi/consensus-trend` | $0.05 | Kalshi consensus slope + acceleration |
| `POST /kalshi-polymarket/spread` | $0.05 | Kalshi-Polymarket cross-source spread |
| `POST /kalshi/sports-live-edge` | $0.05 | Live sports mispricing detector |
| `POST /predmarket/spread` | $0.05 | Polymarket-Limitless cross-venue spread with arbitrage direction |
| `POST /ask` | $0.05 | Natural-language Q&A over the x402 Base settlements warehouse |
| `POST /onchain-x402/address` | $0.01 | Decentralized x402 address lookup via subgraph |

## Quick Reference

```bash
# Free
POST /route        # plain-English -> routed query + live data
POST /chat         # web-chat-style interaction
GET  /quota        # remaining free-tier today

# Discovery (no-charge)
GET /.well-known/agent-card.json   # A2A agent card
GET /agents/capabilities.json      # machine-readable capability list
GET /llms.txt                      # LLM-friendly discovery file
GET /SKILL.md                      # this skill manifest (live source)

# Paid (x402 on Base, USDC) — see table above
```

## Procedure

1. **Classify the request** against `## When to Use`: subgraph/GraphQL
   discovery, Token API balances, Polymarket/Hyperliquid trader intel,
   cross-venue spread, or x402 settlements Q&A.
2. **Route it.** POST the plain-English question to `/route` with a stable
   `sender` address. Free for the first 3 calls per sender per day.
3. **Read the response.** Use `recommendation` (chosen service), `reason`, and
   `query_ready` (a ready-to-run tool call + args). If `execution_result` is
   present, the live data is already included.
4. **Run the query if needed.** If only `query_ready` is returned, execute it
   against the named service (subgraph gateway, Token API, etc.).
5. **For paid endpoints,** expect a `402 Payment Required` with an
   `output_example` in the body. Proceed only if your runtime is configured to
   accept x402 challenges (see `## Pitfalls`); otherwise the call stops there.
6. **Cache** results for `cache_for_seconds` before re-querying the same thing.

## Pitfalls

- **Free tier is 3 routing calls per sender per day,** then `/route` charges
  $0.01 via x402. Pass a stable `sender` address so quota tracks correctly.
- **Paid endpoints do nothing without x402 configured.** Without a funded Base
  wallet and an x402-capable client, paid endpoints return `402` and stop.
- **Spend safety before autonomous use:** fund a dedicated low-balance wallet,
  surface 402 challenges for per-call approval (each 402 body includes
  `output_example` so you can decide on the payload shape), and set
  `maxAmountPerCall` / `maxTotalSpend` or an invocation counter.
- **Never send secrets.** Queries go to `graphadvocate.com` over HTTPS; do not
  include private keys, seed phrases, or sensitive strategy details.
- **`/hyperliquid/screen` caps N at 10;** requesting more silently truncates.
- **Stateless:** no session persists between calls; include all needed context
  in each request.

## Verification

- **Quota check (no charge):**
  `curl -s "https://graphadvocate.com/quota?sender=0xYourAgentAddress"` returns
  JSON with your remaining free-tier calls.
- **Routing works:** a `/route` POST returns JSON containing `recommendation`
  and `query_ready`.
- **Discovery resolves:**
  `curl -s https://graphadvocate.com/.well-known/agent-card.json` returns the
  A2A agent card, and `https://graphadvocate.com/SKILL.md` serves the live
  manifest this file mirrors.
- **Paid preview:** hitting any paid endpoint without payment returns `402`
  with an `output_example` body, confirming the endpoint is live and priced.
