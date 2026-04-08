---
name: avalanche
description: Query Avalanche C-Chain blockchain data with USD pricing — wallet balances, token info, transaction details, gas analysis, contract inspection, whale detection, and live network stats. Uses Avalanche RPC + CoinGecko. No API key required.
version: 0.1.0
author: Community (Hermes Agent)
license: MIT
metadata:
  hermes:
    tags: [Avalanche, AVAX, Blockchain, Crypto, Web3, RPC, DeFi, EVM, C-Chain]
    related_skills: []
---

# Avalanche C-Chain Blockchain Skill

Query Avalanche C-Chain on-chain data enriched with USD pricing via CoinGecko.
8 commands: wallet portfolio, token info, transactions, gas analysis,
contract inspection, whale detection, network stats, and price lookup.

No API key needed. Uses only Python standard library (urllib, json, argparse).

---

## When to Use

- User asks for an Avalanche wallet balance, token holdings, or portfolio value
- User wants to inspect a specific transaction by hash
- User wants ERC-20 token metadata, price, supply, or market cap
- User wants to understand Avalanche gas costs
- User wants to inspect a contract (ERC type detection, proxy resolution)
- User wants to find large AVAX transfers (whale detection)
- User wants Avalanche network health, gas price, or AVAX price
- User asks "what's the price of JOE/PNG/QI/AVAX?"

---

## Prerequisites

The helper script uses only Python standard library (urllib, json, argparse).
No external packages required.

Pricing data comes from CoinGecko's free API (no key needed, rate-limited
to ~10-30 requests/minute). For faster lookups, use `--no-prices` flag.

---

## Quick Reference

RPC endpoint (default): https://api.avax.network/ext/bc/C/rpc
Override: export AVAX_RPC_URL=https://your-private-rpc.com

Helper script path: ~/.hermes/skills/blockchain/avalanche/scripts/avax_client.py

```
python3 avax_client.py wallet   <address> [--limit N] [--all] [--no-prices]
python3 avax_client.py tx       <hash>
python3 avax_client.py token    <contract_address>
python3 avax_client.py gas
python3 avax_client.py contract <address>
python3 avax_client.py whales   [--min-avax N]
python3 avax_client.py stats
python3 avax_client.py price    <contract_address_or_symbol>
```

---

## Procedure

### 0. Setup Check

```bash
python3 --version

# Optional: set a private RPC for better rate limits
export AVAX_RPC_URL="https://api.avax.network/ext/bc/C/rpc"

# Confirm connectivity
python3 ~/.hermes/skills/blockchain/avalanche/scripts/avax_client.py stats
```

### 1. Wallet Portfolio

Get AVAX balance and ERC-20 token holdings with USD values.
Checks ~20 well-known Avalanche tokens (USDC, USDT, JOE, PNG, QI, etc.)
via on-chain `balanceOf` calls. Tokens sorted by value, dust filtered.

```bash
python3 ~/.hermes/skills/blockchain/avalanche/scripts/avax_client.py \
  wallet 0xd8dA6BF26964aF9D7eEd9e03E53415D37aA96045
```

Flags:
- `--limit N` — show top N tokens (default: 20)
- `--all` — show all tokens, no dust filter, no limit
- `--no-prices` — skip CoinGecko price lookups (faster, RPC-only)

Output includes: AVAX balance + USD value, token list with prices sorted
by value, dust count, total portfolio value in USD.

Note: Only checks known tokens. Unknown ERC-20s are not discovered.
Use the `token` command with a specific contract address for any token.

### 2. Transaction Details

Inspect a full transaction by its hash. Shows AVAX value transferred,
gas used, fee in AVAX/USD, status, and decoded ERC-20/ERC-721 transfers.

```bash
python3 ~/.hermes/skills/blockchain/avalanche/scripts/avax_client.py \
  tx 0xabc123...your_tx_hash_here
```

Output: hash, block, from, to, value (AVAX + USD), gas price, gas used,
fee, status, contract creation address (if any), token transfers.

### 3. Token Info

Get ERC-20 token metadata: name, symbol, decimals, total supply, price,
market cap, and contract code size.

```bash
python3 ~/.hermes/skills/blockchain/avalanche/scripts/avax_client.py \
  token 0x6e84a6216eA6dACC71eE8E6b0a5B7322EEbC0fDd
```

Output: name, symbol, decimals, total supply, price, market cap.
Reads name/symbol/decimals directly from the contract via eth_call.

### 4. Gas Analysis

Detailed gas analysis with cost estimates for common operations.
Shows current gas price, base fee trends over 10 blocks, block
utilization, and estimated costs for AVAX transfers, ERC-20 transfers,
and swaps.

```bash
python3 ~/.hermes/skills/blockchain/avalanche/scripts/avax_client.py gas
```

Output: current gas price, base fee, block utilization, 10-block trend,
cost estimates in AVAX and USD.

### 5. Contract Inspection

Inspect an address: determine if it's an EOA or contract, detect
ERC-20/ERC-721/ERC-1155 interfaces, resolve EIP-1967 proxy
implementation addresses.

```bash
python3 ~/.hermes/skills/blockchain/avalanche/scripts/avax_client.py \
  contract 0x6e84a6216eA6dACC71eE8E6b0a5B7322EEbC0fDd
```

Output: is_contract, code size, AVAX balance, detected interfaces
(ERC-20, ERC-721, ERC-1155), ERC-20 metadata, proxy implementation
address.

### 6. Whale Detector

Scan the most recent block for large AVAX transfers with USD values.

```bash
python3 ~/.hermes/skills/blockchain/avalanche/scripts/avax_client.py \
  whales --min-avax 1000
```

Note: scans the latest block only — point-in-time snapshot, not historical.
Default threshold is 1000 AVAX.

### 7. Network Stats

Live Avalanche C-Chain network health: latest block, chain ID, gas price,
base fee, block utilization, transaction count, and AVAX price.

```bash
python3 ~/.hermes/skills/blockchain/avalanche/scripts/avax_client.py stats
```

### 8. Price Lookup

Quick price check for any token by contract address or known symbol.

```bash
python3 ~/.hermes/skills/blockchain/avalanche/scripts/avax_client.py price AVAX
python3 ~/.hermes/skills/blockchain/avalanche/scripts/avax_client.py price JOE
python3 ~/.hermes/skills/blockchain/avalanche/scripts/avax_client.py price PNG
python3 ~/.hermes/skills/blockchain/avalanche/scripts/avax_client.py price USDC
python3 ~/.hermes/skills/blockchain/avalanche/scripts/avax_client.py price 0x6e84a6216eA6dACC71eE8E6b0a5B7322EEbC0fDd
```

Known symbols: AVAX, WAVAX, USDC, USDT, JOE, PNG, QI, XAVA, SUSHI,
AAVE, LINK, WBTC, DAI, FRAX, sAVAX, COQ, KIMBO, TECH, GEC, HUSKY.

---

## Pitfalls

- **Hermes output redaction** — Do NOT use `"token"` as a JSON key name in output.
  Hermes redacts values under that key to `***` (security feature for API tokens).
  Use `"symbol"` instead. This applies to all Hermes tool output, not just this skill.
- **CoinGecko rate-limits** — free tier allows ~10-30 requests/minute.
  Price lookups use 1 request per token. Use `--no-prices` for speed.
  When running multiple commands in sequence, add 1-2 second delays or prices
  will return null.
- **Public RPC rate-limits** — Avalanche public RPC limits requests.
  For production use, set AVAX_RPC_URL to a private endpoint
  (Alchemy, Infura, ANKR, Moralis).
- **Wallet shows known tokens only** — unlike Solana, EVM chains have no
  built-in "get all tokens" RPC. The wallet command checks ~20 popular
  Avalanche tokens via `balanceOf`. Unknown ERC-20s won't appear. Use the
  `token` command for any specific contract.
- **Token names read from contract** — if a contract doesn't implement
  `name()` or `symbol()`, these fields may be empty. Known tokens have
  hardcoded labels as fallback.
- **Whale detector scans latest block only** — not historical. Results
  vary by the moment you query. Default threshold is 1000 AVAX.
- **Proxy detection** — only EIP-1967 proxies are detected. Other proxy
  patterns (EIP-1167 minimal proxy, custom storage slots) are not checked.
- **Retry on 429** — both RPC and CoinGecko calls retry up to 2 times
  with exponential backoff on rate-limit errors.
- **Avalanche subnets** — this skill queries C-Chain (EVM) only. For
  X-Chain or P-Chain, use the Avalanche API directly.

---

## Verification

```bash
# Should print Avalanche chain ID (43114), latest block, gas price, and AVAX price
python3 ~/.hermes/skills/blockchain/avalanche/scripts/avax_client.py stats
```
