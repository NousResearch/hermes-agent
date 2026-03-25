---
name: ethereum
description: Query Ethereum mainnet blockchain data with USD pricing — wallet balances, token info, transaction details, gas analysis, contract inspection, whale detection, and live network stats. Uses Ethereum RPC + CoinGecko. No API key required.
version: 0.1.0
author: dieutx
license: MIT
metadata:
  hermes:
    tags: [Ethereum, Blockchain, Crypto, Web3, RPC, DeFi, EVM, L1, EIP-1559]
    related_skills: [base, solana]
---

# Ethereum Mainnet Blockchain Skill

Query Ethereum mainnet on-chain data enriched with USD pricing via CoinGecko.
8 commands: wallet portfolio, token info, transactions, gas analysis,
contract inspection, whale detection, network stats, and price lookup.

No API key needed. Uses only Python standard library (urllib, json, argparse).

---

## When to Use

- User asks for an Ethereum wallet balance, token holdings, or portfolio value
- User wants to inspect a specific transaction by hash
- User wants ERC-20 token metadata, price, supply, or market cap
- User wants to understand Ethereum gas costs (EIP-1559 base fee + priority fee)
- User wants to inspect a contract (ERC type detection, proxy resolution)
- User wants to find large ETH transfers (whale detection)
- User wants Ethereum network health, gas price, or ETH price
- User asks "what's the price of UNI/LINK/AAVE/ETH?"

---

## Prerequisites

The helper script uses only Python standard library (urllib, json, argparse).
No external packages required.

Pricing data comes from CoinGecko's free API (no key needed, rate-limited
to ~10-30 requests/minute). For faster lookups, use `--no-prices` flag.

---

## Quick Reference

RPC endpoint (default): https://ethereum-rpc.publicnode.com
Override: export ETH_RPC_URL=https://your-private-rpc.com

Helper script path: ~/.hermes/skills/blockchain/ethereum/scripts/ethereum_client.py

```
python3 ethereum_client.py wallet   <address> [--limit N] [--all] [--no-prices]
python3 ethereum_client.py tx       <hash>
python3 ethereum_client.py token    <contract_address>
python3 ethereum_client.py gas
python3 ethereum_client.py contract <address>
python3 ethereum_client.py whales   [--min-eth N]
python3 ethereum_client.py stats
python3 ethereum_client.py price    <contract_address_or_symbol>
```

---

## Procedure

### 0. Setup Check

```bash
python3 --version

# Optional: set a private RPC for better rate limits
export ETH_RPC_URL="https://ethereum-rpc.publicnode.com"

# Confirm connectivity
python3 ~/.hermes/skills/blockchain/ethereum/scripts/ethereum_client.py stats
```

### 1. Wallet Portfolio

Get ETH balance and ERC-20 token holdings with USD values.
Checks ~15 well-known Ethereum tokens (USDC, USDT, DAI, WBTC, UNI, LINK, AAVE, etc.)
via on-chain `balanceOf` calls. Tokens sorted by value, dust filtered.

```bash
python3 ~/.hermes/skills/blockchain/ethereum/scripts/ethereum_client.py \
  wallet 0xd8dA6BF26964aF9D7eEd9e03E53415D37aA96045
```

Flags:
- `--limit N` — show top N tokens (default: 20)
- `--all` — show all tokens, no dust filter, no limit
- `--no-prices` — skip CoinGecko price lookups (faster, RPC-only)

Output includes: ETH balance + USD value, token list with prices sorted
by value, dust count, total portfolio value in USD.

Note: Only checks known tokens. Unknown ERC-20s are not discovered.
Use the `token` command with a specific contract address for any token.

### 2. Transaction Details

Inspect a full transaction by its hash. Shows ETH value transferred,
gas used, fee in ETH/USD, status, and decoded ERC-20/ERC-721 transfers.

```bash
python3 ~/.hermes/skills/blockchain/ethereum/scripts/ethereum_client.py \
  tx 0xabc123...your_tx_hash_here
```

Output: hash, block, from, to, value (ETH + USD), gas price, gas used,
fee, status, contract creation address (if any), token transfers.

### 3. Token Info

Get ERC-20 token metadata: name, symbol, decimals, total supply, price,
market cap, and contract code size.

```bash
python3 ~/.hermes/skills/blockchain/ethereum/scripts/ethereum_client.py \
  token 0xa0b86991c6218b36c1d19d4a2e9eb0ce3606eb48
```

Output: name, symbol, decimals, total supply, price, market cap.
Reads name/symbol/decimals directly from the contract via eth_call.

### 4. Gas Analysis

Detailed gas analysis with EIP-1559 fee breakdown.
Shows current gas price, base fee trends over 10 blocks, block
utilization, and estimated costs for ETH transfers, ERC-20 transfers,
and swaps.

```bash
python3 ~/.hermes/skills/blockchain/ethereum/scripts/ethereum_client.py gas
```

Output: current gas price, base fee, block utilization, 10-block trend,
cost estimates in ETH and USD.

Note: Ethereum mainnet uses EIP-1559 pricing — total tx cost =
(base fee + priority fee) x gas used. The priority fee (tip) varies
by congestion. Estimates use the current base fee only.

### 5. Contract Inspection

Inspect an address: determine if it's an EOA or contract, detect
ERC-20/ERC-721/ERC-1155 interfaces, resolve EIP-1967 proxy
implementation addresses.

```bash
python3 ~/.hermes/skills/blockchain/ethereum/scripts/ethereum_client.py \
  contract 0xa0b86991c6218b36c1d19d4a2e9eb0ce3606eb48
```

Output: is_contract, code size, ETH balance, detected interfaces
(ERC-20, ERC-721, ERC-1155), ERC-20 metadata, proxy implementation
address.

### 6. Whale Detector

Scan the most recent block for large ETH transfers with USD values.

```bash
python3 ~/.hermes/skills/blockchain/ethereum/scripts/ethereum_client.py \
  whales --min-eth 10.0
```

Note: scans the latest block only — point-in-time snapshot, not historical.
Default threshold is 10.0 ETH (higher than L2 defaults since mainnet
whales typically move larger amounts).

### 7. Network Stats

Live Ethereum network health: latest block, chain ID, gas price, base fee,
block utilization, transaction count, and ETH price.

```bash
python3 ~/.hermes/skills/blockchain/ethereum/scripts/ethereum_client.py stats
```

### 8. Price Lookup

Quick price check for any token by contract address or known symbol.

```bash
python3 ~/.hermes/skills/blockchain/ethereum/scripts/ethereum_client.py price ETH
python3 ~/.hermes/skills/blockchain/ethereum/scripts/ethereum_client.py price USDC
python3 ~/.hermes/skills/blockchain/ethereum/scripts/ethereum_client.py price UNI
python3 ~/.hermes/skills/blockchain/ethereum/scripts/ethereum_client.py price LINK
python3 ~/.hermes/skills/blockchain/ethereum/scripts/ethereum_client.py price 0xa0b86991c6218b36c1d19d4a2e9eb0ce3606eb48
```

Known symbols: ETH, WETH, USDC, USDT, DAI, WBTC, wstETH, rETH, cbETH,
UNI, LINK, AAVE, CRV, LDO, SNX, FRAX.

---

## Pitfalls

- **CoinGecko rate-limits** — free tier allows ~10-30 requests/minute.
  Price lookups use 1 request per token. Use `--no-prices` for speed.
- **Public RPC rate-limits** — Ethereum public RPCs limit requests.
  For production use, set ETH_RPC_URL to a private endpoint
  (Alchemy, QuickNode, Infura).
- **Wallet shows known tokens only** — EVM chains have no built-in
  "get all tokens" RPC. The wallet command checks ~15 popular Ethereum
  tokens via `balanceOf`. Unknown ERC-20s won't appear. Use the
  `token` command for any specific contract.
- **Token names read from contract** — if a contract doesn't implement
  `name()` or `symbol()`, these fields may be empty. Known tokens have
  hardcoded labels as fallback.
- **Gas estimates use base fee only** — actual costs include a priority
  fee (tip) that varies by network congestion. During high congestion,
  real costs can be 2-10x the base fee estimate.
- **Whale detector scans latest block only** — not historical. Results
  vary by the moment you query. Default threshold is 10.0 ETH.
- **Proxy detection** — only EIP-1967 proxies are detected. Other proxy
  patterns (EIP-1167 minimal proxy, custom storage slots) are not checked.
- **Retry on 429** — both RPC and CoinGecko calls retry up to 2 times
  with exponential backoff on rate-limit errors.

---

## Verification

```bash
# Should print Ethereum chain ID (1), latest block, gas price, and ETH price
python3 ~/.hermes/skills/blockchain/ethereum/scripts/ethereum_client.py stats
```
