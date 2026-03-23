---
name: evm
description: Query EVM blockchain data with USD pricing — wallet balances, ERC-20 token portfolios with values, transaction details, gas prices, and live network stats. Supports Ethereum, BNB Chain, Base, Arbitrum, and Polygon. Uses public RPCs + CoinGecko. No API key required.
version: 0.1.0
author: 0xFrank
license: MIT
metadata:
  hermes:
    tags: [EVM, Ethereum, BNB Chain, BSC, Base, Arbitrum, Polygon, Blockchain, Crypto, Web3, DeFi, NFT]
    category: blockchain
    related_skills: [solana]
    requires_toolsets: [terminal]
---

# EVM Blockchain Skill

Query EVM-compatible blockchain data enriched with USD pricing via CoinGecko.
7 commands: wallet portfolio, token info, transactions, activity, gas tracker,
network stats, and price lookup.

Supports 5 chains: Ethereum, BNB Chain (BSC), Base, Arbitrum One, and Polygon.

No API key needed. Uses only Python standard library (urllib, json, argparse).

---

## When to Use
- User asks for an Ethereum/BNB/Base/Arbitrum/Polygon wallet balance or portfolio
- User wants to inspect a specific transaction by hash
- User wants ERC-20 token metadata, price, supply, or contract info
- User wants recent transaction history for an address
- User wants current gas prices (gwei) for any EVM chain
- User wants network health: block number, gas price, native token price
- User asks "what's the price of ETH/BNB/USDC/PEPE?"
- User asks about a token on any EVM chain
- User wants to compare gas prices across chains

---

## Prerequisites
The helper script uses only Python standard library (urllib, json, argparse).
No external packages required.

Pricing data comes from CoinGecko's free API (no key needed, rate-limited
to ~10-30 requests/minute).

---

## Quick Reference
Default chain: Ethereum mainnet
Override chain: `--chain bsc` / `--chain base` / `--chain arbitrum` / `--chain polygon`
Override RPC: `export EVM_RPC_URL=https://your-private-rpc.com`

Helper script path: ~/.hermes/skills/blockchain/evm/scripts/evm_client.py

```
python3 evm_client.py stats                                    # Network stats
python3 evm_client.py wallet   <address> [--limit N] [--no-prices]  # Portfolio
python3 evm_client.py tx       <hash>                          # Transaction details
python3 evm_client.py token    <contract_address>              # ERC-20 token info
python3 evm_client.py activity <address> [--limit N]           # Recent transactions
python3 evm_client.py gas                                      # Gas prices (gwei)
python3 evm_client.py price    <symbol_or_address>             # Price lookup

# Chain selection (add to any command):
python3 evm_client.py stats --chain bsc
python3 evm_client.py wallet <address> --chain base
python3 evm_client.py gas --chain arbitrum
```

---

## Procedure

### 0. Setup Check
```bash
python3 --version

# Optional: set a private RPC for better rate limits
export EVM_RPC_URL="https://your-rpc-endpoint.com"

# Confirm connectivity
python3 ~/.hermes/skills/blockchain/evm/scripts/evm_client.py stats
```

### 1. Wallet Portfolio
Get native token balance (ETH/BNB/etc.) with USD value, plus ERC-20 token
holdings with prices. Tokens sorted by value, known tokens labeled by name.

```bash
python3 ~/.hermes/skills/blockchain/evm/scripts/evm_client.py \
  wallet 0xd8dA6BF26964aF9D7eEd9e03E53415D37aA96045
```

Flags:
- `--limit N` — show top N tokens (default: 20)
- `--no-prices` — skip CoinGecko price lookups (faster, RPC-only)
- `--chain bsc` — query BNB Chain instead of Ethereum

Output includes: native balance + USD value, ERC-20 token list with prices,
total portfolio value in USD.

### 2. Transaction Details
Inspect a full transaction by its hash. Shows value, gas used, status,
from/to addresses, and contract interactions.

```bash
python3 ~/.hermes/skills/blockchain/evm/scripts/evm_client.py \
  tx 0x5c504ed432cb51138bcf09aa5e8a410dd4a1e204ef84bfed1be16dfba1b22060
```

Output: block number, timestamp, from, to, value (native + USD),
gas used, gas price, status, input data preview.

### 3. Token Info
Get ERC-20 token metadata: name, symbol, decimals, total supply,
current price, and market cap.

```bash
python3 ~/.hermes/skills/blockchain/evm/scripts/evm_client.py \
  token 0xdAC17F958D2ee523a2206206994597C13D831ec7
```

Output: name, symbol, decimals, total supply, price, market cap.

### 4. Recent Activity
List recent transactions for an address via block scanning.

```bash
python3 ~/.hermes/skills/blockchain/evm/scripts/evm_client.py \
  activity 0xd8dA6BF26964aF9D7eEd9e03E53415D37aA96045 --limit 10
```

Note: Uses eth_getBlockByNumber to scan recent blocks. For deep history,
use a block explorer.

### 5. Gas Tracker
Current gas prices in gwei — slow, standard, fast estimates.

```bash
python3 ~/.hermes/skills/blockchain/evm/scripts/evm_client.py gas
python3 ~/.hermes/skills/blockchain/evm/scripts/evm_client.py gas --chain bsc
```

Output: gas price in gwei, estimated costs for common operations
(transfer, ERC-20 transfer, swap) in native token and USD.

### 6. Network Stats
Live network health: latest block, gas price, native token price and
market cap.

```bash
python3 ~/.hermes/skills/blockchain/evm/scripts/evm_client.py stats
python3 ~/.hermes/skills/blockchain/evm/scripts/evm_client.py stats --chain bsc
```

### 7. Price Lookup
Quick price check for any token by symbol or contract address.

```bash
python3 ~/.hermes/skills/blockchain/evm/scripts/evm_client.py price ETH
python3 ~/.hermes/skills/blockchain/evm/scripts/evm_client.py price BNB
python3 ~/.hermes/skills/blockchain/evm/scripts/evm_client.py price USDC
python3 ~/.hermes/skills/blockchain/evm/scripts/evm_client.py price 0xdAC17F958D2ee523a2206206994597C13D831ec7
```

Known symbols: ETH, BNB, USDC, USDT, WETH, WBTC, DAI, LINK, UNI, AAVE,
PEPE, SHIB, ARB, OP, MATIC, DOGE, MKR, LDO, CRV, COMP, SNX.

---

## Pitfalls
- **CoinGecko rate-limits** — free tier allows ~10-30 requests/minute.
  Use sparingly for wallets with many tokens.
- **Public RPC rate-limits** — free public RPCs may throttle requests.
  For production use, set EVM_RPC_URL to a private endpoint
  (Alchemy, Infura, QuickNode).
- **ERC-20 token discovery** — Without an indexer, the script uses known
  token lists to check balances. Not all tokens will be detected.
  Use a block explorer for complete token lists.
- **Activity scanning** — Transaction history uses recent block scanning,
  which is limited. For full history, use Etherscan/BSCScan APIs.
- **Token names** — ~20 well-known tokens are labeled by name. Others
  show abbreviated contract addresses.
- **Retry on 429** — All API calls retry up to 2 times with exponential
  backoff on rate-limit errors.

---

## Verification
```bash
# Should print current block number, gas price, and ETH price
python3 ~/.hermes/skills/blockchain/evm/scripts/evm_client.py stats
```
