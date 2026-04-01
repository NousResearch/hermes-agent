---
name: ethereum
description: Query Ethereum mainnet blockchain data with USD pricing — wallet balances (ENS supported), token portfolios, transaction details, EIP-1559 gas analysis, contract inspection, whale detection, ENS resolution, and live network stats. Uses Ethereum RPC + CoinGecko. No API key required.
version: 1.0.0
author: maymuneth
license: MIT
metadata:
  hermes:
    tags: [Ethereum, Blockchain, Crypto, Web3, RPC, DeFi, EVM, L1, ENS, EIP-1559]
    related_skills: [base, solana]
---

# Ethereum Mainnet Blockchain Skill

Query Ethereum mainnet on-chain data enriched with USD pricing via CoinGecko.
9 commands: wallet portfolio (ENS-aware), token info, transactions, EIP-1559 gas
analysis, contract inspection, whale detection, ENS resolution, network stats,
and price lookup.

No API key needed. Uses only Python standard library (urllib, json, argparse).

---

## When to Use

- User asks for an Ethereum wallet balance, token holdings, or portfolio value
- User asks "what's in vitalik.eth's wallet?" (ENS names supported)
- User wants to inspect a specific transaction by hash
- User wants ERC-20 token metadata, price, supply, or market cap
- User wants to understand Ethereum gas costs and EIP-1559 fee structure
- User wants to inspect a contract (ERC type detection, proxy resolution)
- User wants to find large ETH transfers (whale detection) in the latest block
- User wants to resolve an ENS name to address, or reverse-resolve an address to ENS
- User wants Ethereum network health, gas price, or ETH price
- User asks "what's the price of USDC/WETH/UNI/AAVE/...?"

---

## Prerequisites

The helper script uses only Python standard library (urllib, json, argparse, hashlib).
No external packages required.

Pricing data comes from CoinGecko's free API (no key needed, rate-limited
to ~10-30 requests/minute). For faster lookups, use `--no-prices` flag.

---

## Quick Reference

RPC endpoint (default): https://ethereum.publicnode.com
Override: export ETH_RPC_URL=https://your-private-rpc.com

Helper script path: ~/.hermes/skills/blockchain/ethereum/scripts/ethereum_client.py

```
python3 ethereum_client.py stats
python3 ethereum_client.py wallet   <address_or_ens> [--limit N] [--all] [--no-prices]
python3 ethereum_client.py tx       <hash>
python3 ethereum_client.py token    <contract_address>
python3 ethereum_client.py gas
python3 ethereum_client.py contract <address>
python3 ethereum_client.py whales   [--min-eth N]
python3 ethereum_client.py ens      <name_or_address>
python3 ethereum_client.py price    <contract_address_or_symbol>
```

---

## Procedure

### 0. Setup Check

```bash
python3 --version

# Optional: set a private RPC for better rate limits
export ETH_RPC_URL="https://ethereum.publicnode.com"

# Confirm connectivity
python3 ~/.hermes/skills/blockchain/ethereum/scripts/ethereum_client.py stats
```

### 1. Wallet Portfolio

Get ETH balance and ERC-20 token holdings with USD values.
Accepts both 0x addresses and ENS names (e.g. `vitalik.eth`).
Checks ~20 well-known Ethereum tokens via on-chain `balanceOf` calls.

```bash
python3 ~/.hermes/skills/blockchain/ethereum/scripts/ethereum_client.py \
  wallet vitalik.eth

python3 ~/.hermes/skills/blockchain/ethereum/scripts/ethereum_client.py \
  wallet 0xd8dA6BF26964aF9D7eEd9e03E53415D37aA96045
```

Flags:
- `--limit N` — show top N tokens (default: 20)
- `--all` — show all tokens, no dust filter
- `--no-prices` — skip CoinGecko price lookups (faster, RPC-only)

Output includes: ETH balance + USD value, ERC-20 tokens sorted by value,
ENS name if available, portfolio total in USD.

### 2. Transaction Details

Inspect a full transaction by its hash. Shows EIP-1559 fee fields
(maxFeePerGas, maxPriorityFeePerGas, effectiveGasPrice), ETH transferred,
gas used, fee in ETH/USD, status, and decoded ERC-20/ERC-721 transfers.

```bash
python3 ~/.hermes/skills/blockchain/ethereum/scripts/ethereum_client.py \
  tx 0xabc123...your_tx_hash_here
```

### 3. Token Info

Get ERC-20 token metadata: name, symbol, decimals, total supply, price,
and market cap. Reads data directly from the contract via eth_call.

```bash
python3 ~/.hermes/skills/blockchain/ethereum/scripts/ethereum_client.py \
  token 0xa0b86991c6218b36c1d19d4a2e9eb0ce3606eb48
```

### 4. Gas Analysis (EIP-1559)

Detailed gas analysis including EIP-1559 base fee trends over 10 blocks,
suggested priority fee, block utilization, and cost estimates for common
Ethereum operations (ETH transfer, ERC-20, Uniswap swap, NFT mint).

```bash
python3 ~/.hermes/skills/blockchain/ethereum/scripts/ethereum_client.py gas
```

Output: current gas price, EIP-1559 base fee, suggested priority fee,
block utilization, 10-block trend, cost estimates in ETH and USD.

### 5. Contract Inspection

Inspect an address: determine if it's an EOA or contract, detect
ERC-20/ERC-721/ERC-1155 interfaces, resolve EIP-1967 proxy addresses,
and reverse-resolve to ENS name.

```bash
python3 ~/.hermes/skills/blockchain/ethereum/scripts/ethereum_client.py \
  contract 0xa0b86991c6218b36c1d19d4a2e9eb0ce3606eb48
```

### 6. Whale Detector

Scan the most recent Ethereum block for large ETH transfers with USD values.
Default threshold is 10 ETH (higher than L2 defaults due to mainnet scale).

```bash
python3 ~/.hermes/skills/blockchain/ethereum/scripts/ethereum_client.py \
  whales --min-eth 10.0
```

### 7. ENS Resolution

Resolve an ENS name to an Ethereum address, or reverse-resolve an address
to its ENS name. Uses the on-chain ENS registry — no external API needed.

```bash
# Forward resolution: ENS name -> address
python3 ~/.hermes/skills/blockchain/ethereum/scripts/ethereum_client.py \
  ens vitalik.eth

# Reverse resolution: address -> ENS name
python3 ~/.hermes/skills/blockchain/ethereum/scripts/ethereum_client.py \
  ens 0xd8dA6BF26964aF9D7eEd9e03E53415D37aA96045
```

### 8. Network Stats

Live Ethereum mainnet health: latest block, chain ID, gas price, EIP-1559
base fee and priority fee, block utilization, transaction count, and ETH price.

```bash
python3 ~/.hermes/skills/blockchain/ethereum/scripts/ethereum_client.py stats
```

### 9. Price Lookup

Quick price check for any token by contract address or known symbol.

```bash
python3 ~/.hermes/skills/blockchain/ethereum/scripts/ethereum_client.py price ETH
python3 ~/.hermes/skills/blockchain/ethereum/scripts/ethereum_client.py price USDC
python3 ~/.hermes/skills/blockchain/ethereum/scripts/ethereum_client.py price WBTC
python3 ~/.hermes/skills/blockchain/ethereum/scripts/ethereum_client.py price UNI
python3 ~/.hermes/skills/blockchain/ethereum/scripts/ethereum_client.py price 0xa0b86991c6218b36c1d19d4a2e9eb0ce3606eb48
```

Known symbols: ETH, WETH, USDC, USDT, DAI, WBTC, UNI, AAVE, LINK,
stETH, wstETH, cbETH, CRV, MKR, COMP, YFI, BAL, 1INCH, LDO, ARB, OP.

---

## Pitfalls

- **CoinGecko rate-limits** — free tier allows ~10-30 requests/minute.
  Use `--no-prices` for faster wallet queries.
- **Public RPC rate-limits** — publicnode.com is rate-limited.
  Set `ETH_RPC_URL` to Alchemy, Infura, or QuickNode for production use.
- **ENS resolution uses on-chain keccak256 (sha3_256)** — Python's
  `hashlib.sha3_256` implements Keccak-256 (not NIST SHA-3). This is correct
  for Ethereum. Available in Python 3.6+.
- **Wallet shows known tokens only** — Ethereum has no built-in
  "get all tokens" RPC. The wallet command checks ~20 popular tokens.
  Use the `token` command for any specific contract address.
- **Whale detector scans latest block only** — not historical.
  Default threshold is 10 ETH (higher than L2 skills due to mainnet scale).
- **EIP-1559 estimates are L1 execution cost** — no L1 data fee since this
  is already Ethereum mainnet. Gas estimates may vary from actual cost.
- **Proxy detection** — only EIP-1967 proxies are detected.
  Other proxy patterns (EIP-1167, custom storage slots) are not checked.
- **Retry on 429** — both RPC and CoinGecko calls retry up to 2 times
  with exponential backoff on rate-limit errors.

---

## Verification

```bash
# Should print chain ID (1), latest block, base fee, and ETH price
python3 ~/.hermes/skills/blockchain/ethereum/scripts/ethereum_client.py stats

# Resolve a well-known ENS name
python3 ~/.hermes/skills/blockchain/ethereum/scripts/ethereum_client.py ens vitalik.eth
```
