---
name: solana
description: Query Solana blockchain data — wallet balances, SPL token holdings, transaction details, NFT portfolios, whale detection, and live network stats via public Solana RPC API. No API key required for basic usage.
version: 0.1.0
author: Deniz Alagoz (gizdusum)
license: MIT
metadata:
  hermes:
    tags: [Solana, Blockchain, Crypto, Web3, RPC, DeFi, NFT]
    related_skills: []
---

# Solana Blockchain Skill

Query Solana on-chain data using the public Solana JSON-RPC API.
Includes 7 intelligence tools: wallet info, transactions, token metadata,
recent activity, NFT portfolios, whale detection, and network stats.

No API key needed for mainnet public endpoint.
For high-volume use, set SOLANA_RPC_URL to a private RPC (Helius, QuickNode, etc.).

---

## When to Use

- User asks for a Solana wallet balance or token holdings
- User wants to inspect a specific transaction by signature
- User wants SPL token metadata, supply, or top holders
- User wants recent transaction history for an address
- User wants NFTs owned by a wallet
- User wants to find large SOL transfers (whale detection)
- User wants Solana network health, TPS, epoch, or slot info

---

## Prerequisites

The helper script uses only Python standard library (urllib, json, argparse).
No external packages required for basic operation.

Optional: httpx (faster async I/O) and base58 (address validation).
Install via your project's dependency manager before use if needed.

---

## Quick Reference

RPC endpoint (default): https://api.mainnet-beta.solana.com
Override: export SOLANA_RPC_URL=https://your-private-rpc.com

Helper script path: ~/.hermes/skills/blockchain/solana/scripts/solana_client.py

  python3 solana_client.py wallet   <address>
  python3 solana_client.py tx       <signature>
  python3 solana_client.py token    <mint_address>
  python3 solana_client.py activity <address> [--limit N]
  python3 solana_client.py nft      <address>
  python3 solana_client.py whales   [--min-sol N]
  python3 solana_client.py stats

---

## Procedure

### 0. Setup Check

```bash
# Verify Python 3 is available
python3 --version

# Optional: set a private RPC for better rate limits
export SOLANA_RPC_URL="https://api.mainnet-beta.solana.com"

# Confirm connectivity
python3 ~/.hermes/skills/blockchain/solana/scripts/solana_client.py stats
```

### 1. Wallet Info

Get SOL balance and all SPL token holdings for an address.

```bash
python3 ~/.hermes/skills/blockchain/solana/scripts/solana_client.py \
  wallet 9WzDXwBbmkg8ZTbNMqUxvQRAyrZzDsGYdLVL9zYtAWWM
```

Output: SOL balance (human readable), list of SPL tokens with mint + amount.

### 2. Transaction Details

Inspect a full transaction by its base58 signature.

```bash
python3 ~/.hermes/skills/blockchain/solana/scripts/solana_client.py \
  tx 5j7s8K...your_signature_here
```

Output: slot, timestamp, fee, status, balance changes, program invocations.

### 3. Token Info

Get SPL token metadata, supply, decimals, mint/freeze authorities, top holders.

```bash
python3 ~/.hermes/skills/blockchain/solana/scripts/solana_client.py \
  token DezXAZ8z7PnrnRJjz3wXBoRgixCa6xjnB7YaB1pPB263
```

Output: decimals, supply (human readable), top 5 holders and their percentages.

### 4. Recent Activity

List recent transactions for an address (default: last 10, max: 25).

```bash
python3 ~/.hermes/skills/blockchain/solana/scripts/solana_client.py \
  activity 9WzDXwBbmkg8ZTbNMqUxvQRAyrZzDsGYdLVL9zYtAWWM --limit 25
```

Output: list of transaction signatures with slot and timestamp.

### 5. NFT Portfolio

List NFTs owned by a wallet (heuristic: SPL tokens with amount=1, decimals=0).

```bash
python3 ~/.hermes/skills/blockchain/solana/scripts/solana_client.py \
  nft 9WzDXwBbmkg8ZTbNMqUxvQRAyrZzDsGYdLVL9zYtAWWM
```

Output: list of NFT mint addresses.
Note: Compressed NFTs (cNFTs) are not detected by this heuristic.

### 6. Whale Detector

Scan the most recent block for large SOL transfers (default threshold: 1000 SOL).

```bash
python3 ~/.hermes/skills/blockchain/solana/scripts/solana_client.py \
  whales --min-sol 500
```

Output: list of large transfers with sender, receiver, amount in SOL.
Note: scans the latest block only — point-in-time snapshot.

### 7. Network Stats

Live Solana network health: current slot, epoch, TPS, supply, validator version.

```bash
python3 ~/.hermes/skills/blockchain/solana/scripts/solana_client.py \
  stats
```

Output: slot, epoch, transactions per second, total/circulating supply, node version.

---

## Raw curl Examples (no script needed)

SOL balance:
```bash
curl -s https://api.mainnet-beta.solana.com \
  -H "Content-Type: application/json" \
  -d '{
    "jsonrpc":"2.0","id":1,"method":"getBalance",
    "params":["9WzDXwBbmkg8ZTbNMqUxvQRAyrZzDsGYdLVL9zYtAWWM"]
  }' | python3 -c "
import sys,json
r=json.load(sys.stdin)
lamports=r['result']['value']
print(f'Balance: {lamports/1e9:.4f} SOL')
"
```

Network slot check:
```bash
curl -s https://api.mainnet-beta.solana.com \
  -H "Content-Type: application/json" \
  -d '{"jsonrpc":"2.0","id":1,"method":"getSlot"}' \
  | python3 -c "import sys,json; print('Slot:', json.load(sys.stdin)['result'])"
```

---

## Pitfalls

- Public RPC rate-limits apply. For production use, get a private endpoint (Helius, QuickNode, Triton).
- NFT detection is heuristic (amount=1, decimals=0). Compressed NFTs (cNFTs) won't appear.
- Transactions older than ~2 days may not be on the public RPC history.
- Whale detector scans only the latest block; old large transfers won't show.
- Token supply is a raw integer — divide by 10^decimals for human-readable value.
- Some RPC methods (e.g. getTokenLargestAccounts) may require commitment=finalized.

---

## Verification

```bash
# Should print current Solana slot number if RPC is reachable
curl -s https://api.mainnet-beta.solana.com \
  -H "Content-Type: application/json" \
  -d '{"jsonrpc":"2.0","id":1,"method":"getSlot"}' \
  | python3 -c "import sys,json; r=json.load(sys.stdin); print('OK, slot:', r['result'])"
```
