---
name: hedera
description: "Read-only Hedera client: accounts, tokens, NFTs, HCS topics."
version: 1.0.0
author: Narbeh Shahnazarian (@narbs91), Hermes Agent
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [Hedera, HBAR, HTS, HCS, Blockchain, Crypto, Web3, DeFi, NFT, Hashgraph]
    category: blockchain
    related_skills: [evm, solana]
    requires_toolsets: [terminal]
---

# Hedera Blockchain Skill

Query Hedera on-chain data enriched with USD pricing via CoinGecko.
10 commands: network stats, account portfolio, token metadata, transaction
details, activity history, NFT holdings, price lookup, fee schedule,
HCS topic inspection, and smart contract info.

No API key needed. Uses only the Python standard library (urllib, json,
argparse, base64, threading). All data comes from Hedera's public Mirror
Node REST API and CoinGecko's free tier.

Supports both mainnet and testnet via `--network`.

---

## When to Use

- User asks for a Hedera account balance or HBAR holdings
- User wants HTS token holdings and portfolio value for an account
- User wants HTS token metadata (type, supply, treasury, key configuration)
- User wants to inspect a Hedera transaction by ID
- User wants recent transaction history for an account
- User wants NFTs held by an account (grouped by collection)
- User asks "what's the price of HBAR?" or a known HTS token
- User wants to know the HBAR cost of a Hedera operation (transfer, mint, etc.)
- User wants to inspect an HCS topic: its metadata or recent messages
- User wants smart contract info: EVM address ↔ account ID mapping, bytecode size, admin keys
- User is working on testnet and needs the same queries against testnet data

---

## Prerequisites

Python 3.8+ standard library only. No pip installs required.

Pricing: CoinGecko free API (rate-limited, ~10-30 req/min). Use `--no-prices`
on the `account` command to skip price lookups when speed matters.

Mirror Node (default):
- Mainnet: https://mainnet-public.mirrornode.hedera.com
- Testnet: https://testnet.mirrornode.hedera.com

Override endpoint: `export HEDERA_MIRROR_URL=https://your-private-mirror.com`

Helper script path: `~/.hermes/skills/blockchain/hedera/scripts/hedera_client.py`

---

## Quick Reference

```
SCRIPT=~/.hermes/skills/blockchain/hedera/scripts/hedera_client.py

# Network stats
python3 $SCRIPT stats
python3 $SCRIPT --network testnet stats

# Account
python3 $SCRIPT account 0.0.1234
python3 $SCRIPT account 0.0.1234 --no-prices          # faster, skip CoinGecko
python3 $SCRIPT --network testnet account 0.0.9999

# Token
python3 $SCRIPT token 0.0.731861                      # HTS token metadata

# Transactions
python3 $SCRIPT tx 0.0.1234-1234567890-000000000
python3 $SCRIPT tx 0.0.1234@1234567890.000000000      # SDK form also accepted
python3 $SCRIPT activity 0.0.1234 --limit 10

# NFTs
python3 $SCRIPT nft 0.0.1234

# Price
python3 $SCRIPT price HBAR
python3 $SCRIPT price SAUCE
python3 $SCRIPT price 0.0.731861

# Fees
python3 $SCRIPT fees

# HCS topic
python3 $SCRIPT topic 0.0.1234
python3 $SCRIPT topic 0.0.1234 --messages 25

# Smart contract
python3 $SCRIPT contract 0.0.5678
python3 $SCRIPT contract 0xabcdef1234567890abcdef1234567890abcdef12
```

---

## Procedure

### 0. Setup Check

```bash
python3 --version   # 3.8+ required

# Confirm mainnet connectivity
python3 ~/.hermes/skills/blockchain/hedera/scripts/hedera_client.py stats

# Confirm testnet connectivity
python3 ~/.hermes/skills/blockchain/hedera/scripts/hedera_client.py --network testnet stats
```

### 1. Network Stats

Current block, HBAR price, circulating supply, market cap, and active node count.

```bash
python3 $SCRIPT stats
python3 $SCRIPT --network testnet stats
```

Output: `latest_block`, `hbar_price_usd`, `released_supply_hbar`, `total_supply_hbar`,
`market_cap_usd`, `node_count`, `network`, `explorer`.

### 2. Account Portfolio

HBAR balance in HBAR and USD, plus HTS token holdings. Token metadata is fetched
concurrently; enrichment is capped at 10 tokens to stay within rate limits.

```bash
python3 $SCRIPT account 0.0.1234567
python3 $SCRIPT account 0.0.1234567 --no-prices         # skip CoinGecko, raw balances only
python3 $SCRIPT account 0xYourEvmAddress                # EVM alias also accepted
```

Output: `hbar_balance`, `hbar_value_usd`, `tokens` (sorted by USD value desc),
`total_portfolio_usd`, `evm_address`, `hashscan_url`.

If the account has more than 10 associated tokens, `tokens_omitted` shows how many
were skipped. Use `token` + `read_file` to inspect specific tokens.

### 3. Token Metadata

Full HTS token info: name, type (fungible or NFT), supply, treasury, custom fees,
and key configuration. Key presence is shown as `true`/`false`; raw key material is
never exposed.

```bash
python3 $SCRIPT token 0.0.731861
```

Output: `token_id`, `name`, `symbol`, `type`, `decimals`, `total_supply`,
`treasury_account_id`, `admin_key` (bool), `supply_key` (bool), `freeze_key` (bool),
`kyc_key` (bool), `wipe_key` (bool), `pause_key` (bool), `custom_fees`, `hashscan_url`.

Token types:
- `FUNGIBLE_COMMON` — divisible fungible token (like ERC-20)
- `NON_FUNGIBLE_UNIQUE` — NFT collection (like ERC-721); each unit has a serial number

### 4. Transaction Details

Look up any transaction by its ID. Both the canonical dash form and the SDK
`@`-delimited form are accepted and normalized automatically.

```bash
# Canonical form
python3 $SCRIPT tx 0.0.1234-1234567890-000000000

# SDK form (same result)
python3 $SCRIPT tx 0.0.1234@1234567890.000000000
```

Output: `transaction_id`, `type`, `result`, `consensus_timestamp`,
`charged_tx_fee_hbar`, `charged_tx_fee_usd`, `memo` (decoded), `transfers`,
`token_transfers` (if any), `nft_transfers` (if any), `hashscan_url`.

Common `type` values: `CRYPTOTRANSFER`, `TOKENMINT`, `TOKENTRANSFERS`,
`CONSENSUSSUBMITMESSAGE`, `CONTRACTCALL`, `TOKENCREATION`.

### 5. Recent Activity

Recent transactions for an account, most recent first.

```bash
python3 $SCRIPT activity 0.0.1234567
python3 $SCRIPT activity 0.0.1234567 --limit 50
```

Output: list of `{transaction_id, type, result, consensus_timestamp, fee_hbar}`.

### 6. NFT Holdings

NFTs held by an account, decoded and grouped by token collection. NFT metadata
is base64-encoded in the mirror node; the script decodes to UTF-8 text when valid
(typically an IPFS URI or JSON), otherwise shows hex.

```bash
python3 $SCRIPT nft 0.0.1234567
python3 $SCRIPT nft 0.0.1234567 --limit 100
```

Output: `total_nfts`, `collections` (grouped by `token_id` with `serial_number`
and `metadata` per NFT).

### 7. Price Lookup

HBAR price, or any HTS token price for tokens in the known registry.

```bash
python3 $SCRIPT price HBAR
python3 $SCRIPT price SAUCE
python3 $SCRIPT price 0.0.731861     # by token ID
```

Returns `price_usd` from CoinGecko for known tokens; `price_usd: null` with a note
for tokens not in the registry. To add a token, update `KNOWN_TOKENS` in the script.

### 8. Fee Schedule

Current cost in USD and HBAR for the most common Hedera operations. HBAR cost is
computed at the live exchange rate from the mirror node.

```bash
python3 $SCRIPT fees
```

Output: `hbar_per_usd`, `operations` list with `cost_usd` and `cost_hbar` per operation,
`fee_schedule_version` (the published schedule the USD costs come from),
`exchange_rate_source`.

Operations covered: `CryptoCreate`, `CryptoTransfer (HBAR/HTS/NFT)`, `TokenCreate`,
`TokenAssociate`, `TokenMint (fungible/NFT)`, `ConsensusCreateTopic`,
`ConsensusSubmitMessage`, `ContractCreate`, `ContractCall`, `FileCreate`.

Note: `ContractCreate` and `ContractCall` show the base fee only. Actual cost also
includes a per-gas charge; see the Hedera fee schedule URL in the output.

### 9. HCS Topic Inspection

Hedera Consensus Service (HCS) lets anyone publish an ordered, timestamped message
log to a public topic. Topics are used for audit trails, DID registries, supply-chain
records, and more.

```bash
python3 $SCRIPT topic 0.0.1234567
python3 $SCRIPT topic 0.0.1234567 --messages 25
```

Output: `topic_id`, `memo`, `admin_key` (bool), `submit_key` (bool),
`auto_renew_period`, `created_timestamp`, `recent_messages` (most recent first).

Each message includes `sequence_number`, `consensus_timestamp`, `message` (decoded
from base64 — UTF-8 text or hex for binary content), and `running_hash_prefix`.

### 10. Smart Contract Info

Maps between the `0.0.XXXX` Hedera account ID and the `0x...` EVM address for a
deployed contract, and shows its bytecode size, admin key, and auto-renew settings.

```bash
python3 $SCRIPT contract 0.0.5678
python3 $SCRIPT contract 0xabcdef1234567890abcdef1234567890abcdef12
```

Output: `contract_id`, `evm_address`, `admin_key` (bool), `auto_renew_account_id`,
`auto_renew_period`, `bytecode_size_bytes`, `balance_hbar`, `hashscan_url`.

---

## Supported Networks

| Key       | Mirror Node                                        | Explorer                        |
|-----------|----------------------------------------------------|---------------------------------|
| mainnet   | mainnet-public.mirrornode.hedera.com               | hashscan.io/mainnet             |
| testnet   | testnet.mirrornode.hedera.com                      | hashscan.io/testnet             |

Override either with `HEDERA_MIRROR_URL` (takes precedence over `--network`).

---

## Pitfalls

- **Token cap on `account`**: only the first 10 associated tokens are enriched with
  metadata and prices. Accounts with many token associations show `tokens_omitted`.
  Use the `token` command to inspect specific tokens individually.
- **Account ID vs. EVM address**: a Hedera account has both a `0.0.XXXX` ID and a
  derived EVM address. Both forms are accepted; the mirror node resolves either.
- **Transaction ID formats**: the canonical form uses dashes (`0.0.X-SSSS-NNNNN`);
  the Hedera SDK uses `@` and `.` (`0.0.X@SSSS.NNNNN`). The script normalizes both.
- **CoinGecko rate limits**: the free tier allows ~10-30 req/min. Use `--no-prices`
  on `account` to skip price lookups entirely. Unknown tokens return `price_usd: null`.
- **`fees` shows base costs only**: `ContractCreate` and `ContractCall` list the base
  fee; actual cost includes a per-gas charge that depends on execution complexity.
  The fee schedule URL in the output links to the full Hedera pricing documentation.
- **HCS `message` field is base64**: the script decodes to UTF-8 when valid; raw binary
  content (e.g., protobuf payloads) appears as a hex string instead.
- **Mirror node history window**: the public mirror nodes retain full history, but
  very old transactions (pre-mirror-node era) may be unavailable.
- **Testnet account IDs are independent of mainnet**: `0.0.1234` on testnet is a
  different account than `0.0.1234` on mainnet. Always pass `--network testnet`
  explicitly when working with testnet data.

---

## Verification

```bash
# Should print latest Hedera block, node count, and HBAR price
python3 ~/.hermes/skills/blockchain/hedera/scripts/hedera_client.py stats

# Should print testnet block (different from mainnet)
python3 ~/.hermes/skills/blockchain/hedera/scripts/hedera_client.py --network testnet stats

# Should print fee table with HBAR costs at live rate
python3 ~/.hermes/skills/blockchain/hedera/scripts/hedera_client.py fees
```