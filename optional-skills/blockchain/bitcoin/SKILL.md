---
name: bitcoin
description: "Read-only Bitcoin research client for on-chain data."
version: 1.0.0
author: Hermes Agent
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [Bitcoin, Blockchain, Crypto, On-chain, Research, OSINT]
    category: blockchain
    related_skills: [evm, solana]
    requires_toolsets: [terminal]
---

# Bitcoin Skill

Read-only Bitcoin research tool for verification and on-chain analysis.
15 commands: `address`, `txs`, `tx`, `utxo`, `block`, `mempool`, `fees`, `stats`, `price`, `whale`, `verify`, `report`, `compare`, `fee-history`, `editorial`.

No API key needed. Python standard library only (`urllib`, `json`, `argparse`).
Export to JSON, CSV, or TSV. Automatic fallback to blockstream.info when
mempool.space is unreachable.

Data sources:
- **mempool.space** public API for on-chain data, mempool, fees, hashrate, difficulty
- **blockstream.info** public API as automatic fallback when mempool.space is unreachable or rate-limited
- **CoinGecko** public API for BTC price and market data

> **Editorial use only.** This skill does not provide financial advice, price
> predictions, or trading signals. Always cite sources when writing for publication.

---

## When to Use

- User asks to verify a Bitcoin address or transaction mentioned in an article.
- User wants balance, transaction count, or fiat value of an address.
- User wants details of a specific transaction: inputs, outputs, fee, fee rate.
- User wants block data: height, timestamp, transaction count, subsidy.
- User wants current mempool congestion and recommended fee rates.
- User wants network stats: hashrate, difficulty, retarget estimate, BTC price.
- User wants a quick whale scan of large unconfirmed mempool transactions.
- User wants to fact-check a claim and needs a checklist for Bitcoin stories.
- User is preparing an article or report and needs verifiable on-chain data.

---

## Prerequisites

Python 3.8+ standard library only. No pip installs required.

Pricing: CoinGecko free API (rate-limited, ~10-30 req/min).
On-chain data: mempool.space public API (rate-limited, free).

Helper script path: `~/.hermes/skills/blockchain/bitcoin/scripts/bitcoin_client.py`

---

## Quick Reference

```bash
SCRIPT=~/.hermes/skills/blockchain/bitcoin/scripts/bitcoin_client.py

# Address (confirmed + unconfirmed balance, tx counts, fiat value)
python3 $SCRIPT address <address> [--currency usd|eur] [--no-fiat]

# Recent transactions for an address
python3 $SCRIPT txs <address> [--limit 25] [--currency usd|eur] [--no-fiat] [--format json|csv|tsv]

# Spendable UTXOs for an address
python3 $SCRIPT utxo <address> [--currency usd|eur] [--no-fiat] [--format json|csv|tsv]

# Transaction details
python3 $SCRIPT tx <txid> [--verbose] [--currency usd|eur] [--no-fiat]

# Block (by height or hash)
python3 $SCRIPT block <height_or_hash>

# Mempool summary
python3 $SCRIPT mempool

# Recommended fees
python3 $SCRIPT fees

# Network stats + BTC price
python3 $SCRIPT stats [--currency usd|eur] [--no-fiat]

# BTC price
python3 $SCRIPT price [--currencies usd,eur]

# Large unconfirmed transactions (whale watch)
python3 $SCRIPT whale [--threshold 1.0] [--limit 10] [--currency usd|eur] [--no-fiat]

# Verify address or txid existence
python3 $SCRIPT verify <address_or_txid>

# Combined report for an address or transaction
python3 $SCRIPT report <address_or_txid> [--currency usd|eur] [--no-fiat]

# Compare multiple addresses
python3 $SCRIPT compare <address1> <address2> ... [--currency usd|eur] [--no-fiat] [--format json|csv|tsv]

# Historical fee rate distribution
python3 $SCRIPT fee-history [--period 1w|1m|3m|6m|1y] [--format json|csv|tsv]

# Editorial fact-check checklist
python3 $SCRIPT editorial [general|address|txs|tx|utxo|fees|mining|price|compare|report]
```

---

## Procedure

### 0. Setup Check

```bash
python3 --version   # 3.8+ required
python3 ~/.hermes/skills/blockchain/bitcoin/scripts/bitcoin_client.py fees
```

### 1. Address Lookup

```bash
python3 ~/.hermes/skills/blockchain/bitcoin/scripts/bitcoin_client.py \
  address bc1qxy2kgdygjrsqtzq2n0yrf2493p83kkfjhx0wlh
```

Output: confirmed balance, unconfirmed balance, transaction counts,
fiat value at current BTC price.

### 2. Address Transaction History

```bash
python3 ~/.hermes/skills/blockchain/bitcoin/scripts/bitcoin_client.py \
  txs bc1qxy2kgdygjrsqtzq2n0yrf2493p83kkfjhx0wlh --limit 25 --format csv
```

Lists the most recent confirmed transactions for the address. For each
transaction reports txid, block height, time, fee rate, and the net value
moved to/from the address. Defaults to the latest 25 transactions returned
by mempool.space. Use `--format csv` or `--format tsv` for spreadsheet import.

### 3. Address UTXOs

```bash
python3 ~/.hermes/skills/blockchain/bitcoin/scripts/bitcoin_client.py \
  utxo bc1qxy2kgdygjrsqtzq2n0yrf2493p83kkfjhx0wlh
```

Lists unspent transaction outputs (UTXOs) for the address, with value,
confirmation status, and block reference. Useful for deeper address analysis
and wallet research.

### 4. Transaction Details

```bash
python3 ~/.hermes/skills/blockchain/bitcoin/scripts/bitcoin_client.py \
  tx 4a5e1e4baab89f3a32518a88c31bc87f618f76673e2cc77ab2127b7afdeda33b
```

Use `--verbose` to include full input and output lists. Output includes
confirmation status, block, fee, fee rate, size, input/output totals.

### 5. Block Lookup

```bash
python3 ~/.hermes/skills/blockchain/bitcoin/scripts/bitcoin_client.py block 840000
python3 ~/.hermes/skills/blockchain/bitcoin/scripts/bitcoin_client.py \
  block 0000000000000000000320283a032748cef8227873ff4872689bf23f1cda83a5
```

Output includes height, timestamp, transaction count, difficulty, and
calculated coinbase subsidy. Total reward (subsidy + fees) is **not**
returned by mempool.space for this endpoint anymore, so fees and total
reward are left empty with an explanatory note.

### 6. Mempool and Fees

```bash
python3 ~/.hermes/skills/blockchain/bitcoin/scripts/bitcoin_client.py mempool
python3 ~/.hermes/skills/blockchain/bitcoin/scripts/bitcoin_client.py fees
```

Use these together to understand network congestion before writing about
fee spikes or transaction delays.

### 7. Network Stats

```bash
python3 ~/.hermes/skills/blockchain/bitcoin/scripts/bitcoin_client.py stats
```

Output: current tip height/hash, hashrate (formatted in EH/s), 7-day
average hashrate, difficulty, remaining blocks to retarget, estimated
retarget change, BTC price, 24h change.

### 8. Price Check

```bash
python3 ~/.hermes/skills/blockchain/bitcoin/scripts/bitcoin_client.py price
python3 ~/.hermes/skills/blockchain/bitcoin/scripts/bitcoin_client.py price --currencies usd,eur,gbp
```

Output: BTC price, market cap, 24h volume, and 24h change for each
currency requested.

### 9. Whale Watch

```bash
python3 ~/.hermes/skills/blockchain/bitcoin/scripts/bitcoin_client.py whale
python3 ~/.hermes/skills/blockchain/bitcoin/scripts/bitcoin_client.py whale --threshold 5 --limit 5
```

Scans the ~10 most recently arrived mempool transactions and returns those
above the threshold. Useful for spotting large unconfirmed movements, but
**not exhaustive** — it does not scan the full mempool.

### 10. Verify an Address or Transaction

```bash
python3 ~/.hermes/skills/blockchain/bitcoin/scripts/bitcoin_client.py \
  verify bc1qxy2kgdygjrsqtzq2n0yrf2493p83kkfjhx0wlh
python3 ~/.hermes/skills/blockchain/bitcoin/scripts/bitcoin_client.py \
  verify 4a5e1e4baab89f3a32518a88c31bc87f618f76673e2cc77ab2127b7afdeda33b
```

Returns `valid: true` if the identifier exists on-chain. Useful for
fact-checking claims in articles or social media posts.

### 11. Combined Report

```bash
python3 ~/.hermes/skills/blockchain/bitcoin/scripts/bitcoin_client.py \
  report bc1qxy2kgdygjrsqtzq2n0yrf2493p83kkfjhx0wlh
python3 ~/.hermes/skills/blockchain/bitcoin/scripts/bitcoin_client.py \
  report 4a5e1e4baab89f3a32518a88c31bc87f618f76673e2cc77ab2127b7afdeda33b
```

Returns a single-page summary of an address (balance, recent transactions)
or a transaction (fee, inputs/outputs), convenient for quick fact-checking.

### 12. Compare Addresses

```bash
python3 ~/.hermes/skills/blockchain/bitcoin/scripts/bitcoin_client.py \
  compare addr1 addr2 addr3 --format csv
```

Compares balances, transaction counts, and unconfirmed activity across
multiple addresses. Useful for clustering research or exchange analysis.

### 13. Historical Fee Rates

```bash
python3 ~/.hermes/skills/blockchain/bitcoin/scripts/bitcoin_client.py fee-history --period 1m
```

Returns per-block fee-rate distributions (min, p10, median, p90, max) for
the selected period. Includes summary statistics and a sample of recent
blocks to give context on whether current fees are high or low.

### 14. Editorial Checklist

```bash
python3 ~/.hermes/skills/blockchain/bitcoin/scripts/bitcoin_client.py editorial
python3 ~/.hermes/skills/blockchain/bitcoin/scripts/bitcoin_client.py editorial tx
python3 ~/.hermes/skills/blockchain/bitcoin/scripts/bitcoin_client.py editorial mining
```

Prints a JSON checklist tailored to the topic. Use before publishing
Bitcoin stories to catch common sourcing and interpretation mistakes.

---

## Supported Data and Sources

| Data | Source | Notes |
|------|--------|-------|
| Address balance, tx counts | mempool.space `/address` | Confirmed vs unconfirmed split. Automatic fallback to blockstream.info. |
| Address transaction history | mempool.space `/address/{addr}/txs` | Last 50 confirmed transactions (default return); `--limit N` shows first N from the list. |
| Address UTXOs | mempool.space `/address/{addr}/utxo` | Spendable outputs with confirmation status. |
| Transaction details | mempool.space `/tx` | Includes fee, fee rate, confirmation status. |
| Block metadata | mempool.space `/block` | Subsidy calculated locally; fees/reward no longer provided by API. |
| Mempool summary | mempool.space `/mempool` | vsize, count, fee histogram. |
| Fee rates | mempool.space `/v1/fees/recommended` | fastest/halfHour/hour/economy/minimum. |
| Fee history | mempool.space `/v1/mining/blocks/fee-rates/{period}` | min/p10/median/p90/max per block. |
| Hashrate / difficulty | mempool.space `/v1/mining/hashrate` and difficulty-adjustments | 7-day average from daily buckets. |
| BTC price | CoinGecko `/simple/price` | Market cap, 24h volume, 24h change. |
| Whale watch | mempool.space `/mempool/recent` | Last ~10 arrivals only. |

---

## Editorial Guidelines

When using this skill to support articles, reports, or fact-checks:

1. **Always cite the source.** For on-chain data, mention mempool.space or
the block explorer used. For price data, mention CoinGecko.
2. **Distinguish confirmed from unconfirmed.** Unconfirmed balances and
mempool data can change quickly.
3. **Do not infer ownership.** An address balance does not identify who
controls it unless there is public, verifiable attribution.
4. **Avoid price predictions.** The skill reports current and historical
data; it does not forecast.
5. **Cross-check sensitive claims.** For high-stakes stories, verify
addresses and transactions against more than one explorer.
6. **Use the editorial checklist.** Run `editorial <topic>` before final
review to catch sloppy claims about fees, mining, or large transactions.

---

## Pitfalls

- **mempool.space rate limits and fallbacks**: the client automatically falls back to blockstream.info when mempool.space returns 5xx errors or is unreachable. For 429 throttling, wait a minute and retry, or use `--no-fiat` to reduce total requests.
- **CSV/TSV format**: several commands (`txs`, `utxo`, `compare`, `fee-history`) support `--format csv` and `--format tsv` for spreadsheet import.
- **CoinGecko rate limits**: price lookups share the same free-tier pool.
Avoid rapid sequential calls with many currencies. Use `--no-fiat` to skip
price conversion entirely when only BTC values are needed.
- **Address verification is not validation**: `verify` confirms the address
was used on-chain, not that it is currently safe or legitimate.
- **Transaction verification depends on network propagation**: a very recent
transaction may not appear immediately across all API endpoints.
- **Block reward vs subsidy**: the `subsidy` field is the fixed coinbase
emission calculated from block height. The total block reward also includes
fees, which are no longer returned by the mempool.space `/block` endpoint
used here.
- **Privacy heuristics are out of scope**: this skill does not perform
clustering, taint analysis, or address de-anonymization.
- **Whale watch is not exhaustive**: it only inspects the last ~10
recently-arrived mempool transactions.
- **No testnet support**: this version queries mainnet only.

---

## Verification

```bash
# Should print current recommended Bitcoin fee rates
python3 ~/.hermes/skills/blockchain/bitcoin/scripts/bitcoin_client.py fees

# Should print the latest block height, hashrate, and BTC price
python3 ~/.hermes/skills/blockchain/bitcoin/scripts/bitcoin_client.py stats
```

---

## License

MIT
