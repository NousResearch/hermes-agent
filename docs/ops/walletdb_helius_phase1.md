# WalletDB Helius Phase 1 — Bundle Tracing (Spec + Notes)

## Scope
Phase 1 focuses on **Helius-first ingestion** + **bundle detection** using enhanced transactions and webhook payloads. It persists transactions, wallet graph edges, token metadata, and produces low-noise bundle alerts.

## Bundle Detection Spec (Phase 1)
A **bundle candidate** is a connected component of wallets linked by token transfer edges derived from Helius enhanced transactions.

**Inputs**
- Helius Enhanced Transactions (REST) or Enhanced Webhook payloads.
- Token transfer edges: `(from_user_account → to_user_account, mint, amount, signature)`.

**Graph**
- Build an undirected graph per token mint.
- Nodes: wallet addresses.
- Edges: token transfer edges (from/to).

**Candidate**
- Connected component per mint.
- Metrics: `wallet_count`, `edge_count`, `max_hop` (graph distance capped at N).

**Low-noise alert rule (Phase 1)**
Trigger an alert when:
- `wallet_count >= 5`
- `edge_count >= 4`
- `max_hop <= 2`
- and the candidate has not been alerted before for the same rule.

These thresholds can be tuned without schema changes.

## Ingestion Flow
1. **Enhanced TX REST** (`/v0/addresses/{address}/transactions`)
2. **Enhanced Webhook** payloads (`transactions` list)
3. Persist into:
   - `helius_enhanced_transactions` (raw JSON + meta)
   - `walletdb_wallet_edges` (token transfer edges)
   - `walletdb_token_meta` (mint metadata as seen)

## Hop Tracking
- `track_hops(seed_wallets, token_mint, max_hops=2)` computes BFS hops from seed wallets.
- Results stored in `walletdb_wallet_hops`.

## Tables (Phase 1)
- `helius_enhanced_transactions`
- `walletdb_wallet_edges`
- `walletdb_token_meta`
- `walletdb_bundle_candidates`
- `walletdb_wallet_hops`
- `walletdb_bundle_alerts`

## Smoke Check
Run a quick dry-run from the WalletDB repo:

```bash
cd /home/yeqiuqiu/projects/walletdb
PYTHONPATH=src python -m walletdb.bundles.helius_phase1_cli --help
```

## CLI Runner
Phase 1 runner (Helius fetch + ingest + bundle detection + alert dispatch):

```bash
export HELIUS_API_KEY=... 
PYTHONPATH=src python -m walletdb.bundles.helius_phase1_cli \
  --address <solana_address> \
  --ca <token_mint> \
  --limit 50 \
  --min-wallets 5 \
  --min-edges 4 \
  --max-hop 2
```

**Address guidance:** `--address` should be a Solana account that Helius indexes for transactions
(e.g., wallet, program, **pair/LP account**). For token analysis, prefer the **pairAddress**
from DexScreener (or the pool account on your DEX). Using the **mint** can return sparse or
incomplete transaction history depending on the token/DEX.

`--ca` is an alias for `--token-mint` to filter bundle detection by mint.

Convenience runner (repo-local):

```bash
cd /home/yeqiuqiu/projects/walletdb
scripts/run_walletdb.sh -m walletdb.bundles.helius_phase1_cli -- --address <solana_address>
```

Alerts are deduped by (rule + cluster_id) and gated by a cooldown window
(default 60 minutes). Alerts are written to `walletdb_alert_spool.jsonl`.

## Cron Example

```bash
export HELIUS_API_KEY=...
export WALLETDB_PHASE1_ADDRESSES=addr1,addr2
*/5 * * * * /home/yeqiuqiu/projects/walletdb/scripts/cron_watch_runner.sh >>/tmp/walletdb_watch_runner.log 2>&1
```
