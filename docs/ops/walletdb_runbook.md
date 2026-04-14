# WalletDB Runbook (Framework Upgrades)

## Quick Start
**Prereqs:** `PYTHONPATH=src` and a WalletDB SQLite path (default `state/walletdb_helius_phase1.sqlite`).

```bash
export PYTHONPATH=src
export WALLETDB_PHASE1_DB=state/walletdb_helius_phase1.sqlite
```

---

## 0) Optional NGMI context (DISABLED by default)

WalletDB should not depend on NGMI Terminal.

If you *explicitly* want NGMI chat-log context (e.g., legacy Trend Digest tooling or `Analyze` button callbacks), you must opt-in:

```bash
export WALLETDB_ENABLE_NGMI_CONTEXT=1
export NGMI_DB_PATH=/path/to/ngmi_terminal.db  # or WALLETDB_NGMI_DB
```

If `WALLETDB_ENABLE_NGMI_CONTEXT` is unset/0, NGMI-dependent flows should be treated as disabled.

See also: `docs/ops/walletdb_ngmi_context.md`.

---

## 1) Cielo Telegram Alert Ingestion (Telethon)
Listen to Cielo bot alerts from Telegram group `-5106082943` and ingest into WalletDB:

```bash
export PYTHONPATH=src
export WALLETDB_PHASE1_DB=state/walletdb_helius_phase1.sqlite
export TELETHON_API_ID=...      # Telegram API ID
export TELETHON_API_HASH=...    # Telegram API hash
export WALLETDB_CIELO_TELEGRAM_GROUP=-5106082943
export WALLETDB_CIELO_TELETHON_SESSION=state/cielo_telethon
export WALLETDB_CIELO_ALERT_MAX_HOPS=2

python -m walletdb.telegram.cielo_telethon_cli
```

Notes:
- Alerts are stored in `walletdb_cielo_alerts` + `walletdb_cielo_alert_wallets`.
- Optional expansion writes to `walletdb_cielo_expanded_wallets` and triggers CA alerts for dead-end wallets.

---

## 2) Cielo Sync Queue (Auto‑tracking)
When alerts are ingested (RAVN or Cielo), wallets are automatically enqueued for Cielo tracking
sync. Run the sync worker to upload queued wallets to Cielo (rate‑limited, retryable, resumable).

```bash
export PYTHONPATH=src
export WALLETDB_PHASE1_DB=state/walletdb_helius_phase1.sqlite
export CIELO_API_KEY=...            # required
export CIELO_BASE_URL=...          # optional (default: https://feed-api.cielo.finance)
# legacy: export CIELO_API_URL=... # still supported
export CIELO_USE_AUTHORIZATION=1   # optional: also send Authorization header
export WALLETDB_CIELO_LIST_ID=...  # optional: target list
export CIELO_TRACK_BUDGET=200      # optional daily credit cap
export CIELO_RATE_LIMIT_PER_SEC=10 # optional credit/sec limiter
export CIELO_POLL_MAX_S=60         # optional 202 poll budget
export CIELO_PROXY=...             # optional proxy (overrides HTTPS_PROXY/HTTP_PROXY)
# or: export HTTPS_PROXY=... / HTTP_PROXY=...

python -m walletdb.jobs.cielo_sync_cli --run --max-jobs 25 --min-delay 1.5
```

Check queue:
```bash
python -m walletdb.jobs.cielo_sync_cli --list --status queued
```

Schema notes:
- Cielo wallet_id <-> address mappings stored in `walletdb_cielo_wallet_map` (used for v1 update/remove).

---

## 3) Label Ingestion (Multi‑source)
Labels are produced automatically in the RAVN pipeline (`run_ravn_report`).
Tune sources via env:

```bash
export WALLETDB_LABEL_SOURCES=authorities,pair_lp,top_holders,bundle_candidates,holder_clusters,edge_activity,cielo_tags
export WALLETDB_LABEL_HOLDER_LIMIT=25
```

---

## 4) Backfill Tokens
Backfill: long‑horizon history + holders + top traders + traces.

```bash
PYTHONPATH=src python -m walletdb.jobs.backfill_cli \
  --token <MINT> \
  --token <MINT2>
```

Token list file:
```bash
PYTHONPATH=src python -m walletdb.jobs.backfill_cli --token-file tokens.txt
```

---

## 5) History Jobs (Queue)
Enqueue addresses:
```bash
PYTHONPATH=src python -m walletdb.jobs.history_cli --enqueue <ADDRESS> --token-mint <MINT>
```

Run jobs (requires Helius API key in env):
```bash
export HELIUS_API_KEY=... # keep private
PYTHONPATH=src python -m walletdb.jobs.history_cli --run --max-jobs 2 --pages 6
```

List queue:
```bash
PYTHONPATH=src python -m walletdb.jobs.history_cli --list
```

---

## 6) Case Notes / Triage
Create a case + notes from Python:

```python
from walletdb.case.triage import create_case, add_case_note
import sqlite3

conn = sqlite3.connect("state/walletdb_helius_phase1.sqlite")
case_id = create_case(conn, token_mint="<MINT>", summary="Cluster flagged", priority="high")
add_case_note(conn, case_id=case_id, note="Initial triage complete")
```

Queue triage items from alerts:
```python
from walletdb.case.triage import upsert_triage_item
upsert_triage_item(conn, token_mint="<MINT>", severity="medium", reason="holder_cluster")
```

---

## Troubleshooting
- **No labels:** confirm label sources enabled and data exists in `walletdb_wallet_edges` and holders snapshots.
- **History jobs stuck:** verify API key, and ensure the address is indexed by Helius.
- **Backfill has empty traders:** check that history ingestion has populated `walletdb_wallet_edges`.
