# WalletDB — NGMI Context (Opt-in)

WalletDB is intended to run independently.

Some legacy workflows can optionally use **NGMI Terminal** context (e.g., reading
Telegram chat logs from NGMI's sqlite DB, or handling legacy `analyze:*` Telegram
callbacks).

## Default: OFF

By default, NGMI coupling is disabled.

Enable it only when you explicitly want it:

```bash
export WALLETDB_ENABLE_NGMI_CONTEXT=1
export NGMI_DB_PATH=/home/yeqiuqiu/projects/ngmi-terminal/data/ngmi_terminal.db  # or WALLETDB_NGMI_DB
```

## CA pipeline

The CA pipeline's NGMI DB lookup is opt-in:

```bash
PYTHONPATH=src python -m walletdb.ca.solana_pipeline_cli \
  --token-mint <MINT> \
  --ngmi-db "$NGMI_DB_PATH"
```

If `--ngmi-db` is not provided, WalletDB will not attempt to read any NGMI DB.
