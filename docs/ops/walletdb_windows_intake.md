# WalletDB Windows Intake + Adapter Runbook

## Blueprint

**Goal:** ingest Telegram alerts on Windows, write a local SQLite intake log with idempotency, then have a Linux/WalletDB adapter read that log and seed RAVN ingest + Cielo tracking (and optional RAVN report seeding).

### Components

1. **Windows Intake (Telegram)**
   - `walletdb/telegram/windows_intake.py`
   - Two modes: `telethon` (user session) or `bot` (Bot API polling).
   - Writes to `walletdb_windows_intake.sqlite` with `UNIQUE(source, chat_id, message_id)`.
   - Parses messages into:
     - `ravn_alert` (via `parse_ravn_alert_text`)
     - `cielo_track` (basic wallet extraction when text contains “track”/“cielo”)
     - `message` (default)

2. **WalletDB Adapter (Linux/Server)**
   - CLI: `walletdb/intake/windows_adapter_cli.py`
   - HTTP: `walletdb/intake/windows_adapter_http.py`
   - Reads pending intake rows and:
     - `ravn_alert` → `ingest_ravn_alert()`
     - `cielo_track` → `walletdb_cielo_tracking` + `enqueue_cielo_sync()`
     - Optional `run_ravn_report()` to seed labels/alerts/traces.

3. **Guardrails**
   - SQLite `busy_timeout` + retry with backoff for locked DBs.
   - Separate intake and WalletDB DBs to avoid write contention.

---

## Windows Intake Runbook

### Install (Telethon mode)

```bash
pip install telethon
python -m walletdb.telegram.windows_intake telethon \
  --api-id <id> --api-hash <hash> --group-id <tg_group_id>
```

### Install (Bot mode)

```bash
set TELEGRAM_BOT_TOKEN=<token>
python -m walletdb.telegram.windows_intake bot --group-id <tg_group_id>
```

### Output

- Default SQLite log: `state/walletdb_windows_intake.sqlite`
- Table: `walletdb_windows_intake_messages`
- Idempotent via unique `(source, chat_id, message_id)`.

---

## Adapter Runbook (Linux)

### One-shot CLI

```bash
python -m walletdb.intake.windows_adapter_cli \
  --intake-db /path/to/walletdb_windows_intake.sqlite \
  --walletdb state/walletdb_helius_phase1.sqlite \
  --seed-reports
```

### Looping CLI

```bash
python -m walletdb.intake.windows_adapter_cli --loop --interval 15 --seed-reports
```

### HTTP service

```bash
python -m walletdb.intake.windows_adapter_http --port 7845 --seed-reports
```

- `POST /process` → process intake rows
- `POST /ingest` → parse a single event payload for debugging

---

## Service Definitions

### Windows Task Scheduler (example)

- Program: `python`
- Args:

```text
-m walletdb.telegram.windows_intake telethon --api-id <id> --api-hash <hash> --group-id <tg_group_id>
```

### systemd: Adapter loop

```ini
[Unit]
Description=WalletDB Windows intake adapter
After=network.target

[Service]
Type=simple
WorkingDirectory=/home/yeqiuqiu/clawd-architect
ExecStart=/usr/bin/python -m walletdb.intake.windows_adapter_cli --loop --interval 15 --seed-reports
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
```

### systemd: Adapter HTTP

```ini
[Unit]
Description=WalletDB Windows adapter HTTP
After=network.target

[Service]
Type=simple
WorkingDirectory=/home/yeqiuqiu/clawd-architect
ExecStart=/usr/bin/python -m walletdb.intake.windows_adapter_http --port 7845 --seed-reports
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
```
