#!/usr/bin/env python3
"""Verify runner for WalletDB Intel Ops MVP bots.

Dry-runs only; does not contact Telegram.

Usage:
  python scripts/verify_walletdb_bots.py
"""

from __future__ import annotations

import json
import pathlib
import sqlite3
import sys

# Allow running from repo checkout without installation.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from walletdb.telegram.cielo_ingest import ingest_cielo_alert
from walletdb.telegram.ravn_dm_bot import ensure_ravn_dm_tables, handle_incoming_dm
from walletdb.telegram.alert_poster import ensure_alert_poster_tables, iter_pending_outbound_alerts


def main() -> None:
    db_path = "state/verify_walletdb_bots.sqlite"
    pathlib.Path("state").mkdir(exist_ok=True)
    conn = sqlite3.connect(db_path)

    # Ensure schemas
    ensure_ravn_dm_tables(conn)
    ensure_alert_poster_tables(conn)

    # 1) Cielo ingest fixture
    cielo_text = pathlib.Path("tests/fixtures/cielo_forwarded_sample.txt").read_text(encoding="utf-8")
    res = ingest_cielo_alert(
        conn,
        text=cielo_text,
        chat_id="-5106082943",
        message_id="fixture-1",
        source="fixture",
        auto_expand=False,
        alert_dead_ends=False,
    )

    # 2) RAVN DM ingest fixture (Bot A): pretend a private DM message
    dm_message = {
        "message_id": 1,
        "chat": {"id": 123, "type": "private"},
        "from": {"id": 123, "username": "fixture_user"},
        "text": "track these: https://solscan.io/account/So11111111111111111111111111111111111111112",
    }
    ack = handle_incoming_dm(conn, dm_message)

    # 3) Alert poster (Bot B) dry-run: should find BUY candidates from cielo fixture
    pending = iter_pending_outbound_alerts(conn, limit=10)

    out = {
        "db_path": db_path,
        "cielo_ingest": res.__dict__ if res else None,
        "ravn_dm_ack": ack,
        "pending_outbound": [p.__dict__ for p in pending],
    }
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
