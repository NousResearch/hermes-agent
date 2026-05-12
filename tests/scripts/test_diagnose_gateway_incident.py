from __future__ import annotations

import importlib.util
import sqlite3
import sys
from datetime import datetime
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "diagnose_gateway_incident.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("diagnose_gateway_incident", SCRIPT_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules["diagnose_gateway_incident"] = module
    spec.loader.exec_module(module)
    return module


def test_detects_flush_without_inbound_with_near_stale_restart_and_send_failure(tmp_path):
    mod = _load_script()
    log = tmp_path / "gateway.log"
    log.write_text(
        "\n".join(
            [
                "2026-05-03 12:17:30,041 INFO [Telegram] Flushing text batch telegram:dm:-1001234567890 (21 chars)",
                "2026-05-03 12:17:30,042 INFO Stale-code restart triggered by user message; event queued for replay: platform=telegram chat=-1001234567890 update_id=10 journal_id=abc",
                "2026-05-03 12:17:30,043 ERROR [Telegram] Failed to send media (): invalid target -1001234567890",
                "2026-05-03 12:25:50,000 INFO inbound message: platform=telegram chat=-1001234567890 user=Example User",
            ]
        ),
        encoding="utf-8",
    )

    result = mod.diagnose(
        log_path=log,
        db_path=tmp_path / "missing.db",
        since=datetime(2026, 5, 3, 12, 17, 0),
        until=datetime(2026, 5, 3, 12, 18, 0),
        chat_id="-1001234567890",
        window_seconds=20,
    )

    assert result["incident_detected"] is True
    gap = result["log_scan"]["flush_without_inbound"][0]
    assert "Flushing text batch" in gap["flush"]["line"]
    assert gap["near_stale_restarts"]
    assert gap["near_send_failures"]
    assert "Potential flush-without-inbound" in result["summary"][0]


def test_no_gap_when_inbound_follows_flush(tmp_path):
    mod = _load_script()
    log = tmp_path / "gateway.log"
    log.write_text(
        "\n".join(
            [
                "2026-05-03 12:17:30,041 INFO [Telegram] Flushing text batch telegram:dm:-1001234567890 (21 chars)",
                "2026-05-03 12:17:31,000 INFO inbound message: platform=telegram chat=-1001234567890 user=Example User",
            ]
        ),
        encoding="utf-8",
    )

    result = mod.diagnose(
        log_path=log,
        db_path=tmp_path / "missing.db",
        since=datetime(2026, 5, 3, 12, 17, 0),
        until=datetime(2026, 5, 3, 12, 18, 0),
        chat_id="-1001234567890",
        window_seconds=20,
    )

    assert result["incident_detected"] is False
    assert result["log_scan"]["flush_without_inbound"] == []


def test_counts_state_db_telegram_rows_in_window(tmp_path):
    mod = _load_script()
    db = tmp_path / "state.db"
    conn = sqlite3.connect(db)
    conn.executescript(
        """
        CREATE TABLE sessions (id TEXT PRIMARY KEY, source TEXT NOT NULL, user_id TEXT);
        CREATE TABLE messages (id INTEGER PRIMARY KEY, session_id TEXT NOT NULL, role TEXT NOT NULL, content TEXT, timestamp REAL NOT NULL);
        INSERT INTO sessions (id, source, user_id) VALUES ('s1', 'telegram', 'u1'), ('s2', 'cli', 'u1');
        INSERT INTO messages (session_id, role, content, timestamp) VALUES ('s1', 'user', 'hello', 1777810650.0), ('s2', 'user', 'cli', 1777810650.0);
        """
    )
    conn.commit()
    conn.close()

    result = mod.count_state_rows(
        db,
        since=datetime.fromtimestamp(1777810640.0),
        until=datetime.fromtimestamp(1777810660.0),
    )

    assert result["telegram_message_rows_in_window"] == 1
