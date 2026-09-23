"""Recording a reply opens exactly one SQLite connection.

``record_obligation`` runs on every outbound final response. Its retention prune must run on the
recording connection, inside the recording transaction — the way the cron ledgers prune — not on a
second connection opened after the first one closed.
"""
from __future__ import annotations

import sqlite3

from gateway import delivery_ledger as dl


def test_recording_a_reply_does_not_open_a_second_connection(tmp_path, monkeypatch):
    monkeypatch.setattr(dl, "_db_path", lambda: tmp_path / "state.db")
    real_connect = sqlite3.connect
    opened: list[str] = []

    def counting_connect(*args, **kwargs):
        opened.append(str(args[0]) if args else str(kwargs.get("database")))
        return real_connect(*args, **kwargs)

    monkeypatch.setattr(sqlite3, "connect", counting_connect)
    dl.record_obligation(obligation_id="ob-warm", session_key="s", platform="p", chat_id="c",
                         thread_id=None, content="warm-up (schema)")
    opened.clear()

    dl.record_obligation(obligation_id="ob-1", session_key="s", platform="p", chat_id="c",
                         thread_id=None, content="hello")

    assert len(opened) == 1, opened
