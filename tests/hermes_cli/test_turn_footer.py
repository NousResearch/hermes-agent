"""Tests for hermes_cli.turn_footer (built-in per-turn footer line)."""

import json
import os
import socket
import sqlite3
import time
from datetime import datetime, timezone

import pytest

from hermes_cli import turn_footer as tf


def _make_state_db(path):
    c = sqlite3.connect(path)
    c.executescript(
        """
        CREATE TABLE sessions (
            id TEXT PRIMARY KEY,
            last_activity_at REAL NOT NULL,
            started_at REAL NOT NULL
        );
        CREATE TABLE session_model_usage (
            session_id TEXT NOT NULL,
            model TEXT NOT NULL,
            billing_provider TEXT NOT NULL DEFAULT '',
            billing_base_url TEXT NOT NULL DEFAULT '',
            billing_mode TEXT NOT NULL DEFAULT '',
            task TEXT NOT NULL DEFAULT '',
            api_call_count INTEGER NOT NULL DEFAULT 0,
            input_tokens INTEGER NOT NULL DEFAULT 0,
            output_tokens INTEGER NOT NULL DEFAULT 0,
            cache_read_tokens INTEGER NOT NULL DEFAULT 0,
            cache_write_tokens INTEGER NOT NULL DEFAULT 0,
            reasoning_tokens INTEGER NOT NULL DEFAULT 0,
            estimated_cost_usd REAL NOT NULL DEFAULT 0,
            actual_cost_usd REAL NOT NULL DEFAULT 0,
            cost_status TEXT,
            cost_source TEXT,
            first_seen REAL,
            last_seen REAL,
            PRIMARY KEY (session_id, model, billing_provider, billing_base_url, billing_mode, task)
        );
        CREATE TABLE messages (
            id INTEGER PRIMARY KEY,
            session_id TEXT NOT NULL,
            role TEXT NOT NULL,
            timestamp REAL NOT NULL
        );
        """
    )
    return c


def _make_9r_db(path):
    c = sqlite3.connect(path)
    c.executescript(
        """
        CREATE TABLE usageHistory (
            id INTEGER PRIMARY KEY,
            timestamp TEXT NOT NULL,
            model TEXT NOT NULL,
            promptTokens INTEGER NOT NULL DEFAULT 0
        );
        """
    )
    return c


def test_fmt():
    assert tf._fmt(0) == "0"
    assert tf._fmt(5) == "5"
    assert tf._fmt(999) == "999"
    assert tf._fmt(1000) == "1.0k"
    assert tf._fmt(1500) == "1.5k"
    assert tf._fmt(166700) == "166.7k"


def test_resolve_session_prefers_last_activity_at(tmp_path, monkeypatch):
    db = tmp_path / "state.db"
    c = _make_state_db(db)
    # session A has earlier started_at but later last_activity_at
    c.execute("INSERT INTO sessions VALUES ('A', 200.0, 10.0)")
    c.execute("INSERT INTO sessions VALUES ('B', 100.0, 20.0)")
    c.commit()
    c.close()
    monkeypatch.setattr(tf, "SESSION_DB", str(db))
    conn = sqlite3.connect(str(db))
    sid = tf._resolve_session_id(conn)
    conn.close()
    assert sid == "A"


def test_disabled_returns_empty(tmp_path, monkeypatch):
    state = tmp_path / "state.db"
    _make_state_db(state).close()
    monkeypatch.setattr(tf, "SESSION_DB", str(state))
    monkeypatch.setattr(tf, "STATE_FILE", str(tmp_path / "footer-tok.json"))
    assert tf.compute_turn_footer_line(config={"display": {"turn_footer": {"enabled": False}}}) == ""


def test_compute_aggregates_sum_and_9r_model(tmp_path, monkeypatch):
    state = tmp_path / "state.db"
    c = _make_state_db(state)
    now = time.time()
    c.execute("INSERT INTO sessions VALUES ('S1', ?, ?)", (now, now - 100))
    # two rows for same session, different models/tasks
    c.execute(
        "INSERT INTO session_model_usage (session_id, model, input_tokens, output_tokens, last_seen) "
        "VALUES ('S1', 'combo', 100, 211, ?)",
        (now - 50,),
    )
    c.execute(
        "INSERT INTO session_model_usage (session_id, model, input_tokens, output_tokens, last_seen) "
        "VALUES ('S1', 'other', 1000, 2000, ?)",
        (now - 10,),
    )
    # last user message 30s ago
    c.execute("INSERT INTO messages (session_id, role, timestamp) VALUES ('S1', 'user', ?)", (now - 30,))
    c.commit()
    c.close()

    nine = tmp_path / "9r.sqlite"
    nc = _make_9r_db(nine)
    nc.execute("INSERT INTO usageHistory (timestamp, model, promptTokens) VALUES (?, ?, ?)",
               ("2026-11-06T10:04:42.241Z", "real-model-x", 5000))
    nc.commit()
    nc.close()

    monkeypatch.setattr(tf, "SESSION_DB", str(state))
    monkeypatch.setattr(tf, "STATE_FILE", str(tmp_path / "footer-tok.json"))
    monkeypatch.setattr(tf, "LOCAL_9R_DB", str(nine))
    monkeypatch.setattr(socket, "gethostname", lambda: tf.VPS_HOSTNAME)

    line = tf.compute_turn_footer_line(session_id="S1")
    print("LINE:", line)
    # model from 9r
    assert "m: real-model-x" in line
    # session totals = 1100 / 2211
    assert "session: in=1.1k out=2.2k" in line
    # clock present
    assert " | " in line


def test_turn_tokens_omitted_when_no_delta(tmp_path, monkeypatch):
    state = tmp_path / "state.db"
    c = _make_state_db(state)
    now = time.time()
    c.execute("INSERT INTO sessions VALUES ('S1', ?, ?)", (now, now - 100))
    c.execute(
        "INSERT INTO session_model_usage (session_id, model, input_tokens, output_tokens, last_seen) "
        "VALUES ('S1', 'combo', 100, 200, ?)",
        (now - 10,),
    )
    c.execute("INSERT INTO messages (session_id, role, timestamp) VALUES ('S1', 'user', ?)", (now - 5,))
    c.commit()
    c.close()

    nine = tmp_path / "9r.sqlite"
    nc = _make_9r_db(nine)
    nc.execute("INSERT INTO usageHistory (timestamp, model, promptTokens) VALUES (?, ?, ?)",
               ("2026-11-06T10:04:42.241Z", "real-model-x", 5000))
    nc.commit()
    nc.close()

    sf = tmp_path / "footer-tok.json"
    # pre-populate state file so delta is 0
    sf.write_text(json.dumps({"in": 100, "out": 200, "sid": "S1"}))

    monkeypatch.setattr(tf, "SESSION_DB", str(state))
    monkeypatch.setattr(tf, "STATE_FILE", str(sf))
    monkeypatch.setattr(tf, "LOCAL_9R_DB", str(nine))
    monkeypatch.setattr(socket, "gethostname", lambda: tf.VPS_HOSTNAME)

    line = tf.compute_turn_footer_line(session_id="S1")
    print("LINE2:", line)
    # turn: omitted when 0
    assert "turn:" not in line
    assert "session: in=100 out=200" in line