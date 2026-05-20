from __future__ import annotations

import importlib.util
import json
import sqlite3
from pathlib import Path


def _load_footer_metrics_module(tmp_path: Path):
    spec = importlib.util.spec_from_file_location(
        "_footer_metrics_under_test",
        Path(__file__).resolve().parents[2] / "scripts" / "footer_metrics.py",
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_compute_footer_metrics_reads_state_db_row(tmp_path):
    mod = _load_footer_metrics_module(tmp_path)
    db_path = tmp_path / "state.db"
    conn = sqlite3.connect(db_path)
    conn.execute(
        """
        CREATE TABLE sessions (
            id TEXT PRIMARY KEY,
            input_tokens INTEGER DEFAULT 0,
            output_tokens INTEGER DEFAULT 0,
            cache_read_tokens INTEGER DEFAULT 0,
            cache_write_tokens INTEGER DEFAULT 0,
            reasoning_tokens INTEGER DEFAULT 0,
            estimated_cost_usd REAL,
            actual_cost_usd REAL,
            cost_status TEXT,
            api_call_count INTEGER DEFAULT 0
        )
        """
    )
    conn.execute(
        """
        INSERT INTO sessions (
            id, input_tokens, output_tokens, cache_read_tokens, cache_write_tokens,
            reasoning_tokens, estimated_cost_usd, actual_cost_usd, cost_status, api_call_count
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        ("sess-1", 1200, 300, 50, 10, 25, 0.012, None, "estimated", 4),
    )
    conn.commit()
    conn.close()

    payload = {
        "session_id": "sess-1",
        "hermes_home": str(tmp_path),
    }
    result = mod.compute_footer_metrics(payload)
    assert result == {
        "total_tokens": 1500,
        "api_calls": 4,
        "estimated_cost_usd": 0.012,
        "cost_status": "estimated",
        "prompt_tokens": 1200,
        "completion_tokens": 300,
        "cache_read_tokens": 50,
        "cache_write_tokens": 10,
        "reasoning_tokens": 25,
    }


def test_compute_footer_metrics_prefers_actual_cost_when_present(tmp_path):
    mod = _load_footer_metrics_module(tmp_path)
    db_path = tmp_path / "state.db"
    conn = sqlite3.connect(db_path)
    conn.execute(
        """
        CREATE TABLE sessions (
            id TEXT PRIMARY KEY,
            input_tokens INTEGER DEFAULT 0,
            output_tokens INTEGER DEFAULT 0,
            cache_read_tokens INTEGER DEFAULT 0,
            cache_write_tokens INTEGER DEFAULT 0,
            reasoning_tokens INTEGER DEFAULT 0,
            estimated_cost_usd REAL,
            actual_cost_usd REAL,
            cost_status TEXT,
            api_call_count INTEGER DEFAULT 0
        )
        """
    )
    conn.execute(
        """
        INSERT INTO sessions (
            id, input_tokens, output_tokens, cache_read_tokens, cache_write_tokens,
            reasoning_tokens, estimated_cost_usd, actual_cost_usd, cost_status, api_call_count
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        ("sess-2", 200, 40, 0, 0, 0, 0.0, 0.021, "actual", 2),
    )
    conn.commit()
    conn.close()

    result = mod.compute_footer_metrics({"session_id": "sess-2", "hermes_home": str(tmp_path)})
    assert result["estimated_cost_usd"] == 0.021
    assert result["cost_status"] == "actual"


def test_compute_footer_metrics_returns_empty_without_session_id(tmp_path):
    mod = _load_footer_metrics_module(tmp_path)
    assert mod.compute_footer_metrics({"hermes_home": str(tmp_path)}) == {}


def test_main_writes_json_metrics_from_stdin(tmp_path, monkeypatch, capsys):
    mod = _load_footer_metrics_module(tmp_path)
    db_path = tmp_path / "state.db"
    conn = sqlite3.connect(db_path)
    conn.execute(
        """
        CREATE TABLE sessions (
            id TEXT PRIMARY KEY,
            input_tokens INTEGER DEFAULT 0,
            output_tokens INTEGER DEFAULT 0,
            cache_read_tokens INTEGER DEFAULT 0,
            cache_write_tokens INTEGER DEFAULT 0,
            reasoning_tokens INTEGER DEFAULT 0,
            estimated_cost_usd REAL,
            actual_cost_usd REAL,
            cost_status TEXT,
            api_call_count INTEGER DEFAULT 0
        )
        """
    )
    conn.execute(
        """
        INSERT INTO sessions (
            id, input_tokens, output_tokens, cache_read_tokens, cache_write_tokens,
            reasoning_tokens, estimated_cost_usd, actual_cost_usd, cost_status, api_call_count
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        ("sess-3", 800, 200, 100, 0, 9, 0.015, None, "estimated", 3),
    )
    conn.commit()
    conn.close()

    input_path = tmp_path / "input.json"
    input_path.write_text(json.dumps({"session_id": "sess-3", "hermes_home": str(tmp_path)}))
    monkeypatch.setattr("sys.stdin.read", lambda: input_path.read_text())

    mod.main()
    out = capsys.readouterr().out.strip()
    assert json.loads(out) == {
        "total_tokens": 1000,
        "api_calls": 3,
        "estimated_cost_usd": 0.015,
        "cost_status": "estimated",
        "prompt_tokens": 800,
        "completion_tokens": 200,
        "cache_read_tokens": 100,
        "cache_write_tokens": 0,
        "reasoning_tokens": 9,
    }
