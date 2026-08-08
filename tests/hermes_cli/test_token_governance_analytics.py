import json
import sqlite3

from hermes_cli import web_server


def _make_db(tmp_path):
    db = tmp_path / "state.db"
    conn = sqlite3.connect(db)
    conn.row_factory = sqlite3.Row
    conn.execute("""
        CREATE TABLE sessions (
          model TEXT, billing_provider TEXT, input_tokens INTEGER,
          output_tokens INTEGER, cache_read_tokens INTEGER,
          cache_write_tokens INTEGER, reasoning_tokens INTEGER,
          estimated_cost_usd REAL, actual_cost_usd REAL,
          api_call_count INTEGER, tool_call_count INTEGER, started_at REAL
        )
    """)
    conn.execute("""
        INSERT INTO sessions VALUES
        ('m','p',10,7,30,5,2,NULL,NULL,2,0,9999999999)
    """)
    conn.commit()
    return conn


def test_usage_dashboard_enrichment_uses_cache_in_prompt():
    row = web_server._enrich_usage_row({
        "input_tokens": 100,
        "output_tokens": 5,
        "cache_read_tokens": 200,
        "cache_write_tokens": 20,
        "api_calls": 4,
    })
    assert row["prompt_tokens"] == 320
    assert row["processed_tokens"] == 325
    assert row["avg_prompt_tokens_per_call"] == 80


def test_models_analytics_propagates_cache_write_and_prompt(monkeypatch, tmp_path):
    conn = _make_db(tmp_path)
    class DB:
        _conn = conn
        def close(self):
            pass
    monkeypatch.setattr(web_server, "_open_session_db_for_profile", lambda *a, **k: DB())
    monkeypatch.setattr(web_server, "_aux_usage_rows", lambda *a, **k: [])

    result = web_server._get_models_analytics(1)

    model = result["models"][0]
    assert model["cache_write_tokens"] == 5
    assert model["prompt_tokens"] == 45
    assert model["processed_tokens"] == 52
    assert result["totals"]["total_cache_write"] == 5
    assert result["totals"]["prompt_tokens"] == 45
