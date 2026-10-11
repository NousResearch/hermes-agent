"""Dashboard timeline API for per-model usage."""

import time

from fastapi import FastAPI
from fastapi.testclient import TestClient

from hermes_cli.web_routers import analytics
from hermes_state import SessionDB


def test_model_usage_timeline_returns_estimated_historical_model_bucket(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    db = SessionDB(db_path=tmp_path / "state.db")
    try:
        db.create_session("s1", source="discord", model="gpt-5.6-terra")
        now = time.time()
        assert db._conn is not None
        db._conn.execute("""
            INSERT INTO session_model_usage (
                session_id, model, billing_provider, task, api_call_count,
                input_tokens, output_tokens, cache_read_tokens, cache_write_tokens,
                reasoning_tokens, first_seen, last_seen
            ) VALUES (?, ?, ?, '', ?, ?, ?, ?, ?, ?, ?, ?)
        """, ("s1", "gpt-5.6-terra", "openai-codex", 1, 11, 4, 3, 0, 2, now, now))
        db._conn.commit()
    finally:
        db.close()

    import hermes_state
    monkeypatch.setattr(hermes_state, "_default_db_path", lambda: tmp_path / "state.db")
    app = FastAPI()
    app.include_router(analytics.router)
    response = TestClient(app).get("/api/analytics/model-usage?days=1&bucket=day")

    assert response.status_code == 200
    payload = response.json()
    assert payload["bucket"] == "day"
    assert payload["period_days"] == 1
    assert len(payload["series"]) == 1
    series = payload["series"][0]
    assert series["model"] == "gpt-5.6-terra"
    assert series["provider"] == "openai-codex"
    assert series["points"] == [{
        "start": "2026-09-23",
        "input_tokens": 11,
        "output_tokens": 4,
        "cache_read_tokens": 3,
        "cache_write_tokens": 0,
        "reasoning_tokens": 2,
        "api_calls": 1,
        "total_tokens": 20,
        "provenance": "estimated",
    }]


def test_model_usage_timeline_marks_new_per_call_usage_exact(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    db = SessionDB(db_path=tmp_path / "state.db")
    try:
        db.create_session("s1", source="discord", model="gpt-5.6-terra")
        db.update_token_counts(
            "s1",
            input_tokens=11,
            output_tokens=4,
            cache_read_tokens=3,
            reasoning_tokens=2,
            api_call_count=1,
            model="gpt-5.6-terra",
            billing_provider="openai-codex",
        )
    finally:
        db.close()

    import hermes_state
    monkeypatch.setattr(hermes_state, "_default_db_path", lambda: tmp_path / "state.db")
    app = FastAPI()
    app.include_router(analytics.router)
    payload = TestClient(app).get("/api/analytics/model-usage?days=1&bucket=day").json()

    assert payload["series"][0]["points"][0]["provenance"] == "exact"
