"""``/api/analytics/usage`` returns a per-(day, model) series for the stacked chart (#20412).

Each model's days must add up to its ``by_model`` totals, including auxiliary usage that lives
only in ``session_model_usage``, so the chart and the per-model table never disagree.
"""

import time

import pytest

from hermes_cli.web_routers import analytics
from hermes_state import SessionDB


@pytest.fixture
def db(tmp_path):
    return SessionDB(tmp_path / "state.db")


@pytest.fixture
def usage(monkeypatch, db):
    monkeypatch.setattr(analytics, "_open_session_db_for_profile", lambda profile, read_only=True: db)
    monkeypatch.setattr(db, "close", lambda: None)
    return lambda: analytics._get_usage_analytics(days=30)


def _session(db, session_id, model, started_at, input_tokens, output_tokens):
    db.create_session(session_id, source="cli", model=model)
    with db._lock:
        db._conn.execute(
            "UPDATE sessions SET started_at = ?, input_tokens = ?, output_tokens = ? WHERE id = ?",
            (started_at, input_tokens, output_tokens, session_id),
        )
        db._conn.commit()


def _local_day(ts):
    return time.strftime("%Y-%m-%d", time.localtime(ts))


def test_daily_by_model_splits_days_and_models_and_matches_by_model(db, usage):
    now = time.time()
    earlier = now - 3 * 86400
    _session(db, "a", "model-a", earlier, 1000, 10)
    _session(db, "b", "model-a", now, 2000, 20)
    _session(db, "c", "model-b", now, 500, 50)
    db.record_auxiliary_usage("c", "vision", model="vision-model", input_tokens=300, output_tokens=3)

    result = usage()
    series = {(r["day"], r["model"]): (r["input_tokens"], r["output_tokens"]) for r in result["daily_by_model"]}

    assert series == {
        (_local_day(earlier), "model-a"): (1000, 10),
        (_local_day(now), "model-a"): (2000, 20),
        (_local_day(now), "model-b"): (500, 50),
        (_local_day(now), "vision-model"): (300, 3),
    }
    for model in result["by_model"]:
        days = [r for r in result["daily_by_model"] if r["model"] == model["model"]]
        assert sum(r["input_tokens"] for r in days) == model["input_tokens"]
        assert sum(r["output_tokens"] for r in days) == model["output_tokens"]


def test_daily_by_model_skips_sessions_outside_the_window(db, usage):
    _session(db, "old", "model-a", time.time() - 60 * 86400, 1000, 10)
    assert usage()["daily_by_model"] == []
