import datetime

from hermes_state import SessionDB


def _timestamp(day: int) -> float:
    return datetime.datetime(2026, 1, day, 12, tzinfo=datetime.timezone.utc).timestamp()


def test_usage_analytics_buckets_by_activity_day_and_matches_totals(tmp_path, monkeypatch):
    import hermes_state_usage
    from hermes_cli.web_routers import analytics

    db_path = tmp_path / "state.db"
    db = SessionDB(db_path)
    try:
        db.create_session("long-running", "cli")
        db._write_sql("UPDATE sessions SET started_at = ? WHERE id = ?", (_timestamp(1), "long-running"))

        monkeypatch.setattr(hermes_state_usage.time, "time", lambda: _timestamp(1))
        db.update_token_counts("long-running", input_tokens=100, output_tokens=10, api_call_count=1)
        monkeypatch.setattr(hermes_state_usage.time, "time", lambda: _timestamp(10))
        db.update_token_counts(
            "long-running", input_tokens=300, output_tokens=30, api_call_count=3, absolute=True,
        )
    finally:
        db.close()

    monkeypatch.setattr(analytics, "_open_session_db_for_profile", lambda *_args, **_kwargs: SessionDB(db_path))
    monkeypatch.setattr(analytics.time, "time", lambda: _timestamp(10))

    result = analytics._get_usage_analytics(days=3)

    assert result["daily"] == [{
        "day": "2026-01-10",
        "input_tokens": 200,
        "output_tokens": 20,
        "cache_read_tokens": 0,
        "reasoning_tokens": 0,
        "estimated_cost": 0.0,
        "actual_cost": 0.0,
        "sessions": 1,
        "api_calls": 2,
    }]
    daily = result["daily"]
    assert result["totals"] == {
        "total_input": sum(row["input_tokens"] for row in daily),
        "total_output": sum(row["output_tokens"] for row in daily),
        "total_cache_read": sum(row["cache_read_tokens"] for row in daily),
        "total_reasoning": sum(row["reasoning_tokens"] for row in daily),
        "total_estimated_cost": sum(row["estimated_cost"] for row in daily),
        "total_actual_cost": sum(row["actual_cost"] for row in daily),
        "total_sessions": 1,
        "total_api_calls": sum(row["api_calls"] for row in daily),
    }