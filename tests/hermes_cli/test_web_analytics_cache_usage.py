"""Cache token buckets exposed by desktop usage analytics (#112700)."""

from hermes_cli.web_routers import analytics
from hermes_state import SessionDB


def test_daily_analytics_includes_cache_write_tokens(tmp_path, monkeypatch):
    db_path = tmp_path / "state.db"
    db = SessionDB(db_path=db_path)
    db.create_session("cache-usage", source="cli", model="test")
    db.update_token_counts(
        "cache-usage",
        input_tokens=100,
        cache_read_tokens=800,
        cache_write_tokens=100,
    )
    db.close()

    monkeypatch.setattr(analytics, "_open_session_db_for_profile", lambda *_args, **_kwargs: SessionDB(db_path=db_path))

    daily = analytics._get_usage_analytics(days=1)["daily"]

    assert daily[0]["cache_read_tokens"] == 800
    assert daily[0]["cache_write_tokens"] == 100
