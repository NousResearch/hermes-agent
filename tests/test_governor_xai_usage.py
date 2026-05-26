from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from agent.account_usage import (
    AccountUsageWindow,
    fetch_account_usage,
    maybe_observe_xai_rate_limit_headers,
    observe_xai_rate_limit_headers,
)
from agent.conversation_loop import _observe_xai_headers_best_effort
from agent.governor_state import (
    _parse_reset,
    compute_governor_band,
    ensure_governor_schema,
    get_transition_rate,
)


def _build_g1_db(path: Path) -> None:
    with sqlite3.connect(path) as db:
        db.execute("PRAGMA user_version = 1")
        db.executescript(
            """
            CREATE TABLE provider_state (
              provider TEXT PRIMARY KEY,
              band TEXT NOT NULL,
              daily_used_pct REAL,
              weekly_used_pct REAL,
              daily_reset_at TEXT,
              weekly_reset_at TEXT,
              observed_429_count_60min INTEGER,
              last_polled_at TEXT,
              source TEXT
            );
            CREATE TABLE mind_allocation (
              mind TEXT PRIMARY KEY,
              provider TEXT NOT NULL REFERENCES provider_state(provider),
              daily_share_pct REAL,
              weekly_share_pct REAL,
              daily_used_pct REAL,
              weekly_used_pct REAL
            );
            CREATE TABLE transition_rates (
              transition_key TEXT PRIMARY KEY,
              base_rate_per_hour INTEGER,
              current_rate_per_hour INTEGER,
              last_updated_at TEXT
            );
            CREATE TABLE governor_decisions (
              event_id TEXT PRIMARY KEY,
              ts TEXT NOT NULL,
              decision TEXT NOT NULL,
              reason TEXT,
              layer TEXT NOT NULL,
              provider TEXT REFERENCES provider_state(provider),
              mind TEXT REFERENCES mind_allocation(mind),
              card_id TEXT,
              defer_until TEXT
            );
            INSERT INTO provider_state(provider, band, daily_used_pct, weekly_used_pct, observed_429_count_60min, source)
            VALUES ('xai', 'green', 0.0, 0.0, 0, 'observed');
            INSERT INTO transition_rates(transition_key, base_rate_per_hour, current_rate_per_hour)
            VALUES ('PlatformOps.PlatformChange->*', 6, 6), ('SDLC.Research->Spec', 4, 4);
            """
        )


def test_ensure_governor_schema_migrates_g1_without_recreating(tmp_path):
    db_path = tmp_path / "governor.db"
    _build_g1_db(db_path)

    with sqlite3.connect(db_path) as db:
        db.execute("INSERT INTO governor_decisions(event_id, ts, decision, layer) VALUES ('keep-me', '2026-05-18T16:00:00Z', 'admit', 'A')")
        before_version = db.execute("PRAGMA user_version").fetchone()[0]

    ensure_governor_schema(db_path)

    with sqlite3.connect(db_path) as db:
        assert before_version == 1
        assert db.execute("PRAGMA user_version").fetchone()[0] == 2
        assert db.execute("SELECT COUNT(*) FROM governor_decisions WHERE event_id='keep-me'").fetchone()[0] == 1
        assert db.execute("SELECT COUNT(*) FROM xai_buckets").fetchone()[0] == 0
        assert db.execute("SELECT source FROM provider_state WHERE provider='xai'").fetchone()[0] == "observed"


def test_xai_header_observations_update_buckets_and_fetch_snapshot(tmp_path, monkeypatch):
    db_path = tmp_path / "governor.db"
    _build_g1_db(db_path)
    fixed_now = datetime(2026, 5, 18, 16, 0, tzinfo=timezone.utc)
    monkeypatch.setattr("agent.governor_state._utc_now", lambda: fixed_now)
    monkeypatch.setattr("agent.account_usage._utc_now", lambda: fixed_now)

    observe_xai_rate_limit_headers(
        {
            "x-ratelimit-limit-requests": "1000",
            "x-ratelimit-remaining-requests": "50",
            "x-ratelimit-reset-requests": "1900000000",
        },
        db_path=db_path,
    )

    snapshot = fetch_account_usage("xai", governor_db_path=db_path)

    assert snapshot is not None
    assert snapshot.provider == "xai"
    assert snapshot.source == "observed_headers"
    assert snapshot.windows == (
        AccountUsageWindow(
            label="xAI request bucket",
            used_percent=95.0,
            reset_at=datetime.fromtimestamp(1_900_000_000, tz=timezone.utc),
            detail="50 of 1000 requests remaining",
        ),
    )
    assert "Governor band: black" in snapshot.details

    with sqlite3.connect(db_path) as db:
        provider = db.execute(
            "SELECT band, daily_used_pct, weekly_used_pct, source FROM provider_state WHERE provider='xai'"
        ).fetchone()
        bucket = db.execute(
            "SELECT bucket_scope, limit_value, remaining_value, used_pct FROM xai_buckets"
        ).fetchone()

    assert provider == ("black", 95.0, 95.0, "observed")
    assert bucket == ("requests", 1000.0, 50.0, 95.0)


def test_xai_fetch_honours_observed_source_and_missing_headers(tmp_path):
    db_path = tmp_path / "governor.db"
    _build_g1_db(db_path)
    ensure_governor_schema(db_path)

    snapshot = fetch_account_usage("xai", governor_db_path=db_path)

    assert snapshot is not None
    assert snapshot.provider == "xai"
    assert snapshot.source == "observed_headers"
    assert snapshot.unavailable_reason == "No xAI rate-limit headers have been observed yet."


def test_xai_reset_parsing_uses_named_relative_cutoff(monkeypatch):
    fixed_now = datetime(2026, 5, 18, 16, 0, tzinfo=timezone.utc)
    monkeypatch.setattr("agent.governor_state._utc_now", lambda: fixed_now)

    assert _parse_reset("86400") == datetime.fromtimestamp(
        fixed_now.timestamp() + 86_400, tz=timezone.utc
    )
    assert _parse_reset("0") == fixed_now
    assert _parse_reset("-1") == datetime.fromtimestamp(-1, tz=timezone.utc)
    assert _parse_reset(str(30 * 86_400)) == datetime.fromtimestamp(
        fixed_now.timestamp() + 30 * 86_400, tz=timezone.utc
    )
    assert _parse_reset(str(30 * 86_400 + 1)) == datetime.fromtimestamp(
        30 * 86_400 + 1, tz=timezone.utc
    )
    assert _parse_reset("10000000") == datetime.fromtimestamp(10_000_000, tz=timezone.utc)
    assert _parse_reset("10000001") == datetime.fromtimestamp(10_000_001, tz=timezone.utc)
    assert _parse_reset("1900000000") == datetime.fromtimestamp(1_900_000_000, tz=timezone.utc)


class _MockXaiResponse:
    def __init__(self, headers):
        self.headers = headers


class _MockHttpError(Exception):
    def __init__(self, headers):
        super().__init__("429 Too Many Requests")
        self.response = _MockXaiResponse(headers)


def test_conversation_loop_xai_header_hook_records_real_shaped_response(tmp_path, monkeypatch):
    db_path = tmp_path / "governor.db"
    _build_g1_db(db_path)
    fixed_now = datetime(2026, 5, 18, 16, 0, tzinfo=timezone.utc)
    monkeypatch.setattr("agent.governor_state._utc_now", lambda: fixed_now)

    response = _MockXaiResponse(
        {
            "x-ratelimit-limit-requests": "1000",
            "x-ratelimit-remaining-requests": "100",
            "x-ratelimit-reset-requests": "1900000000",
        }
    )

    assert _observe_xai_headers_best_effort("xai", "grok-4", response, db_path=db_path) is True

    with sqlite3.connect(db_path) as db:
        assert db.execute("SELECT used_pct FROM xai_buckets WHERE bucket_key='requests:observed'").fetchone()[0] == 90.0
        assert db.execute("SELECT source FROM provider_state WHERE provider='xai'").fetchone()[0] == "observed"


def test_conversation_loop_xai_header_hook_records_429_error_response(tmp_path, monkeypatch):
    db_path = tmp_path / "governor.db"
    _build_g1_db(db_path)
    fixed_now = datetime(2026, 5, 18, 16, 0, tzinfo=timezone.utc)
    monkeypatch.setattr("agent.governor_state._utc_now", lambda: fixed_now)

    error = _MockHttpError(
        {
            "x-ratelimit-limit-requests": "1000",
            "x-ratelimit-remaining-requests": "0",
            "x-ratelimit-reset-requests": "86400",
        }
    )

    assert _observe_xai_headers_best_effort("xai", "grok-4", error, db_path=db_path) is True

    with sqlite3.connect(db_path) as db:
        bucket = db.execute(
            "SELECT used_pct, window_label FROM xai_buckets WHERE bucket_key='requests:day'"
        ).fetchone()
    assert bucket == (100.0, "day")


def test_conversation_loop_xai_header_hook_short_circuits_non_xai_without_db_touch(tmp_path):
    missing_db_path = tmp_path / "missing-governor.db"
    response = _MockXaiResponse(
        {
            "x-ratelimit-limit-requests": "1000",
            "x-ratelimit-remaining-requests": "0",
            "x-ratelimit-reset-requests": "1900000000",
        }
    )

    assert (
        maybe_observe_xai_rate_limit_headers(
            "anthropic", "claude-sonnet", response, db_path=missing_db_path
        )
        is False
    )
    assert _observe_xai_headers_best_effort("anthropic", "claude-sonnet", response, db_path=missing_db_path) is False
    assert not missing_db_path.exists()


def test_band_derivation_uses_daily_or_weekly_thresholds():
    assert compute_governor_band(69.9, 0.0) == "green"
    assert compute_governor_band(70.0, 0.0) == "amber"
    assert compute_governor_band(85.0, 0.0) == "red"
    assert compute_governor_band(95.0, 0.0) == "black"
    assert compute_governor_band(98.0, 0.0) == "post-reserve"
    assert compute_governor_band(0.0, 96.0) == "black"


def test_platformops_wildcard_transition_lookup_is_literal_lexical_fallback(tmp_path):
    db_path = tmp_path / "governor.db"
    _build_g1_db(db_path)
    ensure_governor_schema(db_path)

    assert get_transition_rate("SDLC.Research->Spec", db_path=db_path) == 4
    assert get_transition_rate("PlatformOps.PlatformChange->Review", db_path=db_path) == 6
    assert get_transition_rate("PlatformOps.PlatformChange->Deployed", db_path=db_path) == 6
    assert get_transition_rate("PlatformOps.Other->Review", db_path=db_path) is None
    assert get_transition_rate("PlatformOps.PlatformChange->*", db_path=db_path) == 6
