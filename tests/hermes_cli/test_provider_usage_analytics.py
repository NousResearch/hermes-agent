"""Provider usage preserves main-session accounting while adding auxiliary spend (#27533)."""

import sqlite3
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from hermes_cli.web_routers import analytics
from hermes_state import SessionDB


@pytest.fixture
def home(tmp_path, monkeypatch):
    import hermes_state

    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(hermes_state, "DEFAULT_DB_PATH", home / "state.db")
    return home


@pytest.fixture
def db(home):
    db = SessionDB(home / "state.db")
    yield db
    db.close()


@pytest.fixture
def client(home):
    app = FastAPI()
    app.include_router(analytics.router)
    with TestClient(app) as client:
        yield client


def _usage(client, **params):
    response = client.get("/api/analytics/usage", params=params)
    assert response.status_code == 200, response.text
    return response.json()


def _providers(result):
    rows = result["by_provider"]
    assert len(rows) == len({row["provider"] for row in rows})
    return {row["provider"]: row for row in rows}


def _counts(amount):
    return {
        "input_tokens": amount,
        "output_tokens": amount // 2,
        "cache_read_tokens": amount * 3 // 10,
        "cache_write_tokens": amount // 5,
        "reasoning_tokens": amount // 10,
        "estimated_cost_usd": amount / 10,
        "actual_cost_usd": amount / 5,
        "api_call_count": amount // 10,
    }


def _assert_counts(row, amount):
    names = {
        "estimated_cost_usd": "estimated_cost",
        "actual_cost_usd": "actual_cost",
        "api_call_count": "api_calls",
    }
    for key, expected in _counts(amount).items():
        assert row[names.get(key, key)] == pytest.approx(expected), key


def _main(db, session_id, provider, amount, *, task=""):
    db.update_token_counts(
        session_id, model="main-model", billing_provider=provider,
        task=task, **_counts(amount),
    )


def _summary(db, session_id, provider, amount):
    db.update_token_counts(
        session_id, model="main-model", billing_provider=provider,
        absolute=True, **_counts(amount),
    )
    db.update_session_billing_route(session_id, provider=provider, base_url="")


def _aux(db, session_id, provider, amount, *, task="vision"):
    counts = _counts(amount)
    actual = counts.pop("actual_cost_usd")
    db.record_auxiliary_usage(
        session_id, task, model="aux-model", billing_provider=provider, **counts,
    )
    # The persisted schema also accepts actual costs from other accounting writers.
    db._conn.execute(
        "UPDATE session_model_usage SET actual_cost_usd = ? WHERE session_id = ? AND task = ?",
        (actual, session_id, task),
    )
    db._conn.commit()


@pytest.mark.parametrize("primary_detail", [0, 40, 100])
def test_primary_residual_and_auxiliary_usage_are_additive(db, client, primary_detail):
    """Main 100 plus aux 60 remains 160 with absent, partial, or complete detail."""
    db.create_session("mixed", source="cli", model="main-model")
    if primary_detail:
        _main(db, "mixed", "recorded-route", primary_detail)
    _summary(db, "mixed", "session-route", 100)
    _aux(db, "mixed", "aux-route", 60)

    rows = _providers(_usage(client))

    expected = {"aux-route": 60}
    if primary_detail:
        expected["recorded-route"] = primary_detail
    if primary_detail < 100:
        expected["session-route"] = 100 - primary_detail
    assert rows.keys() == expected.keys()
    for provider, amount in expected.items():
        _assert_counts(rows[provider], amount)
        assert rows[provider]["sessions"] == 1
    assert sum(row["input_tokens"] for row in rows.values()) == 160


def test_session_count_is_distinct_across_primary_routes_and_aux_tasks(db, client):
    db.create_session("mixed", source="cli", model="main-model")
    _main(db, "mixed", "shared", 100)
    _main(db, "mixed", "other", 20)
    for task in ("vision", "compression", "title_generation"):
        _aux(db, "mixed", "shared", 10, task=task)
    db.create_session("second", source="cli", model="main-model")
    _main(db, "second", "shared", 10)

    rows = _providers(_usage(client))

    assert rows.keys() == {"shared", "other"}
    _assert_counts(rows["shared"], 140)
    assert rows["shared"]["sessions"] == 2
    _assert_counts(rows["other"], 20)
    assert rows["other"]["sessions"] == 1


def test_voice_route_is_primary_usage_already_in_the_session_total(db, client):
    db.create_session("voice", source="cli", model="main-model")
    _main(db, "voice", "text-route", 100)
    _main(db, "voice", "voice-route", 20, task="voice_chat")
    _aux(db, "voice", "vision-route", 60)

    rows = _providers(_usage(client))

    assert rows.keys() == {"text-route", "voice-route", "vision-route"}
    for provider, amount in (("text-route", 100), ("voice-route", 20), ("vision-route", 60)):
        _assert_counts(rows[provider], amount)
        assert rows[provider]["sessions"] == 1
    assert sum(row["input_tokens"] for row in rows.values()) == 180


def test_primary_residual_is_nonnegative_for_each_counter(db, client):
    db.create_session("partial", source="cli", model="main-model")
    db.update_token_counts(
        "partial", model="main-model", billing_provider="recorded-route",
        input_tokens=40, output_tokens=20, cache_read_tokens=4,
        cache_write_tokens=3, reasoning_tokens=2, estimated_cost_usd=2,
        actual_cost_usd=0.5, api_call_count=2,
    )
    db.update_token_counts(
        "partial", absolute=True, input_tokens=100, output_tokens=10,
        cache_read_tokens=2, cache_write_tokens=8, reasoning_tokens=1,
        estimated_cost_usd=1, actual_cost_usd=2, api_call_count=1,
    )
    db.update_session_billing_route("partial", provider="session-route", base_url="")

    rows = _providers(_usage(client))

    assert rows["session-route"] == {
        "provider": "session-route", "input_tokens": 60, "output_tokens": 0,
        "cache_read_tokens": 0, "cache_write_tokens": 5, "reasoning_tokens": 0,
        "estimated_cost": 0, "actual_cost": 1.5, "api_calls": 0, "sessions": 1,
    }
    assert rows["recorded-route"]["output_tokens"] == 20
    assert rows["recorded-route"]["estimated_cost"] == 2


def test_float_cost_roundoff_does_not_create_a_phantom_provider(db, client):
    for provider, cost in (("first", 0.1), ("second", 0.2), ("second", 0.3)):
        db.update_token_counts(
            "rounded", model="main-model", billing_provider=provider,
            input_tokens=10, estimated_cost_usd=cost, actual_cost_usd=cost,
            api_call_count=1,
        )
    db.update_session_billing_route("rounded", provider="never-called", base_url="")

    rows = _providers(_usage(client))

    assert rows.keys() == {"first", "second"}
    assert rows["first"]["estimated_cost"] == pytest.approx(0.1)
    assert rows["second"]["actual_cost"] == pytest.approx(0.5)


def test_real_cost_only_residual_remains_attributed(db, client):
    db.update_token_counts(
        "cost-only", model="main-model", billing_provider="recorded-route",
        input_tokens=10, estimated_cost_usd=0.02, actual_cost_usd=0.03, api_call_count=1,
    )
    db.update_token_counts(
        "cost-only", absolute=True, input_tokens=10,
        estimated_cost_usd=0.07, actual_cost_usd=0.11, api_call_count=1,
    )
    db.update_session_billing_route("cost-only", provider="session-route", base_url="")

    row = _providers(_usage(client))["session-route"]

    assert row["estimated_cost"] == pytest.approx(0.05)
    assert row["actual_cost"] == pytest.approx(0.08)
    assert row["input_tokens"] == row["output_tokens"] == row["api_calls"] == 0
    assert row["sessions"] == 1


def test_aux_only_session_is_attributed_without_a_zero_summary_provider(db, client):
    _aux(db, "aux-only", "aux-route", 50)

    rows = _providers(_usage(client))

    assert rows.keys() == {"aux-route"}
    _assert_counts(rows["aux-route"], 50)
    assert rows["aux-route"]["sessions"] == 1


def test_missing_and_empty_provider_names_share_unknown_bucket(db, client):
    for session_id, provider in (("missing", None), ("empty", "")):
        db.create_session(session_id, source="cli", model="main-model")
        db.update_token_counts(
            session_id, model="main-model", billing_provider=provider,
            input_tokens=10, api_call_count=1, absolute=True,
        )
    db.create_session("known", source="cli", model="main-model")
    _main(db, "known", "known-route", 10)
    _aux(db, "known", "", 20)
    _main(db, "unknown-detail", None, 10)

    rows = _providers(_usage(client))

    assert rows.keys() == {"unknown", "known-route"}
    assert rows["unknown"]["input_tokens"] == 50
    assert rows["unknown"]["sessions"] == 4
    assert rows["unknown"]["api_calls"] == 5
    assert rows["known-route"]["sessions"] == 1


@pytest.mark.parametrize("days", [1, 30, 365])
def test_window_uses_strict_session_start_cutoff_for_all_usage(db, client, monkeypatch, days):
    now = time.time()
    cutoff = now - days * 86400
    monkeypatch.setattr(analytics, "time", SimpleNamespace(time=lambda: now))
    for session_id, started_at in (("old", cutoff - 1), ("boundary", cutoff), ("recent", cutoff + 1)):
        db.create_session(session_id, source="cli", model="main-model")
        _main(db, session_id, "main-route", 10)
        _summary(db, session_id, "session-route", 30)
        _aux(db, session_id, "aux-route", 20)
        db._conn.execute("UPDATE sessions SET started_at = ? WHERE id = ?", (started_at, session_id))
        # Detail timestamps deliberately disagree with the owning session's window.
        detail_time = cutoff - 100 if session_id == "recent" else now
        db._conn.execute(
            "UPDATE session_model_usage SET first_seen = ?, last_seen = ? WHERE session_id = ?",
            (detail_time, detail_time, session_id),
        )
    db._conn.commit()

    result = _usage(client, days=days)
    rows = _providers(result)

    assert result["period_days"] == days
    assert result["totals"]["total_sessions"] == 1
    _assert_counts(rows["main-route"], 10)
    _assert_counts(rows["session-route"], 20)
    _assert_counts(rows["aux-route"], 20)
    assert all(row["sessions"] == 1 for row in rows.values())


def test_empty_window_has_no_provider_rows(db, client):
    assert _usage(client)["by_provider"] == []
    _main(db, "old", "old-route", 10)
    db._conn.execute("UPDATE sessions SET started_at = ?", (time.time() - 400 * 86400,))
    db._conn.commit()
    result = _usage(client, days=365)
    assert result["by_provider"] == []
    assert result["totals"]["total_sessions"] == 0


@pytest.mark.parametrize("days", [0, -1, 366, "invalid"])
def test_invalid_period_is_rejected_before_database_open(db, client, monkeypatch, days):
    def unexpected_open(*args, **kwargs):
        pytest.fail("invalid periods must not open the database")

    monkeypatch.setattr(analytics, "_open_session_db_for_profile", unexpected_open)
    assert client.get("/api/analytics/usage", params={"days": days}).status_code == 422


@pytest.mark.parametrize("legacy_schema", ["missing_table", "missing_task"])
def test_read_only_legacy_usage_schema_falls_back_without_migration(db, client, monkeypatch, legacy_schema):
    _main(db, "legacy", "recorded-route", 40)
    _summary(db, "legacy", "session-route", 100)
    if legacy_schema == "missing_task":
        columns = [row[1] for row in db._conn.execute("PRAGMA table_info(session_model_usage)") if row[1] != "task"]
        db._conn.execute(f"CREATE TABLE legacy_usage AS SELECT {', '.join(columns)} FROM session_model_usage")
    db._conn.execute("DROP TABLE session_model_usage")
    if legacy_schema == "missing_task":
        db._conn.execute("ALTER TABLE legacy_usage RENAME TO session_model_usage")
    db._conn.commit()
    before = list(db._conn.execute("SELECT name, sql FROM sqlite_master ORDER BY name"))

    # Skip the server's optional schema-heal opener to exercise an older read-only snapshot.
    def open_legacy(profile, *, read_only):
        assert read_only is True
        return SessionDB(db.db_path, read_only=True)

    monkeypatch.setattr(analytics, "_open_session_db_for_profile", open_legacy)
    rows = _providers(_usage(client))

    if legacy_schema == "missing_table":
        assert rows.keys() == {"session-route"}
        _assert_counts(rows["session-route"], 100)
    else:
        assert rows.keys() == {"recorded-route", "session-route"}
        _assert_counts(rows["recorded-route"], 40)
        _assert_counts(rows["session-route"], 60)
    assert list(db._conn.execute("SELECT name, sql FROM sqlite_master ORDER BY name")) == before


def test_provider_read_errors_reach_existing_corrupt_store_response(db, client, monkeypatch):
    _main(db, "broken", "main-route", 10)
    raw_conn = db._conn

    class CorruptUsageConnection:
        def execute(self, sql, *args, **kwargs):
            if "session_model_usage" in sql:
                raise sqlite3.DatabaseError("database disk image is malformed")
            return raw_conn.execute(sql, *args, **kwargs)

        def __getattr__(self, name):
            return getattr(raw_conn, name)

    monkeypatch.setattr(db, "_conn", CorruptUsageConnection())
    monkeypatch.setattr(analytics, "_open_session_db_for_profile", lambda profile, read_only: db)

    response = client.get("/api/analytics/usage")

    assert response.status_code == 503
    assert response.json()["detail"]["error"] == "state_db_corrupt"


def test_provider_details_do_not_change_existing_usage_sections(db, client):
    _main(db, "mixed", "recorded-route", 40)
    _summary(db, "mixed", "session-route", 100)
    _aux(db, "mixed", "aux-route", 60)
    db.append_message("mixed", role="assistant", content="Loading a skill.", tool_calls=[
        {"function": {"name": "skill_view", "arguments": '{"name":"test-skill"}'}},
        {"function": {"name": "terminal", "arguments": '{"command":"true"}'}},
    ])
    before = _usage(client)
    assert before["totals"]["total_input"] == 100
    assert before["daily"][0]["input_tokens"] == 100
    assert {row["model"]: row["input_tokens"] for row in before["by_model"]} == {
        "main-model": 100, "aux-model": 60,
    }
    assert before["by_task"][0]["task"] == "vision"
    assert before["by_task"][0]["input_tokens"] == 60
    assert before["skills"]["summary"]["total_skill_loads"] == 1
    assert {row["tool"]: row["count"] for row in before["tools"]} == {"skill_view": 1, "terminal": 1}

    db._conn.execute("DELETE FROM session_model_usage WHERE task = ''")
    db._conn.commit()
    after = _usage(client)

    for section in ("daily", "totals", "by_model", "by_task", "skills", "tools", "period_days"):
        assert after[section] == before[section], section
    assert _providers(before)["recorded-route"]["input_tokens"] == 40
    assert _providers(after)["session-route"]["input_tokens"] == 100


def test_profile_selection_is_isolated_and_healthy_reads_are_read_only(home, db, client, monkeypatch):
    _main(db, "shared-id", "default-route", 10)
    profile_home = home / "profiles" / "work"
    profile_home.mkdir(parents=True)
    profile_home.joinpath("config.yaml").write_text("model: main-model\n")
    other = SessionDB(profile_home / "state.db")
    try:
        _main(other, "shared-id", "work-route", 20)
        _aux(other, "shared-id", "work-aux", 30)
    finally:
        other.close()
    opened = []
    original_init = SessionDB.__init__

    def record_open(self, db_path=None, read_only=False):
        opened.append((Path(db_path), read_only))
        original_init(self, db_path=db_path, read_only=read_only)

    monkeypatch.setattr(SessionDB, "__init__", record_open)

    default_rows = _providers(_usage(client))
    work_rows = _providers(_usage(client, profile="work"))

    assert default_rows.keys() == {"default-route"}
    assert work_rows.keys() == {"work-route", "work-aux"}
    _assert_counts(work_rows["work-route"], 20)
    _assert_counts(work_rows["work-aux"], 30)
    assert opened == [(home / "state.db", True), (profile_home / "state.db", True)]
