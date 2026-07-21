"""Kanban dispatcher admission caps: rolling task starts and daily spend."""
from __future__ import annotations

import os
import sqlite3
import sys
import tempfile
from pathlib import Path

import pytest


@pytest.fixture()
def isolated_kanban_home(monkeypatch):
    test_home = tempfile.mkdtemp(prefix="kanban_admission_caps_")
    os.makedirs(os.path.join(test_home, "profiles", "alpha"), exist_ok=True)
    monkeypatch.setenv("HERMES_HOME", test_home)
    monkeypatch.delenv("HERMES_KANBAN_DB", raising=False)
    monkeypatch.delenv("HERMES_KANBAN_BOARD", raising=False)
    for mod in list(sys.modules.keys()):
        if mod.startswith("hermes_cli") or mod == "hermes_constants":
            del sys.modules[mod]
    from hermes_cli import kanban_db

    assert kanban_db.kanban_db_path() != Path("/home/hermes/.hermes/kanban/slbm-operations.db")

    yield Path(test_home), kanban_db


def _fake_spawn(*args, **kwargs):
    return 12345


def _make_ready_tasks(kb, conn, count=3):
    return [kb.create_task(conn, title=f"task {i}", assignee="alpha") for i in range(count)]


def _make_review_task(kb, conn, title="review task", assignee="alpha"):
    task_id = kb.create_task(conn, title=title, assignee=assignee)
    with kb.write_txn(conn):
        conn.execute("UPDATE tasks SET status = 'review' WHERE id = ?", (task_id,))
    return task_id


def test_defaults_preserve_dispatch_behavior(isolated_kanban_home):
    _, kb = isolated_kanban_home
    with kb.connect_closing() as conn:
        _make_ready_tasks(kb, conn, 2)
    with kb.connect_closing() as conn:
        res = kb.dispatch_once(conn, spawn_fn=_fake_spawn, dry_run=True)
    assert len(res.spawned) == 2
    assert not res.skipped_rolling_start_capped
    assert not res.skipped_daily_spend_capped
    assert not res.skipped_spend_ledger_unavailable


def test_rolling_start_cap_holds_at_exact_threshold_without_mutating_tasks(isolated_kanban_home):
    _, kb = isolated_kanban_home
    with kb.connect_closing() as conn:
        task_ids = _make_ready_tasks(kb, conn, 2)
        now = 1_700_000_000
        with kb.write_txn(conn):
            conn.execute(
                "INSERT INTO task_runs (task_id, profile, status, started_at) VALUES (?, ?, ?, ?)",
                ("previous", "alpha", "completed", now - 10),
            )
        monkeypatch_time = pytest.MonkeyPatch()
        monkeypatch_time.setattr(kb.time, "time", lambda: now)
        try:
            res = kb.dispatch_once(
                conn,
                spawn_fn=_fake_spawn,
                dry_run=True,
                max_task_starts_per_hour=1,
            )
        finally:
            monkeypatch_time.undo()
        assert [row[0] for row in res.skipped_rolling_start_capped] == task_ids
        rows = conn.execute(
            "SELECT status, consecutive_failures FROM tasks ORDER BY created_at"
        ).fetchall()
        assert [(r["status"], r["consecutive_failures"]) for r in rows] == [("ready", 0), ("ready", 0)]


def test_rolling_start_cap_allows_remaining_capacity_then_holds_rest(isolated_kanban_home):
    _, kb = isolated_kanban_home
    with kb.connect_closing() as conn:
        task_ids = _make_ready_tasks(kb, conn, 3)
        now = 1_700_000_000
        with kb.write_txn(conn):
            conn.execute(
                "INSERT INTO task_runs (task_id, profile, status, started_at) VALUES (?, ?, ?, ?)",
                ("previous", "alpha", "completed", now - 10),
            )
        monkeypatch_time = pytest.MonkeyPatch()
        monkeypatch_time.setattr(kb.time, "time", lambda: now)
        try:
            res = kb.dispatch_once(
                conn,
                spawn_fn=_fake_spawn,
                dry_run=True,
                max_task_starts_per_hour=2,
            )
        finally:
            monkeypatch_time.undo()
        assert [s[0] for s in res.spawned] == [task_ids[0]]
        assert [h[0] for h in res.skipped_rolling_start_capped] == task_ids[1:]


def test_review_tasks_obey_global_max_in_progress_when_ready_queue_empty(isolated_kanban_home):
    _, kb = isolated_kanban_home
    with kb.connect_closing() as conn:
        running = _make_ready_tasks(kb, conn, 3)
        for task_id in running:
            kb.claim_task(conn, task_id)
        review_id = _make_review_task(kb, conn)

        res = kb.dispatch_once(conn, spawn_fn=_fake_spawn, dry_run=True, max_in_progress=3)

        assert not res.spawned
        assert kb.get_task(conn, review_id).status == "review"


def test_review_tasks_obey_per_profile_max_in_progress(isolated_kanban_home):
    _, kb = isolated_kanban_home
    with kb.connect_closing() as conn:
        running = _make_ready_tasks(kb, conn, 3)
        for task_id in running:
            kb.claim_task(conn, task_id)
        review_id = _make_review_task(kb, conn)

        res = kb.dispatch_once(
            conn,
            spawn_fn=_fake_spawn,
            dry_run=True,
            max_in_progress_per_profile=3,
        )

        assert not res.spawned
        assert res.skipped_per_profile_capped == [(review_id, "alpha", 3)]
        assert kb.get_task(conn, review_id).status == "review"


def test_ready_and_review_tasks_share_global_and_profile_capacity(isolated_kanban_home):
    _, kb = isolated_kanban_home
    with kb.connect_closing() as conn:
        running = _make_ready_tasks(kb, conn, 2)
        for task_id in running:
            kb.claim_task(conn, task_id)
        ready_id = kb.create_task(conn, title="new ready", assignee="alpha")
        review_id = _make_review_task(kb, conn)

        res = kb.dispatch_once(
            conn,
            spawn_fn=_fake_spawn,
            dry_run=True,
            max_in_progress=3,
            max_in_progress_per_profile=4,
        )

        assert [row[0] for row in res.spawned] == [ready_id]
        assert res.skipped_per_profile_capped == []
        assert kb.get_task(conn, review_id).status == "review"


def _write_state_db(home: Path, rows, *, include_billing_columns: bool = True):
    db = home / "state.db"
    conn = sqlite3.connect(db)
    try:
        conn.execute(
            "CREATE TABLE sessions (id TEXT PRIMARY KEY, source TEXT, started_at REAL, ended_at REAL)"
        )
        billing_columns = "billing_mode TEXT DEFAULT ''," if include_billing_columns else ""
        cost_status_column = "cost_status TEXT," if include_billing_columns else ""
        conn.execute(
            f"""CREATE TABLE session_model_usage (
                   session_id TEXT, model TEXT, billing_provider TEXT DEFAULT '', billing_base_url TEXT DEFAULT '',
                   {billing_columns} task TEXT DEFAULT '', api_call_count INTEGER DEFAULT 0,
                   input_tokens INTEGER DEFAULT 0, output_tokens INTEGER DEFAULT 0,
                   cache_read_tokens INTEGER DEFAULT 0, cache_write_tokens INTEGER DEFAULT 0,
                   reasoning_tokens INTEGER DEFAULT 0, estimated_cost_usd REAL DEFAULT 0,
                   actual_cost_usd REAL DEFAULT 0, {cost_status_column} cost_source TEXT,
                   first_seen REAL, last_seen REAL
               )"""
        )
        for idx, row in enumerate(rows):
            sid = f"s{idx}"
            conn.execute(
                "INSERT INTO sessions (id, source, started_at, ended_at) VALUES (?, 'cli', ?, ?)",
                (sid, row["seen"], row["seen"]),
            )
            columns = [
                "session_id",
                "model",
                "estimated_cost_usd",
                "actual_cost_usd",
                "last_seen",
            ]
            values = [
                sid,
                row.get("model", "m"),
                row.get("estimated", 0),
                row.get("actual", 0),
                row["seen"],
            ]
            columns.extend(["billing_provider", "billing_base_url"])
            values.extend([row.get("billing_provider", ""), row.get("billing_base_url", "")])
            if include_billing_columns:
                columns.extend(["billing_mode", "cost_status"])
                values.extend([row.get("billing_mode", ""), row.get("cost_status")])
            placeholders = ", ".join("?" for _ in values)
            conn.execute(
                f"INSERT INTO session_model_usage ({', '.join(columns)}) VALUES ({placeholders})",
                values,
            )
        conn.commit()
    finally:
        conn.close()


def _write_corrupt_state_db(home: Path):
    home.mkdir(parents=True, exist_ok=True)
    (home / "state.db").write_bytes(b"not a sqlite database")


def test_daily_spend_actual_wins_over_estimated_and_multi_profile(isolated_kanban_home):
    home, kb = isolated_kanban_home
    profile_home = home / "profiles" / "alpha"
    now = 1_700_000_000
    _write_state_db(home, [{"seen": now, "estimated": 5, "actual": 2}])
    _write_state_db(profile_home, [{"seen": now, "estimated": 3, "actual": 0}])
    cfg = kb.SpendAdmissionConfig(cap_usd=10, timezone_name="UTC")
    summary = kb.read_daily_spend_ledger(cfg, now=now)
    assert summary.ledger_readable is True
    assert summary.known_metered_cost_usd == pytest.approx(5.0)
    assert summary.actual_cost_usd == pytest.approx(2.0)
    assert summary.estimated_cost_usd == pytest.approx(3.0)


def test_subscription_included_zero_cost_rows_do_not_trigger_unknown_hold(isolated_kanban_home):
    home, kb = isolated_kanban_home
    now = 1_700_000_000
    _write_state_db(
        home,
        [{
            "seen": now,
            "actual": 0,
            "estimated": 0,
            "billing_mode": "subscription_included",
            "cost_status": "included",
        }],
    )
    cfg = kb.SpendAdmissionConfig(
        cap_usd=1,
        timezone_name="UTC",
        unknown_cost_policy="hold",
    )

    summary = kb.read_daily_spend_ledger(cfg, now=now, profile_homes=[home])

    assert summary.ledger_readable is True
    assert summary.included_cost_rows == 1
    assert summary.unknown_cost_rows == 0
    assert summary.known_metered_cost_usd == pytest.approx(0.0)


def test_subscription_included_zero_cost_rows_allow_dispatch_under_hold(isolated_kanban_home):
    home, kb = isolated_kanban_home
    now = 1_700_000_000
    _write_state_db(
        home,
        [{
            "seen": now,
            "actual": 0,
            "estimated": 0,
            "billing_mode": "subscription_included",
            "cost_status": "included",
        }],
    )
    with kb.connect_closing() as conn:
        task_ids = _make_ready_tasks(kb, conn, 1)
        monkeypatch_time = pytest.MonkeyPatch()
        monkeypatch_time.setattr(kb.time, "time", lambda: now)
        try:
            res = kb.dispatch_once(
                conn,
                spawn_fn=_fake_spawn,
                dry_run=True,
                spend_config=kb.SpendAdmissionConfig(
                    cap_usd=1,
                    timezone_name="UTC",
                    unknown_cost_policy="hold",
                ),
            )
        finally:
            monkeypatch_time.undo()

    assert [row[0] for row in res.spawned] == task_ids
    assert not res.skipped_unknown_cost_policy
    assert res.spend_telemetry["included_cost_rows"] == 1


def test_native_chatgpt_codex_blank_zero_rows_allow_dispatch_under_hold_across_profiles(isolated_kanban_home):
    home, kb = isolated_kanban_home
    now = 1_700_000_000
    profile_homes = [home, home / "profiles" / "coder", home / "profiles" / "reviewer"]
    for idx, profile_home in enumerate(profile_homes):
        profile_home.mkdir(parents=True, exist_ok=True)
        _write_state_db(
            profile_home,
            [{
                "seen": now,
                "actual": 0,
                "estimated": 0,
                "billing_provider": "auto",
                "billing_base_url": "https://chatgpt.com/backend-api/codex/",
                "billing_mode": "",
                "cost_status": "",
                "model": ["gpt-5.5", "gpt-5.6-sol", "gpt-5.5"][idx],
            }],
        )
    with kb.connect_closing() as conn:
        task_ids = _make_ready_tasks(kb, conn, 1)
        monkeypatch_time = pytest.MonkeyPatch()
        monkeypatch_time.setattr(kb.time, "time", lambda: now)
        try:
            res = kb.dispatch_once(
                conn,
                spawn_fn=_fake_spawn,
                dry_run=True,
                spend_config=kb.SpendAdmissionConfig(
                    cap_usd=1,
                    timezone_name="UTC",
                    unknown_cost_policy="hold",
                ),
            )
        finally:
            monkeypatch_time.undo()

    assert [row[0] for row in res.spawned] == task_ids
    assert not res.skipped_unknown_cost_policy
    assert res.spend_telemetry["included_cost_rows"] == 3
    assert res.spend_telemetry["explicit_included_cost_rows"] == 0
    assert res.spend_telemetry["transport_inferred_included_rows"] == 3


@pytest.mark.parametrize(
    "base_url",
    [
        "https://evilchatgpt.com/backend-api/codex/",
        "https://chatgpt.com.evil/backend-api/codex/",
        "https://user@chatgpt.com/backend-api/codex/",
        "http://chatgpt.com/backend-api/codex/",
        "https://api.openai.com/v1",
        "https://chatgpt.com/backend-api/codexish/",
        "https://chatgpt.com/backend-api/codex/../metered/",
    ],
)
def test_native_chatgpt_codex_transport_inference_rejects_spoofed_and_custom_urls(
    isolated_kanban_home,
    base_url,
):
    home, kb = isolated_kanban_home
    now = 1_700_000_000
    _write_state_db(
        home,
        [{
            "seen": now,
            "actual": 0,
            "estimated": 0,
            "billing_provider": "auto",
            "billing_base_url": base_url,
            "billing_mode": "",
            "cost_status": "",
            "model": "gpt-5.5",
        }],
    )

    summary = kb.read_daily_spend_ledger(
        kb.SpendAdmissionConfig(cap_usd=1, timezone_name="UTC", unknown_cost_policy="hold"),
        now=now,
        profile_homes=[home],
    )

    assert summary.unknown_cost_rows == 1
    assert summary.included_cost_rows == 0
    assert summary.transport_inferred_included_rows == 0


def test_provider_or_model_alone_do_not_mark_blank_zero_rows_included(isolated_kanban_home):
    home, kb = isolated_kanban_home
    now = 1_700_000_000
    _write_state_db(
        home,
        [
            {
                "seen": now,
                "actual": 0,
                "estimated": 0,
                "billing_provider": "auto",
                "billing_base_url": "",
                "billing_mode": "",
                "cost_status": "",
                "model": "gpt-5.5",
            },
            {
                "seen": now,
                "actual": 0,
                "estimated": 0,
                "billing_provider": "chatgpt",
                "billing_base_url": "",
                "billing_mode": "",
                "cost_status": "",
                "model": "other-model",
            },
        ],
    )

    summary = kb.read_daily_spend_ledger(
        kb.SpendAdmissionConfig(cap_usd=1, timezone_name="UTC", unknown_cost_policy="hold"),
        now=now,
        profile_homes=[home],
    )

    assert summary.unknown_cost_rows == 2
    assert summary.included_cost_rows == 0


def test_native_chatgpt_codex_blank_positive_cost_still_counts_as_metered(isolated_kanban_home):
    home, kb = isolated_kanban_home
    now = 1_700_000_000
    _write_state_db(
        home,
        [{
            "seen": now,
            "actual": 1.25,
            "estimated": 0,
            "billing_provider": "auto",
            "billing_base_url": "https://chatgpt.com/backend-api/codex/",
            "billing_mode": "",
            "cost_status": "",
            "model": "gpt-5.5",
        }],
    )

    summary = kb.read_daily_spend_ledger(
        kb.SpendAdmissionConfig(cap_usd=10, timezone_name="UTC"),
        now=now,
        profile_homes=[home],
    )

    assert summary.known_cost_rows == 1
    assert summary.known_metered_cost_usd == pytest.approx(1.25)
    assert summary.included_cost_rows == 0
    assert summary.transport_inferred_included_rows == 0


def test_subscription_included_rows_do_not_increase_known_metered_spend(isolated_kanban_home):
    home, kb = isolated_kanban_home
    now = 1_700_000_000
    _write_state_db(
        home,
        [{
            "seen": now,
            "actual": 4,
            "estimated": 7,
            "billing_mode": "subscription_included",
            "cost_status": "included",
        }],
    )
    cfg = kb.SpendAdmissionConfig(cap_usd=10, timezone_name="UTC")

    summary = kb.read_daily_spend_ledger(cfg, now=now, profile_homes=[home])

    assert summary.included_cost_rows == 1
    assert summary.known_cost_rows == 0
    assert summary.known_metered_cost_usd == pytest.approx(0.0)
    assert summary.actual_cost_usd == pytest.approx(0.0)
    assert summary.estimated_cost_usd == pytest.approx(0.0)


def test_known_metered_actual_and_estimated_rows_still_count(isolated_kanban_home):
    home, kb = isolated_kanban_home
    now = 1_700_000_000
    _write_state_db(
        home,
        [
            {
                "seen": now,
                "actual": 1.25,
                "estimated": 8,
                "billing_mode": "metered",
                "cost_status": "actual",
            },
            {
                "seen": now,
                "actual": 0,
                "estimated": 2.5,
                "billing_mode": "metered",
                "cost_status": "estimated",
            },
        ],
    )
    cfg = kb.SpendAdmissionConfig(cap_usd=10, timezone_name="UTC")

    summary = kb.read_daily_spend_ledger(cfg, now=now, profile_homes=[home])

    assert summary.known_cost_rows == 2
    assert summary.known_metered_cost_usd == pytest.approx(3.75)
    assert summary.actual_cost_usd == pytest.approx(1.25)
    assert summary.estimated_cost_usd == pytest.approx(2.5)


def test_unknown_metered_rows_still_obey_hold_policy(isolated_kanban_home):
    home, kb = isolated_kanban_home
    now = 1_700_000_000
    _write_state_db(
        home,
        [{
            "seen": now,
            "actual": 0,
            "estimated": 0,
            "billing_mode": "metered",
            "cost_status": None,
        }],
    )

    summary = kb.read_daily_spend_ledger(
        kb.SpendAdmissionConfig(
            cap_usd=1,
            timezone_name="UTC",
            unknown_cost_policy="hold",
        ),
        now=now,
        profile_homes=[home],
    )

    assert summary.unknown_cost_rows == 1
    assert summary.known_metered_cost_usd == pytest.approx(0.0)


def test_malformed_metered_cost_rows_are_unknown_not_crashes(isolated_kanban_home):
    home, kb = isolated_kanban_home
    now = 1_700_000_000
    _write_state_db(
        home,
        [{
            "seen": now,
            "actual": "not-a-number",
            "estimated": 0,
            "billing_mode": "metered",
            "cost_status": "actual",
        }],
    )

    summary = kb.read_daily_spend_ledger(
        kb.SpendAdmissionConfig(cap_usd=1, timezone_name="UTC"),
        now=now,
        profile_homes=[home],
    )

    assert summary.ledger_readable is True
    assert summary.unknown_cost_rows == 1
    assert summary.known_metered_cost_usd == pytest.approx(0.0)


def test_older_ledger_without_billing_columns_is_backward_compatible_fail_closed(isolated_kanban_home):
    home, kb = isolated_kanban_home
    now = 1_700_000_000
    _write_state_db(
        home,
        [{"seen": now, "actual": 0, "estimated": 0}],
        include_billing_columns=False,
    )
    cfg = kb.SpendAdmissionConfig(
        cap_usd=1,
        timezone_name="UTC",
        unknown_cost_policy="hold",
    )

    summary = kb.read_daily_spend_ledger(cfg, now=now, profile_homes=[home])

    assert summary.ledger_readable is True
    assert summary.unknown_cost_rows == 1
    assert summary.known_metered_cost_usd == pytest.approx(0.0)
    assert summary.errors == ()


def test_daily_spend_cap_holds_without_state_mutation(isolated_kanban_home):
    _, kb = isolated_kanban_home
    with kb.connect_closing() as conn:
        task_ids = _make_ready_tasks(kb, conn, 2)
        summary = kb.SpendLedgerSummary(
            ledger_readable=True,
            known_metered_cost_usd=12.5,
            day_start_ts=0,
            day_end_ts=1,
        )
        res = kb.dispatch_once(
            conn,
            spawn_fn=_fake_spawn,
            dry_run=True,
            spend_config=kb.SpendAdmissionConfig(cap_usd=10),
            spend_ledger_reader=lambda _cfg: summary,
        )
        assert [h[0] for h in res.skipped_daily_spend_capped] == task_ids
        assert not res.spawned
        assert [r["status"] for r in conn.execute("SELECT status FROM tasks ORDER BY created_at")] == ["ready", "ready"]


def test_partial_unreadable_spend_ledger_fails_closed_without_start_mutation(isolated_kanban_home):
    home, kb = isolated_kanban_home
    now = 1_700_000_000
    _write_state_db(home, [{"seen": now, "estimated": 3, "actual": 0}])
    _write_corrupt_state_db(home / "profiles" / "alpha")

    with kb.connect_closing() as conn:
        task_ids = _make_ready_tasks(kb, conn, 2)
        monkeypatch_time = pytest.MonkeyPatch()
        monkeypatch_time.setattr(kb.time, "time", lambda: now)
        try:
            res = kb.dispatch_once(
                conn,
                spawn_fn=_fake_spawn,
                spend_config=kb.SpendAdmissionConfig(cap_usd=25, timezone_name="UTC"),
            )
        finally:
            monkeypatch_time.undo()

        assert res.skipped_spend_ledger_unavailable == task_ids
        assert not res.skipped_daily_spend_capped
        assert not res.spawned
        assert res.spend_telemetry["ledger_readable"] is False
        assert res.spend_telemetry["ledgers_read"] == 1
        assert res.spend_telemetry["ledgers_unavailable"] == 1
        assert res.spend_telemetry["known_metered_cost_usd"] == pytest.approx(3.0)
        assert res.spend_telemetry["errors"] == ["ledger_1:DatabaseError"]
        assert conn.execute("SELECT COUNT(*) FROM task_runs").fetchone()[0] == 0
        statuses = conn.execute("SELECT status FROM tasks ORDER BY created_at").fetchall()
        assert [row["status"] for row in statuses] == ["ready", "ready"]


def test_unreadable_ledger_takes_precedence_over_over_cap_healthy_ledger(isolated_kanban_home):
    home, kb = isolated_kanban_home
    now = 1_700_000_000
    _write_state_db(home, [{"seen": now, "estimated": 30, "actual": 0}])
    _write_corrupt_state_db(home / "profiles" / "alpha")

    with kb.connect_closing() as conn:
        task_ids = _make_ready_tasks(kb, conn, 1)
        monkeypatch_time = pytest.MonkeyPatch()
        monkeypatch_time.setattr(kb.time, "time", lambda: now)
        try:
            res = kb.dispatch_once(
                conn,
                spawn_fn=_fake_spawn,
                spend_config=kb.SpendAdmissionConfig(cap_usd=25, timezone_name="UTC"),
            )
        finally:
            monkeypatch_time.undo()

    assert res.skipped_spend_ledger_unavailable == task_ids
    assert not res.skipped_daily_spend_capped
    assert not res.spawned
    assert res.spend_telemetry["ledger_readable"] is False
    assert res.spend_telemetry["known_metered_cost_usd"] == pytest.approx(30.0)


def test_query_failing_ledger_schema_fails_closed_even_with_healthy_ledger(isolated_kanban_home):
    home, kb = isolated_kanban_home
    now = 1_700_000_000
    _write_state_db(home, [{"seen": now, "estimated": 2, "actual": 0}])
    profile_home = home / "profiles" / "alpha"
    profile_home.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(profile_home / "state.db")
    try:
        conn.execute("CREATE TABLE sessions (id TEXT PRIMARY KEY)")
        conn.commit()
    finally:
        conn.close()

    with kb.connect_closing() as board_conn:
        task_ids = _make_ready_tasks(kb, board_conn, 1)
        monkeypatch_time = pytest.MonkeyPatch()
        monkeypatch_time.setattr(kb.time, "time", lambda: now)
        try:
            res = kb.dispatch_once(
                board_conn,
                spawn_fn=_fake_spawn,
                spend_config=kb.SpendAdmissionConfig(cap_usd=25, timezone_name="UTC"),
            )
        finally:
            monkeypatch_time.undo()

    assert res.skipped_spend_ledger_unavailable == task_ids
    assert not res.spawned
    assert res.spend_telemetry["ledger_readable"] is False
    assert res.spend_telemetry["ledgers_unavailable"] == 1
    assert res.spend_telemetry["errors"] == ["ledger_1:missing session_model_usage"]


def test_duplicate_spend_ledger_paths_are_counted_once(isolated_kanban_home):
    home, kb = isolated_kanban_home
    now = 1_700_000_000
    _write_state_db(home, [{"seen": now, "estimated": 2, "actual": 0}])

    summary = kb.read_daily_spend_ledger(
        kb.SpendAdmissionConfig(cap_usd=25, timezone_name="UTC"),
        now=now,
        profile_homes=[home, home.resolve(), home],
    )

    assert summary.ledger_readable is True
    assert summary.profiles_read == 1
    assert summary.ledgers_read == 1
    assert summary.known_metered_cost_usd == pytest.approx(2.0)


def test_all_healthy_multi_profile_spend_ledgers_allow_dispatch(isolated_kanban_home):
    home, kb = isolated_kanban_home
    now = 1_700_000_000
    _write_state_db(home, [{"seen": now, "estimated": 2, "actual": 0}])
    _write_state_db(home / "profiles" / "alpha", [{"seen": now, "estimated": 3, "actual": 0}])

    with kb.connect_closing() as conn:
        task_ids = _make_ready_tasks(kb, conn, 1)
        monkeypatch_time = pytest.MonkeyPatch()
        monkeypatch_time.setattr(kb.time, "time", lambda: now)
        try:
            res = kb.dispatch_once(
                conn,
                spawn_fn=_fake_spawn,
                dry_run=True,
                spend_config=kb.SpendAdmissionConfig(cap_usd=25, timezone_name="UTC"),
            )
        finally:
            monkeypatch_time.undo()

    assert [row[0] for row in res.spawned] == task_ids
    assert not res.skipped_spend_ledger_unavailable
    assert res.spend_telemetry["ledger_readable"] is True
    assert res.spend_telemetry["ledgers_read"] == 2
    assert res.spend_telemetry["ledgers_unavailable"] == 0
    assert res.spend_telemetry["known_metered_cost_usd"] == pytest.approx(5.0)


def test_positive_spend_cap_fails_closed_when_no_ledger_readable(isolated_kanban_home):
    _, kb = isolated_kanban_home
    with kb.connect_closing() as conn:
        task_ids = _make_ready_tasks(kb, conn, 1)
        summary = kb.SpendLedgerSummary(ledger_readable=False)
        res = kb.dispatch_once(
            conn,
            spawn_fn=_fake_spawn,
            dry_run=True,
            spend_config=kb.SpendAdmissionConfig(cap_usd=1),
            spend_ledger_reader=lambda _cfg: summary,
        )
        assert res.skipped_spend_ledger_unavailable == task_ids
        assert not res.spawned


def test_unknown_cost_policy_hold_defers_and_reports_rows(isolated_kanban_home):
    _, kb = isolated_kanban_home
    with kb.connect_closing() as conn:
        task_ids = _make_ready_tasks(kb, conn, 1)
        summary = kb.SpendLedgerSummary(
            ledger_readable=True,
            known_metered_cost_usd=0.25,
            unknown_cost_rows=2,
            day_start_ts=0,
            day_end_ts=1,
        )
        res = kb.dispatch_once(
            conn,
            spawn_fn=_fake_spawn,
            dry_run=True,
            spend_config=kb.SpendAdmissionConfig(cap_usd=1, unknown_cost_policy="hold"),
            spend_ledger_reader=lambda _cfg: summary,
        )
        assert res.skipped_unknown_cost_policy == [(task_ids[0], 2)]
        assert res.spend_telemetry["unknown_cost_rows"] == 2
        assert not res.spawned


def test_daily_spend_day_boundary_uses_local_timezone_dst(isolated_kanban_home):
    _, kb = isolated_kanban_home
    # 2026-03-08 in America/Chicago is a spring-forward 23-hour local day.
    noon_chicago_utc = 1_772_989_200
    start, end, tz = kb._day_bounds_for_timezone("America/Chicago", now=noon_chicago_utc)
    assert tz == "America/Chicago"
    assert end - start == 23 * 60 * 60
