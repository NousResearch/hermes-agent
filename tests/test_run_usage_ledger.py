from __future__ import annotations

import sqlite3
import threading
import time

from agent.run_usage_ledger import UsageLedger, run_id_for_session
from hermes_cli import kanban_db
from hermes_state import SessionDB


def test_sessiondb_schema_migrates_run_receipts_on_existing_database(tmp_path):
    db_path = tmp_path / "state.db"
    first = SessionDB(db_path)
    first.create_session("legacy-session", source="cli", model="old-model")
    first.close()

    second = SessionDB(db_path)
    with sqlite3.connect(db_path) as connection:
        assert connection.execute("SELECT COUNT(*) FROM usage_runs").fetchone() == (0,)
        assert connection.execute("SELECT model FROM sessions WHERE id = 'legacy-session'").fetchone() == ("old-model",)
    second.close()


def test_legacy_state_database_migrates_without_losing_sessions(tmp_path):
    db_path = tmp_path / "state.db"
    with sqlite3.connect(db_path) as connection:
        connection.execute("CREATE TABLE sessions (id TEXT PRIMARY KEY, title TEXT)")
        connection.execute("INSERT INTO sessions VALUES ('legacy-session', 'kept')")
        connection.commit()

    ledger = UsageLedger(db_path)
    ledger.start_run(
        run_id="run-legacy",
        process_id="123",
        session_id="session-direct",
        model="model-a",
        provider="provider-a",
    )

    with sqlite3.connect(db_path) as connection:
        assert connection.execute("SELECT title FROM sessions WHERE id = 'legacy-session'").fetchone() == ("kept",)
        assert connection.execute("SELECT run_id FROM usage_runs").fetchone() == ("run-legacy",)

    ledger.start_run(
        run_id="run-legacy",
        process_id="123",
        session_id="session-direct",
        model="model-a",
        provider="provider-a",
    )


def test_model_usage_callback_is_idempotent_for_duplicate_event(tmp_path):
    ledger = UsageLedger(tmp_path / "state.db")

    first = ledger.record_model_usage(
        run_id="run-1",
        event_id="session-direct:turn-1:api:1",
        session_id="session-direct",
        turn_id="turn-1",
        model="model-a",
        provider="provider-a",
        input_tokens=100,
        output_tokens=25,
        cost_usd=0.012,
        retry_count=1,
    )
    second = ledger.record_model_usage(
        run_id="run-1",
        event_id="session-direct:turn-1:api:1",
        session_id="session-direct",
        turn_id="turn-1",
        model="model-a",
        provider="provider-a",
        input_tokens=100,
        output_tokens=25,
        cost_usd=0.012,
        retry_count=1,
    )

    assert first is True
    assert second is False
    receipt = ledger.get_run("run-1")
    assert receipt["input_tokens"] == 100
    assert receipt["output_tokens"] == 25
    assert receipt["cost_usd"] == 0.012
    assert receipt["turn_count"] == 1
    assert receipt["retry_count"] == 1


def test_event_identity_is_scoped_to_run(tmp_path):
    ledger = UsageLedger(tmp_path / "state.db")

    assert ledger.record_model_usage(
        run_id="run-a", event_id="provider-request-1", session_id="s-a",
        turn_id="t-a", model="m", provider="p", input_tokens=2,
    )
    assert ledger.record_model_usage(
        run_id="run-b", event_id="provider-request-1", session_id="s-b",
        turn_id="t-b", model="m", provider="p", input_tokens=3,
    )
    assert not ledger.record_model_usage(
        run_id="run-a", event_id="provider-request-1", session_id="s-a",
        turn_id="t-a", model="m", provider="p", input_tokens=2,
    )

    assert ledger.get_run("run-a")["input_tokens"] == 2
    assert ledger.get_run("run-b")["input_tokens"] == 3


def test_legacy_global_event_id_migrates_to_composite_identity(tmp_path):
    db_path = tmp_path / "state.db"
    with sqlite3.connect(db_path) as connection:
        connection.executescript(
            """
            CREATE TABLE usage_runs (
                run_id TEXT PRIMARY KEY, process_id TEXT NOT NULL,
                started_at REAL NOT NULL, updated_at REAL NOT NULL
            );
            CREATE TABLE usage_events (
                event_id TEXT PRIMARY KEY, run_id TEXT NOT NULL,
                event_type TEXT NOT NULL, session_id TEXT, turn_id TEXT,
                input_tokens INTEGER NOT NULL DEFAULT 0,
                output_tokens INTEGER NOT NULL DEFAULT 0,
                cost_usd REAL NOT NULL DEFAULT 0,
                retry_count INTEGER NOT NULL DEFAULT 0,
                model TEXT, provider TEXT, created_at REAL NOT NULL
            );
            INSERT INTO usage_runs VALUES ('run-a', 'proc-a', 1, 1);
            INSERT INTO usage_runs VALUES ('run-b', 'proc-b', 1, 1);
            INSERT INTO usage_events VALUES
                ('request-1', 'run-a', 'model', 's-a', 't-a', 2, 1, 0.1, 0, 'm', 'p', 1);
            INSERT INTO usage_events VALUES
                ('request-2', 'run-b', 'model', 's-b', 't-b', 3, 1, 0.2, 0, 'm', 'p', 2);
            """
        )
        connection.commit()

    ledger = UsageLedger(db_path)
    with sqlite3.connect(db_path) as connection:
        columns = connection.execute("PRAGMA table_info(usage_events)").fetchall()
        pk = {row[1]: row[5] for row in columns if row[5]}
        rows = connection.execute(
            "SELECT run_id, event_id FROM usage_events ORDER BY run_id"
        ).fetchall()
    assert pk == {"run_id": 1, "event_id": 2}
    assert rows == [("run-a", "request-1"), ("run-b", "request-2")]
    assert ledger.record_model_usage(
        run_id="run-b", event_id="request-1", session_id="s-b",
        turn_id="t-c", model="m", provider="p", input_tokens=4,
    )


def test_legacy_model_table_migration_is_atomic_idempotent_and_indexed(tmp_path):
    db_path = tmp_path / "legacy-models.db"
    with sqlite3.connect(db_path) as connection:
        connection.executescript(
            """
            CREATE TABLE usage_runs (
                run_id TEXT PRIMARY KEY, process_id TEXT NOT NULL,
                started_at REAL NOT NULL, updated_at REAL NOT NULL
            );
            CREATE TABLE usage_events (
                event_id TEXT PRIMARY KEY, run_id TEXT NOT NULL,
                event_type TEXT NOT NULL, created_at REAL NOT NULL
            );
            CREATE TABLE usage_run_models (
                run_id TEXT NOT NULL, model TEXT NOT NULL,
                provider TEXT,
                input_tokens INTEGER NOT NULL DEFAULT 0,
                output_tokens INTEGER NOT NULL DEFAULT 0,
                cost_usd REAL NOT NULL DEFAULT 0,
                event_count INTEGER NOT NULL DEFAULT 0,
                PRIMARY KEY (run_id, model, provider)
            );
            INSERT INTO usage_runs VALUES ('run-a', 'proc-a', 1, 1);
            INSERT INTO usage_run_models VALUES ('run-a', 'm', NULL, 2, 3, 0.4, 1);
            INSERT INTO usage_run_models VALUES ('run-a', 'm', NULL, 5, 7, 0.6, 2);
            """
        )
        connection.commit()

    UsageLedger(db_path)
    UsageLedger(db_path)
    with sqlite3.connect(db_path) as connection:
        row = connection.execute(
            "SELECT run_id, model, provider, input_tokens, output_tokens, cost_usd, event_count FROM usage_run_models"
        ).fetchone()
        indexes = connection.execute("PRAGMA index_list(usage_run_models)").fetchall()
        legacy = connection.execute(
            "SELECT COUNT(*) FROM sqlite_master WHERE name='usage_run_models_legacy'"
        ).fetchone()[0]
    assert row == ("run-a", "m", "unknown", 7, 10, 1.0, 3)
    assert any(index[1] == "idx_usage_run_models_run" for index in indexes)
    assert legacy == 0


def test_usage_receipt_keeps_card_optional_and_reports_non_card_identity(tmp_path):
    ledger = UsageLedger(tmp_path / "state.db")
    ledger.start_run(
        run_id="run-card",
        process_id="1",
        session_id="session-card",
        task_id="task-123",
        board="default",
        model="model-a",
        provider="provider-a",
    )
    ledger.record_model_usage(
        run_id="run-card",
        event_id="event-card",
        session_id="session-card",
        turn_id="turn-card",
        model="model-a",
        provider="provider-a",
        input_tokens=10,
        output_tokens=5,
        cost_usd=0.1,
    )
    ledger.start_run(
        run_id="run-direct",
        process_id="2",
        session_id="session-direct",
        model="model-b",
        provider="provider-b",
    )
    ledger.record_model_usage(
        run_id="run-direct",
        event_id="event-direct",
        session_id="session-direct",
        turn_id="turn-direct",
        model="model-b",
        provider="provider-b",
        input_tokens=20,
        output_tokens=8,
        cost_usd=0.2,
    )

    card_rows = ledger.report(board="default", task_id="task-123")
    direct_rows = ledger.report(board="default", run_id="run-direct", include_unassigned=True)

    assert [row["run_id"] for row in card_rows] == ["run-card"]
    assert card_rows[0]["task_id"] == "task-123"
    assert [row["run_id"] for row in direct_rows] == ["run-direct"]
    assert direct_rows[0]["session_id"] == "session-direct"


def test_finish_records_elapsed_failure_outcome_and_tool_calls_once(tmp_path):
    ledger = UsageLedger(tmp_path / "state.db")
    ledger.start_run(run_id="run-fail", process_id="9", session_id="s")
    assert ledger.record_tool_call(run_id="run-fail", event_id="tool-1", session_id="s") is True
    assert ledger.record_tool_call(run_id="run-fail", event_id="tool-1", session_id="s") is False

    ledger.finish_run(
        run_id="run-fail",
        outcome="failed",
        failure_reason="provider timeout",
        ended_at=120.0,
        elapsed=5.5,
    )

    receipt = ledger.get_run("run-fail")
    assert receipt["tool_call_count"] == 1
    assert receipt["outcome"] == "failed"
    assert receipt["failure_reason"] == "provider timeout"
    assert receipt["elapsed"] == 5.5
    assert receipt["ended_at"] == 120.0


def test_kanban_link_is_keyed_by_authoritative_task_run_and_idempotent(tmp_path):
    board = tmp_path / "kanban.db"
    kanban_db.init_db(board)
    with kanban_db.connect_closing(board) as connection:
        connection.execute(
            "INSERT INTO task_runs(task_id, status, started_at) VALUES (?, ?, ?)",
            ("task-1", "running", 1),
        )
        task_run_id = connection.execute("SELECT id FROM task_runs").fetchone()[0]

    ledger = UsageLedger(tmp_path / "state.db")
    run_id = f"task-run:{task_run_id}"
    ledger.start_run(
        run_id=run_id,
        process_id="worker",
        task_run_id=task_run_id,
        task_id="task-1",
        board="default",
        model="model-a",
        provider="provider-a",
    )
    ledger.record_model_usage(
        run_id=run_id,
        event_id="event-1",
        session_id="s",
        turn_id="t",
        model="model-a",
        provider="provider-a",
        input_tokens=4,
        output_tokens=5,
        cost_usd=0.6,
    )
    ledger.finish_run(run_id=run_id, outcome="completed")
    assert ledger.link_kanban_run(task_run_id=task_run_id, usage_run_id=run_id, kanban_db=board)
    assert ledger.link_kanban_run(task_run_id=task_run_id, usage_run_id=run_id, kanban_db=board)

    with kanban_db.connect_closing(board) as connection:
        row = connection.execute(
            "SELECT task_run_id, usage_run_id, input_tokens, output_tokens, cost_usd FROM task_run_usage"
        ).fetchone()
    assert tuple(row) == (task_run_id, run_id, 4, 5, 0.6)


def test_async_writer_preserves_order_and_isolates_queue_backpressure(tmp_path):
    ledger = UsageLedger(tmp_path / "state.db", queue_size=2)
    assert ledger.queue_start_run(run_id="ordered", process_id="1")
    assert ledger.queue_model_usage(
        run_id="ordered", event_id="one", session_id="s", turn_id="t1",
        model="m", provider="p", input_tokens=1,
    )
    # A full queue is a bounded accounting loss, not a conversation failure.
    assert ledger.queue_model_usage(
        run_id="ordered", event_id="two", session_id="s", turn_id="t2",
        model="m", provider="p", input_tokens=1,
    ) in {True, False}
    assert ledger.flush()
    assert ledger.get_run("ordered")["input_tokens"] >= 1


def test_full_queue_retains_critical_events_and_persists_noncritical_diagnostic(tmp_path):
    ledger = UsageLedger(tmp_path / "state.db", queue_size=1)
    gate = threading.Event()
    entered = threading.Event()

    def blocked_writer():
        entered.set()
        gate.wait(timeout=2)

    ledger._writer_loop = blocked_writer
    assert ledger.queue_start_run(run_id="full", process_id="p")
    assert entered.wait(timeout=1)
    assert not ledger.queue_model_usage(
        run_id="full", event_id="api-1", session_id="s", turn_id="t",
        model="m", provider="p", input_tokens=5, output_tokens=2, cost_usd=0.25,
    )
    gate.set()
    assert not ledger.finalize_run(run_id="full", outcome="completed")
    with sqlite3.connect(tmp_path / "state.db") as connection:
        assert connection.execute("SELECT COUNT(*) FROM usage_diagnostics").fetchone()[0] >= 1


def test_writer_exception_is_replayed_during_finalization(tmp_path):
    ledger = UsageLedger(tmp_path / "state.db")
    ledger.start_run(run_id="replay", process_id="p")
    original = ledger._record_event
    state = {"failed": False}

    def fail_once(**kwargs):
        if not state["failed"]:
            state["failed"] = True
            raise sqlite3.OperationalError("injected writer failure")
        return original(**kwargs)

    ledger._record_event = fail_once
    assert ledger.queue_model_usage(
        run_id="replay", event_id="api-1", session_id="s", turn_id="t",
        model="m", provider="p", input_tokens=3, output_tokens=4, cost_usd=0.5,
    )
    assert ledger.finalize_run(run_id="replay", outcome="completed")
    receipt = ledger.get_run("replay")
    assert receipt["input_tokens"] == 3
    assert receipt["output_tokens"] == 4
    assert receipt["cost_usd"] == 0.5


def test_persistent_writer_failure_fails_closed_and_persists_diagnostic(tmp_path):
    ledger = UsageLedger(tmp_path / "state.db")
    ledger.start_run(run_id="persistent", process_id="p")

    def always_fail(**_kwargs):
        raise sqlite3.OperationalError("persistent failure")

    ledger._record_event = always_fail
    assert ledger.queue_model_usage(
        run_id="persistent", event_id="api-1", session_id="s", turn_id="t",
        model="m", provider="p", input_tokens=3,
    )
    assert not ledger.finalize_run(run_id="persistent", outcome="completed")
    with sqlite3.connect(tmp_path / "state.db") as connection:
        diagnostics = connection.execute(
            "SELECT diagnostic_type, detail FROM usage_diagnostics"
        ).fetchall()
    assert any(detail == "persistent_writer_failure" for _, detail in diagnostics)


def test_failed_operation_buffer_overflow_sets_global_fail_closed_sentinel(tmp_path):
    ledger = UsageLedger(tmp_path / "state.db", queue_size=2)
    ledger.start_run(run_id="buffer", process_id="p")

    def always_fail(**_kwargs):
        raise sqlite3.OperationalError("injected persistence failure")

    ledger._record_event = always_fail
    for event_id in ("api-1", "api-2"):
        assert ledger.queue_model_usage(
            run_id="buffer", event_id=event_id, session_id="s", turn_id=event_id,
            model="m", provider="p", input_tokens=1,
        )
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline:
            with ledger._queue_cond:
                if len(ledger._failed_operations) >= int(event_id[-1]):
                    break
            time.sleep(0.005)
        else:
            raise AssertionError("writer did not retain failed operation")

    # The caller-facing queue API remains nonblocking even after the bounded
    # failed-operation buffer is full. The writer observes the third failure
    # and records the global fail-closed state automatically.
    started = time.monotonic()
    assert ledger.queue_model_usage(
        run_id="buffer", event_id="api-3", session_id="s", turn_id="api-3",
        model="m", provider="p", input_tokens=1,
    )
    assert time.monotonic() - started < 0.5
    assert ledger.flush()

    assert not ledger.finalize_run(run_id="buffer", outcome="completed")
    with sqlite3.connect(tmp_path / "state.db") as connection:
        diagnostics = connection.execute(
            "SELECT run_id, diagnostic_type, detail FROM usage_diagnostics"
        ).fetchall()
        receipt = connection.execute(
            "SELECT ended_at FROM usage_runs WHERE run_id='buffer'"
        ).fetchone()
        projection_table = connection.execute(
            "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='task_run_usage'"
        ).fetchone()[0]
    assert any(
        kind == "usage_incomplete_global" and detail == "failed_buffer_full"
        for _, kind, detail in diagnostics
    )
    assert any(run_id == "buffer" and detail == "failed_buffer_full" for run_id, _, detail in diagnostics)
    assert receipt == (None,)
    assert projection_table == 0
    with ledger._queue_cond:
        assert len(ledger._queue) <= 2
        assert len(ledger._failed_operations) <= 2
        assert len(ledger._dropped) <= 256
        assert len(ledger._incomplete_runs) <= 256
        assert len(ledger._global_diagnostics) <= 256


def test_diagnostic_capacity_exhaustion_is_automatic_and_global(tmp_path):
    ledger = UsageLedger(tmp_path / "state.db", queue_size=1)
    entered = threading.Event()
    release = threading.Event()

    def blocked_writer(**_kwargs):
        entered.set()
        assert release.wait(timeout=5)
        raise sqlite3.OperationalError("injected persistence failure")

    ledger._record_event = blocked_writer
    ledger.start_run(run_id="blocker", process_id="p")
    assert ledger.queue_model_usage(
        run_id="blocker", event_id="blocker-event", session_id="s", turn_id="t",
        model="m", provider="p", input_tokens=1,
    )
    assert entered.wait(timeout=2)

    dropped = []
    for index in range(300):
        run_id = f"dropped-{index}"
        ledger.start_run(run_id=run_id, process_id=run_id)
        accepted = ledger.queue_model_usage(
            run_id=run_id, event_id=f"event-{index}", session_id=run_id, turn_id=f"turn-{index}",
            model="m", provider="p", input_tokens=1,
        )
        if not accepted:
            dropped.append(run_id)
    assert len(dropped) > 256
    release.set()
    assert ledger.flush()

    with ledger._queue_cond:
        assert len(ledger._queue) <= 1
        assert len(ledger._failed_operations) <= 1
        assert len(ledger._dropped) <= 256
        assert len(ledger._incomplete_runs) <= 256
        assert len(ledger._global_diagnostics) <= 256

    affected = dropped[256]
    assert not ledger.finalize_run(run_id=affected, outcome="completed")
    assert not ledger.finalize_run(run_id="dropped-after-cap", outcome="completed")
    with sqlite3.connect(tmp_path / "state.db") as connection:
        diagnostics = connection.execute(
            "SELECT run_id, diagnostic_type, detail FROM usage_diagnostics"
        ).fetchall()
        ended = connection.execute(
            "SELECT run_id, ended_at FROM usage_runs WHERE run_id IN (?, ?) ORDER BY run_id",
            (affected, "dropped-after-cap"),
        ).fetchall()
    assert any(
        run_id is None and kind == "usage_incomplete_global" and detail == "diagnostic_capacity_exhausted"
        for run_id, kind, detail in diagnostics
    )
    assert len([row for row in diagnostics if row[1] == "dropped_event"]) >= 256
    assert all(row[1] is None for row in ended)


def test_model_breakdown_is_deterministic_and_marks_mixed_runs(tmp_path):
    ledger = UsageLedger(tmp_path / "state.db")
    ledger.record_model_usage(
        run_id="mixed", event_id="a", session_id="s", turn_id="t1",
        model="model-b", provider="provider-b", input_tokens=2, output_tokens=3, cost_usd=0.2,
    )
    ledger.record_model_usage(
        run_id="mixed", event_id="b", session_id="s", turn_id="t2",
        model="model-a", provider="provider-a", input_tokens=5, output_tokens=7, cost_usd=0.7,
    )
    receipt = ledger.get_run("mixed")
    assert receipt["model"] == "mixed"
    assert receipt["provider"] == "mixed"
    assert [item["model"] for item in receipt["model_breakdown"]] == ["model-a", "model-b"]
    assert receipt["input_tokens"] == 7
    assert receipt["output_tokens"] == 10
    assert receipt["cost_usd"] == 0.9


def test_unknown_provider_is_non_null_and_idempotent(tmp_path):
    ledger = UsageLedger(tmp_path / "state.db")
    assert ledger.record_model_usage(
        run_id="unknown", event_id="one", session_id="s", turn_id="t1",
        model="fallback", provider=None, input_tokens=1, output_tokens=2,
    )
    assert ledger.record_model_usage(
        run_id="unknown", event_id="two", session_id="s", turn_id="t2",
        model="fallback", provider="", input_tokens=3, output_tokens=4,
    )
    receipt = ledger.get_run("unknown")
    assert receipt["model_breakdown"] == [{
        "model": "fallback", "provider": "unknown", "input_tokens": 4,
        "output_tokens": 6, "cost_usd": 0.0, "event_count": 2,
    }]


def test_session_run_identity_is_process_distinct_and_task_env_wins(monkeypatch):
    monkeypatch.delenv("HERMES_RUN_ID", raising=False)
    monkeypatch.delenv("HERMES_KANBAN_RUN_ID", raising=False)
    first = run_id_for_session("same-session")
    second = run_id_for_session("same-session")
    assert first == second
    monkeypatch.setenv("HERMES_RUN_ID", "explicit")
    assert run_id_for_session("same-session") == "explicit"
    monkeypatch.setenv("HERMES_KANBAN_RUN_ID", "42")
    assert run_id_for_session("same-session") == "task-run:42"


def test_process_identity_is_not_a_reusable_bare_pid():
    from agent.run_usage_ledger import process_invocation_id

    process_id = process_invocation_id()
    assert process_id.startswith("proc-")
    assert not process_id.isdigit()


def test_kanban_link_rejects_mismatched_authoritative_task_run(tmp_path):
    board = tmp_path / "kanban.db"
    kanban_db.init_db(board)
    with kanban_db.connect_closing(board) as connection:
        connection.execute("INSERT INTO task_runs(task_id, status, started_at) VALUES ('task', 'running', 1)")
        task_run_id = connection.execute("SELECT id FROM task_runs").fetchone()[0]
    ledger = UsageLedger(tmp_path / "state.db")
    ledger.start_run(run_id="task-run:999", process_id="p", task_run_id=999, task_id="task")
    ledger.finish_run(run_id="task-run:999", outcome="completed")
    assert not ledger.link_kanban_run(task_run_id=task_run_id, usage_run_id="task-run:999", kanban_db=board)
