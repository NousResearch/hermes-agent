"""Tests for the derived project usage ledger and Kanban usage CLI."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from hermes_state import SessionDB
from hermes_cli import kanban as kc
from hermes_cli import kanban_db as kb
from hermes_cli import project_usage_ledger as usage


@pytest.fixture
def usage_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb._INITIALIZED_PATHS.clear()
    kb.init_db(board="default")
    return home


def _seed_session(home: Path, session_id: str, *, in_tok: int, out_tok: int, cost: float) -> None:
    db = SessionDB(home / "state.db")
    try:
        db.create_session(session_id, "kanban", model="gpt-test", user_id="u-test")
        db.update_token_counts(
            session_id,
            input_tokens=in_tok,
            output_tokens=out_tok,
            reasoning_tokens=7,
            estimated_cost_usd=cost,
            billing_provider="test-provider",
            billing_mode="api",
            api_call_count=2,
            absolute=True,
        )
        db.end_session(session_id, "done")
    finally:
        db.close()


def _seed_completed_run(session_id: str, *, title: str = "usage task") -> str:
    with kb.connect(board="default") as conn:
        tid = kb.create_task(conn, title=title, assignee="worker")
        assert kb.claim_task(conn, tid)
        assert kb.complete_task(
            conn,
            tid,
            summary="finished",
            metadata={"worker_session_id": session_id},
        )
        return tid


def test_project_usage_backfill_correlates_kanban_run_to_session(usage_home):
    _seed_session(usage_home, "sess-usage-1", in_tok=100, out_tok=50, cost=0.0123)
    tid = _seed_completed_run("sess-usage-1")

    data = usage.get_summary(board="default", task_id=tid, refresh=True)

    assert data["ledger_path"].endswith("usage/project_usage.db")
    assert data["totals"]["input_tokens"] == 100
    assert data["totals"]["output_tokens"] == 50
    assert data["totals"]["reasoning_tokens"] == 7
    assert data["totals"]["estimated_cost_usd"] == pytest.approx(0.0123)
    assert [b["board_slug"] for b in data["boards"]] == ["default"]
    assert data["tasks"][0]["task_id"] == tid
    assert data["tasks"][0]["sessions"] == 1
    assert data["runs"][0]["session_id"] == "sess-usage-1"


def test_project_usage_backfill_batches_upserts_in_transaction(usage_home, monkeypatch):
    _seed_session(usage_home, "sess-txn", in_tok=100, out_tok=50, cost=0.0123)
    _seed_completed_run("sess-txn")
    real_upsert = usage._upsert_entry
    saw_upsert = False

    def assert_transaction(conn, entry):
        nonlocal saw_upsert
        saw_upsert = True
        assert conn.in_transaction is True
        return real_upsert(conn, entry)

    monkeypatch.setattr(usage, "_upsert_entry", assert_transaction)

    usage.backfill()

    assert saw_upsert is True


def test_project_usage_backfill_counts_session_usage_once_across_runs(usage_home):
    _seed_session(usage_home, "sess-retry", in_tok=100, out_tok=50, cost=0.0123)
    tid = _seed_completed_run("sess-retry")
    with kb.connect(board="default") as conn:
        conn.execute(
            """
            INSERT INTO task_runs(
                task_id, profile, status, started_at, ended_at, outcome,
                summary, metadata
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                tid,
                "worker",
                "completed",
                10,
                20,
                "done",
                "retry terminal row",
                json.dumps({"worker_session_id": "sess-retry"}),
            ),
        )

    data = usage.get_summary(board="default", task_id=tid, refresh=True)

    assert data["totals"]["entries"] == 2
    assert data["totals"]["sessions"] == 1
    assert data["totals"]["input_tokens"] == 100
    assert data["totals"]["output_tokens"] == 50
    assert data["totals"]["estimated_cost_usd"] == pytest.approx(0.0123)
    assert data["tasks"][0]["runs"] == 2
    assert data["tasks"][0]["sessions"] == 1
    assert data["tasks"][0]["input_tokens"] == 100


def test_project_usage_backfill_dedupes_per_task_not_globally(usage_home):
    _seed_session(usage_home, "sess-shared", in_tok=100, out_tok=50, cost=0.0123)
    tid1 = _seed_completed_run("sess-shared", title="first shared task")
    tid2 = _seed_completed_run("sess-shared", title="second shared task")

    data1 = usage.get_summary(board="default", task_id=tid1, refresh=True)
    data2 = usage.get_summary(board="default", task_id=tid2, refresh=False)

    assert data1["totals"]["input_tokens"] == 100
    assert data2["totals"]["input_tokens"] == 100


def test_get_task_rollups_finds_specific_task_beyond_summary_limit(usage_home):
    conn = usage.connect()
    try:
        now = 123.0
        for idx in range(501):
            usage._upsert_entry(conn, {
                "source_type": "task_run",
                "source_id": f"default:{idx}",
                "board_slug": "default",
                "board_name": "Default",
                "task_id": f"t-{idx}",
                "task_title": f"task {idx}",
                "task_status": "done",
                "run_id": idx,
                "run_status": "completed",
                "run_outcome": "done",
                "session_id": f"sess-{idx}",
                "session_title": None,
                "session_source": "kanban",
                "user_id": None,
                "model": "gpt-test",
                "billing_provider": "test",
                "billing_mode": "api",
                "input_tokens": idx + 1,
                "output_tokens": 0,
                "cache_read_tokens": 0,
                "cache_write_tokens": 0,
                "reasoning_tokens": 0,
                "api_call_count": 0,
                "tool_call_count": 0,
                "estimated_cost_usd": float(idx + 1),
                "actual_cost_usd": 0.0,
                "cost_status": None,
                "started_at": now + idx,
                "ended_at": now + idx,
                "metadata": None,
                "backfilled_at": now,
            })
    finally:
        conn.close()

    summary = usage.get_summary(board="default", refresh=False)
    assert "t-0" not in {row["task_id"] for row in summary["tasks"]}

    rows = usage.get_task_rollups(board="default", task_ids=["t-0"], refresh=False)
    assert rows[0]["task_id"] == "t-0"
    assert rows[0]["input_tokens"] == 1




def test_backfill_closes_owned_ledger_when_state_open_fails(usage_home, monkeypatch):
    class FakeLedgerConn:
        def __init__(self):
            self.closed = False

        def close(self):
            self.closed = True

    fake_ledger = FakeLedgerConn()
    usage.state_db_path().touch()
    real_sqlite_connect = usage.sqlite3.connect

    def fake_sqlite_connect(path, *args, **kwargs):
        if str(path).endswith("state.db"):
            raise sqlite3.OperationalError("state open failed")
        return real_sqlite_connect(path, *args, **kwargs)

    monkeypatch.setattr(usage, "connect", lambda: fake_ledger)
    monkeypatch.setattr(usage.sqlite3, "connect", fake_sqlite_connect)

    with pytest.raises(sqlite3.OperationalError, match="state open failed"):
        usage.backfill()

    assert fake_ledger.closed is True

def test_backfill_records_board_query_error_without_aborting(usage_home, monkeypatch):
    _seed_session(usage_home, "sess-board-error", in_tok=21, out_tok=8, cost=0.004)
    _seed_completed_run("sess-board-error")
    real_connect = kb.connect

    class FlakyConn:
        def __init__(self, conn):
            self._conn = conn

        def execute(self, sql, *args, **kwargs):
            if "FROM task_runs" in sql:
                raise sqlite3.OperationalError("synthetic task_runs failure")
            return self._conn.execute(sql, *args, **kwargs)

        def close(self):
            self._conn.close()

    def fake_connect(*args, board=None, **kwargs):
        conn = real_connect(*args, board=board, **kwargs)
        if board == "default":
            return FlakyConn(conn)
        return conn

    monkeypatch.setattr(usage.kanban_db, "connect", fake_connect)

    data = usage.get_summary(refresh=True)

    assert data["last_backfill_error"]
    assert "default: synthetic task_runs failure" in data["last_backfill_error"]
    # The skipped board should not abort the whole backfill; raw sessions remain visible.
    assert data["totals"]["input_tokens"] == 21
    assert any(b["board_slug"] == "__unassigned__" for b in data["boards"])

def test_project_usage_summary_keeps_unassigned_sessions_visible(usage_home):
    _seed_session(usage_home, "sess-unassigned", in_tok=11, out_tok=22, cost=0.003)

    data = usage.get_summary(refresh=True)

    unassigned = next(b for b in data["boards"] if b["board_slug"] == "__unassigned__")
    assert unassigned["board_name"] == "Unassigned sessions"
    assert unassigned["sessions"] == 1
    assert data["totals"]["input_tokens"] == 11
    assert data["totals"]["output_tokens"] == 22


def test_backfill_removes_stale_unassigned_row_after_session_links_to_task(usage_home):
    _seed_session(usage_home, "sess-later-linked", in_tok=11, out_tok=22, cost=0.003)
    first = usage.get_summary(refresh=True)
    assert any(b["board_slug"] == "__unassigned__" for b in first["boards"])

    tid = _seed_completed_run("sess-later-linked")
    linked = usage.get_summary(board="default", task_id=tid, refresh=True)
    all_data = usage.get_summary(refresh=False)

    assert linked["totals"]["input_tokens"] == 11
    assert not any(b["board_slug"] == "__unassigned__" for b in all_data["boards"])
    assert all_data["totals"]["input_tokens"] == 11


def test_stamp_usage_metadata_embeds_worker_session_snapshot(usage_home):
    _seed_session(usage_home, "sess-snapshot", in_tok=5, out_tok=6, cost=0.0009)

    stamped = usage.stamp_usage_metadata({"keep": True}, "sess-snapshot")

    assert stamped["keep"] is True
    assert stamped["worker_session_id"] == "sess-snapshot"
    assert stamped["usage_snapshot"]["input_tokens"] == 5
    assert stamped["usage_snapshot"]["output_tokens"] == 6
    assert stamped["usage_snapshot"]["estimated_cost_usd"] == pytest.approx(0.0009)


def test_cli_worker_usage_metadata_stamping_is_best_effort(usage_home, monkeypatch, capsys):
    monkeypatch.setenv("HERMES_KANBAN_TASK", "t-safe")
    monkeypatch.setenv("HERMES_SESSION_ID", "sess-safe")

    def boom(metadata, session_id):
        raise sqlite3.OperationalError("database is locked")

    import sqlite3
    monkeypatch.setattr(usage, "stamp_usage_metadata", boom)

    assert kc._stamp_worker_usage_metadata("t-safe", {"keep": True}) == {
        "keep": True,
        "worker_session_id": "sess-safe",
    }
    assert "usage metadata unavailable" in capsys.readouterr().err


def test_kanban_usage_cli_outputs_json_summary(usage_home):
    _seed_session(usage_home, "sess-cli", in_tok=13, out_tok=17, cost=0.0042)
    tid = _seed_completed_run("sess-cli", title="cli usage task")

    out = kc.run_slash("usage --refresh --json")
    data = json.loads(out)

    assert data["totals"]["input_tokens"] == 13
    assert data["totals"]["output_tokens"] == 17
    row = next(t for t in data["tasks"] if t["task_id"] == tid)
    assert row["task_title"] == "cli usage task"
    assert row["estimated_cost_usd"] == pytest.approx(0.0042)
