from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

from agent.run_usage_ledger import UsageLedger
from agent import run_usage_report
from hermes_cli import kanban_db


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "card-cost.sh"


def test_card_cost_reports_card_and_non_card_runs_from_explicit_board(tmp_path):
    db_path = tmp_path / "state.db"
    ledger = UsageLedger(db_path)
    ledger.start_run(run_id="card-run", process_id="1", task_id="task-1", board="board-a")
    ledger.record_model_usage(
        run_id="card-run",
        event_id="card-event",
        session_id="s-card",
        turn_id="t-card",
        model="model-a",
        provider="provider-a",
        input_tokens=11,
        output_tokens=7,
        cost_usd=0.3,
    )
    ledger.start_run(run_id="direct-run", process_id="2", session_id="s-direct")
    ledger.record_model_usage(
        run_id="direct-run",
        event_id="direct-event",
        session_id="s-direct",
        turn_id="t-direct",
        model="model-b",
        provider="provider-b",
        input_tokens=13,
        output_tokens=5,
        cost_usd=0.2,
    )

    env = {**os.environ, "PYTHONPATH": str(ROOT)}
    result = subprocess.run(
        [str(SCRIPT), "--db", str(db_path), "--board", "board-a", "--include-unassigned"],
        cwd=ROOT,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    rows = json.loads(result.stdout)
    assert [row["run_id"] for row in rows] == ["card-run", "direct-run"]
    assert rows[0]["task_id"] == "task-1"
    assert rows[1]["session_id"] == "s-direct"


def test_card_cost_requires_explicit_board(tmp_path):
    result = subprocess.run(
        [str(SCRIPT), "--db", str(tmp_path / "state.db")],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "--board" in result.stderr


def test_report_all_selected_profiles_deduplicates_identical_global_run_ids(tmp_path, capsys, monkeypatch):
    first = tmp_path / "default.db"
    second = tmp_path / "reviewer.db"
    UsageLedger(first).start_run(run_id="direct-a", process_id="1", session_id="s", started_at=1)
    UsageLedger(second).start_run(run_id="direct-b", process_id="2", session_id="s2")
    monkeypatch.setattr(
        run_usage_report,
        "_selected_ledgers",
        lambda args: [("default", first), ("reviewer", second)],
    )
    assert run_usage_report.main(["--all-profiles", "--run-id", "direct-a"]) == 0
    row = json.loads(capsys.readouterr().out)[0]
    assert row["source_profile"] == "default"

    UsageLedger(second).start_run(run_id="direct-a", process_id="3", session_id="s", started_at=1)
    assert run_usage_report.main(["--all-profiles", "--run-id", "direct-a"]) == 0
    rows = json.loads(capsys.readouterr().out)
    assert len(rows) == 1
    assert rows[0]["source_profile"] == "default"
    assert rows[0]["source_profiles"] == ["default", "reviewer"]


def test_report_all_profiles_rejects_conflicting_duplicate_global_run_ids(tmp_path, capsys, monkeypatch):
    first = tmp_path / "default.db"
    second = tmp_path / "reviewer.db"
    UsageLedger(first).start_run(run_id="conflict", process_id="1")
    UsageLedger(second).start_run(run_id="conflict", process_id="2")
    UsageLedger(second).record_model_usage(
        run_id="conflict", event_id="event", session_id="s", turn_id="t",
        model="m", provider="p", input_tokens=9,
    )
    monkeypatch.setattr(
        run_usage_report,
        "_selected_ledgers",
        lambda args: [("default", first), ("reviewer", second)],
    )

    assert run_usage_report.main(["--all-profiles", "--run-id", "conflict"]) == 2
    assert "conflicting duplicate" in capsys.readouterr().err


def test_all_profiles_maps_root_default_and_named_profiles_once(tmp_path, capsys, monkeypatch):
    root = tmp_path / "hermes"
    named = root / "profiles" / "reviewer"
    named.mkdir(parents=True)
    root.mkdir(exist_ok=True)
    UsageLedger(root / "state.db").start_run(run_id="same", process_id="default", started_at=1)
    UsageLedger(named / "state.db").start_run(run_id="same", process_id="reviewer", started_at=1)
    monkeypatch.setattr(run_usage_report, "get_default_hermes_root", lambda: root)
    monkeypatch.setattr("hermes_cli.profiles.list_profiles", lambda: [
        type("P", (), {"name": "default", "path": root})(),
        type("P", (), {"name": "reviewer", "path": named})(),
    ])
    assert run_usage_report.main(["--all-profiles", "--run-id", "same"]) == 0
    rows = json.loads(capsys.readouterr().out)
    assert len(rows) == 1
    assert rows[0]["source_profile"] == "default"
    assert rows[0]["source_profiles"] == ["default", "reviewer"]


def test_card_query_joins_exact_task_run_usage(tmp_path, capsys):
    board = tmp_path / "kanban.db"
    kanban_db.init_db(board)
    with kanban_db.connect_closing(board) as connection:
        connection.execute(
            "INSERT INTO task_runs(task_id, status, started_at) VALUES (?, ?, ?)",
            ("task-exact", "running", 1),
        )
        task_run_id = connection.execute("SELECT id FROM task_runs").fetchone()[0]
    state = tmp_path / "state.db"
    ledger = UsageLedger(state)
    run_id = f"task-run:{task_run_id}"
    ledger.start_run(run_id=run_id, process_id="p", task_run_id=task_run_id, task_id="task-exact", board="board-a")
    ledger.record_model_usage(
        run_id=run_id, event_id="exact-event", session_id="s", turn_id="t",
        model="m", provider="p", input_tokens=2, output_tokens=3, cost_usd=0.4,
    )
    ledger.finish_run(run_id=run_id, outcome="completed")
    assert ledger.link_kanban_run(task_run_id=task_run_id, usage_run_id=run_id, kanban_db=board)
    assert run_usage_report.main([
        "--db", str(state), "--board", "board-a", "--task-id", "task-exact",
        "--kanban-db", str(board),
    ]) == 0
    rows = json.loads(capsys.readouterr().out)
    assert [row["run_id"] for row in rows] == [run_id]
