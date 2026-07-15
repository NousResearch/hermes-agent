from __future__ import annotations

import json

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_metrics as km
from hermes_cli import kanban as kc
from hermes_cli.config import DEFAULT_CONFIG


def test_ship_crew_flags_default_disabled_and_explicitly_enableable():
    flags = DEFAULT_CONFIG["kanban"]["ship_crew"]
    assert flags == {
        "enforce_contracts": False,
        "compact_worker_context": False,
        "enforce_route_policy": False,
        "enforce_output_budgets": False,
        "quota_domain_scheduling": False,
    }
    enabled = {**flags, "compact_worker_context": True}
    assert enabled["compact_worker_context"] is True
    assert flags["compact_worker_context"] is False


def test_empty_board_metrics_are_read_only(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    kb.init_db()
    db_path = home / "kanban.db"
    before_mtime = db_path.stat().st_mtime_ns
    with kb.connect_closing() as conn:
        before = conn.execute("SELECT name, sql FROM sqlite_master ORDER BY name").fetchall()
        before_counts = {
            table: conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            for table in ("tasks", "task_runs", "task_comments", "task_events")
        }
        snapshot = km.aggregate_connection(conn)
        after = conn.execute("SELECT name, sql FROM sqlite_master ORDER BY name").fetchall()
        after_counts = {
            table: conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            for table in ("tasks", "task_runs", "task_comments", "task_events")
        }
    assert snapshot["read_only"] is True
    assert snapshot["tasks"]["total"] == 0
    assert snapshot["runs"]["total"] == 0
    assert snapshot["protocol_violations"] == 0
    assert [(r["name"], r["sql"]) for r in before] == [(r["name"], r["sql"]) for r in after]
    assert before_counts == after_counts
    assert db_path.stat().st_mtime_ns == before_mtime


def test_metrics_cover_retries_context_routing_and_events():
    tasks = [
        {"id": "t1", "assignee": "engineer", "status": "done", "created_at": 100,
         "body": "hello", "goal_mode": 1},
        {"id": "t2", "assignee": "pirate", "status": "blocked", "created_at": 100,
         "body": "world", "goal_mode": 0},
    ]
    runs = [
        {"id": 1, "task_id": "t1", "profile": "engineer", "started_at": 110,
         "ended_at": 120, "outcome": "completed", "metadata": json.dumps({
             "model": "gemini-flash", "reasoning_effort": "medium",
             "quota_domain": "agy", "goal_mode": True, "parent_summary_bytes": 7,
             "assembled_context_bytes": 33, "output_class": "inline",
             "inline_output_bytes": 12,
         })},
        {"id": 2, "task_id": "t1", "profile": "engineer", "started_at": 130,
         "ended_at": 140, "outcome": "timed_out", "metadata": "{}"},
        {"id": 3, "task_id": "t2", "profile": "pirate", "started_at": 120,
         "ended_at": 125, "outcome": "blocked", "metadata": json.dumps({
             "output_class": "artifact", "output_bytes": 20,
         })},
    ]
    comments = [{"body": "a comment"}]
    events = [
        {"kind": "protocol_violation", "payload": "{}"},
        {"kind": "quota_blocked", "payload": "{}"},
        {"kind": "rate_limit_requeued", "payload": "{}"},
    ]
    snapshot = km.aggregate_metrics(tasks, runs, comments, events, board="fixture")
    assert snapshot["board"] == "fixture"
    assert snapshot["tasks"]["by_profile"] == {"engineer": 1, "pirate": 1}
    assert snapshot["runs"]["retries"] == 1
    assert snapshot["runs"]["failure_outcomes"] == 2
    assert snapshot["runs"]["goal_mode_runs"] == 2
    assert snapshot["runs"]["goal_mode_tasks"] == 1
    assert snapshot["protocol_violations"] == 1
    assert snapshot["events"] == {"rate_limit": 1, "quota_block": 1}
    assert snapshot["routing"]["quota_domain"] == {"agy": 1, "unknown": 2}
    assert snapshot["context_bytes"]["task_body"] == 10
    assert snapshot["context_bytes"]["comments"] == 9
    assert snapshot["timing"]["queue_to_start"]["count"] == 3


def test_metrics_command_emits_json(tmp_path, monkeypatch, capsys):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    kb.init_db()
    output = kc.run_slash("metrics --json")
    payload = json.loads(output)
    assert payload["schema_version"] == "ship-crew/diagnostics/v1"
    assert payload["read_only"] is True
