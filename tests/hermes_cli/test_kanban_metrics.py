import json
import sqlite3
from pathlib import Path

from hermes_cli import kanban_metrics

NOW = 10_000


def make_db(path: Path) -> None:
    conn = sqlite3.connect(path)
    conn.executescript(
        """
        CREATE TABLE tasks (
            id TEXT PRIMARY KEY,
            title TEXT NOT NULL,
            body TEXT,
            assignee TEXT,
            status TEXT NOT NULL,
            priority INTEGER DEFAULT 0,
            created_by TEXT,
            created_at INTEGER NOT NULL,
            started_at INTEGER,
            completed_at INTEGER,
            workspace_kind TEXT DEFAULT 'scratch',
            workspace_path TEXT,
            tenant TEXT,
            result TEXT,
            current_run_id INTEGER,
            skills TEXT,
            max_runtime_seconds INTEGER
        );
        CREATE TABLE task_links (parent_id TEXT NOT NULL, child_id TEXT NOT NULL, PRIMARY KEY(parent_id, child_id));
        CREATE TABLE task_events (id INTEGER PRIMARY KEY AUTOINCREMENT, task_id TEXT NOT NULL, run_id INTEGER, kind TEXT NOT NULL, payload TEXT, created_at INTEGER NOT NULL);
        CREATE TABLE task_runs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            task_id TEXT NOT NULL,
            profile TEXT,
            step_key TEXT,
            status TEXT NOT NULL,
            started_at INTEGER NOT NULL,
            ended_at INTEGER,
            outcome TEXT,
            summary TEXT,
            metadata TEXT,
            error TEXT,
            max_runtime_seconds INTEGER
        );
        """
    )
    conn.close()


def add_task(conn, task_id, title, assignee, status, created_at, **kw):
    cols = {
        "id": task_id,
        "title": title,
        "body": kw.pop("body", None),
        "assignee": assignee,
        "status": status,
        "priority": kw.pop("priority", 0),
        "created_by": kw.pop("created_by", None),
        "created_at": created_at,
        "started_at": kw.pop("started_at", None),
        "completed_at": kw.pop("completed_at", None),
        "workspace_kind": "scratch",
        "workspace_path": None,
        "tenant": None,
        "result": kw.pop("result", None),
        "current_run_id": None,
        "skills": kw.pop("skills", None),
        "max_runtime_seconds": kw.pop("max_runtime_seconds", None),
    }
    assert not kw
    conn.execute(
        f"INSERT INTO tasks ({', '.join(cols)}) VALUES ({', '.join(['?'] * len(cols))})",
        list(cols.values()),
    )


def add_run(conn, task_id, status, started_at, ended_at=None, outcome=None, summary=None, metadata=None):
    conn.execute(
        "INSERT INTO task_runs (task_id, profile, status, started_at, ended_at, outcome, summary, metadata) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (task_id, "backend-eng", status, started_at, ended_at, outcome, summary, json.dumps(metadata) if isinstance(metadata, dict) else metadata),
    )


def add_event(conn, task_id, kind, created_at, payload=None):
    conn.execute(
        "INSERT INTO task_events (task_id, kind, payload, created_at) VALUES (?, ?, ?, ?)",
        (task_id, kind, payload, created_at),
    )


def compute(tmp_path):
    db = tmp_path / "kanban.db"
    return kanban_metrics.compute_metrics(db, generated_at=NOW, window_seconds=10_000, immutable=True)


def test_root_task_queue_and_active_time(tmp_path):
    db = tmp_path / "kanban.db"
    make_db(db)
    conn = sqlite3.connect(db)
    add_task(conn, "t_root", "Implement example", "backend-eng", "done", 100, started_at=150, completed_at=250)
    add_run(conn, "t_root", "done", 150, 250, "completed")
    conn.commit(); conn.close()

    metrics = compute(tmp_path)
    task = metrics["tasks"][0]
    assert task["durations"]["created_to_first_claim_seconds"] == 50
    assert task["durations"]["ready_to_first_claim_seconds"] == 50
    assert task["durations"]["active_seconds_total"] == 100
    assert task["stage"] == "implementation"


def test_child_dependency_wait_and_ready_queue(tmp_path):
    db = tmp_path / "kanban.db"
    make_db(db)
    conn = sqlite3.connect(db)
    add_task(conn, "t_parent", "Spec plan", "pm", "done", 100, completed_at=300)
    add_task(conn, "t_child", "Build child", "backend-eng", "done", 150, started_at=360, completed_at=460)
    conn.execute("INSERT INTO task_links(parent_id, child_id) VALUES ('t_parent', 't_child')")
    add_run(conn, "t_child", "done", 360, 460, "completed")
    conn.commit(); conn.close()

    child = {t["task_id"]: t for t in compute(tmp_path)["tasks"]}["t_child"]
    assert child["dependency_ready_at"] == 300
    assert child["durations"]["dependency_wait_seconds"] == 150
    assert child["durations"]["ready_to_first_claim_seconds"] == 60


def test_block_unblock_and_open_blocked_intervals(tmp_path):
    db = tmp_path / "kanban.db"
    make_db(db)
    conn = sqlite3.connect(db)
    add_task(conn, "t_block", "Fix blocker", "backend-eng", "done", 100, completed_at=500)
    add_run(conn, "t_block", "blocked", 100, 120, "blocked")
    add_event(conn, "t_block", "blocked", 120)
    add_event(conn, "t_block", "unblocked", 220)
    add_run(conn, "t_block", "done", 300, 500, "completed")
    add_task(conn, "t_open_block", "Verify blocked", "verifier", "blocked", 900)
    add_event(conn, "t_open_block", "blocked", 950)
    conn.commit(); conn.close()

    by_id = {t["task_id"]: t for t in compute(tmp_path)["tasks"]}
    assert by_id["t_block"]["durations"]["blocked_seconds_total"] == 100
    assert by_id["t_open_block"]["durations"]["blocked_seconds_total"] == NOW - 950


def test_crash_reclaim_retry_splits_active_by_outcome(tmp_path):
    db = tmp_path / "kanban.db"
    make_db(db)
    conn = sqlite3.connect(db)
    add_task(conn, "t_retry", "Fix retry", "backend-eng", "done", 100, completed_at=500)
    add_run(conn, "t_retry", "crashed", 120, 170, "crashed")
    add_run(conn, "t_retry", "reclaimed", 180, 200, "reclaimed")
    add_run(conn, "t_retry", "done", 300, 500, "completed")
    conn.commit(); conn.close()

    task = compute(tmp_path)["tasks"][0]
    assert task["durations"]["active_crashed_seconds"] == 50
    assert task["durations"]["active_reclaimed_seconds"] == 20
    assert task["run_counts"]["total"] == 3


def test_review_failure_remediation_re_review_success(tmp_path):
    db = tmp_path / "kanban.db"
    make_db(db)
    conn = sqlite3.connect(db)
    add_task(conn, "t_review", "Review feature", "reviewer", "done", 100, completed_at=200)
    add_event(conn, "t_review", "completion_blocked_failed_gate", 200)
    add_task(conn, "t_remediate", "Remediate review findings", "backend-eng", "done", 210, started_at=220, completed_at=320)
    add_task(conn, "t_rereview", "Re-review remediation", "reviewer", "done", 330, started_at=340, completed_at=360)
    conn.execute("INSERT INTO task_links VALUES ('t_review', 't_remediate')")
    conn.execute("INSERT INTO task_links VALUES ('t_remediate', 't_rereview')")
    add_run(conn, "t_remediate", "done", 220, 320, "completed")
    add_run(conn, "t_rereview", "done", 340, 360, "completed")
    conn.commit(); conn.close()

    metrics = compute(tmp_path)
    tasks_by_id = {t["task_id"]: t for t in metrics["tasks"]}
    assert metrics["summaries"]["rework"]["failed_gate_count"] == 1
    assert metrics["summaries"]["rework"]["remediation_loop_count"] >= 1
    assert tasks_by_id["t_remediate"]["stage"] == "remediation"
    assert tasks_by_id["t_rereview"]["stage"] == "review"
    assert "remediation" in {t["stage"] for t in metrics["tasks"] if t["rework"]}


def test_re_review_and_re_verify_count_as_waiting_gates(tmp_path):
    db = tmp_path / "kanban.db"
    make_db(db)
    conn = sqlite3.connect(db)
    add_task(conn, "t_rereview", "Re-review remediation for t_parent", "reviewer", "ready", 100)
    add_task(conn, "t_reverify", "Re-verify remediation for t_parent", "verifier", "ready", 200)
    conn.commit(); conn.close()

    metrics = compute(tmp_path)
    tasks_by_id = {t["task_id"]: t for t in metrics["tasks"]}
    assert tasks_by_id["t_rereview"]["stage"] == "review"
    assert tasks_by_id["t_reverify"]["stage"] == "verification"
    assert {g["task_id"]: g["stage"] for g in metrics["oldest_waiting_gates"]} == {
        "t_rereview": "review",
        "t_reverify": "verification",
    }
    assert metrics["summaries"]["wip"]["counts_by_assignee_stage"]["reviewer"] == {"review": 1}
    assert metrics["summaries"]["wip"]["counts_by_assignee_stage"]["verifier"] == {"verification": 1}


def test_open_running_task_active_open_age(tmp_path):
    db = tmp_path / "kanban.db"
    make_db(db)
    conn = sqlite3.connect(db)
    add_task(conn, "t_running", "Implement running", "backend-eng", "running", 100, started_at=900)
    add_run(conn, "t_running", "running", 900, None, None)
    conn.commit(); conn.close()

    task = compute(tmp_path)["tasks"][0]
    assert task["durations"]["active_seconds_total"] == NOW - 900
    assert task["durations"]["active_open_age_seconds"] == NOW - 900
    assert task["open_interval"] is True


def test_archived_done_graph_cycle_time(tmp_path):
    db = tmp_path / "kanban.db"
    make_db(db)
    conn = sqlite3.connect(db)
    add_task(conn, "t_a", "Spec", "pm", "done", 100, completed_at=200)
    add_task(conn, "t_b", "Build", "backend-eng", "archived", 150, completed_at=500)
    conn.execute("INSERT INTO task_links VALUES ('t_a', 't_b')")
    conn.commit(); conn.close()

    graph = compute(tmp_path)["graphs"][0]
    assert graph["open_interval"] is False
    assert graph["cycle_time_seconds"] == 400


def test_unknown_event_bad_payload_no_crash(tmp_path):
    db = tmp_path / "kanban.db"
    make_db(db)
    conn = sqlite3.connect(db)
    add_task(conn, "t_unknown", "Mystery", "nobody", "ready", 100)
    add_event(conn, "t_unknown", "weird_kind", 120, "{bad json")
    conn.commit(); conn.close()

    metrics = compute(tmp_path)
    assert metrics["unknown_event_kinds"] == {"weird_kind": 1}
    assert "payload_parse_errors=1" in metrics["unknowns"]


def test_reserve_backlog_candidates_and_markdown(tmp_path):
    db = tmp_path / "kanban.db"
    make_db(db)
    conn = sqlite3.connect(db)
    add_task(conn, "t_ci", "Fix CI lint tests", "backend-eng", "ready", 100, priority=0, body="cleanup docs", max_runtime_seconds=600)
    add_task(conn, "t_urgent", "Implement urgent", "backend-eng", "ready", 100, priority=50)
    conn.commit(); conn.close()

    metrics = compute(tmp_path)
    reserve = metrics["summaries"]["reserve_backlog"]
    assert reserve["reserve_backlog_ready_count"] == 1
    assert reserve["reserve_backlog_estimated_minutes"] == 10
    assert reserve["coverage_15m_blocks"] == 0
    md = kanban_metrics.render_markdown(metrics)
    assert "Reserve coverage" in md
    assert "Kanban metrics" in md


def test_guard_does_not_open_real_home_kanban_db(tmp_path, monkeypatch):
    fake_home = tmp_path / "home"
    fake_home.mkdir()
    monkeypatch.setenv("HOME", str(fake_home))
    realish = fake_home / ".hermes" / "kanban.db"
    assert not realish.exists()
    db = tmp_path / "explicit.db"
    make_db(db)
    metrics = kanban_metrics.compute_metrics(db, generated_at=NOW, immutable=True)
    assert metrics["db_path"] == str(db)
    assert not realish.exists()


def test_cli_writes_json_and_markdown_outputs(tmp_path):
    db = tmp_path / "kanban.db"
    make_db(db)
    conn = sqlite3.connect(db)
    add_task(conn, "t_root", "Implement example", "backend-eng", "done", 100, started_at=150, completed_at=250)
    add_run(conn, "t_root", "done", 150, 250, "completed")
    conn.commit(); conn.close()
    json_out = tmp_path / "metrics.json"
    md_out = tmp_path / "metrics.md"
    args = kanban_metrics.build_arg_parser().parse_args([
        "--db", str(db), "--window", "1d", "--format", "both", "--json-out", str(json_out), "--markdown-out", str(md_out), "--generated-at", str(NOW), "--immutable"
    ])
    assert kanban_metrics.run_metrics_command(args) == 0
    assert json.loads(json_out.read_text())["tasks"][0]["task_id"] == "t_root"
    assert "Active by assignee" in md_out.read_text()
