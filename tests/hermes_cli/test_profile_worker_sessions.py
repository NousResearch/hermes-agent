import json
import sqlite3

from hermes_cli import kanban_db
from hermes_cli.web_routers import profiles


def _board(path, rows):
    conn = sqlite3.connect(path)
    conn.executescript(kanban_db.SCHEMA_SQL)
    for row in rows:
        conn.execute(
            """INSERT INTO tasks (id, title, status, created_at)
               VALUES (?, ?, ?, ?)""",
            (row["task_id"], row["task_title"], row["task_status"], row["started_at"]),
        )
        conn.execute(
            """INSERT INTO task_runs
               (task_id, profile, status, started_at, ended_at, metadata)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (
                row["task_id"],
                row["profile"],
                row["run_status"],
                row["started_at"],
                row.get("ended_at"),
                json.dumps({"worker_session_id": row["session_id"]}),
            ),
        )
    conn.commit()
    conn.close()


def test_enrich_kanban_sessions_uses_task_run_lifecycle_truth(tmp_path):
    board = tmp_path / "board.db"
    _board(
        board,
        [
            {
                "task_id": "t_live",
                "task_title": "Live task",
                "task_status": "running",
                "profile": "clientops",
                "run_status": "running",
                "session_id": "live-session",
                "started_at": 100,
            },
            {
                "task_id": "t_done",
                "task_title": "Done task",
                "task_status": "done",
                "profile": "agentops",
                "run_status": "done",
                "session_id": "done-session",
                "started_at": 90,
                "ended_at": 120,
            },
        ],
    )
    sessions = [
        {
            "id": "live-session",
            "source": "kanban",
            "profile": "clientops",
            "ended_at": None,
            "is_active": False,
        },
        {
            "id": "done-session",
            "source": "kanban",
            "profile": "agentops",
            "ended_at": None,
            "is_active": True,
        },
        {
            "id": "ordinary",
            "source": "desktop",
            "profile": "default",
            "ended_at": None,
            "is_active": True,
        },
    ]

    profiles._enrich_kanban_session_rows(sessions, [("control-room", board)])

    assert sessions[0] == {
        "id": "live-session",
        "source": "kanban",
        "profile": "clientops",
        "ended_at": None,
        "is_active": True,
        "kanban_board": "control-room",
        "kanban_run_status": "running",
        "kanban_task_id": "t_live",
        "kanban_task_status": "running",
        "kanban_task_title": "Live task",
    }
    assert sessions[1] == {
        "id": "done-session",
        "source": "kanban",
        "profile": "agentops",
        "ended_at": 120,
        "is_active": False,
        "kanban_board": "control-room",
        "kanban_run_status": "done",
        "kanban_task_id": "t_done",
        "kanban_task_status": "done",
        "kanban_task_title": "Done task",
    }
    assert sessions[2]["is_active"] is True
    assert "kanban_task_id" not in sessions[2]


def test_enrich_kanban_sessions_dedupes_by_profile_and_session_using_newest_run(tmp_path):
    older = tmp_path / "older.db"
    newer = tmp_path / "newer.db"
    common = {
        "profile": "clientops",
        "run_status": "done",
        "session_id": "same-session",
        "task_status": "done",
    }
    _board(
        older,
        [
            {
                **common,
                "task_id": "t_old",
                "task_title": "Old duplicate",
                "started_at": 100,
                "ended_at": 110,
            }
        ],
    )
    _board(
        newer,
        [
            {
                **common,
                "task_id": "t_new",
                "task_title": "Newest duplicate",
                "started_at": 200,
                "ended_at": 220,
            }
        ],
    )
    sessions = [
        {"id": "same-session", "source": "kanban", "profile": "clientops", "ended_at": None, "is_active": True},
        {"id": "same-session", "source": "kanban", "profile": "agentops", "ended_at": None, "is_active": True},
    ]

    profiles._enrich_kanban_session_rows(sessions, [("older", older), ("newer", newer)])

    assert sessions[0]["kanban_task_id"] == "t_new"
    assert sessions[0]["kanban_board"] == "newer"
    assert sessions[0]["ended_at"] == 220
    assert "kanban_task_id" not in sessions[1]


def test_enrich_kanban_sessions_ignores_missing_profile_board_db(tmp_path):
    sessions = [
        {"id": "worker", "source": "kanban", "profile": "clientops", "ended_at": None, "is_active": True}
    ]

    profiles._enrich_kanban_session_rows(sessions, [("missing", tmp_path / "missing.db")])

    assert sessions == [
        {"id": "worker", "source": "kanban", "profile": "clientops", "ended_at": None, "is_active": True}
    ]
