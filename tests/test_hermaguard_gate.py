import importlib.util
import sqlite3
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "hermaguard-gate.py"


def _module():
    scripts_dir = str(SCRIPT_PATH.parent)
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    spec = importlib.util.spec_from_file_location("hermaguard_gate", SCRIPT_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _create_current_schema_db(path: Path, timestamp: int) -> None:
    with sqlite3.connect(path) as connection:
        connection.execute(
            """
            CREATE TABLE tasks (
                id TEXT PRIMARY KEY,
                title TEXT,
                body TEXT,
                assignee TEXT,
                status TEXT,
                tier TEXT,
                pipeline_stage TEXT,
                created_at INTEGER,
                started_at INTEGER,
                completed_at INTEGER,
                done_at INTEGER,
                archived_at INTEGER
            )
            """
        )
        connection.execute(
            """
            INSERT INTO tasks (
                id, title, body, assignee, status, tier, pipeline_stage,
                created_at, started_at, completed_at, done_at, archived_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "task-1",
                "Backend API change",
                "",
                "octacon",
                "review",
                "full",
                "review",
                timestamp - 60,
                timestamp,
                None,
                None,
                None,
            ),
        )


def test_scan_boards_supports_current_schema_without_updated_at(tmp_path):
    module = _module()
    observed_now = datetime(2026, 7, 27, 2, 0, tzinfo=timezone(timedelta(hours=1)))
    database = tmp_path / "kanban.db"
    _create_current_schema_db(database, int(observed_now.timestamp()))
    setattr(module, "BOARDS", {"test": database})
    setattr(module, "now", observed_now)

    tasks = module._scan_boards()

    assert [task["id"] for task in tasks] == ["task-1"]
    assert tasks[0]["updated_at"] == int(observed_now.timestamp())
