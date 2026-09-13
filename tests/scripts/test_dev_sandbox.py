from __future__ import annotations

import json
import os
import shutil
import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "dev-sandbox.sh"


def test_dev_sandbox_drops_live_kanban_pins(tmp_path):
    bash = shutil.which("bash")
    if bash is None:
        pytest.skip("dev-sandbox.sh requires bash")

    live_root = tmp_path / "live"
    live_root.mkdir()
    live_db = live_root / "kanban.db"
    with sqlite3.connect(live_db) as conn:
        conn.execute("CREATE TABLE sentinel (value TEXT)")
        conn.execute("INSERT INTO sentinel VALUES ('untouched')")

    pin_names = (
        "HERMES_KANBAN_DB",
        "HERMES_KANBAN_BOARD",
        "HERMES_KANBAN_HOME",
        "HERMES_KANBAN_WORKSPACES_ROOT",
        "HERMES_KANBAN_ATTACHMENTS_ROOT",
        "HERMES_KANBAN_LOGS_ROOT",
        "HERMES_KANBAN_TASK",
        "HERMES_KANBAN_WORKSPACE",
        "HERMES_KANBAN_BRANCH",
        "HERMES_KANBAN_RUN_ID",
        "HERMES_KANBAN_CLAIM_LOCK",
        "HERMES_KANBAN_GOAL_MODE",
        "HERMES_KANBAN_GOAL_MAX_TURNS",
        "HERMES_KANBAN_FUTURE_PIN",
    )
    env = os.environ.copy()
    env.update({name: "inherited-live-value" for name in pin_names})
    env["HERMES_KANBAN_DB"] = str(live_db)
    env["HERMES_KANBAN_HOME"] = str(live_root)
    env["PYTHONPATH"] = str(ROOT)

    probe = """
import json
import os
from hermes_cli import kanban_db as kb
from hermes_cli.kanban_db_connect import connect

db = kb.kanban_db_path()
with connect() as conn:
    task = kb.create_task(conn, title="sandbox-only")
print(json.dumps({
    "db": str(db),
    "home": os.environ["HERMES_HOME"],
    "pins": {name: os.environ.get(name) for name in %r},
    "task": task,
}))
""" % (pin_names,)

    result = subprocess.run(
        [bash, str(SCRIPT), Path(sys.executable).as_posix(), "-c", probe],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        timeout=30,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    assert all(value is None for value in report["pins"].values())
    assert Path(report["db"]) == Path(report["home"]) / "kanban.db"
    assert str(live_db) not in result.stdout
    assert str(live_db) not in result.stderr
    with sqlite3.connect(live_db) as conn:
        assert conn.execute("SELECT value FROM sentinel").fetchone() == ("untouched",)
        assert conn.execute(
            "SELECT count(*) FROM sqlite_master WHERE type = 'table' AND name = 'tasks'"
        ).fetchone() == (0,)