from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from agent.delegation_context import DELEGATED_CHILD_ENV_MARKER
from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def fenced_multiboard(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, str]:
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("HERMES_HOME", str(home))
    for key in (
        "HERMES_KANBAN_DB",
        "HERMES_KANBAN_BOARD",
        "HERMES_KANBAN_WORKSPACES_ROOT",
        "HERMES_DELEGATED_CHILD_CONTEXT",
    ):
        monkeypatch.delenv(key, raising=False)

    kb.init_db(board="alpha")
    kb.init_db(board="beta")
    with kbc.connect_closing(board="alpha") as conn:
        alpha_ready = kb.create_task(conn, title="alpha ready", assignee="default")
        conn.execute("UPDATE tasks SET status = 'ready' WHERE id = ?", (alpha_ready,))
        conn.commit()
    with kbc.connect_closing(board="beta") as conn:
        beta_done_1 = kb.create_task(conn, title="beta done", assignee="default")
        beta_done_2 = kb.create_task(conn, title="beta done 2", assignee="default")
        conn.execute(
            "UPDATE tasks SET status = 'done' WHERE id IN (?, ?)",
            (beta_done_1, beta_done_2),
        )
        conn.commit()

    alpha_db = kb.kanban_db_path(board="alpha")
    marker = str(kb.kanban_home())
    monkeypatch.setenv("HERMES_KANBAN_DB", str(alpha_db))
    monkeypatch.setenv("HERMES_KANBAN_BOARD", "alpha")
    monkeypatch.setenv(DELEGATED_CHILD_ENV_MARKER, marker)
    return {"home": str(home), "alpha_db": str(alpha_db), "alpha_ready": alpha_ready}


def _run_kanban(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "hermes_cli.main", "kanban", *args],
        cwd=ROOT,
        env=dict(os.environ),
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=45,
    )


def test_delegated_child_can_list_fenced_board_read_only(fenced_multiboard: dict[str, str]) -> None:
    proc = _run_kanban("list", "--json")

    assert proc.returncode == 0, (proc.stdout, proc.stderr)
    rows = json.loads(proc.stdout)
    assert [row["id"] for row in rows] == [fenced_multiboard["alpha_ready"]]
    assert rows[0]["status"] == "ready"


def test_delegated_child_dry_run_dispatch_is_read_only(fenced_multiboard: dict[str, str]) -> None:
    proc = _run_kanban("dispatch", "--dry-run", "--json")

    assert proc.returncode == 0, (proc.stdout, proc.stderr)
    payload = json.loads(proc.stdout)
    assert payload["spawned"] == [
        {"task_id": fenced_multiboard["alpha_ready"], "assignee": "default", "workspace": ""}
    ]
    with kbc.connect_closing(db_path=Path(fenced_multiboard["alpha_db"])) as conn:
        task = kb.get_task(conn, str(fenced_multiboard["alpha_ready"]))
    assert task is not None
    assert task.status == "ready"
    assert task.current_run_id is None


def test_boards_list_counts_each_board_despite_pinned_worker_env(
    fenced_multiboard: dict[str, str],
) -> None:
    proc = _run_kanban("boards", "list", "--json")

    assert proc.returncode == 0, (proc.stdout, proc.stderr)
    boards = {row["slug"]: row for row in json.loads(proc.stdout)}
    assert boards["default"]["counts"] == {}
    assert boards["alpha"]["counts"] == {"ready": 1}
    assert boards["beta"]["counts"] == {"done": 2}
    assert boards["beta"]["db_path"].endswith("/kanban/boards/beta/kanban.db")
