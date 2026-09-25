"""``hermes kanban boards pause|resume``: a per-board dispatch switch.

Pausing must stop new spawns on the next tick without touching running
workers (no kill, no reclaim, no retry budget spent), must be read live
(no gateway restart), and must be scoped to one board.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_dispatch as kbd

_WORKTREE = Path(__file__).resolve().parents[2]


@pytest.fixture
def home(tmp_path, monkeypatch):
    home = tmp_path / "hermes_home"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    for var in ("HERMES_KANBAN_DB", "HERMES_KANBAN_WORKSPACES_ROOT", "HERMES_KANBAN_HOME", "HERMES_KANBAN_BOARD"):
        monkeypatch.delenv(var, raising=False)
    kb._INITIALIZED_PATHS.clear()
    monkeypatch.setattr(kbd, "_memory_pressure_level", lambda sample=None: "unknown")
    monkeypatch.setattr(kbd, "_profile_exists_fn", lambda: None)  # trust the assignee
    return home


def _tick(board: str, spawns: list) -> kbd.DispatchResult:
    def spawn(task, workspace, board=None):
        spawns.append(task.id)
        return os.getpid()  # a live pid: the running task must survive the next tick
    with kbc.connect_closing(board=board) as conn:
        return kbd.dispatch_once(conn, spawn_fn=spawn, board=board)


def test_pause_stops_spawning_and_leaves_running_workers_alone(home):
    kb.create_board("xg")
    kb.create_board("other")
    with kbc.connect_closing(board="xg") as conn:
        running = kb.create_task(conn, title="running", assignee="dev")
    with kbc.connect_closing(board="other") as conn:
        elsewhere = kb.create_task(conn, title="elsewhere", assignee="dev")

    spawns: list = []
    _tick("xg", spawns)
    assert spawns == [running]

    kb.write_board_metadata("xg", paused=True, paused_reason="provider window exhausted")
    meta = kb.read_board_metadata("xg")
    assert meta["paused"] is True and meta["paused_reason"] == "provider window exhausted"
    with kbc.connect_closing(board="xg") as conn:
        queued = kb.create_task(conn, title="queued", assignee="dev")

    spawns.clear()
    result = _tick("xg", spawns)
    assert result.paused is True
    assert spawns == []
    assert "paused=1" in kbd.describe_suppression([result])
    with kbc.connect_closing(board="xg") as conn:
        assert kb.get_task(conn, running).status == "running"
        assert kb.get_task(conn, queued).status == "ready"

    # Scoped to one board.
    other: list = []
    assert _tick("other", other).paused is False
    assert other == [elsewhere]

    kb.write_board_metadata("xg", paused=False)
    assert kb.read_board_metadata("xg")["paused_reason"] == ""
    spawns.clear()
    assert _tick("xg", spawns).paused is False
    assert spawns == [queued]


def test_cli_pause_resume_round_trip(tmp_path):
    env = {**os.environ, "HERMES_HOME": str(tmp_path), "PYTHONPATH": str(_WORKTREE)}

    def cli(*args):
        return subprocess.run([sys.executable, "-m", "hermes_cli.main", "kanban", "boards", *args],
                              env=env, capture_output=True, text=True, cwd=str(_WORKTREE), timeout=60)

    assert cli("create", "xg").returncode == 0
    r = cli("pause", "xg", "--reason", "kill switch")
    assert r.returncode == 0, r.stderr
    board_json = tmp_path / "kanban" / "boards" / "xg" / "board.json"
    assert '"paused": true' in board_json.read_text()
    assert cli("resume", "xg").returncode == 0
    assert '"paused": false' in board_json.read_text()
    assert cli("pause", "missing").returncode != 0
