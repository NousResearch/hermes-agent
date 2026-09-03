"""Blocked→unblocked workers must resume the prior worker session (#75830).

Workers stamp ``task_runs.metadata.worker_session_id`` on complete/block.
``_default_spawn`` must pass ``--resume <sid>`` so a fresh process continues
the prior conversation instead of starting blank.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb._INITIALIZED_PATHS.clear()
    kb.init_db()
    return home


def _make_task(task_id: str = "t_resume_blocked", *, assignee: str = "default") -> kb.Task:
    return kb.Task(
        id=task_id,
        title="resume blocked",
        body=None,
        assignee=assignee,
        status="running",
        priority=0,
        created_by="test",
        created_at=1,
        started_at=None,
        completed_at=None,
        workspace_kind="dir",
        workspace_path=None,
        claim_lock="lock",
        claim_expires=None,
        tenant=None,
        current_run_id=7,
    )


def _capture_spawn(monkeypatch, tmp_path, task: kb.Task, *, board=None):
    monkeypatch.setattr(kb, "_resolve_hermes_argv", lambda: ["hermes"])
    monkeypatch.setattr(kb, "_retag_legacy_worker_sessions", lambda _root: None)
    monkeypatch.setattr(kb, "worker_logs_dir", lambda board=None: tmp_path / "logs")

    captured: dict = {}

    class FakeProc:
        pid = 9001

    def fake_popen(cmd, *args, **kwargs):
        captured["cmd"] = list(cmd)
        captured["env"] = dict(kwargs.get("env") or {})
        return FakeProc()

    monkeypatch.setattr(subprocess, "Popen", fake_popen)

    workspace = tmp_path / "workspace"
    workspace.mkdir(exist_ok=True)
    pid = kb._default_spawn(task, str(workspace), board=board)
    captured["pid"] = pid
    return captured


def test_resume_session_id_none_when_no_prior_runs(kanban_home):
    with kb.connect() as conn:
        tid = kb.create_task(conn, title="first try", assignee="worker")
    assert kb._resume_session_id_for_task(tid) is None


def test_resume_session_id_from_blocked_run_metadata(kanban_home):
    with kb.connect() as conn:
        tid = kb.create_task(conn, title="needs input", assignee="worker")
        assert kb.claim_task(conn, tid, claimer="worker:1") is not None
        assert kb.complete_task(
            conn,
            tid,
            result="blocked for decision",
            metadata={"worker_session_id": "20260801_113226_8fb971"},
        )
    assert kb._resume_session_id_for_task(tid) == "20260801_113226_8fb971"


def test_resume_session_id_prefers_newest_ended_run(kanban_home):
    with kb.connect() as conn:
        tid = kb.create_task(conn, title="multi run", assignee="worker")
        assert kb.claim_task(conn, tid, claimer="worker:1") is not None
        assert kb.complete_task(
            conn,
            tid,
            result="first attempt",
            metadata={"worker_session_id": "session_old"},
        )
        # Re-open as ready for a second claim (simulates unblock → re-dispatch).
        conn.execute(
            "UPDATE tasks SET status = 'ready', completed_at = NULL, result = NULL "
            "WHERE id = ?",
            (tid,),
        )
        conn.commit()
        assert kb.claim_task(conn, tid, claimer="worker:2") is not None
        assert kb.complete_task(
            conn,
            tid,
            result="second attempt",
            metadata={"worker_session_id": "session_new"},
        )
    assert kb._resume_session_id_for_task(tid) == "session_new"


def test_resume_session_id_skips_runs_without_stamp(kanban_home):
    with kb.connect() as conn:
        tid = kb.create_task(conn, title="no stamp", assignee="worker")
        assert kb.claim_task(conn, tid, claimer="worker:1") is not None
        assert kb.complete_task(conn, tid, result="ok", metadata={"files": 2})
    assert kb._resume_session_id_for_task(tid) is None


def test_resume_session_id_fail_open_on_db_error(monkeypatch, kanban_home):
    def boom(*_a, **_k):
        raise RuntimeError("db down")

    monkeypatch.setattr(kb, "connect", boom)
    assert kb._resume_session_id_for_task("t_any") is None


def test_default_spawn_passes_resume_when_prior_session_exists(kanban_home, monkeypatch, tmp_path):
    with kb.connect() as conn:
        tid = kb.create_task(conn, title="unblock me", assignee="default")
        assert kb.claim_task(conn, tid, claimer="worker:1") is not None
        assert kb.complete_task(
            conn,
            tid,
            result="need design decision",
            metadata={"worker_session_id": "20260801_113226_8fb971"},
        )

    (kanban_home / "profiles" / "default").mkdir(parents=True, exist_ok=True)
    captured = _capture_spawn(monkeypatch, tmp_path, _make_task(tid, assignee="default"))

    assert captured["pid"] == 9001
    cmd = captured["cmd"]
    assert "--resume" in cmd
    assert cmd[cmd.index("--resume") + 1] == "20260801_113226_8fb971"
    # Resume is a top-level flag, before the chat subcommand.
    assert cmd.index("--resume") < cmd.index("chat")
    assert "chat" in cmd
    assert "-q" in cmd


def test_default_spawn_omits_resume_on_first_dispatch(kanban_home, monkeypatch, tmp_path):
    with kb.connect() as conn:
        tid = kb.create_task(conn, title="fresh", assignee="default")

    (kanban_home / "profiles" / "default").mkdir(parents=True, exist_ok=True)
    captured = _capture_spawn(monkeypatch, tmp_path, _make_task(tid, assignee="default"))

    assert "--resume" not in captured["cmd"]
    assert "chat" in captured["cmd"]


def test_default_spawn_omits_resume_when_lookup_returns_none(
    kanban_home, monkeypatch, tmp_path
):
    (kanban_home / "profiles" / "default").mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(kb, "_resume_session_id_for_task", lambda *_a, **_k: None)
    captured = _capture_spawn(
        monkeypatch, tmp_path, _make_task("t_no_resume", assignee="default")
    )
    assert "--resume" not in captured["cmd"]
