"""Timeout grace, empty-timeout brake, and progress-stall reclaim.

Live Council 2026-09-20/21: DB 0.1 / 6.1 were SIGTERM'd after the artifact
existed; 5.5-B heartbeated for two empty hours then blocked on the second
timeout. Heartbeat-only is not progress.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_dispatch as kbd


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_KANBAN_CRASH_GRACE_SECONDS", "0")
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


def _overrun(conn, *, elapsed: int, max_runtime: int = 1, workspace: Path | None = None) -> str:
    tid = kb.create_task(
        conn, title="job", assignee="worker",
        max_runtime_seconds=max_runtime,
        workspace_kind="scratch",
        workspace_path=str(workspace) if workspace else None,
    )
    kb.claim_task(conn, tid)
    kbd._set_worker_pid(conn, tid, os.getpid())
    old = int(time.time()) - elapsed
    with kb.write_txn(conn):
        conn.execute(
            "UPDATE tasks SET started_at = ?, workspace_kind = 'scratch', workspace_path = ? WHERE id = ?",
            (old, str(workspace) if workspace else None, tid),
        )
        conn.execute(
            "UPDATE task_runs SET started_at = ? "
            "WHERE id = (SELECT current_run_id FROM tasks WHERE id = ?)",
            (old, tid),
        )
    return tid


def test_timeout_grace_skips_kill_when_workspace_just_wrote(kanban_home, monkeypatch):
    """A file written inside the grace window after cap must not SIGTERM yet."""
    original_alive = kb._pid_alive
    kb._pid_alive = lambda pid: False
    ws = kanban_home / "ws"
    ws.mkdir()
    (ws / "probe.md").write_text("done", encoding="utf-8")
    try:
        conn = kbc.connect()
        try:
            tid = _overrun(conn, elapsed=kbd.TIMEOUT_PROGRESS_GRACE_SECONDS - 10, workspace=ws)
            killed = []
            assert kbd.enforce_max_runtime(conn, signal_fn=lambda pid, sig: killed.append((pid, sig))) == []
            assert killed == []
            assert kb.get_task(conn, tid).status == "running"
        finally:
            conn.close()
    finally:
        kb._pid_alive = original_alive


def test_timeout_grace_expires_even_with_fresh_write(kanban_home, monkeypatch):
    original_alive = kb._pid_alive
    kb._pid_alive = lambda pid: False
    ws = kanban_home / "ws"
    ws.mkdir()
    (ws / "probe.md").write_text("done", encoding="utf-8")
    try:
        conn = kbc.connect()
        try:
            tid = _overrun(conn, elapsed=kbd.TIMEOUT_PROGRESS_GRACE_SECONDS + 30, workspace=ws)
            timed = kbd.enforce_max_runtime(conn, signal_fn=lambda *_: None)
            assert tid in timed
            # Has progress, so this is a retryable timeout, not an empty-run block.
            assert kb.get_task(conn, tid).status == "ready"
        finally:
            conn.close()
    finally:
        kb._pid_alive = original_alive


def test_empty_timeout_blocks_instead_of_retry(kanban_home):
    """No comments and only bootstrap files -> first timeout trips the breaker."""
    original_alive = kb._pid_alive
    kb._pid_alive = lambda pid: False
    ws = kanban_home / "ws"
    ws.mkdir()
    (ws / "tsconfig.json").write_text("{}", encoding="utf-8")
    try:
        conn = kbc.connect()
        try:
            tid = _overrun(conn, elapsed=90, workspace=ws)
            timed = kbd.enforce_max_runtime(conn, signal_fn=lambda *_: None)
            assert tid in timed
            task = kb.get_task(conn, tid)
            assert task.status == "blocked"
            events = kb.list_events(conn, tid)
            assert any(e.kind == "timed_out" for e in events)
            assert any(e.kind == "gave_up" for e in events)
            to_event = next(e for e in events if e.kind == "timed_out")
            assert to_event.payload.get("empty_run") is True
        finally:
            conn.close()
    finally:
        kb._pid_alive = original_alive


def test_tsconfig_is_not_progress(kanban_home):
    ws = kanban_home / "ws"
    ws.mkdir()
    (ws / "tsconfig.json").write_text("{}", encoding="utf-8")
    assert kbd.newest_workspace_progress_mtime(str(ws)) is None
    (ws / "notes.md").write_text("x", encoding="utf-8")
    assert kbd.newest_workspace_progress_mtime(str(ws)) is not None


def test_progress_stall_blocks_heartbeat_only_worker(kanban_home, monkeypatch):
    monkeypatch.setattr(kbd, "PROGRESS_STALL_SECONDS", 1)
    original_alive = kb._pid_alive
    kb._pid_alive = lambda pid: False
    ws = kanban_home / "ws"
    ws.mkdir()
    try:
        conn = kbc.connect()
        try:
            tid = _overrun(conn, elapsed=90, max_runtime=3600, workspace=ws)
            stalled = kbd.detect_progress_stall(conn, signal_fn=lambda *_: None)
            assert tid in stalled
            assert kb.get_task(conn, tid).status == "blocked"
            assert any(e.kind == "progress_stalled" for e in kb.list_events(conn, tid))
        finally:
            conn.close()
    finally:
        kb._pid_alive = original_alive


def test_empty_timeout_does_not_block_review_runs(kanban_home, monkeypatch):
    """Review hops are often comment-late; empty-timeout must retry in review."""
    original_alive = kb._pid_alive
    kb._pid_alive = lambda pid: False
    try:
        conn = kbc.connect()
        try:
            tid = kb.create_task(conn, title="rev", assignee="builder", max_runtime_seconds=1)
            impl = kb.claim_task(conn, tid)
            assert kb.request_review(
                conn, tid, summary="ready", reviewer="reviewer",
                expected_run_id=impl.current_run_id,
            )
            review = kb.claim_review_task(conn, tid)
            assert review is not None
            kbd._set_worker_pid(conn, tid, os.getpid())
            old = int(time.time()) - 90
            with kb.write_txn(conn):
                conn.execute("UPDATE tasks SET started_at = ? WHERE id = ?", (old, tid))
                conn.execute(
                    "UPDATE task_runs SET started_at = ? WHERE id = ?",
                    (old, review.current_run_id),
                )
            assert tid in kbd.enforce_max_runtime(conn, signal_fn=lambda *_: None)
            assert kb.get_task(conn, tid).status == "review"
        finally:
            conn.close()
    finally:
        kb._pid_alive = original_alive


def test_progress_stall_skips_when_comment_exists(kanban_home, monkeypatch):
    monkeypatch.setattr(kbd, "PROGRESS_STALL_SECONDS", 1)
    original_alive = kb._pid_alive
    kb._pid_alive = lambda pid: False
    ws = kanban_home / "ws"
    ws.mkdir()
    try:
        conn = kbc.connect()
        try:
            tid = _overrun(conn, elapsed=90, max_runtime=3600, workspace=ws)
            kb.add_comment(conn, tid, "worker", "still going")
            assert kbd.detect_progress_stall(conn, signal_fn=lambda *_: None) == []
            assert kb.get_task(conn, tid).status == "running"
        finally:
            conn.close()
    finally:
        kb._pid_alive = original_alive
