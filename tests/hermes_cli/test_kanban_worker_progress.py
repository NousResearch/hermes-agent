from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

from hermes_cli import kanban as kc
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


def _make_running_task(conn, tmp_path, *, now: int, title: str = "Quick MOVE extract"):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    tid = kb.create_task(
        conn,
        title=title,
        body="D:/QUICK/MOVE public-video extraction with local transcription fallback",
        assignee="reels",
        workspace_kind="dir",
        workspace_path=str(workspace),
    )
    task = kb.claim_task(conn, tid)
    assert task is not None
    run_id = kb.get_task(conn, tid).current_run_id
    assert run_id is not None
    conn.execute(
        "UPDATE tasks SET last_heartbeat_at = ?, claim_expires = ?, session_id = ? WHERE id = ?",
        (now - 45, now + 600, "sess_quick_move", tid),
    )
    conn.execute(
        "UPDATE task_runs SET last_heartbeat_at = ?, claim_expires = ?, metadata = ? WHERE id = ?",
        (
            now - 45,
            now + 600,
            json.dumps({"prior_evidence": "Quick MOVE transcript extraction submitted"}),
            run_id,
        ),
    )
    conn.commit()
    return tid, workspace


def _touch(path: Path, *, mtime: int, text: str = "evidence") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    os.utime(path, (mtime, mtime))
    return path


def test_quick_move_fresh_heartbeat_without_visible_progress_is_backend_wait(
    kanban_home, tmp_path
):
    from hermes_cli import kanban_progress as progress

    now = int(time.time())
    with kb.connect() as conn:
        tid, workspace = _make_running_task(conn, tmp_path, now=now)
        _touch(
            workspace / "transcript.partial.txt",
            mtime=now - 240,
            text="partial transcript from Quick MOVE video",
        )
        log_path = kb.worker_log_path(tid)
        _touch(
            log_path,
            mtime=now - 240,
            text="Quick MOVE extraction started\nwaiting for backend response\n",
        )

        result = progress.classify_task_progress(
            conn,
            tid,
            board="default",
            now=now,
            progress_grace_seconds=120,
        )

    assert result.state == "backend_wait"
    assert "comment or keep monitoring" in result.recommended_action.lower()
    assert "kill" not in result.recommended_action.lower()
    assert result.evidence["heartbeat_age_seconds"] == 45
    assert result.evidence["worker_log"]["encoding"] == "utf-8"
    assert result.evidence["workspace"]["latest_mtime_age_seconds"] >= 240


def test_progress_cli_emits_json_for_subcommand_board_flag(kanban_home, tmp_path):
    now = int(time.time())
    with kb.connect() as conn:
        tid, workspace = _make_running_task(conn, tmp_path, now=now)
        _touch(workspace / "artifact.txt", mtime=now - 240)
        _touch(
            kb.worker_log_path(tid),
            mtime=now - 240,
            text="Quick MOVE extraction started\nwaiting for backend response\n",
        )

    out = kc.run_slash(f"progress {tid} --board default --json")

    assert not out.startswith("⚠ /kanban usage error")
    payload = json.loads(out)
    assert payload["task_id"] == tid
    assert payload["state"] == "backend_wait"
    assert payload["recommended_action"].lower().startswith("comment or keep monitoring")


def test_idle_external_media_subprocess_is_distinguished_from_backend_wait(
    kanban_home, tmp_path
):
    pytest.importorskip("psutil")
    from hermes_cli import kanban_progress as progress

    now = int(time.time())
    with kb.connect() as conn:
        tid, workspace = _make_running_task(conn, tmp_path, now=now)
        final_output = workspace / "clip.mp4"
        part_output = Path(str(final_output) + ".part")
        _touch(part_output, mtime=now - 300, text="partial media bytes")
        _touch(
            kb.worker_log_path(tid),
            mtime=now - 300,
            text="Quick MOVE extraction started\nyt-dlp downloading clip\n",
        )
        proc = subprocess.Popen(
            [
                sys.executable,
                "-c",
                "import time; time.sleep(60)",
                "yt-dlp",
                "-o",
                str(final_output),
            ],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        try:
            conn.execute(
                "UPDATE tasks SET worker_pid = ? WHERE id = ?",
                (proc.pid, tid),
            )
            conn.execute(
                "UPDATE task_runs SET worker_pid = ? WHERE task_id = ?",
                (proc.pid, tid),
            )
            conn.commit()

            result = progress.classify_task_progress(
                conn,
                tid,
                board="default",
                now=now,
                progress_grace_seconds=120,
                external_idle_seconds=60,
            )

            assert result.state == "external_subprocess_stuck"
            assert "automatically" in result.recommended_action.lower()
            assert "kill" not in result.recommended_action.lower()
            media = result.evidence["process"]["external_media_processes"][0]
            assert media["command_name"] == "yt-dlp"
            assert media["output_path"] == str(part_output)
            assert media["is_idle"] is True
            assert proc.poll() is None
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=3)


def test_claim_stale_when_heartbeat_exceeds_existing_reclaim_window(
    kanban_home, tmp_path
):
    from hermes_cli import kanban_progress as progress

    now = int(time.time())
    with kb.connect() as conn:
        tid, _workspace = _make_running_task(conn, tmp_path, now=now, title="stale")
        stale_hb = now - kb.DEFAULT_CLAIM_HEARTBEAT_MAX_STALE_SECONDS - 1
        conn.execute(
            "UPDATE tasks SET last_heartbeat_at = ?, claim_expires = ? WHERE id = ?",
            (stale_hb, now - 1, tid),
        )
        conn.execute(
            "UPDATE task_runs SET last_heartbeat_at = ?, claim_expires = ? WHERE task_id = ?",
            (stale_hb, now - 1, tid),
        )
        conn.commit()

        result = progress.classify_task_progress(conn, tid, board="default", now=now)

    assert result.state == "claim_stale"
    assert "dispatch" in result.recommended_action.lower()


def test_progress_states_all_have_operator_actions():
    from hermes_cli import kanban_progress as progress

    assert progress.PROGRESS_STATES == {
        "productive",
        "backend_wait",
        "external_subprocess_stuck",
        "no_tool_spin",
        "claim_stale",
        "unknown",
    }
    assert set(progress.RECOMMENDED_ACTIONS) == progress.PROGRESS_STATES
    assert "kill" not in progress.RECOMMENDED_ACTIONS["backend_wait"].lower()
    assert "comment or keep monitoring" in progress.RECOMMENDED_ACTIONS["backend_wait"].lower()
