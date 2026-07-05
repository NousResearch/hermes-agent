from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb


POLISH_TEXT = "Za\u017c\u00f3\u0142\u0107 g\u0119\u015bl\u0105 ja\u017a\u0144"


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb._INITIALIZED_PATHS.clear()
    kb.init_db()
    return home


def _touch(path: Path, *, mtime: int, content: bytes = b"partial") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    os.utime(path, (mtime, mtime))


def test_quick_move_reliability_chain_canary(
    kanban_home, tmp_path, monkeypatch, all_assignees_spawnable
):
    pytest.importorskip("psutil")
    from hermes_cli import kanban_progress as progress

    now = int(time.time())

    def profile_config(assignee: str):
        if assignee == "zero-specialist":
            return {
                "goals": {"max_turns": 0},
                "agent": {"max_turns": 90},
                "delegation": {"max_iterations": 90},
            }
        return {
            "goals": {"max_turns": 12},
            "agent": {"max_turns": 12},
            "delegation": {"max_iterations": 12},
        }

    monkeypatch.setattr(kb, "_load_worker_profile_config", profile_config)

    spawned: list[str] = []

    def spawn_fn(task, workspace):
        spawned.append(task.id)
        return 4242

    with kb.connect() as conn:
        zero_a = kb.create_task(
            conn,
            title="extract public video A",
            body="synthetic public video extraction lane",
            assignee="zero-specialist",
            goal_mode=True,
        )
        zero_b = kb.create_task(
            conn,
            title="extract public video B",
            body="synthetic public video extraction lane",
            assignee="zero-specialist",
            goal_mode=True,
        )
        healthy_extract = kb.create_task(
            conn,
            title="extract public video C",
            body="synthetic public video extraction lane",
            assignee="healthy-c",
            goal_mode=True,
        )
        synthesis = kb.create_task(
            conn,
            title="synthesize extracted findings",
            assignee="synth",
            parents=[zero_a, zero_b, healthy_extract],
        )
        review = kb.create_task(
            conn,
            title="review synthesis",
            assignee="reviewer",
            parents=[synthesis],
        )
        archive = kb.create_task(
            conn,
            title="archive MOVE batch",
            assignee="archiver",
            parents=[review],
        )

        dispatch = kb.dispatch_once(conn, spawn_fn=spawn_fn)

        assert zero_a in dispatch.spawn_blocked_zero_budget
        assert zero_b in dispatch.spawn_blocked_zero_budget
        assert zero_a not in spawned
        assert zero_b not in spawned
        assert healthy_extract in spawned
        assert kb.get_task(conn, zero_a).status == "blocked"
        assert kb.get_task(conn, zero_b).status == "blocked"
        assert kb.get_task(conn, synthesis).status == "todo"
        assert kb.get_task(conn, review).status == "todo"
        assert kb.get_task(conn, archive).status == "todo"

        candidates, dry_actions = kb.repair_zero_budget_failures(conn, dry_run=True)
        assert {c.task_id for c in candidates} == {zero_a, zero_b}
        assert {a.task_id for a in dry_actions} == {zero_a, zero_b}
        assert all(a.mutated is False for a in dry_actions)

        _candidates, reroute_actions = kb.repair_zero_budget_failures(
            conn,
            dry_run=False,
            reroute_profiles=["healthy-a", "healthy-b"],
            author="canary",
        )
        assert [a.reroute_profile for a in reroute_actions] == ["healthy-a", "healthy-b"]
        replacements = [a.replacement_task_id for a in reroute_actions]
        assert all(replacements)
        assert kb.get_task(conn, synthesis).status == "todo"
        assert set(kb.parent_ids(conn, synthesis)) == {
            zero_a,
            zero_b,
            healthy_extract,
            *replacements,
        }

        busy = kb.create_task(conn, title="busy profile", assignee="healthy-a")
        assert kb.claim_task(conn, busy) is not None
        capped = kb.create_task(conn, title="capacity capped", assignee="healthy-a")

        cap = kb.dispatch_once(conn, dry_run=True, max_in_progress_per_profile=1)
        by_kind = {item["kind"]: item for item in cap.ready_explanations}
        assert (capped, "healthy-a", 1) in cap.skipped_per_profile_capped
        healthy_a_cap = by_kind["capacity_limited_per_profile"]["profiles"]["healthy-a"]
        assert healthy_a_cap["limit"] == 1
        assert healthy_a_cap["running"] == 1
        assert capped in healthy_a_cap["task_ids"]

        workspace = tmp_path / "quick-move-worker"
        workspace.mkdir()
        progress_task = kb.create_task(
            conn,
            title="media transcript extraction",
            body="D:/QUICK/MOVE synthetic transcript extraction",
            assignee="reels",
            workspace_kind="dir",
            workspace_path=str(workspace),
        )
        task = kb.claim_task(conn, progress_task)
        assert task is not None
        run_id = kb.get_task(conn, progress_task).current_run_id
        assert run_id is not None

        final_output = workspace / "clip.mp4"
        part_output = Path(str(final_output) + ".part")
        _touch(part_output, mtime=now - 300, content=b"partial media bytes")
        log_path = kb.worker_log_path(progress_task)
        _touch(log_path, mtime=now - 300, content=POLISH_TEXT.encode("cp1250"))

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
                "UPDATE tasks SET worker_pid = ?, last_heartbeat_at = ?, claim_expires = ?, session_id = ? WHERE id = ?",
                (proc.pid, now - 45, now + 600, "canary_session", progress_task),
            )
            conn.execute(
                "UPDATE task_runs SET worker_pid = ?, last_heartbeat_at = ?, claim_expires = ? WHERE id = ?",
                (proc.pid, now - 45, now + 600, run_id),
            )
            conn.commit()

            progress_result = progress.classify_task_progress(
                conn,
                progress_task,
                board="default",
                now=now,
                progress_grace_seconds=120,
                external_idle_seconds=60,
            )

            assert progress_result.state == "external_subprocess_stuck"
            assert "kill" not in progress_result.recommended_action.lower()
            media = progress_result.evidence["process"]["external_media_processes"][0]
            assert media["command_name"] == "yt-dlp"
            assert media["output_path"] == str(part_output)
            assert media["is_idle"] is True

            worker_log = progress_result.evidence["worker_log"]
            assert worker_log["encoding"].lower() == "cp1250"
            assert worker_log["used_fallback"] is True
            assert POLISH_TEXT in worker_log["tail"]
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=3)

    intake = tmp_path / "MOVE"
    archive_dir = tmp_path / "ARCHIVE" / "batch-001"
    intake.mkdir()
    archive_dir.mkdir(parents=True)
    leftover = intake / "video-a.mp4"
    leftover.write_text("raw", encoding="utf-8")
    (archive_dir / "video-a.mp4").write_text("archived", encoding="utf-8")
    (archive_dir / "README.md").write_text("source split\nreview gate\n", encoding="utf-8")

    blocked_archive = kb.verify_archive_move_contract(
        source_dir=intake,
        archive_dir=archive_dir,
        expected_items=["video-a.mp4"],
        require_empty_source=True,
    )
    assert blocked_archive["ok"] is False
    assert {issue["code"] for issue in blocked_archive["issues"]} == {"source_not_empty"}

    leftover.unlink()
    clean_archive = kb.verify_archive_move_contract(
        source_dir=intake,
        archive_dir=archive_dir,
        expected_items=["video-a.mp4"],
        require_empty_source=True,
    )
    assert clean_archive["ok"] is True
