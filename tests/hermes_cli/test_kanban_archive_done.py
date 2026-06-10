from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli.kanban_archive_done import run_archival


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_KANBAN_BOARD", "default")
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


class RecordingDiscordClient:
    def __init__(self, fail_threads: set[str] | None = None, missing_threads: set[str] | None = None):
        self.calls: list[str] = []
        self.metadata_calls: list[dict | None] = []
        self.fail_threads = fail_threads or set()
        self.missing_threads = missing_threads or set()

    def archive_and_lock(self, thread_id: str, *, projection_metadata: dict | None = None):
        from hermes_cli.kanban_archive_done import DiscordResult

        self.calls.append(str(thread_id))
        self.metadata_calls.append(projection_metadata)
        if str(thread_id) in self.fail_threads:
            return DiscordResult(False, "http_500", "boom")
        if str(thread_id) in self.missing_threads:
            return DiscordResult(True, "missing_thread", "404")
        return DiscordResult(True, "archived_locked")


def _done_task(conn, *, title: str, completed_at: int, with_discord_thread: str | None = None) -> str:
    tid = kb.create_task(conn, title=title, assignee="worker", initial_status="running")
    assert kb.complete_task(conn, tid, summary="done")
    conn.execute("UPDATE tasks SET completed_at = ? WHERE id = ?", (completed_at, tid))
    if with_discord_thread is not None:
        kb.add_notify_sub(
            conn,
            task_id=tid,
            platform="discord",
            chat_id="forum-1",
            thread_id=with_discord_thread,
        )
    return tid


def _running_task(conn, *, title: str, created_at: int, with_discord_thread: str | None = None) -> str:
    tid = kb.create_task(conn, title=title, assignee="worker", initial_status="running")
    conn.execute("UPDATE tasks SET created_at = ?, started_at = ? WHERE id = ?", (created_at, created_at, tid))
    if with_discord_thread is not None:
        kb.add_notify_sub(
            conn,
            task_id=tid,
            platform="discord",
            chat_id="forum-1",
            thread_id=with_discord_thread,
        )
    return tid


def _status(conn, task_id: str) -> str:
    return conn.execute("SELECT status FROM tasks WHERE id = ?", (task_id,)).fetchone()["status"]


def test_dry_run_lists_only_old_done_candidates_without_mutating(kanban_home):
    now = int(time.time())
    conn = kb.connect()
    try:
        old_tid = _done_task(conn, title="old", completed_at=now - 8 * 86400, with_discord_thread="111")
        fresh_tid = _done_task(conn, title="fresh", completed_at=now - 2 * 86400, with_discord_thread="222")
        result = run_archival(now=now, dry_run=True, discord_client=RecordingDiscordClient())
        assert result["candidate_count"] == 1
        assert result["archived_count"] == 0
        assert result["tasks"] == [
            {
                "task_id": old_tid,
                "status": "done",
                "action": "would_archive",
                "discord_thread_ids": ["111"],
            }
        ]
        assert _status(conn, old_tid) == "done"
        assert _status(conn, fresh_tid) == "done"
    finally:
        conn.close()


def test_live_run_archives_old_done_and_locks_discord_then_removes_sub(kanban_home):
    now = int(time.time())
    client = RecordingDiscordClient()
    conn = kb.connect()
    try:
        tid = _done_task(conn, title="old", completed_at=now - 9 * 86400, with_discord_thread="333")
        result = run_archival(now=now, discord_client=client)
        assert result["candidate_count"] == 1
        assert result["archived_count"] == 1
        assert result["discord_archived_locked_count"] == 1
        assert result["failure_count"] == 0
        assert client.calls == ["333"]
        assert client.metadata_calls == [{}]
        assert _status(conn, tid) == "archived"
        assert kb.list_notify_subs(conn, tid) == []
    finally:
        conn.close()


def test_live_run_skips_fresh_done_and_non_done_tasks(kanban_home):
    now = int(time.time())
    client = RecordingDiscordClient()
    conn = kb.connect()
    try:
        fresh_tid = _done_task(conn, title="fresh", completed_at=now - 2 * 86400, with_discord_thread="222")
        running_tid = _running_task(conn, title="running", created_at=now - 30 * 86400, with_discord_thread="999")
        original_running_status = _status(conn, running_tid)

        result = run_archival(now=now, discord_client=client)

        assert result["candidate_count"] == 0
        assert result["reconcile_candidate_count"] == 0
        assert result["archived_count"] == 0
        assert result["failure_count"] == 0
        assert result["tasks"] == []
        assert client.calls == []
        assert _status(conn, fresh_tid) == "done"
        assert _status(conn, running_tid) == original_running_status
        assert len(kb.list_notify_subs(conn, fresh_tid)) == 1
        assert len(kb.list_notify_subs(conn, running_tid)) == 1
    finally:
        conn.close()


def test_discord_archive_patch_includes_configured_archived_projection_metadata(kanban_home):
    tag_config = {
        "tags": {"archived": {"emoji": "📦"}},
        "status_to_tag": {"archived": "status-archived"},
        "assignee_to_tag": {"worker": "assignee-worker"},
    }
    cfg_path = kb.board_dir("default") / "discord-forum-tags.json"
    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    cfg_path.write_text(json.dumps(tag_config), encoding="utf-8")

    now = int(time.time())
    client = RecordingDiscordClient()
    conn = kb.connect()
    try:
        _done_task(conn, title="old", completed_at=now - 9 * 86400, with_discord_thread="333")
        result = run_archival(now=now, discord_client=client)
        assert result["discord_archived_locked_count"] == 1
        assert client.metadata_calls == [
            {"name": "📦 [archived] old", "applied_tags": ["status-archived", "assignee-worker"]}
        ]
    finally:
        conn.close()


def test_discord_failure_leaves_subscription_for_next_reconcile_run(kanban_home):
    now = int(time.time())
    conn = kb.connect()
    try:
        tid = _done_task(conn, title="old", completed_at=now - 9 * 86400, with_discord_thread="444")
        first = run_archival(now=now, discord_client=RecordingDiscordClient(fail_threads={"444"}))
        assert first["archived_count"] == 1
        assert first["failure_count"] == 1
        assert _status(conn, tid) == "archived"
        assert len(kb.list_notify_subs(conn, tid)) == 1

        second_client = RecordingDiscordClient()
        second = run_archival(now=now, discord_client=second_client)
        assert second["candidate_count"] == 0
        assert second["already_archived_count"] == 1
        assert second["discord_archived_locked_count"] == 1
        assert second_client.calls == ["444"]
        assert kb.list_notify_subs(conn, tid) == []
    finally:
        conn.close()


def test_already_archived_task_with_subscription_is_reconciled_idempotently(kanban_home):
    now = int(time.time())
    client = RecordingDiscordClient()
    conn = kb.connect()
    try:
        tid = _done_task(conn, title="old", completed_at=now - 9 * 86400, with_discord_thread="777")
        assert kb.archive_task(conn, tid)

        result = run_archival(now=now, discord_client=client)

        assert result["candidate_count"] == 0
        assert result["reconcile_candidate_count"] == 1
        assert result["already_archived_count"] == 1
        assert result["archived_count"] == 0
        assert result["discord_archived_locked_count"] == 1
        assert client.calls == ["777"]
        assert _status(conn, tid) == "archived"
        assert kb.list_notify_subs(conn, tid) == []
    finally:
        conn.close()


def test_missing_discord_thread_404_is_nonfatal_and_removes_subscription(kanban_home):
    now = int(time.time())
    client = RecordingDiscordClient(missing_threads={"555"})
    conn = kb.connect()
    try:
        tid = _done_task(conn, title="old", completed_at=now - 9 * 86400, with_discord_thread="555")

        result = run_archival(now=now, discord_client=client)

        assert result["candidate_count"] == 1
        assert result["archived_count"] == 1
        assert result["failure_count"] == 0
        assert result["skipped_by_reason"]["missing_or_deleted_discord_thread"] == 1
        assert result["tasks"][0]["discord"] == [{"thread_id": "555", "status": "missing_thread"}]
        assert _status(conn, tid) == "archived"
        assert kb.list_notify_subs(conn, tid) == []
    finally:
        conn.close()


def test_malformed_discord_thread_reference_is_logged_nonfatal_and_removed(kanban_home):
    now = int(time.time())
    client = RecordingDiscordClient()
    conn = kb.connect()
    try:
        tid = _done_task(conn, title="old", completed_at=now - 9 * 86400, with_discord_thread="not-a-thread")

        result = run_archival(now=now, discord_client=client)

        assert result["archived_count"] == 1
        assert result["failure_count"] == 0
        assert result["skipped_by_reason"]["malformed_discord_thread_ref"] == 1
        assert client.calls == []
        assert _status(conn, tid) == "archived"
        assert kb.list_notify_subs(conn, tid) == []
    finally:
        conn.close()


def test_missing_thread_reference_is_logged_nonfatal_and_idempotent(kanban_home):
    now = int(time.time())
    conn = kb.connect()
    try:
        tid = _done_task(conn, title="old", completed_at=now - 9 * 86400, with_discord_thread="")
        result = run_archival(now=now, discord_client=RecordingDiscordClient())
        assert result["failure_count"] == 0
        assert result["skipped_by_reason"]["missing_discord_thread_ref"] == 1
        assert _status(conn, tid) == "archived"
        assert kb.list_notify_subs(conn, tid) == []
    finally:
        conn.close()


def test_batch_limit_caps_archives(kanban_home):
    now = int(time.time())
    conn = kb.connect()
    try:
        tids = [
            _done_task(conn, title=f"old {idx}", completed_at=now - (10 + idx) * 86400)
            for idx in range(3)
        ]
        result = run_archival(now=now, batch_size=2, discord_client=RecordingDiscordClient())
        assert result["candidate_count"] == 2
        archived = [tid for tid in tids if _status(conn, tid) == "archived"]
        assert len(archived) == 2
    finally:
        conn.close()
