"""Behavior contracts for gateway session mirrors (issue #116940)."""

from __future__ import annotations

from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli.kanban_db_session_mirror import (
    archive_mirror,
    create_or_get_mirror,
    delete_mirror,
    finish_mirror,
    get_mirror,
    list_mirrors,
    mark_mirror_running,
    promote_mirror,
    prune_expired_mirrors,
)


@pytest.fixture
def mirror_conn(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    conn = kbc.connect(board="default")
    try:
        yield conn
    finally:
        conn.close()


def _create(conn, *, message_id="msg-1", now=100_000):
    return create_or_get_mirror(
        conn,
        profile="default",
        platform="telegram",
        chat_id="chat-1",
        thread_id="topic-2",
        session_id="session-1",
        message_id=message_id,
        now=now,
    )


def test_mirror_turn_identity_is_idempotent_and_terminal_state_is_immutable(mirror_conn):
    mirror_id, created = _create(mirror_conn)
    assert created is True
    duplicate_id, duplicate_created = _create(mirror_conn, now=100_001)
    assert duplicate_id == mirror_id
    assert duplicate_created is False

    assert get_mirror(mirror_conn, mirror_id)["status"] == "received"
    assert mark_mirror_running(mirror_conn, mirror_id, now=100_002) is True
    assert finish_mirror(mirror_conn, mirror_id, "completed", now=100_010) is True
    assert mark_mirror_running(mirror_conn, mirror_id, now=100_020) is False
    assert finish_mirror(mirror_conn, mirror_id, "failed", now=100_030) is False

    row = get_mirror(mirror_conn, mirror_id)
    assert row["status"] == "completed"
    assert row["profile"] == "default"
    assert row["platform"] == "telegram"
    assert row["chat_id"] == "chat-1"
    assert row["thread_id"] == "topic-2"
    assert row["session_id"] == "session-1"
    assert "content" not in row and "transcript" not in row and "user_id" not in row
    # A mirror is not an executable task and is invisible to the dispatcher's task inventory.
    assert kb.list_tasks(mirror_conn) == []


def test_mirrors_can_be_archived_deleted_expired_and_explicitly_promoted(mirror_conn):
    archived_id, _ = _create(mirror_conn, message_id="msg-archive", now=100_000)
    retained_id, _ = _create(mirror_conn, message_id="msg-retain", now=190_000)
    expired_id, _ = _create(mirror_conn, message_id="msg-expired", now=1)

    assert archive_mirror(mirror_conn, archived_id, now=100_001) is True
    assert archive_mirror(mirror_conn, archived_id, now=100_001) is False
    assert [row["id"] for row in list_mirrors(mirror_conn)] == [retained_id, expired_id]
    assert [row["id"] for row in list_mirrors(mirror_conn, include_archived=True)] == [
        retained_id, archived_id, expired_id,
    ]

    assert finish_mirror(mirror_conn, archived_id, "completed", now=100_002) is True
    assert finish_mirror(mirror_conn, expired_id, "failed", now=2) is True
    task_id = promote_mirror(
        mirror_conn,
        retained_id,
        title="Review the Telegram request",
        body="Created explicitly by the user; no chat transcript was copied.",
        created_by="desktop",
    )
    assert promote_mirror(
        mirror_conn,
        retained_id,
        title="A duplicate click must not make a second task",
        body=None,
        created_by="desktop",
    ) == task_id
    task = kb.get_task(mirror_conn, task_id)
    assert task is not None
    assert task.status == "blocked"
    assert task.assignee is None
    assert task.session_id == "session-1"
    assert get_mirror(mirror_conn, retained_id)["promoted_task_id"] == task_id

    # Retention removes only old terminal mirrors; active turns and tasks remain untouched.
    assert finish_mirror(mirror_conn, archived_id, "failed", now=190_002) is False
    assert finish_mirror(mirror_conn, expired_id, "failed", now=190_003) is False
    assert prune_expired_mirrors(mirror_conn, retention_days=1, now=190_004) == 2
    assert get_mirror(mirror_conn, expired_id) is None
    assert get_mirror(mirror_conn, retained_id) is not None
    assert get_mirror(mirror_conn, archived_id) is None
    assert kb.get_task(mirror_conn, task_id) is not None
    assert delete_mirror(mirror_conn, retained_id) is True
    assert delete_mirror(mirror_conn, retained_id) is False


def test_retention_is_scoped_to_the_profile_setting(mirror_conn):
    ids = {}
    for profile, message_id in (("alpha", "alpha-old"), ("beta", "beta-old"), ("alpha", "alpha-active")):
        ids[message_id], _ = create_or_get_mirror(
            mirror_conn, profile=profile, platform="telegram", chat_id="chat-1", thread_id=None,
            session_id=f"session-{message_id}", message_id=message_id, now=1,
        )
    finish_mirror(mirror_conn, ids["alpha-old"], "completed", now=2)
    finish_mirror(mirror_conn, ids["beta-old"], "completed", now=2)
    assert prune_expired_mirrors(mirror_conn, retention_days=1, now=100_000, profile="alpha") == 1
    assert get_mirror(mirror_conn, ids["alpha-old"]) is None
    assert get_mirror(mirror_conn, ids["beta-old"]) is not None
    active = get_mirror(mirror_conn, ids["alpha-active"])
    assert active is not None and active["status"] == "received"
