"""Respawn-guard tests for explicit re-queue after a PR comment."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_dispatch as kbd


@pytest.fixture
def kanban_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


def _comment(conn, task_id: str, created_at: int, body: str) -> None:
    conn.execute(
        "INSERT INTO task_comments (task_id, author, body, created_at) "
        "VALUES (?, ?, ?, ?)",
        (task_id, "worker", body, created_at),
    )


def _event(conn, task_id: str, kind: str, created_at: int) -> None:
    conn.execute(
        "INSERT INTO task_events (task_id, kind, payload, created_at) "
        "VALUES (?, ?, ?, ?)",
        (task_id, kind, json.dumps({"status": "ready"}), created_at),
    )


def test_recent_pr_comment_still_guards_idle_task(
    kanban_home: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    with kbc.connect() as conn:
        task_id = kb.create_task(conn, title="already has a PR", assignee="worker")
        _comment(conn, task_id, 1_000, "PR: https://github.com/NousResearch/hermes-agent/pull/123")
        monkeypatch.setattr(kbd.time, "time", lambda: 1_001)

        assert kbd.check_respawn_guard(conn, task_id) == "active_pr"


def test_requeue_after_latest_pr_comment_allows_respawn(
    kanban_home: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    with kbc.connect() as conn:
        task_id = kb.create_task(conn, title="review rework", assignee="worker")
        _comment(conn, task_id, 1_000, "PR: https://github.com/NousResearch/hermes-agent/pull/123")
        _event(conn, task_id, "changes_requested", 1_001)
        monkeypatch.setattr(kbd.time, "time", lambda: 1_002)

        assert kbd.check_respawn_guard(conn, task_id) is None


@pytest.mark.parametrize(
    "event_kind",
    ["status", "promoted", "promoted_manual", "unblocked", "reclaimed", "changes_requested"],
)
def test_each_explicit_requeue_event_bypasses_active_pr(
    kanban_home: Path, monkeypatch: pytest.MonkeyPatch, event_kind: str,
) -> None:
    with kbc.connect() as conn:
        task_id = kb.create_task(conn, title=event_kind, assignee="worker")
        _comment(conn, task_id, 1_000, "PR: https://github.com/NousResearch/hermes-agent/pull/123")
        _event(conn, task_id, event_kind, 1_001)
        monkeypatch.setattr(kbd.time, "time", lambda: 1_002)

        assert kbd.check_respawn_guard(conn, task_id) is None


def test_requeue_before_latest_pr_comment_does_not_bypass_guard(
    kanban_home: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    with kbc.connect() as conn:
        task_id = kb.create_task(conn, title="stale requeue", assignee="worker")
        _event(conn, task_id, "changes_requested", 1_000)
        _comment(conn, task_id, 1_001, "PR: https://github.com/NousResearch/hermes-agent/pull/123")
        monkeypatch.setattr(kbd.time, "time", lambda: 1_002)

        assert kbd.check_respawn_guard(conn, task_id) == "active_pr"
