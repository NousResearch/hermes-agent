"""Prose-discovered artifacts must belong to the task's own scratch workspace.

A legacy worker names deliverables only by path in its summary/result prose. Those names are
promoted into ``metadata["artifacts"]`` and then uploaded to the chat, so the ownership check
is a delivery authorization, not a convenience. A lexical prefix match against the workspace
is not containment:

* ``<workspace>/../../other/x.txt`` spells the prefix but resolves outside it;
* ``<workspace>/link.txt`` is a symlink living inside the workspace whose target is outside.

Both reached ``complete_task`` / ``request_review`` and were delivered. These are the
regressions; the in-workspace case must keep working.
"""

from __future__ import annotations

import json
import pathlib
import sqlite3

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc


@pytest.fixture()
def board(tmp_path, monkeypatch):
    """Isolated HERMES_HOME with one scratch task and its managed workspace."""
    home = tmp_path / ".hermes"
    (home / "kanban" / "workspaces").mkdir(parents=True)
    (home / "kanban" / "boards").mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(pathlib.Path, "home", lambda: tmp_path)
    kb.init_db()
    conn = kbc.connect()
    task_id = kb.create_task(conn, title="scratch boundary")
    workspace = home / "kanban" / "workspaces" / task_id
    workspace.mkdir(parents=True, exist_ok=True)
    conn.execute(
        "UPDATE tasks SET workspace_kind='scratch', workspace_path=?, status='running' WHERE id=?",
        (str(workspace), task_id),
    )
    conn.commit()

    outside = tmp_path / "outside-the-workspace"
    outside.mkdir()
    secret = outside / "synthetic-private.txt"
    secret.write_text("SYNTHETIC-PRIVATE\n", encoding="utf-8")
    return conn, task_id, workspace, secret


def _completed_event_artifacts(conn: sqlite3.Connection, task_id: str) -> list[str]:
    row = conn.execute(
        "SELECT payload FROM task_events WHERE task_id=? AND kind='completed' ORDER BY id DESC",
        (task_id,),
    ).fetchone()
    if not row:
        return []
    payload = row["payload"]
    payload = json.loads(payload) if isinstance(payload, str) else payload
    return payload.get("artifacts", [])


def test_parent_traversal_out_of_the_workspace_is_not_promoted(board):
    conn, task_id, workspace, secret = board
    summary = f"SUMMARY: wrote {workspace}/../../../../outside-the-workspace/{secret.name}"

    merged = kb._merge_completion_prose_artifacts(conn, task_id, None, summary=summary, result=None)

    assert merged is None or not merged.get("artifacts")


def test_symlink_escaping_the_workspace_is_not_promoted(board):
    conn, task_id, workspace, secret = board
    (workspace / "link-to-outside.txt").symlink_to(secret)
    summary = f"SUMMARY: see {workspace}/link-to-outside.txt"

    merged = kb._merge_completion_prose_artifacts(conn, task_id, None, summary=summary, result=None)

    assert merged is None or not merged.get("artifacts")


def test_sibling_directory_sharing_the_workspace_prefix_is_not_promoted(board):
    conn, task_id, workspace, _secret = board
    sibling = workspace.parent / f"{workspace.name}-evil"
    sibling.mkdir()
    (sibling / "synthetic-other.txt").write_text("SYNTHETIC\n", encoding="utf-8")
    summary = f"SUMMARY: see {sibling}/synthetic-other.txt"

    merged = kb._merge_completion_prose_artifacts(conn, task_id, None, summary=summary, result=None)

    assert merged is None or not merged.get("artifacts")


def test_file_inside_the_workspace_is_still_promoted(board):
    conn, task_id, workspace, _secret = board
    inside = workspace / "report.txt"
    inside.write_text("SYNTHETIC-LEGIT\n", encoding="utf-8")
    summary = f"SUMMARY: see {inside}"

    merged = kb._merge_completion_prose_artifacts(conn, task_id, None, summary=summary, result=None)

    assert [str(pathlib.Path(p)) for p in merged["artifacts"]] == [str(inside)]


def test_complete_task_does_not_publish_an_escaped_prose_path(board):
    """The real entry point: the boundary must hold before the event is written."""
    conn, task_id, workspace, secret = board
    (workspace / "link-to-outside.txt").symlink_to(secret)
    summary = (
        f"SUMMARY: wrote {workspace}/../../../../outside-the-workspace/{secret.name} "
        f"and {workspace}/link-to-outside.txt"
    )

    assert kb.complete_task(conn, task_id, result=summary, summary=summary) is True

    artifacts = _completed_event_artifacts(conn, task_id)
    assert not any("outside-the-workspace" in path for path in artifacts)
    assert not any(path.endswith("link-to-outside.txt") for path in artifacts)


def test_request_review_does_not_publish_an_escaped_prose_path(board):
    conn, task_id, workspace, secret = board
    (workspace / "link-to-outside.txt").symlink_to(secret)
    summary = f"SUMMARY: see {workspace}/../../../../outside-the-workspace/{secret.name}"

    kb.request_review(conn, task_id, summary=summary)

    row = conn.execute(
        "SELECT payload FROM task_events WHERE task_id=? AND kind='review_requested' "
        "ORDER BY id DESC",
        (task_id,),
    ).fetchone()
    payload = json.loads(row["payload"]) if row and isinstance(row["payload"], str) else (
        row["payload"] if row else {}
    )
    assert not any("outside-the-workspace" in path for path in (payload or {}).get("artifacts", []))
