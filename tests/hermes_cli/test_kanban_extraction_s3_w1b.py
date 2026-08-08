"""Regression tests for the wave-1 shard-s3 extraction of hermes_cli/kanban_db.py.

Covers the two clusters moved verbatim out of ``hermes_cli/kanban_db.py``
into the new mixin modules:

* c6  artifact preservation -> ``hermes_cli/artifacts_mixin.py``
  (``ArtifactPreservationError``, ``_merge_completion_prose_artifacts``,
  ``_persist_scratch_completion_artifacts``, ``_insert_completion_attachment``,
  ``_unique_attachment_path``)
* c9  first-use scratch tip -> ``hermes_cli/scratch_tip_mixin.py``
  (``_scratch_tip_sentinel_path``, ``_scratch_tip_shown``,
  ``_mark_scratch_tip_shown``, ``_maybe_emit_scratch_tip``)

Every test drives the functions through the ``hermes_cli.kanban_db``
re-export surface — the API every caller (CLI, dashboard plugin,
dispatcher) uses — so a broken re-export fails loudly here.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def kanban_home_env(tmp_path, monkeypatch):
    """Full kanban environment: real schema on disk under a temp home."""
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_KANBAN_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    conn = kb.connect()
    yield home, conn
    conn.close()


@pytest.fixture
def scratch_workspace(tmp_path, monkeypatch):
    """A managed scratch workspace plus a live sqlite row for one task."""
    root = tmp_path / "workspaces"
    root.mkdir()
    ws = root / "w1"
    ws.mkdir()
    monkeypatch.setenv("HERMES_KANBAN_WORKSPACES_ROOT", str(root))
    monkeypatch.setenv("HERMES_KANBAN_HOME", str(tmp_path / "home"))
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.execute(
        "CREATE TABLE tasks (id TEXT PRIMARY KEY, workspace_kind TEXT, workspace_path TEXT)"
    )
    conn.execute(
        "INSERT INTO tasks VALUES (?, ?, ?)",
        ("t1", "scratch", str(ws)),
    )
    yield root, ws, conn
    conn.close()


def _task_conn(conn, task_id, kind, path):
    conn.execute(
        "INSERT INTO tasks VALUES (?, ?, ?)",
        (task_id, kind, str(path)),
    )


# ---------------------------------------------------------------------------
# c6: artifact preservation
# ---------------------------------------------------------------------------


def test_unique_attachment_path_prefers_free_name(tmp_path):
    d = tmp_path / "att"
    d.mkdir()
    target = d / "report.md"
    target.write_text("x", encoding="utf-8")
    # existing file -> suffixed variant
    assert kb._unique_attachment_path(d, "report.md", set()) == d / "report_1.md"
    # used destinations are skipped even if not on disk
    assert kb._unique_attachment_path(d, "report.md", {d / "report_1.md"}) == d / "report_2.md"
    # free name is used as-is
    assert kb._unique_attachment_path(d, "notes.txt", set()) == d / "notes.txt"
    # only the basename is kept
    assert kb._unique_attachment_path(d, "sub/dir/file.txt", set()).name == "file.txt"
    # suffix survives collision numbering
    assert kb._unique_attachment_path(d, "report.md", set()).suffix == ".md"
    # empty filename falls back to 'artifact'
    assert kb._unique_attachment_path(d, "", set()) == d / "artifact"


def test_merge_completion_prose_artifacts_discovers_scratch_paths(scratch_workspace):
    root, ws, conn = scratch_workspace
    deliverable = ws / "out.txt"
    deliverable.write_text("hello", encoding="utf-8")

    merged = kb._merge_completion_prose_artifacts(
        conn, "t1", None, summary=f"wrote {deliverable}", result=None,
    )
    assert merged == {"artifacts": [str(deliverable)]}

    # repeated mentions are de-duplicated
    merged2 = kb._merge_completion_prose_artifacts(
        conn, "t1", {"artifacts": []},
        summary=f"a {deliverable} b {deliverable}",
        result=None,
    )
    assert merged2 == {"artifacts": [str(deliverable)]}

    # non-scratch workspaces are skipped untouched
    _task_conn(conn, "t2", "worktree", ws)
    assert kb._merge_completion_prose_artifacts(
        conn, "t2", {"artifacts": []}, summary="x", result=None,
    ) == {"artifacts": []}

    # paths outside managed scratch are not discovered
    outside = ws.parent.parent / "outside.txt"
    outside.write_text("x", encoding="utf-8")
    assert kb._merge_completion_prose_artifacts(
        conn, "t1", None, summary=f"wrote {outside}", result=None,
    ) is None


def test_persist_scratch_completion_artifacts_copies_into_attachments(scratch_workspace):
    root, ws, conn = scratch_workspace
    deliverable = ws / "out.bin"
    deliverable.write_bytes(b"payload-123")

    metadata = {"artifacts": [str(deliverable)]}
    kb._persist_scratch_completion_artifacts(conn, "t1", metadata)

    staged = metadata.get("_staged_artifacts")
    assert staged and len(staged) == 1
    staged_path = Path(staged[0])
    assert staged_path.is_file()
    assert staged_path.read_bytes() == b"payload-123"
    assert metadata["artifacts"] == [str(staged_path.resolve())]

    # artifacts outside the workspace are kept verbatim, never copied
    outside = root.parent / "outside.bin"
    outside.write_bytes(b"x")
    metadata2 = {"artifacts": [str(outside)]}
    kb._persist_scratch_completion_artifacts(conn, "t1", metadata2)
    assert metadata2["artifacts"] == [str(outside)]
    assert "_staged_artifacts" not in metadata2


def test_insert_completion_attachment_records_row_and_event(kanban_home_env):
    home, conn = kanban_home_env
    kb._insert_completion_attachment(
        conn, "t1", filename="out.txt", stored_path="C:/tmp/out.txt",
        size=3, created_at=42,
    )
    row = conn.execute(
        "SELECT * FROM task_attachments WHERE task_id = 't1'"
    ).fetchone()
    assert row["filename"] == "out.txt"
    assert row["stored_path"] == "C:/tmp/out.txt"
    assert row["size"] == 3
    assert row["content_type"] is None
    assert row["uploaded_by"] == "kanban_complete"

    ev = conn.execute(
        "SELECT kind, payload FROM task_events WHERE task_id = 't1'"
    ).fetchone()
    assert ev["kind"] == "attached"
    assert '"filename": "out.txt"' in ev["payload"]


# ---------------------------------------------------------------------------
# c9: first-use scratch tip
# ---------------------------------------------------------------------------


def test_scratch_tip_sentinel_lifecycle(kanban_home_env):
    home, conn = kanban_home_env
    sentinel = kb._scratch_tip_sentinel_path()
    assert sentinel == Path(home) / ".scratch_tip_shown"
    assert kb._scratch_tip_shown() is False
    kb._mark_scratch_tip_shown()
    assert sentinel.exists()
    assert kb._scratch_tip_shown() is True


def test_maybe_emit_scratch_tip_emits_once_per_install(kanban_home_env):
    home, conn = kanban_home_env
    sentinel = kb._scratch_tip_sentinel_path()

    # non-scratch kinds never emit, and never create the sentinel
    kb._maybe_emit_scratch_tip(conn, "t2", "worktree")
    kb._maybe_emit_scratch_tip(conn, "t2", "dir")
    count2 = conn.execute(
        "SELECT COUNT(*) FROM task_events WHERE task_id = 't2'"
    ).fetchone()[0]
    assert count2 == 0
    assert not sentinel.exists()

    kb._maybe_emit_scratch_tip(conn, "t1", "scratch")
    rows = conn.execute(
        "SELECT kind FROM task_events WHERE task_id = 't1'"
    ).fetchall()
    assert [r["kind"] for r in rows] == ["tip_scratch_workspace"]
    assert sentinel.exists()

    # second emit is a no-op: the sentinel is present
    kb._maybe_emit_scratch_tip(conn, "t1", "scratch")
    count = conn.execute(
        "SELECT COUNT(*) FROM task_events WHERE task_id = 't1'"
    ).fetchone()[0]
    assert count == 1


def test_maybe_emit_scratch_tip_treats_none_kind_as_scratch(kanban_home_env):
    home, conn = kanban_home_env
    # workspace_kind None is the legacy default and counts as scratch:
    # the code does ``(workspace_kind or "scratch") != "scratch"``.
    kb._maybe_emit_scratch_tip(conn, "t9", None)
    rows = conn.execute(
        "SELECT kind FROM task_events WHERE task_id = 't9'"
    ).fetchall()
    assert [r["kind"] for r in rows] == ["tip_scratch_workspace"]
    assert kb._scratch_tip_sentinel_path().exists()
