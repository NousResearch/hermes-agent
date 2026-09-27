"""Attach-time evidence anchor tests (t_a538586e, Claude ruling 2026-09-14).

Every row entering task_attachments must carry a readable anchor at attach
time: sha256 of the stored blob in the 'attached' event payload AND the
producing run's id in the event's run_id column. Both INSERT sites
(add_attachment via store_attachment_bytes, and _insert_completion_attachment
via complete_task) are covered. No new column, no DDL, no task_runs.metadata
write.
"""

from __future__ import annotations

import hashlib
import sqlite3
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_workspace as kbw


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    """Isolated HERMES_HOME with an empty kanban DB."""
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


def _attached_events(conn, task_id):
    return [e for e in kb.list_events(conn, task_id) if e.kind == "attached"]


def _task_with_ready_task(conn):
    return kb.create_task(conn, title="anchor", assignee="coder")


# ---------------------------------------------------------------------------
# AC-1: add_attachment path — sha256 payload field matches the stored blob
# ---------------------------------------------------------------------------

def test_add_attachment_records_sha256_in_event_payload(kanban_home):
    data = b"anchored blob bytes"
    expected_sha = hashlib.sha256(data).hexdigest()
    with kbc.connect() as conn:
        t = _task_with_ready_task(conn)
        kb.store_attachment_bytes(conn, t, "report.bin", data, uploaded_by="coder")
        events = _attached_events(conn, t)
        atts = kb.list_attachments(conn, t)

    assert len(events) == 1
    assert events[0].payload["sha256"] == expected_sha
    # the anchor matches the bytes actually on disk, not just the caller's copy
    assert hashlib.sha256(Path(atts[0].stored_path).read_bytes()).hexdigest() == expected_sha


# ---------------------------------------------------------------------------
# AC-2: attach while a run is active -> event run_id == that run
# ---------------------------------------------------------------------------

def test_add_attachment_event_run_id_matches_active_run(kanban_home):
    with kbc.connect() as conn:
        t = _task_with_ready_task(conn)
        claimed = kb.claim_task(conn, t, claimer="coder")
        assert claimed is not None
        active_run = kb.latest_run(conn, t).id

        kb.store_attachment_bytes(conn, t, "during-run.txt", b"x", uploaded_by="coder")
        events = _attached_events(conn, t)
    assert events[0].run_id == active_run


# ---------------------------------------------------------------------------
# AC-5: attach with NO active run -> sha present, run_id NULL
# ---------------------------------------------------------------------------

def test_add_attachment_without_active_run_has_sha_null_run(kanban_home):
    data = b"no-run blob"
    expected_sha = hashlib.sha256(data).hexdigest()
    with kbc.connect() as conn:
        t = _task_with_ready_task(conn)  # ready, never claimed -> no run
        kb.store_attachment_bytes(conn, t, "orphan.txt", data, uploaded_by="user")
        events = _attached_events(conn, t)

    assert events[0].payload["sha256"] == expected_sha
    assert events[0].run_id is None


# ---------------------------------------------------------------------------
# AC-3: kanban_complete path (_insert_completion_attachment) carries the anchor
# AC-2 (completion variant): event run_id == the completing run
# ---------------------------------------------------------------------------

def test_completion_artifact_carries_anchor(kanban_home):
    blob = b"completion-artifact-bytes"
    expected_sha = hashlib.sha256(blob).hexdigest()
    with kbc.connect() as conn:
        t = _task_with_ready_task(conn)
        ws = kbw.resolve_workspace(kb.get_task(conn, t))
        kbw.set_workspace_path(conn, t, ws)
        kb.claim_task(conn, t, claimer="coder")
        run_id = kb.latest_run(conn, t).id

        artifact = ws / "out.txt"
        artifact.write_bytes(blob)
        assert kb.complete_task(conn, t, result="ok", metadata={"artifacts": [str(artifact)]})

        events = _attached_events(conn, t)
        atts = kb.list_attachments(conn, t)

    assert len(atts) == 1
    assert atts[0].uploaded_by == "kanban_complete"
    assert len(events) == 1
    assert events[0].payload["sha256"] == expected_sha
    assert hashlib.sha256(Path(atts[0].stored_path).read_bytes()).hexdigest() == expected_sha
    # complete_task stages artifacts BEFORE _end_run, so current_run_id still
    # resolved at attach time: the event is tied to the completing run.
    assert events[0].run_id == run_id


# ---------------------------------------------------------------------------
# AC-4: no write to task_runs.metadata introduced by the anchor
# ---------------------------------------------------------------------------

def test_anchor_does_not_write_task_runs_metadata(kanban_home):
    with kbc.connect() as conn:
        t = _task_with_ready_task(conn)
        ws = kbw.resolve_workspace(kb.get_task(conn, t))
        kbw.set_workspace_path(conn, t, ws)
        kb.claim_task(conn, t, claimer="coder")
        kb.store_attachment_bytes(conn, t, "a.txt", b"abc", uploaded_by="coder")
        # attach alone must leave the active run's metadata NULL
        metadata_after_attach = kb.latest_run(conn, t).metadata

        artifact = ws / "b.txt"
        artifact.parent.mkdir(parents=True, exist_ok=True)
        artifact.write_bytes(b"def")
        # AC-4: the anchor lives on the 'attached' event only. task_runs.metadata
        # keeps its pre-existing meaning (caller handoff fields) and must never
        # gain a sha256 key from the anchoring code.
        assert kb.complete_task(
            conn, t, result="ok", metadata={"artifacts": [str(artifact)]},
        )
        run = kb.latest_run(conn, t)
        attached = [e for e in kb.list_events(conn, t) if e.kind == "attached"]

    assert metadata_after_attach is None
    assert len(attached) == 2
    assert "sha256" not in (run.metadata or {})
    assert attached[1].payload["sha256"] == hashlib.sha256(b"def").hexdigest()


# ---------------------------------------------------------------------------
# AC-1 hardening (reviewer 2026-09-14): an unreadable/missing blob ABORTS the
# attach — no row may enter task_attachments without a sha256 anchor.
# ---------------------------------------------------------------------------

def test_add_attachment_missing_blob_aborts_without_row_or_event(kanban_home):
    with kbc.connect() as conn:
        t = _task_with_ready_task(conn)
        with pytest.raises(kb.AttachmentAnchorError):
            kb.add_attachment(
                conn, t, filename="ghost.txt", stored_path="/nonexistent/ghost.txt",
                content_type="text/plain", size=3, uploaded_by="coder",
            )
        atts = kb.list_attachments(conn, t)
        events = _attached_events(conn, t)
    assert atts == []
    assert events == []


def test_store_attachment_bytes_unreadable_blob_leaves_no_row(tmp_path, kanban_home, monkeypatch):
    # The blob lands on disk, then becomes unreadable before hashing
    # (race / permission loss): the whole attach must abort, the wrapper
    # must clean the orphan blob, and no row/event may survive.
    real_read_bytes = Path.read_bytes

    def _unreadable(self):
        if self.name == "doomed.bin":
            raise OSError("simulated read failure")
        return real_read_bytes(self)

    monkeypatch.setattr(Path, "read_bytes", _unreadable)
    with kbc.connect() as conn:
        t = _task_with_ready_task(conn)
        with pytest.raises(kb.AttachmentAnchorError):
            kb.store_attachment_bytes(conn, t, "doomed.bin", b"payload", uploaded_by="coder")
        atts = kb.list_attachments(conn, t)
        events = _attached_events(conn, t)
        att_dir = kb.task_attachments_dir(t)
    assert atts == []
    assert events == []
    # no orphan blob left behind
    assert not list(att_dir.glob("doomed*"))


def test_completion_attachment_unreadable_blob_aborts_completion(tmp_path, kanban_home, monkeypatch):
    # _insert_completion_attachment hashing failure must roll back the whole
    # complete_task transaction: no attachment row, no attached event, the
    # task must stay running so the worker can retry.
    blob = b"completion-artifact-bytes"
    expected_sha = hashlib.sha256(blob).hexdigest()
    with kbc.connect() as conn:
        t = _task_with_ready_task(conn)
        ws = kbw.resolve_workspace(kb.get_task(conn, t))
        kbw.set_workspace_path(conn, t, ws)
        kb.claim_task(conn, t, claimer="coder")

        artifact = ws / "out.txt"
        artifact.write_bytes(blob)

        real_read_bytes = Path.read_bytes

        def _unreadable(self):
            if "attachments" in self.parts and self.name.startswith("out"):
                raise OSError("simulated read failure at anchor time")
            return real_read_bytes(self)

        monkeypatch.setattr(Path, "read_bytes", _unreadable)
        with pytest.raises(kb.AttachmentAnchorError):
            kb.complete_task(conn, t, result="ok", metadata={"artifacts": [str(artifact)]})
        monkeypatch.setattr(Path, "read_bytes", real_read_bytes)

        task = kb.get_task(conn, t)
        atts = kb.list_attachments(conn, t)
        events = _attached_events(conn, t)
    assert task.status == "running"
    assert atts == []
    assert events == []

    # retry path: with the blob readable again, completion succeeds fully anchored
    with kbc.connect() as conn:
        assert kb.complete_task(conn, t, result="ok", metadata={"artifacts": [str(artifact)]})
        atts = kb.list_attachments(conn, t)
        events = _attached_events(conn, t)
    assert len(atts) == 1
    assert len(events) == 1
    assert events[0].payload["sha256"] == expected_sha
    assert hashlib.sha256(Path(atts[0].stored_path).read_bytes()).hexdigest() == expected_sha


# ---------------------------------------------------------------------------
# No-DDL check: task_attachments schema is untouched by the anchor
# ---------------------------------------------------------------------------

def test_no_new_column_on_task_attachments(kanban_home):
    with kbc.connect() as conn:
        cols = [
            r[1] for r in conn.execute("PRAGMA table_info(task_attachments)").fetchall()
        ]
    assert cols == [
        "id", "task_id", "filename", "stored_path", "content_type",
        "size", "uploaded_by", "created_at",
    ]
