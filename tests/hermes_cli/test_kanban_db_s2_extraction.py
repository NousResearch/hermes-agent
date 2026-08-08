"""Regression tests for the shard-s2 wave-1 extraction of hermes_cli/kanban_db.py.

Covers the two clusters moved verbatim into sibling modules in this wave:

* ``hermes_cli.kanban_db_attachments`` (cluster c10): attachment name
  sanitisation, collision-free paths, size-cap enforcement, and the
  kanban_db re-export wiring.
* ``hermes_cli.kanban_db_links`` (cluster c8): dependency-link cycle
  guards and parent/child queries, plus the re-export wiring.

The identity assertions (``kb.X is module.X``) are the extraction-specific
regression: they fail if the re-export wiring is ever dropped.
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_attachments
from hermes_cli import kanban_db_events
from hermes_cli import kanban_db_links
from hermes_cli import kanban_db_tasks


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    """Isolated HERMES_HOME with an empty kanban DB."""
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


# ---------------------------------------------------------------------------
# Re-export wiring (the extraction-specific regression)
# ---------------------------------------------------------------------------


def test_attachments_cluster_is_reexported_from_kanban_db():
    assert kb.AttachmentTooLarge is kanban_db_attachments.AttachmentTooLarge
    assert kb._safe_attachment_name is kanban_db_attachments._safe_attachment_name
    assert kb._collision_free_path is kanban_db_attachments._collision_free_path
    assert kb.store_attachment_bytes is kanban_db_attachments.store_attachment_bytes
    assert kb.add_attachment is kanban_db_attachments.add_attachment
    assert kb.list_attachments is kanban_db_attachments.list_attachments
    assert kb.get_attachment is kanban_db_attachments.get_attachment
    assert kb.delete_attachment is kanban_db_attachments.delete_attachment


def test_links_cluster_is_reexported_from_kanban_db():
    assert kb.link_tasks is kanban_db_links.link_tasks
    assert kb._would_cycle is kanban_db_links._would_cycle
    assert kb.unlink_tasks is kanban_db_links.unlink_tasks
    assert kb.parent_ids is kanban_db_links.parent_ids
    assert kb.child_ids is kanban_db_links.child_ids
    assert kb.parent_results is kanban_db_links.parent_results


def test_tasks_cluster_is_reexported_from_kanban_db():
    assert kb._canonical_assignee is kanban_db_tasks._canonical_assignee
    assert kb._find_missing_parents is kanban_db_tasks._find_missing_parents
    assert kb._inherit_notify_subs is kanban_db_tasks._inherit_notify_subs
    assert kb.create_task is kanban_db_tasks.create_task
    assert kb.get_task is kanban_db_tasks.get_task
    assert kb.list_tasks is kanban_db_tasks.list_tasks


def test_events_cluster_is_reexported_from_kanban_db():
    assert kb.list_events is kanban_db_events.list_events
    assert kb._append_event is kanban_db_events._append_event
    assert kb._end_run is kanban_db_events._end_run
    assert kb._current_run_id is kanban_db_events._current_run_id
    assert kb._synthesize_ended_run is kanban_db_events._synthesize_ended_run


# ---------------------------------------------------------------------------
# Attachments: pure helpers
# ---------------------------------------------------------------------------


def test_safe_attachment_name_strips_directory_components():
    f = kanban_db_attachments._safe_attachment_name
    assert f("../../etc/passwd") == "passwd"
    assert f("C:\\evil\\payload.txt") == "payload.txt"
    assert f("a/b/c.txt") == "c.txt"
    assert f("plain.pdf") == "plain.pdf"


def test_safe_attachment_name_strips_control_chars_and_leading_dots():
    f = kanban_db_attachments._safe_attachment_name
    assert f("\x00evil.txt") == "evil.txt"
    assert f("..hidden") == "hidden"
    assert f("  spaced out.txt  ") == "spaced out.txt"


def test_safe_attachment_name_rejects_unusable_names():
    f = kanban_db_attachments._safe_attachment_name
    for bad in ("", None, ".", "..", "...", " . ", "/", "\\", "a/.."):
        with pytest.raises(ValueError, match="invalid attachment filename"):
            f(bad)


def test_safe_attachment_name_truncates_to_200_chars():
    f = kanban_db_attachments._safe_attachment_name
    assert len(f("x" * 300)) == 200


def test_collision_free_path_numbers_collisions(tmp_path):
    f = kanban_db_attachments._collision_free_path
    assert f(tmp_path, "foo.pdf") == tmp_path / "foo.pdf"
    (tmp_path / "foo.pdf").write_bytes(b"a")
    assert f(tmp_path, "foo.pdf") == tmp_path / "foo (1).pdf"
    (tmp_path / "foo (1).pdf").write_bytes(b"b")
    assert f(tmp_path, "foo.pdf") == tmp_path / "foo (2).pdf"


# ---------------------------------------------------------------------------
# Attachments: size cap + full roundtrip through the moved module
# ---------------------------------------------------------------------------


def test_store_attachment_bytes_enforces_size_cap(kanban_home):
    conn = kb.connect()
    try:
        task_id = kb.create_task(conn, title="t")
        with pytest.raises(kb.AttachmentTooLarge, match="MB limit"):
            kanban_db_attachments.store_attachment_bytes(
                conn, task_id, "big.bin", b"x" * 10, max_bytes=5,
            )
        # nothing was recorded for the rejected upload
        assert kb.list_attachments(conn, task_id) == []
    finally:
        conn.close()


def test_store_attachment_bytes_roundtrip_via_moved_module(kanban_home):
    conn = kb.connect()
    try:
        task_id = kb.create_task(conn, title="t")
        att_id = kanban_db_attachments.store_attachment_bytes(
            conn, task_id, "report.pdf", b"%PDF-1.4 fake",
            content_type="application/pdf", uploaded_by="tester",
        )
        assert att_id > 0
        att = kb.get_attachment(conn, att_id)
        assert att is not None
        assert att.filename == "report.pdf"
        assert att.content_type == "application/pdf"
        assert att.size == len(b"%PDF-1.4 fake")
        assert att.uploaded_by == "tester"
        # the moved add_attachment path still records the event row
        events = kb.list_events(conn, task_id)
        assert any(e.kind == "attached" for e in events)
        # blob landed under the per-task attachments dir
        blob = Path(att.stored_path)
        assert blob.is_file() and blob.read_bytes() == b"%PDF-1.4 fake"
        # delete through the re-exported name removes row + blob
        removed = kb.delete_attachment(conn, att_id)
        assert removed is not None and removed.id == att_id
        assert not blob.exists()
        assert kb.get_attachment(conn, att_id) is None
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Links: cycle guards, parent/child queries, unlink
# ---------------------------------------------------------------------------


def test_link_self_dependency_rejected(kanban_home):
    conn = kb.connect()
    try:
        t = kb.create_task(conn, title="t")
        with pytest.raises(ValueError, match="cannot depend on itself"):
            kb.link_tasks(conn, t, t)
    finally:
        conn.close()


def test_link_unknown_task_rejected(kanban_home):
    conn = kb.connect()
    try:
        t = kb.create_task(conn, title="t")
        with pytest.raises(ValueError, match="unknown task"):
            kb.link_tasks(conn, t, "t_missing")
    finally:
        conn.close()


def test_link_cycle_rejected(kanban_home):
    conn = kb.connect()
    try:
        a = kb.create_task(conn, title="a")
        b = kb.create_task(conn, title="b")
        c = kb.create_task(conn, title="c")
        kb.link_tasks(conn, a, b)
        kb.link_tasks(conn, b, c)
        # direct guard agrees the edge would create a cycle
        assert kb._would_cycle(conn, c, a)
        assert not kb._would_cycle(conn, a, b)
        with pytest.raises(ValueError, match="would create a cycle"):
            kb.link_tasks(conn, c, a)
        # nothing was inserted for the rejected edge
        assert kb.child_ids(conn, c) == []
    finally:
        conn.close()


def test_unlink_removes_edge(kanban_home):
    conn = kb.connect()
    try:
        a = kb.create_task(conn, title="a")
        b = kb.create_task(conn, title="b")
        kb.link_tasks(conn, a, b)
        assert kb.parent_ids(conn, b) == [a]
        assert kb.child_ids(conn, a) == [b]
        assert kb.unlink_tasks(conn, a, b) is True
        assert kb.parent_ids(conn, b) == []
        assert kb.child_ids(conn, a) == []
        assert kb.unlink_tasks(conn, a, b) is False
    finally:
        conn.close()


def test_parent_results_only_done_parents(kanban_home):
    conn = kb.connect()
    try:
        a = kb.create_task(conn, title="a")
        b = kb.create_task(conn, title="b")
        kb.link_tasks(conn, a, b)
        assert kb.parent_results(conn, b) == []
        conn.execute(
            "UPDATE tasks SET status = 'done', completed_at = ? WHERE id = ?",
            (int(time.time()), a),
        )
        conn.commit()
        assert kb.parent_results(conn, b) == [(a, None)]
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Tasks: create/query through the moved cluster
# ---------------------------------------------------------------------------


def test_create_task_defaults_and_roundtrip(kanban_home):
    conn = kb.connect()
    try:
        task_id = kanban_db_tasks.create_task(conn, title="hello")
        assert task_id.startswith("t_")
        task = kb.get_task(conn, task_id)
        assert task is not None and task.title == "hello"
        assert task.status == "ready"
        assert [t.id for t in kb.list_tasks(conn)] == [task_id]
    finally:
        conn.close()


def test_create_task_with_parents_defers_status(kanban_home):
    conn = kb.connect()
    try:
        parent = kb.create_task(conn, title="parent")
        child = kanban_db_tasks.create_task(conn, title="child", parents=[parent])
        assert kb.get_task(conn, child).status == "todo"
        assert kb.parent_ids(conn, child) == [parent]
    finally:
        conn.close()


def test_list_tasks_filters_by_status_and_assignee(kanban_home):
    conn = kb.connect()
    try:
        a = kb.create_task(conn, title="a")
        b = kb.create_task(conn, title="b")
        kb.assign_task(conn, b, "alice")
        assert {t.id for t in kb.list_tasks(conn, status="ready")} == {a, b}
        assert {t.id for t in kb.list_tasks(conn, assignee="alice")} == {b}
        # canonical assignee normalisation applies to the filter too
        assert {t.id for t in kb.list_tasks(conn, assignee="ALICE")} == {b}
        with pytest.raises(ValueError, match="status must be one of"):
            kb.list_tasks(conn, status="bogus")
    finally:
        conn.close()


def test_canonical_assignee_normalises(kanban_home):
    assert kanban_db_tasks._canonical_assignee(None) is None
    assert kanban_db_tasks._canonical_assignee("  Alice  ") == "alice"


# ---------------------------------------------------------------------------
# Events & runs: append/list, end-run, synthetic run
# ---------------------------------------------------------------------------


def test_append_and_list_events_roundtrip(kanban_home):
    conn = kb.connect()
    try:
        task_id = kb.create_task(conn, title="t")
        kanban_db_events._append_event(
            conn, task_id, "custom_kind", {"k": "v", "n": 1},
        )
        events = kb.list_events(conn, task_id)
        assert [e.kind for e in events].count("custom_kind") == 1
        ev = next(e for e in events if e.kind == "custom_kind")
        assert ev.payload == {"k": "v", "n": 1}
        assert ev.task_id == task_id
    finally:
        conn.close()


def test_end_run_closes_active_run(kanban_home):
    conn = kb.connect()
    try:
        task_id = kb.create_task(conn, title="t")
        # simulate an active run (as claim_task would set up)
        cur = conn.execute(
            "INSERT INTO task_runs (task_id, status, outcome, started_at) "
            "VALUES (?, 'running', 'running', ?)",
            (task_id, int(time.time())),
        )
        run_id = int(cur.lastrowid)
        conn.execute("UPDATE tasks SET current_run_id = ? WHERE id = ?", (run_id, task_id))
        conn.commit()
        assert kb._current_run_id(conn, task_id) == run_id
        closed = kanban_db_events._end_run(
            conn, task_id, outcome="completed", summary="done", status="done",
        )
        assert closed == run_id
        assert kb._current_run_id(conn, task_id) is None
        # second close is a no-op
        assert kanban_db_events._end_run(conn, task_id, outcome="completed") is None
        row = conn.execute("SELECT * FROM task_runs WHERE id = ?", (run_id,)).fetchone()
        assert row["status"] == "done" and row["outcome"] == "completed"
        assert row["ended_at"] is not None
    finally:
        conn.close()


def test_synthesize_ended_run_inserts_closed_run(kanban_home):
    conn = kb.connect()
    try:
        task_id = kb.create_task(conn, title="t")
        run_id = kanban_db_events._synthesize_ended_run(
            conn, task_id, outcome="completed", summary="instant",
        )
        assert run_id > 0
        row = conn.execute("SELECT * FROM task_runs WHERE id = ?", (run_id,)).fetchone()
        assert row["task_id"] == task_id
        assert row["status"] == "completed"
        assert row["summary"] == "instant"
        assert row["started_at"] == row["ended_at"]
        # the task row keeps no dangling run pointer
        assert kb._current_run_id(conn, task_id) is None
    finally:
        conn.close()
