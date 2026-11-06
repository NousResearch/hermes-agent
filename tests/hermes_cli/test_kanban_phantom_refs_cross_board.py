"""Behavioural acceptance tests for cross-board prose-reference diagnostics."""

from __future__ import annotations

from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_boards as kbb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_diagnostics as kd


PHANTOM = "t_deadbeef1"


@pytest.fixture
def cross_board_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    # HERMES_KANBAN_HOME is the one pin kanban_home() honours unconditionally.
    # HERMES_HOME alone is NOT enough: when pytest's tmp_path itself lives inside
    # the live install (a dispatched worker's TMPDIR is under the profile scratch,
    # which is under the real root), get_default_hermes_root() sees the env home
    # *under* the native root, treats it as profile mode, and snaps the root back
    # to the LIVE install -- boards then land in the real kanban/boards/ (the
    # alpha/beta leak this fixture exists to prevent).
    monkeypatch.setenv("HERMES_KANBAN_HOME", str(home))
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    for name in (
        "HERMES_KANBAN_DB",
        "HERMES_KANBAN_BOARD",
        "HERMES_KANBAN_TASK",
        "HERMES_KANBAN_WORKSPACES_ROOT",
        "HERMES_KANBAN_ATTACHMENTS_ROOT",
        "HERMES_DELEGATED_CHILD_CONTEXT",
    ):
        monkeypatch.delenv(name, raising=False)
    kb.init_db()
    # Fail fast if any future resolver change lets a path escape the temp home --
    # tests must never write to the real kanban home.
    assert kb.kanban_home() == home
    assert kb.boards_root() == home / "kanban" / "boards"
    return home


def _create(board: str | None, title: str = "card", **kwargs) -> str:
    with kbc.connect_closing(board=board) as conn:
        return kb.create_task(conn, title=title, **kwargs)


def _complete_with_refs(board: str | None, refs: list[str], **kwargs) -> str:
    with kbc.connect_closing(board=board) as conn:
        task_id = kb.create_task(conn, title="referencing card")
        assert kb.complete_task(conn, task_id, summary="Refs: " + " ".join(refs), **kwargs)
        return task_id


def _diagnostics(board: str | None, task_id: str):
    with kbc.connect_closing(board=board) as conn:
        return kd.compute_task_diagnostics(
            kb.get_task(conn, task_id),
            kb.list_events(conn, task_id),
            kb.list_runs(conn, task_id),
        )


def _phantom_diagnostics(board: str | None, task_id: str):
    return [d for d in _diagnostics(board, task_id) if d.kind == "prose_phantom_refs"]


def test_ac1_real_cross_board_reference_has_no_diagnostic(cross_board_home):
    kb.create_board("alpha")
    kb.create_board("beta")
    referenced = _create("beta")
    referencing = _complete_with_refs("alpha", [referenced])
    assert _phantom_diagnostics("alpha", referencing) == []


def test_ac2_invented_reference_is_flagged_exactly(cross_board_home):
    kb.create_board("alpha")
    kb.create_board("beta")
    _create("beta")
    referencing = _complete_with_refs("alpha", [PHANTOM])
    diagnostics = _phantom_diagnostics("alpha", referencing)
    assert len(diagnostics) == 1
    assert diagnostics[0].data["phantom_refs"] == [PHANTOM]


def test_ac3_mixed_references_list_only_invented_id(cross_board_home):
    kb.create_board("alpha")
    kb.create_board("beta")
    referenced = _create("beta")
    referencing = _complete_with_refs("alpha", [referenced, PHANTOM])
    diagnostics = _phantom_diagnostics("alpha", referencing)
    assert len(diagnostics) == 1
    assert diagnostics[0].data["phantom_refs"] == [PHANTOM]


def test_ac4_default_and_named_boards_resolve_both_directions(cross_board_home):
    kb.create_board("named")
    default_card = _create(None, title="default card")
    named_card = _create("named", title="named card")
    named_ref = _complete_with_refs("named", [default_card])
    default_ref = _complete_with_refs(None, [named_card])
    assert _phantom_diagnostics("named", named_ref) == []
    assert _phantom_diagnostics(None, default_ref) == []


def test_ac5_reference_created_after_completion_resolves_lazily(cross_board_home, monkeypatch):
    kb.create_board("alpha")
    kb.create_board("beta")
    late_id = "t_c0ffee12"
    referencing = _complete_with_refs("alpha", [late_id])
    assert _phantom_diagnostics("alpha", referencing)[0].data["phantom_refs"] == [late_id]
    monkeypatch.setattr(kb, "_new_task_id", lambda: late_id)
    assert _create("beta", title="created later") == late_id
    assert _phantom_diagnostics("alpha", referencing) == []


def test_ac6_archived_board_resolves_but_hard_deleted_board_does_not(cross_board_home):
    kb.create_board("archived-ref")
    archived_card = _create("archived-ref")
    archived_reference = _complete_with_refs(None, [archived_card])
    kb.remove_board("archived-ref", archive=True)
    assert _phantom_diagnostics(None, archived_reference) == []

    kb.create_board("deleted-ref")
    deleted_card = _create("deleted-ref")
    deleted_reference = _complete_with_refs(None, [deleted_card])
    kb.remove_board("deleted-ref", archive=False)
    diagnostics = _phantom_diagnostics(None, deleted_reference)
    assert diagnostics[0].data["phantom_refs"] == [deleted_card]


def test_ac7_verified_created_cards_still_create_no_event_or_diagnostic(cross_board_home):
    with kbc.connect_closing() as conn:
        parent = kb.create_task(conn, title="parent", assignee="worker")
        child = kb.create_task(conn, title="child", created_by="worker")
        assert kb.complete_task(
            conn,
            parent,
            summary=f"Created {child}",
            created_cards=[child],
        )
        kinds = [event.kind for event in kb.list_events(conn, parent)]
    assert "suspected_hallucinated_references" not in kinds
    assert _phantom_diagnostics(None, parent) == []


def test_ac8_result_edit_clears_flag_and_later_edit_does_not_restore_it(cross_board_home):
    task_id = _complete_with_refs(None, [PHANTOM])
    assert len(_phantom_diagnostics(None, task_id)) == 1
    with kbc.connect_closing() as conn:
        assert kb.edit_task(conn, task_id, result="corrected")
        assert kb.edit_task(conn, task_id, title="renamed")
    assert _phantom_diagnostics(None, task_id) == []


def test_ac9_no_flagged_event_opens_no_sibling_board(cross_board_home, monkeypatch):
    kb.create_board("sibling")
    task_id = _complete_with_refs(None, ["plain prose without a task id"])
    with kbc.connect_closing() as conn:
        task = kb.get_task(conn, task_id)
        events = kb.list_events(conn, task_id)
        runs = kb.list_runs(conn, task_id)
    opens = []
    original = kbb.sqlite3.connect

    def counted_connect(*args, **kwargs):
        opens.append(args[0])
        return original(*args, **kwargs)

    monkeypatch.setattr(kbb.sqlite3, "connect", counted_connect)
    assert kd.compute_task_diagnostics(task, events, runs) == []
    assert opens == []


class _TrackedConnection:
    def __init__(self, connection, path, records):
        self._connection = connection
        self._record = {"path": path, "closed": False}
        records.append(self._record)

    def execute(self, *args, **kwargs):
        return self._connection.execute(*args, **kwargs)

    def close(self):
        self._record["closed"] = True
        return self._connection.close()


def test_ac10_each_board_opened_at_most_once_and_every_connection_closed(
    cross_board_home, monkeypatch
):
    kb.create_board("alpha")
    kb.create_board("beta")
    kb.create_board("old")
    _create("beta")
    _create("old")
    kb.remove_board("old", archive=True)
    referencing = _complete_with_refs("alpha", [PHANTOM])
    with kbc.connect_closing(board="alpha") as conn:
        task = kb.get_task(conn, referencing)
        events = kb.list_events(conn, referencing)
        runs = kb.list_runs(conn, referencing)
    records = []
    original = kbb.sqlite3.connect

    def tracked_connect(path, *args, **kwargs):
        return _TrackedConnection(original(path, *args, **kwargs), path, records)

    monkeypatch.setattr(kbb.sqlite3, "connect", tracked_connect)
    diagnostics = [
        diagnostic for diagnostic in kd.compute_task_diagnostics(task, events, runs)
        if diagnostic.kind == "prose_phantom_refs"
    ]
    assert diagnostics[0].data["phantom_refs"] == [PHANTOM]
    opened_paths = [record["path"] for record in records]
    assert len(opened_paths) == len(set(opened_paths))
    assert len(opened_paths) == 4
    assert all(record["closed"] for record in records)


def test_ac11_corrupt_sibling_does_not_raise_or_hide_unresolved_id(cross_board_home):
    kb.create_board("alpha")
    corrupt_dir = cross_board_home / "kanban" / "boards" / "corrupt"
    corrupt_dir.mkdir(parents=True)
    (corrupt_dir / "board.json").write_text("{}", encoding="utf-8")
    (corrupt_dir / "kanban.db").write_bytes(b"not a sqlite database")
    referencing = _complete_with_refs("alpha", [PHANTOM])
    diagnostics = _phantom_diagnostics("alpha", referencing)
    assert len(diagnostics) == 1
    assert diagnostics[0].data["phantom_refs"] == [PHANTOM]


def _tree_listing(root: Path):
    return sorted((str(path.relative_to(root)), path.is_dir()) for path in root.rglob("*"))


def _database_bytes(root: Path):
    return {str(path.relative_to(root)): path.read_bytes() for path in root.rglob("kanban.db")}


def test_ac12_lookup_creates_nothing_and_writes_no_database(cross_board_home):
    kb.create_board("alpha")
    kb.create_board("beta")
    referenced = _create("beta")
    before_listing = _tree_listing(cross_board_home)
    before_databases = _database_bytes(cross_board_home)
    assert kb.task_ids_on_any_board([referenced, PHANTOM]) == {referenced}
    assert _tree_listing(cross_board_home) == before_listing
    assert _database_bytes(cross_board_home) == before_databases


def test_ac13_completion_path_still_appends_one_event_and_succeeds(cross_board_home):
    task_id = _complete_with_refs(None, [PHANTOM])
    with kbc.connect_closing() as conn:
        events = [
            event for event in kb.list_events(conn, task_id)
            if event.kind == "suspected_hallucinated_references"
        ]
        assert kb.get_task(conn, task_id).status == "done"
    assert len(events) == 1


def test_ac14_diagnostics_modify_no_existing_event_rows(cross_board_home):
    task_id = _complete_with_refs(None, [PHANTOM])
    with kbc.connect_closing() as conn:
        before = list(conn.execute(
            "SELECT id, task_id, run_id, kind, payload, created_at "
            "FROM task_events WHERE task_id = ? ORDER BY id",
            (task_id,),
        ).fetchall())
        completed_before = [tuple(row) for row in before if row[3] == "completed"]
    assert len(_phantom_diagnostics(None, task_id)) == 1
    with kbc.connect_closing() as conn:
        after = list(conn.execute(
            "SELECT id, task_id, run_id, kind, payload, created_at "
            "FROM task_events WHERE task_id = ? ORDER BY id",
            (task_id,),
        ).fetchall())
        completed_after = [tuple(row) for row in after if row[3] == "completed"]
    assert [tuple(row) for row in after] == [tuple(row) for row in before]
    assert completed_after == completed_before
