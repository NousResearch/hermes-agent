"""Kanban dependency events record changes to the board (regression for #148861)."""

from contextlib import closing

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc


def test_repeating_a_link_does_not_record_another_event(tmp_path):
    with closing(kbc.connect(tmp_path / "kanban.db")) as conn:
        parent = kb.create_task(conn, title="Parent")
        child = kb.create_task(conn, title="Child")

        kb.link_tasks(conn, parent, child)
        kb.link_tasks(conn, parent, child)
        assert kb.parent_ids(conn, child) == [parent]

        assert kb.unlink_tasks(conn, parent, child)
        kb.link_tasks(conn, parent, child)
        edge = {"parent": parent, "child": child}
        assert [
            (event.kind, event.payload)
            for event in kb.list_events(conn, child)
            if event.kind in {"linked", "unlinked"}
        ] == [("linked", edge), ("unlinked", edge), ("linked", edge)]
