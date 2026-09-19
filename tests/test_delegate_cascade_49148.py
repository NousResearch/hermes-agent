"""Regression tests for delegate-child cascade collection (#49148).

`_collect_delegate_child_ids` walks the ``_delegate_from`` marker chain to
find delegate subagents that should be cascade-deleted with their parent.
The parents themselves are deleted separately by the callers, so they must
never appear in the collected child set. A delegation cycle (or a parent
that is also another parent's delegate child) used to leak the parent into
the deletion set, permanently deleting the parent session and its messages.
"""

import json
import sqlite3

from hermes_state import _collect_delegate_child_ids, _delete_delegate_children


def _make_conn():
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.execute(
        "CREATE TABLE sessions ("
        " id TEXT PRIMARY KEY,"
        " parent_session_id TEXT,"
        " model_config TEXT)"
    )
    conn.execute("CREATE TABLE messages (session_id TEXT)")
    return conn


def _add_session(conn, sid, *, delegate_from=None, parent_session_id=None, messages=0):
    model_config = json.dumps({"_delegate_from": delegate_from}) if delegate_from else None
    conn.execute(
        "INSERT INTO sessions (id, parent_session_id, model_config) VALUES (?, ?, ?)",
        (sid, parent_session_id, model_config),
    )
    for _ in range(messages):
        conn.execute("INSERT INTO messages (session_id) VALUES (?)", (sid,))


class TestCollectDelegateChildIds:
    def test_collects_delegate_child_excludes_parent(self):
        conn = _make_conn()
        _add_session(conn, "P")
        _add_session(conn, "C", delegate_from="P")

        result = _collect_delegate_child_ids(conn, ["P"])

        assert "C" in result
        assert "P" not in result

    def test_multilevel_chain_collects_all_descendants(self):
        conn = _make_conn()
        _add_session(conn, "O")
        _add_session(conn, "A", delegate_from="O")
        _add_session(conn, "B", delegate_from="A")

        result = set(_collect_delegate_child_ids(conn, ["O"]))

        assert result == {"A", "B"}  # parent O excluded, both descendants in

    def test_parent_session_id_branch_with_marker_collected(self):
        # Second OR clause: parent_session_id match AND _delegate_from present.
        conn = _make_conn()
        _add_session(conn, "P")
        _add_session(conn, "C", parent_session_id="P", delegate_from="something")

        assert _collect_delegate_child_ids(conn, ["P"]) == ["C"]

    def test_untagged_child_not_collected(self):
        # No _delegate_from marker -> orphan-don't-delete contract.
        conn = _make_conn()
        _add_session(conn, "P")
        _add_session(conn, "C", parent_session_id="P")

        assert _collect_delegate_child_ids(conn, ["P"]) == []

    def test_cycle_terminates_and_excludes_parent(self):
        # The #49148 bug: A and B reference each other via _delegate_from.
        # Collection must terminate and never return the seed parent A.
        conn = _make_conn()
        _add_session(conn, "A", delegate_from="B")
        _add_session(conn, "B", delegate_from="A")

        result = _collect_delegate_child_ids(conn, ["A"])

        assert "A" not in result  # parent never collected as its own child
        assert result == ["B"]


class TestDeleteDelegateChildrenPreservesParent:
    def test_cycle_does_not_delete_parent_or_its_messages(self):
        conn = _make_conn()
        _add_session(conn, "A", delegate_from="B", messages=3)
        _add_session(conn, "B", delegate_from="A", messages=2)

        removed = _delete_delegate_children(conn, ["A"])

        assert "A" not in removed
        # Parent A and its messages survive; only delegate child B is gone.
        assert conn.execute("SELECT COUNT(*) FROM sessions WHERE id='A'").fetchone()[0] == 1
        assert conn.execute("SELECT COUNT(*) FROM messages WHERE session_id='A'").fetchone()[0] == 3
        assert conn.execute("SELECT COUNT(*) FROM sessions WHERE id='B'").fetchone()[0] == 0
        assert conn.execute("SELECT COUNT(*) FROM messages WHERE session_id='B'").fetchone()[0] == 0
