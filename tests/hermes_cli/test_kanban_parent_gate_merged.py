"""Opt-in parent gate ``"parent_gate": "merged"``: a child waits for its parent's
PR to land on the base branch, not only for the parent's approval.

Measured on board `platform`, 2026-10-01: B81 was dispatched three times after
its parent B75 was approved (``done``) while B75's PR was still open, and each
run found ``apps/sdk`` missing from ``release/platform``. The default gate
(``done``/``archived``) stays for every board that does not opt in.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc

PR = "https://github.com/acme/repo/pull/7"


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    kb._parent_gate_cache.clear()
    return home


def _board_json(conn) -> Path:
    db_file = kbc._main_db_file(conn)
    assert db_file, "test DB must be a file"
    return Path(db_file).with_name("board.json")


def _opt_in(conn, value: str = "merged") -> None:
    path = _board_json(conn)
    meta = json.loads(path.read_text()) if path.exists() else {}
    meta["parent_gate"] = value
    path.write_text(json.dumps(meta))
    kb._parent_gate_cache.clear()


def _parent_done_with_pr(conn, contract: str | None = PR) -> tuple[str, str]:
    parent = kb.create_task(conn, title="parent", assignee="dev")
    child = kb.create_task(conn, title="child", assignee="dev", parents=[parent])
    assert kb.get_task(conn, child).status == "todo"
    with kbc.write_txn(conn):
        conn.execute(
            "UPDATE tasks SET status='done', completion_contract=? WHERE id=?", (contract, parent),
        )
    kb.recompute_ready(conn)
    return parent, child


def test_default_board_promotes_on_done(kanban_home):
    with kbc.connect_closing() as conn:
        _parent, child = _parent_done_with_pr(conn)
        assert kb.get_task(conn, child).status == "ready"


def test_merged_gate_holds_child_until_merged_event(kanban_home):
    with kbc.connect_closing() as conn:
        _opt_in(conn)
        parent, child = _parent_done_with_pr(conn)
        assert kb.get_task(conn, child).status == "todo"
        # The claim path re-checks parents: a child forced to `ready` is demoted.
        with kbc.write_txn(conn):
            conn.execute("UPDATE tasks SET status='ready' WHERE id=?", (child,))
        assert kb.claim_task(conn, child) is None
        assert kb.get_task(conn, child).status == "todo"

        assert kb.record_merged(conn, parent, {"pr": PR, "sha": "a" * 40}) is True
        assert kb.get_task(conn, child).status == "ready"
        # Idempotent.
        assert kb.record_merged(conn, parent) is False


def test_merged_gate_does_not_hold_a_parent_without_a_pr(kanban_home):
    with kbc.connect_closing() as conn:
        _opt_in(conn)
        for contract in (None, "local-only", "acme/repo"):
            _parent, child = _parent_done_with_pr(conn, contract)
            assert kb.get_task(conn, child).status == "ready", contract


def test_merged_gate_archived_parent_releases(kanban_home):
    with kbc.connect_closing() as conn:
        _opt_in(conn)
        parent, child = _parent_done_with_pr(conn)
        with kbc.write_txn(conn):
            conn.execute("UPDATE tasks SET status='archived' WHERE id=?", (parent,))
        kb.recompute_ready(conn)
        assert kb.get_task(conn, child).status == "ready"


def test_any_other_board_json_value_keeps_the_default_gate(kanban_home):
    with kbc.connect_closing() as conn:
        for value in ("done", "", "MERGED"):
            _opt_in(conn, value)
            _parent, child = _parent_done_with_pr(conn)
            assert kb.get_task(conn, child).status == "ready", value


def test_malformed_board_json_keeps_the_default_gate(kanban_home):
    with kbc.connect_closing() as conn:
        _board_json(conn).write_text("{not json")
        kb._parent_gate_cache.clear()
        _parent, child = _parent_done_with_pr(conn)
        assert kb.get_task(conn, child).status == "ready"


def test_link_to_a_done_but_unmerged_parent_demotes_the_child(kanban_home):
    """`link` uses the SAME terminal predicate as the claim: on a merged-gate
    board, linking a `ready` child under a `done` parent whose PR has not
    merged demotes the child to `todo` (2026-10-02: t_b7354264 under B45 was
    left `ready` with PR #182 still open)."""
    with kbc.connect_closing() as conn:
        _opt_in(conn)
        parent = kb.create_task(conn, title="parent", assignee="dev")
        with kbc.write_txn(conn):
            conn.execute(
                "UPDATE tasks SET status='done', completion_contract=? WHERE id=?", (PR, parent),
            )
        child = kb.create_task(conn, title="child", assignee="dev")
        assert kb.get_task(conn, child).status == "ready"
        assert kb.link_tasks(conn, parent, child) is True
        assert kb.get_task(conn, child).status == "todo"

        kb.record_merged(conn, parent, {"pr": PR, "sha": "b" * 40})
        assert kb.get_task(conn, child).status == "ready"

        # A merged parent gates nothing: a second child linked now stays ready.
        other = kb.create_task(conn, title="other", assignee="dev")
        assert kb.link_tasks(conn, parent, other) is False
        assert kb.get_task(conn, other).status == "ready"


def test_record_merged_unknown_task(kanban_home):
    with kbc.connect_closing() as conn:
        assert kb.record_merged(conn, "t_nope") is False
