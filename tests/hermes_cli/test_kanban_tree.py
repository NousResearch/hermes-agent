"""Tests for the kanban task-tree renderer (hermes_cli.kanban_tree + the tree CLI verb)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from hermes_cli import kanban as kc
from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_tree as kt


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


def _seed_parent_child(parent: str = "parent", child: str = "child") -> tuple[str, str]:
    with kbc.connect_closing() as conn:
        parent_id = kb.create_task(conn, title=parent)
        child_id = kb.create_task(conn, title=child)
        kb.link_tasks(conn, parent_id=parent_id, child_id=child_id)
        return parent_id, child_id


# ---------------------------------------------------------------------------
# build_forest — shape + scoping
# ---------------------------------------------------------------------------


def test_build_forest_nests_child_under_parent(kanban_home):
    parent_id, child_id = _seed_parent_child()

    with kbc.connect_closing() as conn:
        forest = kt.build_forest(conn)

    assert len(forest) == 1
    root = forest[0]
    assert root["id"] == parent_id
    assert root["title"] == "parent"
    assert [c["id"] for c in root["children"]] == [child_id]


def test_build_forest_root_subtree_scoped(kanban_home):
    parent_id, child_id = _seed_parent_child()
    # An unrelated task that must NOT appear when the subtree is scoped to parent.
    with kbc.connect_closing() as conn:
        other_id = kb.create_task(conn, title="unrelated")

    with kbc.connect_closing() as conn:
        forest = kt.build_forest(conn, root_id=parent_id)

    all_ids = {root["id"] for root in forest}
    def walk(nodes, acc):
        for n in nodes:
            acc.add(n["id"])
            walk(n["children"], acc)
        return acc
    found = walk(forest, set())

    assert found == {parent_id, child_id}
    assert other_id not in found


def test_build_forest_multiple_roots_are_forest(kanban_home):
    with kbc.connect_closing() as conn:
        a = kb.create_task(conn, title="a")
        b = kb.create_task(conn, title="b")  # independent root (no parent)

    with kbc.connect_closing() as conn:
        forest = kt.build_forest(conn)

    assert {r["id"] for r in forest} == {a, b}


def test_build_forest_unknown_root_raises(kanban_home):
    with kbc.connect_closing() as conn:
        with pytest.raises(ValueError, match="no such task"):
            kt.build_forest(conn, root_id="t_doesnotexist")


def test_build_forest_archived_root_needs_archived_flag(kanban_home):
    with kbc.connect_closing() as conn:
        tid = kb.create_task(conn, title="doomed")
        kb.archive_task(conn, tid)

    with kbc.connect_closing() as conn:
        with pytest.raises(ValueError):
            kt.build_forest(conn, root_id=tid)
        # With --archived-equivalent, the task resolves.
        forest = kt.build_forest(conn, root_id=tid, include_archived=True)
    assert forest[0]["id"] == tid


# ---------------------------------------------------------------------------
# Cycle termination (defensive — DB normally forbids cycles)
# ---------------------------------------------------------------------------


def test_build_forest_terminates_on_cycle(kanban_home):
    with kbc.connect_closing() as conn:
        a = kb.create_task(conn, title="a")
        b = kb.create_task(conn, title="b")
        # Write a raw self-cycle edge, bypassing the DB's would_cycle guard.
        conn.execute("INSERT OR IGNORE INTO task_links (parent_id, child_id) VALUES (?, ?)", (a, a))
        conn.execute("INSERT OR IGNORE INTO task_links (parent_id, child_id) VALUES (?, ?)", (b, a))
        conn.commit()

    with kbc.connect_closing() as conn:
        forest = kt.build_forest(conn)  # must not hang / recurse forever

    assert isinstance(forest, list)


# ---------------------------------------------------------------------------
# render_ascii
# ---------------------------------------------------------------------------


def test_render_ascii_shows_lineage(kanban_home):
    parent_id, child_id = _seed_parent_child("Original epic", "subtask A")

    with kbc.connect_closing() as conn:
        out = kt.render_ascii(kt.build_forest(conn))

    assert parent_id in out
    assert child_id in out
    assert "subtask A" in out
    # The child is indented under the parent (lineage visible), id right after branch.
    assert f"└─ {child_id}" in out or f"├─ {child_id}" in out


def test_render_ascii_multiple_children_uses_continuation(kanban_home):
    with kbc.connect_closing() as conn:
        root = kb.create_task(conn, title="root")
        mid = kb.create_task(conn, title="mid")
        leaf = kb.create_task(conn, title="leaf")
        pal = kb.create_task(conn, title="pal")
        kb.link_tasks(conn, parent_id=root, child_id=mid)
        kb.link_tasks(conn, parent_id=mid, child_id=leaf)
        kb.link_tasks(conn, parent_id=root, child_id=pal)

    with kbc.connect_closing() as conn:
        out = kt.render_ascii(kt.build_forest(conn))

    # mid is the first of two children -> its subtree continues under "├─"; a
    # "│" continuation glyph marks the sibling that follows.
    assert out.index(mid) < out.index(leaf) < out.index(pal)
    assert "│" in out


# ---------------------------------------------------------------------------
# render_json
# ---------------------------------------------------------------------------


def test_render_json_parses_and_nests(kanban_home):
    parent_id, child_id = _seed_parent_child()

    with kbc.connect_closing() as conn:
        raw = kt.render_json(kt.build_forest(conn))

    forest = json.loads(raw)
    assert isinstance(forest, list)
    assert forest[0]["id"] == parent_id
    assert forest[0]["children"][0]["id"] == child_id
    assert "title" in forest[0]
    assert "status" in forest[0]


# ---------------------------------------------------------------------------
# render_mermaid
# ---------------------------------------------------------------------------


def test_render_mermaid_edges_follow_lineage(kanban_home):
    parent_id, child_id = _seed_parent_child()

    with kbc.connect_closing() as conn:
        out = kt.render_mermaid(kt.build_forest(conn))

    assert out.startswith("flowchart TD")
    assert f"{parent_id} --> {child_id}" in out
    assert f'"{parent_id}:' in out


def test_render_mermaid_escapes_label_injection(kanban_home):
    with kbc.connect_closing() as conn:
        tid = kb.create_task(conn, title='evil"]; alert(1); //"')

    with kbc.connect_closing() as conn:
        out = kt.render_mermaid(kt.build_forest(conn, root_id=tid))

    # Quotes and brackets are escaped so the label cannot break out of the node.
    assert '\\"' in out
    assert "\\]" in out
    assert '"]; alert(1); //' not in out


# ---------------------------------------------------------------------------
# CLI surface (run_slash — shared by CLI and gateway)
# ---------------------------------------------------------------------------


def test_cli_tree_text(kanban_home):
    parent_id, child_id = _seed_parent_child()
    out = kc.run_slash("tree")
    assert child_id in out
    assert parent_id in out


def test_cli_tree_json(kanban_home):
    parent_id, child_id = _seed_parent_child()
    out = kc.run_slash("tree --json")
    forest = json.loads(out)
    assert forest[0]["id"] == parent_id
    assert forest[0]["children"][0]["id"] == child_id


def test_cli_tree_mermaid(kanban_home):
    parent_id, child_id = _seed_parent_child()
    out = kc.run_slash("tree --mermaid")
    assert out.startswith("flowchart TD")
    assert f"{parent_id} --> {child_id}" in out


def test_cli_tree_scoped_subtree(kanban_home):
    parent_id, child_id = _seed_parent_child()
    out = kc.run_slash(f"tree {parent_id}")
    assert child_id in out
    assert parent_id in out