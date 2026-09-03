from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb


@pytest.fixture
def board(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    with kb.connect() as conn:
        yield conn


def test_existing_rows_migrate_to_root_task_issues(tmp_path):
    db = tmp_path / "legacy.db"
    conn = sqlite3.connect(db)
    conn.execute("CREATE TABLE tasks (id TEXT PRIMARY KEY, title TEXT NOT NULL, body TEXT, assignee TEXT, status TEXT NOT NULL, priority INTEGER DEFAULT 0, created_by TEXT, created_at INTEGER NOT NULL, started_at INTEGER, completed_at INTEGER, workspace_kind TEXT NOT NULL DEFAULT 'scratch', workspace_path TEXT, claim_lock TEXT, claim_expires INTEGER)")
    conn.execute("CREATE TABLE task_events (id INTEGER PRIMARY KEY AUTOINCREMENT, task_id TEXT NOT NULL, kind TEXT NOT NULL, payload TEXT, created_at INTEGER NOT NULL)")
    conn.execute("INSERT INTO tasks (id, title, status, created_at) VALUES ('legacy', 'Legacy', 'todo', 1)")
    conn.commit()
    conn.close()
    with kb.connect(db) as migrated:
        issue = kb.get_task(migrated, "legacy")
        columns = {row["name"] for row in migrated.execute("PRAGMA table_info(tasks)")}
    assert {"kind", "parent_id", "product_id"} <= columns
    assert (issue.kind, issue.parent_id, issue.product_id) == ("task", None, None)


def test_recursive_hierarchy_is_independent_from_dependencies(board):
    product = kb.create_task(board, title="Hermes", kind="product", product_id="product_hermes")
    project = kb.create_task(board, title="Hierarchy", kind="project", hierarchy_parent_id=product, product_id="product_hermes")
    feature = kb.create_task(board, title="Breadcrumbs", kind="feature", hierarchy_parent_id=project, product_id="product_hermes")
    assert kb.get_task(board, feature).status == "ready"
    assert kb.hierarchy_child_ids(board, product) == [project]
    assert [item.id for item in kb.issue_breadcrumbs(board, feature)] == [product, project, feature]
    blocker = kb.create_task(board, title="Design accepted")
    kb.link_tasks(board, blocker, feature)
    assert kb.get_task(board, feature).status == "todo"
    assert kb.get_task(board, feature).parent_id == project


def test_reparent_rejects_cycles_orphans_and_self_parent(board):
    root = kb.create_task(board, title="Root", kind="product")
    child = kb.create_task(board, title="Child", kind="project", hierarchy_parent_id=root)
    leaf = kb.create_task(board, title="Leaf", kind="feature", hierarchy_parent_id=child)
    for issue, parent in ((root, leaf), (root, root), (leaf, "missing")):
        with pytest.raises(ValueError):
            kb.reparent_issue(board, issue, parent)
    assert kb.get_task(board, root).parent_id is None


def test_archive_and_review_reopen_preserve_hierarchy_and_dependencies(board):
    parent = kb.create_task(board, title="Parent", kind="feature")
    blocker = kb.create_task(board, title="Blocker")
    child = kb.create_task(board, title="Child", hierarchy_parent_id=parent)
    kb.link_tasks(board, blocker, child)
    kb.archive_task(board, child)
    assert kb.get_task(board, child).parent_id == parent
    assert kb.parent_ids(board, child) == [blocker]

    review_child = kb.create_task(board, title="Review child", hierarchy_parent_id=parent)
    kb.request_review(board, review_child)
    kb.reopen_review_task(board, review_child)
    assert kb.get_task(board, review_child).parent_id == parent


def test_hierarchy_validation_and_filters_fail_closed(board):
    with pytest.raises(ValueError, match="kind"):
        kb.create_task(board, title="Bad", kind="epic")
    with pytest.raises(ValueError, match="unknown hierarchy parent"):
        kb.create_task(board, title="Orphan", hierarchy_parent_id="t_missing")
    root = kb.create_task(board, title="Hermes", kind="product", product_id="product_hermes")
    feature = kb.create_task(board, title="Hierarchy", kind="feature", hierarchy_parent_id=root, product_id="product_hermes")
    assert [x.id for x in kb.list_tasks(board, kind="feature")] == [feature]
    assert [x.id for x in kb.list_tasks(board, hierarchy_parent_id=root)] == [feature]
    assert {x.id for x in kb.list_tasks(board, product_id="product_hermes")} == {root, feature}


def test_qualified_issue_reference_resolves_read_only_with_pinned_board(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.create_board("alpha")
    kb.create_board("beta")
    with kb.connect(board="alpha") as alpha:
        issue = kb.create_task(alpha, title="Alpha issue")
    assert kb.resolve_issue_ref(f"alpha:{issue}")["task"].id == issue
    with pytest.raises(ValueError, match="qualified"):
        kb.resolve_issue_ref(issue)
    with pytest.raises(ValueError):
        kb.resolve_issue_ref(f"beta:{issue}")


def test_delete_refuses_to_orphan_hierarchy_children(board):
    parent = kb.create_task(board, title="Parent", kind="feature")
    kb.create_task(board, title="Child", hierarchy_parent_id=parent)
    with pytest.raises(ValueError, match="hierarchy children"):
        kb.delete_task(board, parent)


def test_containment_depth_boundary_is_shared_by_create_reparent_and_breadcrumbs(board):
    chain = [kb.create_task(board, title="root")]
    for depth in range(1, kb.MAX_CONTAINMENT_DEPTH + 1):
        chain.append(kb.create_task(
            board, title=f"level {depth}", hierarchy_parent_id=chain[-1],
        ))

    assert len(kb.issue_breadcrumbs(board, chain[-1])) == kb.MAX_CONTAINMENT_DEPTH + 1
    with pytest.raises(ValueError, match="maximum containment depth"):
        kb.create_task(board, title="too deep", hierarchy_parent_id=chain[-1])

    movable = kb.create_task(board, title="movable")
    with pytest.raises(ValueError, match="maximum containment depth"):
        kb.reparent_issue(board, movable, chain[-1])

    # Corrupt legacy rows are never rendered with silently truncated ancestry.
    with kb.write_txn(board):
        board.execute("UPDATE tasks SET parent_id = ? WHERE id = ?", (movable, chain[0]))
    with pytest.raises(ValueError, match="maximum containment depth"):
        kb.issue_breadcrumbs(board, chain[-1])


def test_breadcrumbs_fail_closed_for_corrupt_orphan_and_cycle_rows(board):
    root = kb.create_task(board, title="root")
    child = kb.create_task(board, title="child", hierarchy_parent_id=root)
    with kb.write_txn(board):
        board.execute("UPDATE tasks SET parent_id = 'missing' WHERE id = ?", (root,))
    with pytest.raises(ValueError, match="unknown hierarchy issue missing"):
        kb.issue_breadcrumbs(board, child)

    with kb.write_txn(board):
        board.execute("UPDATE tasks SET parent_id = ? WHERE id = ?", (child, root))
    with pytest.raises(ValueError, match="hierarchy cycle"):
        kb.issue_breadcrumbs(board, child)


def test_product_scope_ids_deliberately_allow_colon(board):
    product_id = "company:zer0:product:hermes-agent"
    task_id = kb.create_task(board, title="portable", product_id=product_id)
    assert kb.get_task(board, task_id).product_id == product_id
