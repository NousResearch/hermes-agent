"""Behavioural coverage for governed Kanban task creation."""

from __future__ import annotations

import json
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from hermes_cli import kanban as cli
from hermes_cli import kanban_db as kb


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


def test_create_persists_governance_fields_and_json_exposes_them(kanban_home):
    created = json.loads(
        cli.run_slash(
            "create 'governed task' --initial-status todo "
            "--workflow-template-id valor.mission/v2 "
            "--idempotency-key valor-mission:test-governed-create "
            "--max-runtime 30m --max-retries 2 --json"
        )
    )

    assert created["status"] == "todo"
    assert created["workflow_template_id"] == "valor.mission/v2"
    assert created["idempotency_key"] == "valor-mission:test-governed-create"
    assert created["max_runtime_seconds"] == 1800
    assert created["max_retries"] == 2
    assert created["auto_promote"] is False

    with kb.connect_closing() as conn:
        assert kb.recompute_ready(conn) == 0
        assert kb.claim_task(conn, created["id"]) is None
        promoted, reason = kb.promote_task(
            conn,
            created["id"],
            actor="operator",
            reason="governance review complete",
        )
        assert promoted is True
        assert reason is None

    shown = json.loads(cli.run_slash(f"show {created['id']} --json"))["task"]
    assert shown["status"] == "ready"
    assert shown["auto_promote"] is True
    assert {
        key: value
        for key, value in shown.items()
        if key not in {"status", "auto_promote"}
    } == {
        key: value
        for key, value in created.items()
        if key not in {"status", "auto_promote"}
    }


def test_dispatch_allowlist_excludes_legacy_ready_tasks(kanban_home, monkeypatch):
    import hermes_cli.config as config
    import hermes_cli.profiles as profiles

    monkeypatch.setattr(
        config,
        "load_config_readonly",
        lambda: {
            "kanban": {
                "dispatch_workflow_template_allowlist": ["valor.mission/v2"]
            }
        },
    )
    monkeypatch.setattr(profiles, "profile_exists", lambda _name: True)

    with kb.connect_closing() as conn:
        legacy_id = kb.create_task(conn, title="legacy", assignee="valorcore")
        governed_id = kb.create_task(
            conn,
            title="governed",
            assignee="valorcore",
            workflow_template_id="valor.mission/v2",
        )

        result = kb.dispatch_once(conn, dry_run=True, reconcile_orphans=False)
        spawned_ids = [entry[0] for entry in result.spawned]

        assert spawned_ids == [governed_id]
        assert kb.get_task(conn, legacy_id).status == "ready"
        assert kb.has_spawnable_ready(conn) is True

        with kb.write_txn(conn):
            conn.execute(
                "UPDATE tasks SET status = 'done' WHERE id = ?", (governed_id,)
            )
        assert kb.has_spawnable_ready(conn) is False


def test_configured_graph_depth_is_enforced_atomically(kanban_home, monkeypatch):
    monkeypatch.setattr(kb, "_configured_graph_limits", lambda: (3, 6, 20))
    with kb.connect_closing() as conn:
        root = kb.create_task(conn, title="root")
        child = kb.create_task(conn, title="child", parents=(root,))
        grandchild = kb.create_task(conn, title="grandchild", parents=(child,))

        with pytest.raises(ValueError, match="depth limit exceeded"):
            kb.create_task(conn, title="too deep", parents=(grandchild,))

        assert kb.get_task(conn, grandchild) is not None
        assert conn.execute("SELECT COUNT(*) FROM tasks").fetchone()[0] == 3


def test_graph_fanout_and_component_size_are_enforced(kanban_home, monkeypatch):
    monkeypatch.setattr(kb, "_configured_graph_limits", lambda: (10, 2, 10))
    with kb.connect_closing() as conn:
        root = kb.create_task(conn, title="root")
        first = kb.create_task(conn, title="first", parents=(root,))
        second = kb.create_task(conn, title="second", parents=(root,))

        with pytest.raises(ValueError, match="fanout limit exceeded"):
            kb.create_task(conn, title="third", parents=(root,))

        monkeypatch.setattr(kb, "_configured_graph_limits", lambda: (10, 10, 3))
        with pytest.raises(ValueError, match="node limit exceeded"):
            kb.create_task(conn, title="fourth", parents=(second,))

        assert kb.get_task(conn, first) is not None
        assert conn.execute("SELECT COUNT(*) FROM tasks").fetchone()[0] == 3


def test_concurrent_idempotent_creators_converge(kanban_home, monkeypatch):
    barrier = threading.Barrier(2)
    original_new_task_id = kb._new_task_id

    def synchronized_task_id():
        barrier.wait(timeout=5)
        return original_new_task_id()

    monkeypatch.setattr(kb, "_new_task_id", synchronized_task_id)

    def create_once():
        with kb.connect_closing() as conn:
            return kb.create_task(
                conn,
                title="same event",
                idempotency_key="cron:test:same-event",
            )

    with ThreadPoolExecutor(max_workers=2) as executor:
        task_ids = list(executor.map(lambda _: create_once(), range(2)))

    assert task_ids[0] == task_ids[1]
    with kb.connect_closing() as conn:
        assert conn.execute("SELECT COUNT(*) FROM tasks").fetchone()[0] == 1


def test_manual_link_cannot_bypass_graph_limits(kanban_home, monkeypatch):
    monkeypatch.setattr(kb, "_configured_graph_limits", lambda: None)
    with kb.connect_closing() as conn:
        root = kb.create_task(conn, title="root")
        child = kb.create_task(conn, title="child")
        grandchild = kb.create_task(conn, title="grandchild")
        too_deep = kb.create_task(conn, title="too deep")
        kb.link_tasks(conn, root, child)
        kb.link_tasks(conn, child, grandchild)

        monkeypatch.setattr(kb, "_configured_graph_limits", lambda: (3, 6, 20))
        with pytest.raises(ValueError, match="depth limit exceeded"):
            kb.link_tasks(conn, grandchild, too_deep)

        assert kb.parent_ids(conn, too_deep) == []


def test_decomposition_cannot_bypass_graph_limits(kanban_home, monkeypatch):
    monkeypatch.setattr(kb, "_configured_graph_limits", lambda: (3, 6, 20))
    with kb.connect_closing() as conn:
        root = kb.create_task(conn, title="root", triage=True)

        with pytest.raises(ValueError, match="depth limit exceeded"):
            kb.decompose_triage_task(
                conn,
                root,
                root_assignee="default",
                children=[
                    {"title": "first", "parents": []},
                    {"title": "second", "parents": [0]},
                    {"title": "third", "parents": [1]},
                ],
            )

        assert conn.execute("SELECT COUNT(*) FROM tasks").fetchone()[0] == 1
        assert kb.get_task(conn, root).status == "triage"


def test_manual_review_decomposition_parks_children(kanban_home, monkeypatch):
    monkeypatch.setattr(kb, "_configured_graph_limits", lambda: None)
    with kb.connect_closing() as conn:
        root = kb.create_task(conn, title="root", triage=True)
        child_ids = kb.decompose_triage_task(
            conn,
            root,
            root_assignee="default",
            children=[{"title": "review me", "parents": []}],
            auto_promote=False,
        )

        child = kb.get_task(conn, child_ids[0])
        assert child.status == "todo"
        assert child.auto_promote is False
        assert kb.recompute_ready(conn) == 0
