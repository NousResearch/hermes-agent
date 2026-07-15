"""HERMES-ORCH-001C: parent notification subscription inheritance.

Covers:
- subscribed parent -> child gets exact route(s) before readiness
- unsubscribed parent invents nothing
- repeated inheritance is idempotent / duplicate-free
- sibling children each inherit
- create_task (CLI/tool/swarm), link_tasks, and decompose_triage_task paths
- preserve platform/chat/thread/user/notifier_profile; reset last_event_id
- auditable notify_subs_inherited event
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb


FULL_SHA = "e9b8ae6be137abead6d19ed8a67c523f8c527096"


@pytest.fixture
def isolated_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.delenv("HERMES_KANBAN_ADMISSION_ENFORCE", raising=False)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    return home


def _fresh_conn(board: str = "default"):
    path = kb.kanban_db_path(board=board)
    path.parent.mkdir(parents=True, exist_ok=True)
    kb._INITIALIZED_PATHS.discard(str(path.resolve()))
    return kb.connect(board=board)


def _subscribe(
    conn,
    task_id: str,
    *,
    platform: str = "telegram",
    chat_id: str = "12345",
    thread_id: str = "topic-1",
    user_id: str = "u1",
    notifier_profile: str = "default",
) -> None:
    kb.add_notify_sub(
        conn,
        task_id=task_id,
        platform=platform,
        chat_id=chat_id,
        thread_id=thread_id,
        user_id=user_id,
        notifier_profile=notifier_profile,
    )


def _route_keys(subs: list[dict]) -> list[tuple]:
    return sorted(
        (
            s["platform"],
            s["chat_id"],
            s.get("thread_id") or "",
            s.get("user_id"),
            s.get("notifier_profile"),
        )
        for s in subs
    )


def _event_kinds(conn, task_id: str) -> list[str]:
    rows = conn.execute(
        "SELECT kind FROM task_events WHERE task_id = ? ORDER BY id",
        (task_id,),
    ).fetchall()
    return [r["kind"] for r in rows]


def _latest_payload(conn, task_id: str, kind: str) -> dict:
    row = conn.execute(
        "SELECT payload FROM task_events WHERE task_id = ? AND kind = ? "
        "ORDER BY id DESC LIMIT 1",
        (task_id, kind),
    ).fetchone()
    assert row is not None, f"no event kind={kind!r}"
    return json.loads(row["payload"] or "{}")


def _valid_contract(**overrides):
    base = {
        "version": 1,
        "scope": "ORCH-001C parent subscription inheritance only",
        "allowed_files": [
            "hermes_cli/kanban_db.py",
            "tests/hermes_cli/test_kanban_notify_inheritance.py",
        ],
        "forbidden_files": ["hermes_cli/main.py"],
        "base_commit": FULL_SHA,
        "required_evidence": ["pytest output", "commit SHA"],
        "required_commands": [
            "scripts/run_tests.sh tests/hermes_cli/test_kanban_notify_inheritance.py -q"
        ],
        "allow_child_creation": False,
        "forbidden_git_actions": [
            "push",
            "merge",
            "amend",
            "reset",
            "clean",
            "restore",
            "stash",
        ],
        "notification_verified": True,
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# create_task path (covers CLI / tool / swarm which all call create_task)
# ---------------------------------------------------------------------------


def test_create_child_inherits_parent_subscription_exact_route(isolated_home):
    with _fresh_conn() as conn:
        parent = kb.create_task(conn, title="parent", assignee="orch")
        _subscribe(
            conn,
            parent,
            platform="telegram",
            chat_id="999",
            thread_id="t42",
            user_id="alice",
            notifier_profile="ops",
        )
        # Advance parent cursor so inheritance must NOT copy it.
        conn.execute(
            "UPDATE kanban_notify_subs SET last_event_id = 77 "
            "WHERE task_id = ?",
            (parent,),
        )
        conn.commit()

        child = kb.create_task(
            conn, title="child", assignee="worker", parents=[parent]
        )
        child_subs = kb.list_notify_subs(conn, child)
        assert len(child_subs) == 1
        sub = child_subs[0]
        assert sub["platform"] == "telegram"
        assert sub["chat_id"] == "999"
        assert (sub.get("thread_id") or "") == "t42"
        assert sub.get("user_id") == "alice"
        assert sub.get("notifier_profile") == "ops"
        assert int(sub.get("last_event_id") or 0) == 0

        # Inheritance happens before readiness: child is todo (parent not done)
        # but already subscribed.
        assert kb.get_task(conn, child).status == "todo"
        assert "notify_subs_inherited" in _event_kinds(conn, child)
        payload = _latest_payload(conn, child, "notify_subs_inherited")
        assert parent in payload.get("sources", [])
        assert payload.get("count") == 1


def test_unsubscribed_parent_invents_nothing(isolated_home):
    with _fresh_conn() as conn:
        parent = kb.create_task(conn, title="parent", assignee="orch")
        child = kb.create_task(
            conn, title="child", assignee="worker", parents=[parent]
        )
        assert kb.list_notify_subs(conn, child) == []
        assert "notify_subs_inherited" not in _event_kinds(conn, child)


def test_sibling_children_each_inherit(isolated_home):
    with _fresh_conn() as conn:
        parent = kb.create_task(conn, title="parent", assignee="orch")
        _subscribe(conn, parent, chat_id="c1", thread_id="th")
        a = kb.create_task(conn, title="a", assignee="w1", parents=[parent])
        b = kb.create_task(conn, title="b", assignee="w2", parents=[parent])
        assert _route_keys(kb.list_notify_subs(conn, a)) == _route_keys(
            kb.list_notify_subs(conn, b)
        )
        assert len(kb.list_notify_subs(conn, a)) == 1
        # Distinct rows per child (not shared).
        assert a != b
        assert all(s["task_id"] == a for s in kb.list_notify_subs(conn, a))
        assert all(s["task_id"] == b for s in kb.list_notify_subs(conn, b))


def test_multiple_routes_and_parents_union(isolated_home):
    with _fresh_conn() as conn:
        p1 = kb.create_task(conn, title="p1", assignee="orch")
        p2 = kb.create_task(conn, title="p2", assignee="orch")
        _subscribe(conn, p1, chat_id="c1", thread_id="t1")
        _subscribe(conn, p1, chat_id="c1", thread_id="t2")
        _subscribe(
            conn,
            p2,
            platform="discord",
            chat_id="chan",
            thread_id="",
            user_id="u2",
            notifier_profile="d-bot",
        )
        child = kb.create_task(
            conn, title="child", assignee="w", parents=[p1, p2]
        )
        keys = _route_keys(kb.list_notify_subs(conn, child))
        assert len(keys) == 3
        assert ("telegram", "c1", "t1", "u1", "default") in keys
        assert ("telegram", "c1", "t2", "u1", "default") in keys
        assert ("discord", "chan", "", "u2", "d-bot") in keys


def test_repeated_inheritance_no_duplicates(isolated_home):
    with _fresh_conn() as conn:
        parent = kb.create_task(conn, title="parent", assignee="orch")
        _subscribe(conn, parent)
        child = kb.create_task(
            conn, title="child", assignee="w", parents=[parent]
        )
        first = kb.list_notify_subs(conn, child)
        assert len(first) == 1
        # Explicit re-run (idempotent public helper).
        n = kb.inherit_notify_subs_from_parents(
            conn, child_id=child, parent_ids=[parent]
        )
        assert n == 0
        assert len(kb.list_notify_subs(conn, child)) == 1
        # Only one inheritance event from the original create.
        kinds = _event_kinds(conn, child)
        assert kinds.count("notify_subs_inherited") == 1


def test_subscribed_done_parent_enables_create_time_ready_with_contract(
    isolated_home,
):
    """Inheritance before readiness: contracted child can land ready."""
    with _fresh_conn() as conn:
        parent = kb.create_task(conn, title="parent", assignee="orch")
        _subscribe(conn, parent)
        kb.claim_task(conn, parent)
        assert kb.complete_task(conn, parent, result="ok")
        assert kb.get_task(conn, parent).status == "done"

        child = kb.create_task(
            conn,
            title="contracted-child",
            assignee="w",
            parents=[parent],
            contract=_valid_contract(),
        )
        task = kb.get_task(conn, child)
        assert task.status == "ready"
        assert kb.list_notify_subs(conn, child)
        assert "admission_rejected" not in _event_kinds(conn, child)


# ---------------------------------------------------------------------------
# link_tasks path
# ---------------------------------------------------------------------------


def test_link_tasks_inherits_from_new_parent(isolated_home):
    with _fresh_conn() as conn:
        parent = kb.create_task(conn, title="parent", assignee="orch")
        _subscribe(conn, parent, chat_id="link-chat", thread_id="L1")
        child = kb.create_task(conn, title="orphan", assignee="w")
        assert kb.list_notify_subs(conn, child) == []
        kb.link_tasks(conn, parent, child)
        subs = kb.list_notify_subs(conn, child)
        assert len(subs) == 1
        assert subs[0]["chat_id"] == "link-chat"
        assert (subs[0].get("thread_id") or "") == "L1"
        assert "notify_subs_inherited" in _event_kinds(conn, child)
        # Re-link is no-op for duplicates.
        kb.link_tasks(conn, parent, child)
        assert len(kb.list_notify_subs(conn, child)) == 1


# ---------------------------------------------------------------------------
# decompose_triage_task path
# ---------------------------------------------------------------------------


def test_decompose_children_inherit_root_subscriptions_before_ready(
    isolated_home,
):
    with _fresh_conn() as conn:
        root = kb.create_task(
            conn,
            title="root-triage",
            assignee=None,
            triage=True,
        )
        assert kb.get_task(conn, root).status == "triage"
        _subscribe(
            conn,
            root,
            platform="telegram",
            chat_id="tg-1",
            thread_id="topic-9",
            user_id="ops-user",
            notifier_profile="gateway-a",
        )
        children = [
            {"title": "leaf-a", "body": "a", "assignee": "w1", "parents": []},
            {"title": "leaf-b", "body": "b", "assignee": "w2", "parents": []},
            {
                "title": "join",
                "body": "depends on a",
                "assignee": "w3",
                "parents": [0],
            },
        ]
        child_ids = kb.decompose_triage_task(
            conn,
            root,
            root_assignee="orchestrator",
            children=children,
            author="tester",
            auto_promote=True,
        )
        assert child_ids is not None
        assert len(child_ids) == 3

        for cid in child_ids:
            subs = kb.list_notify_subs(conn, cid)
            assert len(subs) == 1, f"{cid} missing inheritance"
            s = subs[0]
            assert s["platform"] == "telegram"
            assert s["chat_id"] == "tg-1"
            assert (s.get("thread_id") or "") == "topic-9"
            assert s.get("user_id") == "ops-user"
            assert s.get("notifier_profile") == "gateway-a"
            assert int(s.get("last_event_id") or 0) == 0
            assert "notify_subs_inherited" in _event_kinds(conn, cid)

        # Parent-free leaves promote to ready and already have subs.
        a, b, join = child_ids
        assert kb.get_task(conn, a).status == "ready"
        assert kb.get_task(conn, b).status == "ready"
        assert kb.get_task(conn, join).status == "todo"


def test_decompose_unsubscribed_root_invents_nothing(isolated_home):
    with _fresh_conn() as conn:
        root = kb.create_task(conn, title="root", triage=True)
        child_ids = kb.decompose_triage_task(
            conn,
            root,
            root_assignee="orch",
            children=[{"title": "only", "assignee": "w", "parents": []}],
            author="t",
            auto_promote=True,
        )
        assert child_ids is not None
        assert kb.list_notify_subs(conn, child_ids[0]) == []
