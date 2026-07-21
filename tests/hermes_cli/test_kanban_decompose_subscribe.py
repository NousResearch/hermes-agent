"""Kanban decomposition notification subscription inheritance tests."""

from __future__ import annotations

import time

from hermes_cli import kanban_db as kb


def _insert_triage_task(conn, task_id: str = "t_parent") -> str:
    now = int(time.time())
    conn.execute(
        """
        INSERT INTO tasks
            (id, title, body, assignee, status, created_by, created_at,
             workspace_kind, tenant)
        VALUES (?, ?, ?, ?, 'triage', ?, ?, 'scratch', ?)
        """,
        (task_id, "Parent", "Root body", "profile-pmo", "tester", now, "demo"),
    )
    conn.commit()
    return task_id


def _subscribe(conn, task_id: str) -> None:
    kb.add_notify_sub(
        conn,
        task_id=task_id,
        platform="discord",
        chat_id="1523615301965447298",
        thread_id="1527382896757964971",
        user_id="458519787346198530",
        notifier_profile="profile-pmo",
    )


def test_decompose_copies_parent_notify_subscriptions_to_children(tmp_path):
    conn = kb.connect(tmp_path / "kanban.db")
    try:
        parent_id = _insert_triage_task(conn)
        _subscribe(conn, parent_id)

        child_ids = kb.decompose_triage_task(
            conn,
            parent_id,
            root_assignee="profile-pmo",
            children=[
                {"title": "Child one", "body": "one", "assignee": "profile-dev", "parents": []},
                {"title": "Child two", "body": "two", "assignee": "profile-qa", "parents": [0]},
            ],
            author="profile-pmo",
            auto_promote=False,
        )

        assert child_ids is not None
        assert len(child_ids) == 2
        for child_id in child_ids:
            rows = kb.list_notify_subs(conn, child_id)
            assert rows == [
                {
                    "task_id": child_id,
                    "platform": "discord",
                    "chat_id": "1523615301965447298",
                    "thread_id": "1527382896757964971",
                    "user_id": "458519787346198530",
                    "notifier_profile": "profile-pmo",
                    "created_at": rows[0]["created_at"],
                    "last_event_id": 0,
                }
            ]
    finally:
        conn.close()


def test_decompose_respects_auto_subscribe_false(tmp_path):
    conn = kb.connect(tmp_path / "kanban.db")
    try:
        parent_id = _insert_triage_task(conn)
        _subscribe(conn, parent_id)

        child_ids = kb.decompose_triage_task(
            conn,
            parent_id,
            root_assignee="profile-pmo",
            children=[{"title": "Child", "body": "one", "assignee": "profile-dev", "parents": []}],
            author="profile-pmo",
            auto_promote=False,
            auto_subscribe=False,
        )

        assert child_ids is not None
        assert kb.list_notify_subs(conn, child_ids[0]) == []
    finally:
        conn.close()


def test_copy_parent_subscriptions_to_children_is_idempotent(tmp_path):
    conn = kb.connect(tmp_path / "kanban.db")
    try:
        parent_id = _insert_triage_task(conn)
        _subscribe(conn, parent_id)
        child_id = "t_child"
        _insert_triage_task(conn, child_id)

        assert kb._copy_parent_subscriptions_to_children(
            conn, parent_task_id=parent_id, child_ids=[child_id]
        ) == 1
        assert kb._copy_parent_subscriptions_to_children(
            conn, parent_task_id=parent_id, child_ids=[child_id]
        ) == 0
        assert len(kb.list_notify_subs(conn, child_id)) == 1
    finally:
        conn.close()


def test_create_task_with_parent_inherits_notify_subscription(tmp_path):
    conn = kb.connect(tmp_path / "kanban.db")
    try:
        parent_id = _insert_triage_task(conn)
        _subscribe(conn, parent_id)

        child_id = kb.create_task(
            conn,
            title="Child from worker",
            body="Needs human gate",
            assignee="profile-pmo",
            created_by="profile-pmo",
            parents=[parent_id],
        )

        rows = kb.list_notify_subs(conn, child_id)
        assert len(rows) == 1
        assert rows[0]["platform"] == "discord"
        assert rows[0]["chat_id"] == "1523615301965447298"
        assert rows[0]["notifier_profile"] == "profile-pmo"
    finally:
        conn.close()


def test_link_tasks_copies_notify_subscription_once(tmp_path):
    conn = kb.connect(tmp_path / "kanban.db")
    try:
        parent_id = _insert_triage_task(conn, "t_parent")
        child_id = _insert_triage_task(conn, "t_linked_child")
        _subscribe(conn, parent_id)

        kb.link_tasks(conn, parent_id, child_id)
        kb.link_tasks(conn, parent_id, child_id)

        assert len(kb.list_notify_subs(conn, child_id)) == 1
    finally:
        conn.close()
