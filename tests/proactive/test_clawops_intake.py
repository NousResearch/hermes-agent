from __future__ import annotations

from hermes_cli import kanban_db as kb
from proactive.clawops_intake import (
    create_clawops_task,
    resolve_clawops_assignee,
    subscribe_clawops_task,
)


def test_resolve_clawops_assignee_prefers_env(monkeypatch):
    monkeypatch.setenv("HERMES_CLAWOPS_ASSIGNEE", "ops-runtime")

    assert resolve_clawops_assignee({"clawops": {"default_assignee": "config-agent"}}) == "ops-runtime"


def test_resolve_clawops_assignee_falls_back_to_default_profile(monkeypatch):
    monkeypatch.delenv("HERMES_CLAWOPS_ASSIGNEE", raising=False)

    assert resolve_clawops_assignee({}) == "default"


def test_create_clawops_task_writes_task_and_created_event(tmp_path, monkeypatch):
    db_path = tmp_path / "kanban.db"
    monkeypatch.setenv("HERMES_KANBAN_DB", str(db_path))
    monkeypatch.setenv("HERMES_CLAWOPS_ASSIGNEE", "clawops-test")

    task = create_clawops_task(
        "verify proactive runtime queue",
        source={"platform": "telegram", "chat_id": "chat-1", "user_id": "kj"},
    )

    with kb.connect_closing(db_path) as conn:
        row = kb.get_task(conn, task.task_id)
        events = kb.list_events(conn, task.task_id)

    assert row is not None
    assert row.title == "ClawOps: verify proactive runtime queue"
    assert row.assignee == "clawops-test"
    assert row.status == "ready"
    assert row.created_by == "hermes-clawops-intake"
    assert "Hermes remains the primary agent" in row.body
    assert "platform: telegram" in row.body
    assert [event.kind for event in events] == ["created"]


def test_subscribe_clawops_task_writes_notify_subscription(tmp_path, monkeypatch):
    db_path = tmp_path / "kanban.db"
    monkeypatch.setenv("HERMES_KANBAN_DB", str(db_path))
    task = create_clawops_task("watch terminal update path", assignee="clawops-test")

    subscribed = subscribe_clawops_task(
        task.task_id,
        platform="telegram",
        chat_id="chat-1",
        thread_id="thread-1",
        user_id="kj",
        notifier_profile="main",
    )

    with kb.connect_closing(db_path) as conn:
        subs = kb.list_notify_subs(conn, task.task_id)

    assert subscribed is True
    assert len(subs) == 1
    assert subs[0]["platform"] == "telegram"
    assert subs[0]["chat_id"] == "chat-1"
    assert subs[0]["thread_id"] == "thread-1"
    assert subs[0]["user_id"] == "kj"
    assert subs[0]["notifier_profile"] == "main"
