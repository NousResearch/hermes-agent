from types import SimpleNamespace

import pytest


class _Adapter:
    def __init__(self):
        self.calls = []

    async def send(self, chat_id, msg, metadata=None):
        self.calls.append((chat_id, msg, metadata))


def test_build_kanban_session_event_text_matches_watcher_wording():
    from gateway.run import GatewayRunner

    text = GatewayRunner._build_kanban_session_event_text(
        board="ts-editor",
        task_id="t_abc123",
        status="Blocked",
    )

    assert text == (
        "[KANBAN_WATCHER_EVENT] board=ts-editor task_id=t_abc123 moved to status=Blocked\n"
        "This is a synthetic watcher event, not a new user instruction: "
        "follow prior user instructions about how to handle Kanban watcher "
        "events in this conversation. If none apply, provide a concise "
        "summary of the task's outcome, and do nothing else.\n"
        "You may inspect the task using `hermes kanban --board ts-editor show t_abc123`."
    )


def test_kanban_status_from_event_kind_titleizes_non_completed_values():
    from gateway.run import GatewayRunner

    assert GatewayRunner._kanban_status_from_event_kind("completed") == "Done"
    assert GatewayRunner._kanban_status_from_event_kind("blocked") == "Blocked"
    assert GatewayRunner._kanban_status_from_event_kind("timed_out") == "Timed Out"


@pytest.mark.asyncio
async def test_session_event_delivery_sends_normal_notification_before_enqueue(monkeypatch):
    from gateway.run import GatewayRunner

    runner = GatewayRunner.__new__(GatewayRunner)
    adapter = _Adapter()
    order = []

    async def _fake_artifacts(**kwargs):
        order.append("artifacts")

    def _fake_enqueue(*, board, task_id, status, session_key):
        order.append(("enqueue", board, task_id, status, session_key))
        return True

    monkeypatch.setattr(runner, "_deliver_kanban_artifacts", _fake_artifacts)
    monkeypatch.setattr(runner, "_enqueue_kanban_session_event", _fake_enqueue)

    sub = {
        "task_id": "t_blocked",
        "chat_id": "chat-1",
        "thread_id": "thread-9",
        "session_key": "agent:main:discord:thread:chat-1:thread-9",
    }
    task = SimpleNamespace(title="Hard reset spec", assignee="coder", result=None)
    ev = SimpleNamespace(kind="blocked", payload={"reason": "waiting for review"})

    ok = await runner._deliver_kanban_session_event(
        adapter=adapter,
        sub=sub,
        task=task,
        ev=ev,
        board_slug="ts-editor",
    )

    assert ok is True
    assert adapter.calls == [
        (
            "chat-1",
            "⏸ @coder Kanban t_blocked blocked: waiting for review",
            {"thread_id": "thread-9"},
        )
    ]
    assert order == [
        ("enqueue", "ts-editor", "t_blocked", "Blocked", "agent:main:discord:thread:chat-1:thread-9")
    ]


@pytest.mark.asyncio
async def test_session_event_delivery_keeps_success_when_enqueue_fails_after_normal_notification(monkeypatch):
    from gateway.run import GatewayRunner

    runner = GatewayRunner.__new__(GatewayRunner)
    adapter = _Adapter()

    async def _fake_artifacts(**kwargs):
        raise AssertionError("artifacts should not run for blocked events")

    monkeypatch.setattr(runner, "_deliver_kanban_artifacts", _fake_artifacts)
    monkeypatch.setattr(runner, "_enqueue_kanban_session_event", lambda **kwargs: False)

    sub = {
        "task_id": "t_blocked",
        "chat_id": "chat-1",
        "thread_id": "thread-9",
        "session_key": "agent:main:discord:thread:chat-1:thread-9",
    }
    task = SimpleNamespace(title="Hard reset spec", assignee="coder", result=None)
    ev = SimpleNamespace(kind="blocked", payload={"reason": "waiting for review"})

    ok = await runner._deliver_kanban_session_event(
        adapter=adapter,
        sub=sub,
        task=task,
        ev=ev,
        board_slug="ts-editor",
    )

    assert ok is True
    assert adapter.calls == [
        (
            "chat-1",
            "⏸ @coder Kanban t_blocked blocked: waiting for review",
            {"thread_id": "thread-9"},
        )
    ]
