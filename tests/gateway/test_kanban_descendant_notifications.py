import asyncio

from gateway.config import Platform
from gateway.run import GatewayRunner
from hermes_cli import kanban_db as kb


class RecordingAdapter:
    def __init__(self, *, fail=False):
        self.sent = []
        self.fail = fail

    async def send(self, chat_id, text, metadata=None):
        if self.fail:
            raise RuntimeError("temporary telegram failure")
        self.sent.append((chat_id, text, metadata or {}))
        return type("Receipt", (), {"message_id": "tg-42"})()


async def _tick(monkeypatch, runner):
    real_sleep = asyncio.sleep

    async def sleep(delay):
        if delay == 5:
            return
        runner._running = False
        await real_sleep(0)

    monkeypatch.setattr(asyncio, "sleep", sleep)
    await runner._kanban_notifier_watcher(interval=1)


def _runner(adapter):
    runner = GatewayRunner.__new__(GatewayRunner)
    runner._running = True
    runner.adapters = {Platform.TELEGRAM: adapter}
    runner._kanban_sub_fail_counts = {}
    return runner


def _graph(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_KANBAN_DB", str(tmp_path / "board.db"))
    kb.init_db()
    conn = kb.connect()
    root = kb.create_task(conn, title="Root project", assignee="orchestrator", tenant="acme")
    child = kb.create_task(conn, title="Configure production", assignee="engineer", tenant="acme")
    kb.link_tasks(conn, root, child)
    # Model a child already claimed by a worker. The parent/child edge normally
    # comes from decomposition before the worker reaches this state.
    conn.execute("UPDATE tasks SET status='running' WHERE id=?", (child,))
    kb.add_notify_sub(conn, task_id=root, platform="telegram", chat_id="chat", thread_id="topic")
    return conn, root, child


def test_child_blocker_rolls_up_and_delivers_once(tmp_path, monkeypatch):
    conn, root, child = _graph(tmp_path, monkeypatch)
    kb.block_task(conn, child, reason="Which region should be used?", kind="needs_input")
    rollup = kb.task_rollup(conn, root)
    assert rollup["effective_status"] == "blocked"
    assert rollup["actionable_blocker"]["task_id"] == child
    conn.close()

    adapter = RecordingAdapter()
    asyncio.run(_tick(monkeypatch, _runner(adapter)))
    assert len(adapter.sent) == 1
    assert adapter.sent[0][2]["thread_id"] == "topic"
    assert "Which region" in adapter.sent[0][1]
    assert "Configure production" in adapter.sent[0][1]

    conn = kb.connect()
    outbox = kb.list_notify_outbox(conn)
    assert [(r["state"], r["message_receipt"], r["task_id"]) for r in outbox] == [("sent", "tg-42", child)]
    conn.close()

    asyncio.run(_tick(monkeypatch, _runner(adapter)))
    assert len(adapter.sent) == 1


def test_changed_blocker_gets_new_version_and_child_success_is_silent(tmp_path, monkeypatch):
    conn, root, child = _graph(tmp_path, monkeypatch)
    kb.block_task(conn, child, reason="first question", kind="needs_input")
    conn.close()
    adapter = RecordingAdapter()
    asyncio.run(_tick(monkeypatch, _runner(adapter)))

    conn = kb.connect()
    kb.unblock_task(conn, child)
    conn.execute("UPDATE tasks SET status='running' WHERE id=?", (child,))
    kb.block_task(conn, child, reason="different question", kind="capability")
    conn.close()
    asyncio.run(_tick(monkeypatch, _runner(adapter)))
    assert len(adapter.sent) == 2
    assert "different question" in adapter.sent[-1][1]


def test_failed_send_is_durable_and_tenant_is_stamped(tmp_path, monkeypatch):
    conn, _, child = _graph(tmp_path, monkeypatch)
    kb.block_task(conn, child, reason="need approval", kind="needs_input")
    conn.close()
    asyncio.run(_tick(monkeypatch, _runner(RecordingAdapter(fail=True))))
    conn = kb.connect()
    rows = kb.list_notify_outbox(conn)
    assert len(rows) == 1
    assert rows[0]["state"] == "failed"
    assert rows[0]["attempts"] == 1
    assert rows[0]["next_attempt_at"] > rows[0]["updated_at"]
    assert rows[0]["tenant"] == "acme"
    conn.close()


def test_failed_send_retries_after_restart_when_backoff_is_due(tmp_path, monkeypatch):
    conn, _, child = _graph(tmp_path, monkeypatch)
    kb.block_task(conn, child, reason="retry this question", kind="needs_input")
    conn.close()
    asyncio.run(_tick(monkeypatch, _runner(RecordingAdapter(fail=True))))

    conn = kb.connect()
    conn.execute("UPDATE kanban_notify_outbox SET next_attempt_at=0 WHERE task_id=?", (child,))
    conn.close()

    replacement_adapter = RecordingAdapter()
    asyncio.run(_tick(monkeypatch, _runner(replacement_adapter)))
    assert len(replacement_adapter.sent) == 1
    conn = kb.connect()
    row = kb.list_notify_outbox(conn)[0]
    assert (row["state"], row["attempts"], row["message_receipt"]) == ("sent", 2, "tg-42")
    conn.close()


def test_correlated_reply_comments_unblocks_and_acknowledges(tmp_path, monkeypatch):
    conn, _, child = _graph(tmp_path, monkeypatch)
    kb.block_task(conn, child, reason="Which region?", kind="needs_input")
    conn.close()
    asyncio.run(_tick(monkeypatch, _runner(RecordingAdapter())))

    conn = kb.connect()
    answered = kb.acknowledge_notify_reply(
        conn, platform="telegram", chat_id="chat", thread_id="topic",
        receipt="tg-42", reply="Use eu-west-1", user_id="sam",
    )
    assert answered == child
    task = kb.get_task(conn, child)
    assert task is not None
    assert task.status == "todo"  # dependency remains; the blocker itself is cleared
    assert kb.list_notify_outbox(conn)[0]["state"] == "acknowledged"
    assert kb.list_comments(conn, child)[-1].body == "Use eu-west-1"
    assert kb.acknowledge_notify_reply(
        conn, platform="telegram", chat_id="other", thread_id="topic",
        receipt="tg-42", reply="wrong chat",
    ) is None
    conn.close()


def test_root_completion_notifies_but_child_completion_does_not(tmp_path, monkeypatch):
    conn, root, child = _graph(tmp_path, monkeypatch)
    kb.complete_task(conn, child, summary="child done")
    conn.close()
    adapter = RecordingAdapter()
    asyncio.run(_tick(monkeypatch, _runner(adapter)))
    assert adapter.sent == []

    conn = kb.connect()
    kb.complete_task(conn, root, summary="all done")
    conn.close()
    asyncio.run(_tick(monkeypatch, _runner(adapter)))
    assert len(adapter.sent) == 1
    assert "all done" in adapter.sent[0][1]


def test_cross_tenant_linkage_does_not_roll_up(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_KANBAN_DB", str(tmp_path / "isolation.db"))
    kb.init_db()
    conn = kb.connect()
    root = kb.create_task(conn, title="A", assignee="x", tenant="a")
    child = kb.create_task(conn, title="B", assignee="x", tenant="b")
    # Simulate a malformed legacy link; lineage queries must still isolate.
    conn.execute("INSERT INTO task_links(parent_id, child_id) VALUES (?, ?)", (root, child))
    kb.block_task(conn, child, reason="secret tenant question", kind="needs_input")
    assert kb.task_rollup(conn, root)["actionable_blocker"] is None
    conn.close()
