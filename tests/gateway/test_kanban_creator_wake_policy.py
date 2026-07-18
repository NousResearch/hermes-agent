import asyncio

from gateway.config import Platform
from gateway.run import GatewayRunner
from hermes_cli import kanban_db as kb


class RecordingAdapter:
    def __init__(self):
        self.sent = []
        self.wakes = []

    async def send(self, chat_id, text, metadata=None):
        self.sent.append({"chat_id": chat_id, "text": text, "metadata": metadata or {}})

    async def handle_message(self, event):
        self.wakes.append(event)


def _make_runner(adapter):
    runner = GatewayRunner.__new__(GatewayRunner)
    runner._running = True
    runner.adapters = {Platform.TELEGRAM: adapter}
    runner._kanban_sub_fail_counts = {}
    return runner


async def _run_one_tick(monkeypatch, runner):
    real_sleep = asyncio.sleep

    async def fake_sleep(delay):
        if delay == 5:
            return None
        runner._running = False
        await real_sleep(0)

    monkeypatch.setattr(asyncio, "sleep", fake_sleep)
    await runner._kanban_notifier_watcher(interval=1)


def _create_wakeable_completion(thread_id=None):
    conn = kb.connect()
    try:
        task_id = kb.create_task(
            conn,
            title="creator wake policy",
            assignee="worker",
            session_id="agent:main:telegram:dm:chat-1",
        )
        kb.add_notify_sub(
            conn,
            task_id=task_id,
            platform="telegram",
            chat_id="chat-1",
            thread_id=thread_id,
        )
        kb.complete_task(conn, task_id, summary="finished")
        return task_id
    finally:
        conn.close()


def _configure(monkeypatch, tmp_path, *, wake_creator_session, thread_id=None):
    db_path = tmp_path / "kanban.db"
    monkeypatch.setenv("HERMES_KANBAN_DB", str(db_path))
    kb.init_db()
    _create_wakeable_completion(thread_id=thread_id)
    monkeypatch.setattr(
        "hermes_cli.config.load_config",
        lambda: {
            "kanban": {
                "dispatch_in_gateway": True,
                "wake_creator_session": wake_creator_session,
            }
        },
    )


def test_default_policy_delivers_notification_without_waking_creator(tmp_path, monkeypatch):
    _configure(monkeypatch, tmp_path, wake_creator_session=False)
    adapter = RecordingAdapter()

    asyncio.run(_run_one_tick(monkeypatch, _make_runner(adapter)))

    assert len(adapter.sent) == 1
    assert adapter.wakes == []


def test_opt_in_still_refuses_root_chat_without_distinct_thread(tmp_path, monkeypatch):
    _configure(monkeypatch, tmp_path, wake_creator_session=True, thread_id=None)
    adapter = RecordingAdapter()

    asyncio.run(_run_one_tick(monkeypatch, _make_runner(adapter)))

    assert len(adapter.sent) == 1
    assert adapter.wakes == []


def test_opt_in_wakes_only_the_explicit_thread(tmp_path, monkeypatch):
    _configure(monkeypatch, tmp_path, wake_creator_session=True, thread_id="topic-77")
    adapter = RecordingAdapter()

    asyncio.run(_run_one_tick(monkeypatch, _make_runner(adapter)))

    assert len(adapter.sent) == 1
    assert len(adapter.wakes) == 1
    assert adapter.wakes[0].internal is True
    assert adapter.wakes[0].source.thread_id == "topic-77"
