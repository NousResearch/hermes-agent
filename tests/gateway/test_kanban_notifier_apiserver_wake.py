"""Kanban notifier behavior on stateless (api_server) subscriptions.

Covers the wrong-session-wake / silent-loss fixes:
* a SendResult(success=False) return (the API server's send() stub) rewinds
  the cursor instead of advancing past a never-delivered event;
* api_server subscriptions wake the creator's REAL session via the
  /v1/chat/completions self-post (raw task.session_id), never via
  handle_message (which would run under a build_session_key()-derived key
  that never matches the raw X-Hermes-Session-Id session real turns use).
"""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

from gateway.config import Platform
from gateway.platforms.base import MessageEvent, MessageType, SendResult
from gateway.run import GatewayRunner
from gateway.session import SessionSource
from hermes_cli import kanban_db as kb


class SoftFailAdapter:
    """Push-capable adapter whose send() returns SendResult(success=False)
    WITHOUT raising — previously treated as delivered (event lost)."""

    def __init__(self):
        self.attempts = 0

    async def send(self, chat_id, text, metadata=None):
        self.attempts += 1
        return SendResult(success=False, error="soft failure")


class ApiServerLikeAdapter:
    supports_async_delivery = False

    def __init__(self):
        self._host = "127.0.0.1"
        self._port = 8642
        self._api_key = "k"
        self._model_name = "hermes"
        self.handle_message_calls = []
        self.send_calls = 0

    async def send(self, chat_id, text, metadata=None):
        self.send_calls += 1
        return SendResult(
            success=False,
            error="API server uses HTTP request/response, not send()",
        )

    async def handle_message(self, event):
        self.handle_message_calls.append(event)


async def _run_one_notifier_tick(monkeypatch, runner):
    real_sleep = asyncio.sleep

    async def fake_sleep(delay):
        if delay == 5:
            return None
        runner._running = False
        await real_sleep(0)

    monkeypatch.setattr(asyncio, "sleep", fake_sleep)
    await runner._kanban_notifier_watcher(interval=1)


def _make_runner(adapters):
    runner = GatewayRunner.__new__(GatewayRunner)
    runner._running = True
    runner.adapters = adapters
    runner._kanban_sub_fail_counts = {}
    runner._kanban_dispatcher_lock_handle = object()
    return runner


def _bind_session_store(runner, session_id):
    store = object()
    runner.session_store = store
    runner._async_session_store = SimpleNamespace(
        _store=store,
        get_or_create_session=AsyncMock(
            return_value=SimpleNamespace(session_id=session_id)
        ),
    )


def _create_completed_subscription(platform, chat_id, session_id=None):
    conn = kb.connect()
    try:
        tid = kb.create_task(
            conn, title="notify once", assignee="worker", session_id=session_id,
        )
        kb.add_notify_sub(conn, task_id=tid, platform=platform, chat_id=chat_id)
        kb.complete_task(conn, tid, summary="done once")
        return tid
    finally:
        conn.close()


def _unseen_terminal_events(tid, platform, chat_id):
    conn = kb.connect()
    try:
        _, events = kb.unseen_events_for_sub(
            conn,
            task_id=tid,
            platform=platform,
            chat_id=chat_id,
            kinds=["completed", "blocked", "gave_up", "crashed", "timed_out"],
        )
        return events
    finally:
        conn.close()


def test_apiserver_sub_wakes_real_session_via_self_post(tmp_path, monkeypatch):
    """An api_server subscription wakes the creator's REAL session by
    self-posting with the task's raw session_id — never handle_message (which
    would run the wake under a build_session_key()-derived key that can't
    match the raw X-Hermes-Session-Id session)."""
    monkeypatch.setenv("HERMES_KANBAN_DB", str(tmp_path / "apiserver.db"))
    kb.init_db()
    tid = _create_completed_subscription(
        "api_server", "raw-sid-123", session_id="raw-sid-123",
    )

    posts = []

    async def fake_self_post(adapter, *, text, session_id):
        posts.append({"text": text, "session_id": session_id})

    import gateway.wake as wake_mod

    monkeypatch.setattr(wake_mod, "_self_post_chat_completion", fake_self_post)

    adapter = ApiServerLikeAdapter()
    runner = _make_runner({Platform.API_SERVER: adapter})
    asyncio.run(_run_one_notifier_tick(monkeypatch, runner))

    assert adapter.handle_message_calls == [], (
        "api_server wake must not go through handle_message (wrong-session bug)"
    )
    assert len(posts) == 1
    assert posts[0]["session_id"] == "raw-sid-123"
    assert tid in posts[0]["text"]
    assert "done once" in posts[0]["text"]
    assert "ask what they would like to move to next" in posts[0]["text"]
    assert "Suggest 2 or 3 concrete candidates" in posts[0]["text"]
    # The wake self-post IS the delivery on this path (no separate text-ping
    # fallback is attempted for stateless api_server subs) — cursor advances
    # once the wake succeeds.
    assert _unseen_terminal_events(tid, "api_server", "raw-sid-123") == []


def test_gateway_slash_create_binds_session_and_wakes_it(tmp_path, monkeypatch):
    """Exercise slash create -> persisted task binding -> completion wake."""
    monkeypatch.setenv("HERMES_KANBAN_DB", str(tmp_path / "slash-create.db"))
    kb.init_db()

    adapter = ApiServerLikeAdapter()
    runner = _make_runner({Platform.API_SERVER: adapter})
    _bind_session_store(runner, "gateway-durable-session")
    runner._kanban_notifier_profile = "default"
    source = SessionSource(
        platform=Platform.API_SERVER,
        chat_id="gateway-durable-session",
        chat_type="dm",
        thread_id=None,
        user_id="api-user",
    )
    event = MessageEvent(
        text='/kanban create "slash wake proof" --assignee worker',
        message_type=MessageType.TEXT,
        source=source,
        message_id="request-1",
    )

    output = asyncio.run(runner._handle_kanban_command(event))
    assert "subscribed" in output.lower()

    conn = kb.connect()
    try:
        task = kb.list_tasks(conn)[0]
        assert task.session_id == "gateway-durable-session"
        kb.complete_task(conn, task.id, summary="slash path complete")
    finally:
        conn.close()

    posts = []

    async def fake_self_post(_adapter, *, text, session_id):
        posts.append({"text": text, "session_id": session_id})

    import gateway.wake as wake_mod

    monkeypatch.setattr(wake_mod, "_self_post_chat_completion", fake_self_post)
    asyncio.run(_run_one_notifier_tick(monkeypatch, runner))

    assert len(posts) == 1
    assert posts[0]["session_id"] == "gateway-durable-session"
    assert task.id in posts[0]["text"]
    assert "slash path complete" in posts[0]["text"]

