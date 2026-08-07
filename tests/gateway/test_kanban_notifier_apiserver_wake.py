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

from gateway.config import Platform
from gateway.platforms.base import SendResult
from gateway.run import GatewayRunner
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


async def _run_notifier_ticks(monkeypatch, runner, *, ticks=1):
    real_sleep = asyncio.sleep
    completed_ticks = 0

    async def fake_sleep(delay):
        nonlocal completed_ticks
        if delay == 5:
            return None
        completed_ticks += 1
        if completed_ticks >= ticks:
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

    async def fake_deliver_wake(
        adapter, *, text, session_id, source=None, producer_identity=None,
    ):
        assert source is None
        posts.append({
            "text": text,
            "session_id": session_id,
            "producer_identity": producer_identity,
        })

    import gateway.wake as wake_mod

    monkeypatch.setattr(wake_mod, "deliver_wake", fake_deliver_wake)

    adapter = ApiServerLikeAdapter()
    runner = _make_runner({Platform.API_SERVER: adapter})
    asyncio.run(_run_notifier_ticks(monkeypatch, runner))

    assert adapter.handle_message_calls == [], (
        "api_server wake must not go through handle_message (wrong-session bug)"
    )
    assert len(posts) == 1
    assert posts[0]["session_id"] == "raw-sid-123"
    assert posts[0]["producer_identity"]
    assert tid in posts[0]["text"]
    # The wake self-post IS the delivery on this path (no separate text-ping
    # fallback is attempted for stateless api_server subs) — cursor advances
    # once the wake succeeds.
    assert _unseen_terminal_events(tid, "api_server", "raw-sid-123") == []


def test_apiserver_rewind_replay_reuses_kanban_producer_identity(
    tmp_path, monkeypatch,
):
    """A failed wake is replayed with the same producer identity."""
    monkeypatch.setenv("HERMES_KANBAN_DB", str(tmp_path / "apiserver-replay.db"))
    kb.init_db()
    tid = _create_completed_subscription(
        "api_server",
        "raw-sid-replay",
        session_id="raw-sid-replay",
    )
    producer_identities = []

    async def flaky_deliver_wake(
        adapter, *, text, session_id, source=None, producer_identity=None,
    ):
        assert session_id == "raw-sid-replay"
        assert source is None
        producer_identities.append(producer_identity)
        if len(producer_identities) == 1:
            raise asyncio.TimeoutError("ambiguous first delivery")

    import gateway.wake as wake_mod

    monkeypatch.setattr(wake_mod, "deliver_wake", flaky_deliver_wake)

    adapter = ApiServerLikeAdapter()
    runner = _make_runner({Platform.API_SERVER: adapter})
    asyncio.run(_run_notifier_ticks(monkeypatch, runner, ticks=2))

    assert len(producer_identities) == 2
    assert producer_identities[0]
    assert producer_identities[0] == producer_identities[1]
    assert adapter.handle_message_calls == []
    assert adapter.send_calls == 0
    assert _unseen_terminal_events(tid, "api_server", "raw-sid-replay") == []
