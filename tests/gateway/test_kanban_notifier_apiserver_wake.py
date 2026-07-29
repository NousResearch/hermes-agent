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


def test_sendresult_failure_rewinds_cursor(tmp_path, monkeypatch):
    """SendResult(success=False) without an exception must count as a failed
    delivery — cursor rewound, event retried on the next tick. Previously the
    cursor advanced and the event was permanently lost."""
    monkeypatch.setenv("HERMES_KANBAN_DB", str(tmp_path / "softfail.db"))
    kb.init_db()
    tid = _create_completed_subscription("telegram", "chat-1")

    adapter = SoftFailAdapter()
    runner = _make_runner({Platform.TELEGRAM: adapter})
    asyncio.run(_run_one_notifier_tick(monkeypatch, runner))

    assert adapter.attempts >= 1
    assert [ev.kind for ev in _unseen_terminal_events(tid, "telegram", "chat-1")] == [
        "completed"
    ]


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
    # The wake self-post IS the delivery on this path (no separate text-ping
    # fallback is attempted for stateless api_server subs) — cursor advances
    # once the wake succeeds.
    assert _unseen_terminal_events(tid, "api_server", "raw-sid-123") == []


def test_apiserver_failed_self_post_rewinds_cursor(tmp_path, monkeypatch):
    """A failed/exhausted wake self-post must NOT advance the cursor: on the
    api_server path the self-post IS the delivery, so advancing first would
    permanently lose the event behind a best-effort except. The claim is
    rewound and the event stays visible for the next tick's retry."""
    monkeypatch.setenv("HERMES_KANBAN_DB", str(tmp_path / "apiserver_fail.db"))
    kb.init_db()
    tid = _create_completed_subscription(
        "api_server", "raw-sid-999", session_id="raw-sid-999",
    )

    async def failing_self_post(adapter, *, text, session_id):
        raise RuntimeError("self-post exhausted retries")

    import gateway.wake as wake_mod

    monkeypatch.setattr(wake_mod, "_self_post_chat_completion", failing_self_post)

    adapter = ApiServerLikeAdapter()
    runner = _make_runner({Platform.API_SERVER: adapter})
    asyncio.run(_run_one_notifier_tick(monkeypatch, runner))

    # Event NOT lost: the cursor was rewound, so the completed event is still
    # unseen and will be re-claimed (and the self-post retried) next tick.
    assert [ev.kind for ev in _unseen_terminal_events(tid, "api_server", "raw-sid-999")] == [
        "completed"
    ]
    # And the failure was counted toward the drop threshold.
    assert list(runner._kanban_sub_fail_counts.values()) == [1]


def test_apiserver_self_post_succeeds_after_earlier_failure(tmp_path, monkeypatch):
    """The rewound event is retried on the next tick; a successful self-post
    then advances the cursor and clears the failure counter."""
    monkeypatch.setenv("HERMES_KANBAN_DB", str(tmp_path / "apiserver_retry.db"))
    kb.init_db()
    tid = _create_completed_subscription(
        "api_server", "raw-sid-777", session_id="raw-sid-777",
    )

    calls = {"n": 0}

    async def flaky_self_post(adapter, *, text, session_id):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("transient outage")

    import gateway.wake as wake_mod

    monkeypatch.setattr(wake_mod, "_self_post_chat_completion", flaky_self_post)

    adapter = ApiServerLikeAdapter()
    runner = _make_runner({Platform.API_SERVER: adapter})
    asyncio.run(_run_one_notifier_tick(monkeypatch, runner))
    assert calls["n"] == 1
    assert len(_unseen_terminal_events(tid, "api_server", "raw-sid-777")) == 1

    # Second tick: the re-claimed event's self-post succeeds → cursor advances.
    runner._running = True
    asyncio.run(_run_one_notifier_tick(monkeypatch, runner))
    assert calls["n"] == 2
    assert _unseen_terminal_events(tid, "api_server", "raw-sid-777") == []
    assert runner._kanban_sub_fail_counts == {}
