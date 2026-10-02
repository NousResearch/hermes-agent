"""Wake-Guard: the kanban notifier must not ping/wake dead cards on catch-up
(t_bdd69e28, port of PR #91's supervisor-brake tests to the upstream line).

Upstream's brake lives in ``_Collector._claim_for_sub``: the claim already
advanced the notify cursor, so a stale batch on a terminal card is dropped
whole (no ping, no wake), while a batch that contains the terminal transition
itself (completed/archived) still delivers. The two wake-destination tests use
the upstream api_server self-post wake path (the fork's supervisor-ack
machinery does not exist upstream); the ack-row assertion is replaced by "no
self-post + cursor advanced".
"""

import asyncio

from gateway.config import Platform
from gateway.run import GatewayRunner
from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_notify as kbn
from gateway import kanban_watchers_notifier as watchers_mod


class RecordingAdapter:
    def __init__(self):
        self.sent = []
        self.handled = []

    async def send(self, chat_id, text, metadata=None):
        self.sent.append({"chat_id": chat_id, "text": text, "metadata": metadata or {}})

    async def handle_message(self, event):
        self.handled.append(event)
        event._gateway_accepted = True


class ApiServerLikeAdapter:
    supports_async_delivery = False

    def __init__(self):
        self._host = "127.0.0.1"
        self._port = 8642
        self._api_key = "k"
        self.handle_message_calls = []
        self.send_calls = 0

    async def send(self, chat_id, text, metadata=None):
        self.send_calls += 1
        return None

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


def _fresh_board(tmp_path, monkeypatch, name="terminal-brake.db"):
    monkeypatch.setenv("HERMES_KANBAN_DB", str(tmp_path / name))
    kb.init_db()
    return kbc.connect()


def _last_event_id(conn, tid):
    row = conn.execute(
        "SELECT MAX(id) AS mid FROM task_events WHERE task_id = ?", (tid,)
    ).fetchone()
    return row["mid"] or 0


def _sub_cursor(conn, tid, platform="telegram", chat_id="chat-1"):
    for sub in kbn.list_notify_subs(conn):
        if sub["task_id"] == tid and sub["platform"] == platform and sub["chat_id"] == chat_id:
            return sub["last_event_id"]
    return None


def test_notifier_terminal_catchup_no_wake(tmp_path, monkeypatch):
    """A stale batch on a done card delivers nothing (no ping, no wake); the
    cursor still advances past the claimed events."""
    conn = _fresh_board(tmp_path, monkeypatch)
    try:
        tid = kb.create_task(conn, title="catch-up target", assignee="worker")
        kbn.add_notify_sub(
            conn, task_id=tid, platform="telegram", chat_id="chat-1",
            delivery_mode="notify+wake",
        )
        kb.complete_task(conn, tid, summary="card is done")
    finally:
        conn.close()

    adapter = RecordingAdapter()
    runner = _make_runner({Platform.TELEGRAM: adapter})
    asyncio.run(_run_one_notifier_tick(monkeypatch, runner))
    assert len(adapter.sent) == 1
    assert len(adapter.handled) == 1

    # Late-arriving stale event on the now-terminal card (no transition in it).
    conn = kbc.connect()
    try:
        kb._append_event(conn, tid, "blocked", {"reason": "late stale noise"})
        late_id = _last_event_id(conn, tid)
        before = watchers_mod._terminal_skip_count
    finally:
        conn.close()

    adapter2 = RecordingAdapter()
    runner2 = _make_runner({Platform.TELEGRAM: adapter2})
    asyncio.run(_run_one_notifier_tick(monkeypatch, runner2))

    assert adapter2.sent == []
    assert adapter2.handled == []
    assert watchers_mod._terminal_skip_count > before
    conn = kbc.connect()
    try:
        cursor = _sub_cursor(conn, tid)
    finally:
        conn.close()
    assert cursor is not None and cursor >= late_id


def test_notifier_terminal_transition_still_wakes(tmp_path, monkeypatch):
    """A batch that contains the terminal transition itself (a fresh
    ``completed`` event after a reopen/re-complete) delivers in full."""
    conn = _fresh_board(tmp_path, monkeypatch, name="terminal-transition.db")
    try:
        tid = kb.create_task(conn, title="recompleted card", assignee="worker")
        kbn.add_notify_sub(
            conn, task_id=tid, platform="telegram", chat_id="chat-2",
            delivery_mode="notify+wake",
        )
        kb.complete_task(conn, tid, summary="first completion")
    finally:
        conn.close()

    adapter = RecordingAdapter()
    runner = _make_runner({Platform.TELEGRAM: adapter})
    asyncio.run(_run_one_notifier_tick(monkeypatch, runner))
    assert len(adapter.sent) == 1 and len(adapter.handled) == 1

    # Reopen/re-complete cycle lands as a fresh terminal transition in the
    # subscription's next batch.
    conn = kbc.connect()
    try:
        kb._append_event(conn, tid, "completed", {"summary": "re-completed after reopen"})
    finally:
        conn.close()

    adapter2 = RecordingAdapter()
    runner2 = _make_runner({Platform.TELEGRAM: adapter2})
    asyncio.run(_run_one_notifier_tick(monkeypatch, runner2))

    assert len(adapter2.sent) == 1
    assert len(adapter2.handled) == 1


def test_notifier_reopen_status_ping_not_suppressed_before_terminality(tmp_path, monkeypatch):
    """While the card is live, ordinary event pings/wakes are untouched by the
    brake."""
    conn = _fresh_board(tmp_path, monkeypatch, name="live-card-ping.db")
    try:
        tid = kb.create_task(conn, title="live card", assignee="worker")
        kbn.add_notify_sub(
            conn, task_id=tid, platform="telegram", chat_id="chat-3",
            delivery_mode="notify+wake",
        )
        kb.block_task(conn, tid, reason="needs operator input", kind="needs_input")
    finally:
        conn.close()

    adapter = RecordingAdapter()
    runner = _make_runner({Platform.TELEGRAM: adapter})
    asyncio.run(_run_one_notifier_tick(monkeypatch, runner))

    assert len(adapter.sent) == 1
    assert len(adapter.handled) == 1


def test_notifier_terminal_catchup_no_wake_wake_destination(tmp_path, monkeypatch):
    """The api_server wake-only destination gets no self-post for a stale batch
    on a terminal card; the cursor advances (the fork's supervisor-ack row does
    not exist upstream — the wake self-post IS the destination contract)."""
    conn = _fresh_board(tmp_path, monkeypatch, name="terminal-wake-dest.db")
    try:
        tid = kb.create_task(conn, title="wake destination", assignee="worker")
        kbn.add_notify_sub(
            conn, task_id=tid, platform="api_server", chat_id="sup-session",
        )
        kb.complete_task(conn, tid, summary="done before catch-up")
    finally:
        conn.close()

    posts = []

    async def fake_self_post(adapter, *, text, session_id):
        posts.append({"text": text, "session_id": session_id})

    import gateway.wake as wake_mod

    monkeypatch.setattr(wake_mod, "_self_post_chat_completion", fake_self_post)

    adapter = ApiServerLikeAdapter()
    runner = _make_runner({Platform.API_SERVER: adapter})
    asyncio.run(_run_one_notifier_tick(monkeypatch, runner))
    assert len(posts) == 1

    conn = kbc.connect()
    try:
        kb._append_event(conn, tid, "blocked", {"reason": "late stale noise"})
        late_id = _last_event_id(conn, tid)
        before = watchers_mod._terminal_skip_count
    finally:
        conn.close()

    runner2 = _make_runner({Platform.API_SERVER: ApiServerLikeAdapter()})
    asyncio.run(_run_one_notifier_tick(monkeypatch, runner2))

    assert len(posts) == 1  # unchanged
    assert watchers_mod._terminal_skip_count > before
    conn = kbc.connect()
    try:
        cursor = _sub_cursor(conn, tid, platform="api_server", chat_id="sup-session")
    finally:
        conn.close()
    assert cursor is not None and cursor >= late_id


def test_notifier_terminal_transition_still_wakes_wake_destination(tmp_path, monkeypatch):
    """A wake-destination subscription still receives the wake for a batch that
    carries the terminal transition itself."""
    conn = _fresh_board(tmp_path, monkeypatch, name="transition-wake-dest.db")
    try:
        tid = kb.create_task(conn, title="transition wake", assignee="worker")
        kbn.add_notify_sub(
            conn, task_id=tid, platform="api_server", chat_id="sup-session",
        )
        kb.complete_task(conn, tid, summary="first completion")
    finally:
        conn.close()

    posts = []

    async def fake_self_post(adapter, *, text, session_id):
        posts.append({"text": text, "session_id": session_id})

    import gateway.wake as wake_mod

    monkeypatch.setattr(wake_mod, "_self_post_chat_completion", fake_self_post)

    runner = _make_runner({Platform.API_SERVER: ApiServerLikeAdapter()})
    asyncio.run(_run_one_notifier_tick(monkeypatch, runner))
    assert len(posts) == 1

    conn = kbc.connect()
    try:
        kb._append_event(conn, tid, "completed", {"summary": "re-completed after reopen"})
    finally:
        conn.close()

    runner2 = _make_runner({Platform.API_SERVER: ApiServerLikeAdapter()})
    asyncio.run(_run_one_notifier_tick(monkeypatch, runner2))

    assert len(posts) == 2
    assert posts[-1]["session_id"] == "sup-session"
