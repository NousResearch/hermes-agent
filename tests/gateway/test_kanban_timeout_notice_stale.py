"""A timeout/crash notice must not promise a retry the live board contradicts.

``timed_out`` and ``crashed`` are true records of what happened to ONE attempt —
the dispatcher appends them under a ``rowcount == 1`` guard on the still-running
row. The notice, though, is rendered later from the event alone, so a card whose
attempt was retried and has since finished (or whose breaker gave up, or that is
already re-running) still produced the event: the owner got

    ⏱ [default] @worker Kanban t_x timed out (max_runtime=0s); will retry

for a card that was ``done``, with nothing retried. That is the false alarm this
pins: the retry claim is derived from the LIVE task row, never from the event.
"""

import asyncio

from gateway.config import Platform
from gateway.kanban_watchers_notifier import _KanbanNotification
from gateway.run import GatewayRunner
from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_notify as kbn


class RecordingAdapter:
    def __init__(self):
        self.sent = []
        self.handled = []

    async def send(self, chat_id, text, metadata=None):
        self.sent.append({"chat_id": chat_id, "text": text, "metadata": metadata or {}})

    async def handle_message(self, event):
        self.handled.append(event)
        event._gateway_accepted = True


def _make_runner(adapter):
    runner = GatewayRunner.__new__(GatewayRunner)
    runner._running = True
    runner.adapters = {Platform.TELEGRAM: adapter}
    runner._kanban_sub_fail_counts = {}
    runner._kanban_dispatcher_lock_handle = object()
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


def _board(tmp_path, monkeypatch, name="stale-timeout.db", *, subscribe=True):
    """A fresh board with one subscribed task; returns (conn, task_id)."""
    monkeypatch.setenv("HERMES_KANBAN_DB", str(tmp_path / name))
    kb.init_db()
    conn = kbc.connect()
    tid = kb.create_task(conn, title="temp-ssh drop lab", assignee="hardware-manager")
    if subscribe:
        kbn.add_notify_sub(conn, task_id=tid, platform="telegram", chat_id="chat-1")
    return conn, tid


def _status(conn, tid) -> str:
    task = kb.get_task(conn, tid)
    assert task is not None, f"task {tid} vanished"
    return task.status


def _timeout_event(conn, tid, *, limit_seconds=0):
    """The exact event ``enforce_max_runtime`` appends for one timed-out attempt."""
    kb._append_event(
        conn, tid, "timed_out",
        {"pid": 4242, "elapsed_seconds": limit_seconds + 1, "limit_seconds": limit_seconds,
         "sigkill": False, "retry_status": "ready"},
    )


def _texts(adapter):
    return [d["text"] for d in adapter.sent]


def test_timed_out_notice_never_promises_a_retry_for_a_finished_card(tmp_path, monkeypatch):
    """The reported instance: the attempt timed out, the retry finished, the ping lied."""
    conn, tid = _board(tmp_path, monkeypatch)
    try:
        _timeout_event(conn, tid)
        # The retry ran and completed before the notifier's next tick.
        assert kb.complete_task(conn, tid, summary="lab dropped") is True
        assert _status(conn, tid) == "done"
    finally:
        conn.close()

    adapter = RecordingAdapter()
    asyncio.run(_run_one_tick(monkeypatch, _make_runner(adapter)))

    texts = _texts(adapter)
    assert texts, "the completed notice must still be delivered"
    assert not any("retried" in t.lower() for t in texts), texts
    assert not any("timed out" in t.lower() for t in texts), texts
    assert any("done" in t.lower() for t in texts), texts


def test_timed_out_event_claimed_after_the_card_finished_stays_silent(tmp_path, monkeypatch):
    """The completion ping already went out; the stale timeout event says nothing at all.

    The event id ordering here is the shape that produced the false alarm in the
    field: the card is terminal, the notifier claims the ``timed_out`` event on a
    later tick, and the only thing it could say would be a retry that never comes.
    """
    conn, tid = _board(tmp_path, monkeypatch)
    try:
        assert kb.complete_task(conn, tid, summary="lab dropped") is True
    finally:
        conn.close()

    adapter = RecordingAdapter()
    asyncio.run(_run_one_tick(monkeypatch, _make_runner(adapter)))
    assert len(_texts(adapter)) == 1, _texts(adapter)  # the completed ping

    conn = kbc.connect()
    try:
        _timeout_event(conn, tid)
    finally:
        conn.close()

    asyncio.run(_run_one_tick(monkeypatch, _make_runner(adapter)))
    assert len(_texts(adapter)) == 1, f"a stale timeout notice must be silent; got {_texts(adapter)}"


def test_timed_out_notice_still_promises_the_retry_when_one_is_really_queued(tmp_path, monkeypatch):
    """Companion case: the task is parked for another spawn, so the claim is true."""
    conn, tid = _board(tmp_path, monkeypatch)
    try:
        _timeout_event(conn, tid, limit_seconds=1800)
        # ``_retry_status_for_run`` leaves a timed-out attempt here for the next tick.
        assert _status(conn, tid) == "ready"
    finally:
        conn.close()

    adapter = RecordingAdapter()
    asyncio.run(_run_one_tick(monkeypatch, _make_runner(adapter)))

    texts = _texts(adapter)
    assert len(texts) == 1, texts
    assert "will be retried automatically" in texts[0], texts
    assert "30-minute" in texts[0], texts


def test_timed_out_notice_states_the_live_status_when_no_retry_is_queued(tmp_path, monkeypatch):
    """A blocked card is not queued for anything — say so instead of promising a retry."""
    conn, tid = _board(tmp_path, monkeypatch)
    try:
        _timeout_event(conn, tid)
        assert kb.block_task(conn, tid, reason="worker died for good") is True
        assert _status(conn, tid) == "blocked"
    finally:
        conn.close()

    adapter = RecordingAdapter()
    asyncio.run(_run_one_tick(monkeypatch, _make_runner(adapter)))

    texts = _texts(adapter)
    assert texts, texts
    assert not any("retried" in t.lower() for t in texts), texts
    timeout_texts = [t for t in texts if "ran past" in t.lower()]
    assert timeout_texts, texts
    assert "no retry is queued" in timeout_texts[0], timeout_texts
    assert "blocked" in timeout_texts[0], timeout_texts


def test_crashed_notice_has_the_same_stale_guard(tmp_path, monkeypatch):
    """The sibling kind renders the same promise, so it needs the same live check."""
    conn, tid = _board(tmp_path, monkeypatch)
    try:
        kb._append_event(conn, tid, "crashed", {"pid": 4242, "claimer": "host:1", "retry_status": "ready"})
        assert kb.complete_task(conn, tid, summary="finished anyway") is True
    finally:
        conn.close()

    adapter = RecordingAdapter()
    asyncio.run(_run_one_tick(monkeypatch, _make_runner(adapter)))

    texts = _texts(adapter)
    assert texts, texts
    assert not any("retried" in t.lower() for t in texts), texts
    assert not any("stopped unexpectedly" in t.lower() for t in texts), texts


def _notification(task, events, *, wake=True):
    """A ``_KanbanNotification`` wired for ``build_wake_text`` only."""
    n = _KanbanNotification.__new__(_KanbanNotification)
    n.task = task
    n.sub = {"task_id": task.id, "platform": "telegram", "chat_id": "chat-1", "thread_id": ""}
    n.d = {"events": events, "sub": n.sub, "task": task}
    n.board_slug = "default"
    n.title = task.title
    n.wake_agent = wake
    n.is_push_adapter = True
    n.wake_handoff = n.wake_review_detail = ""
    return n


class _Ev:
    def __init__(self, kind, payload=None):
        self.kind = kind
        self.payload = payload or {}


def test_a_stale_interruption_does_not_wake_the_creator(tmp_path, monkeypatch):
    """The wake turn renders ``gateway.kanban.wake.timed_out`` ("dispatcher will retry").

    An event whose retry already happened must not reach it, or the false promise
    simply moves into the wake text.
    """
    conn, tid = _board(tmp_path, monkeypatch, name="wake-gate.db", subscribe=False)
    try:
        _timeout_event(conn, tid)
        assert kb.complete_task(conn, tid, summary="done") is True
        finished = kb.get_task(conn, tid)

        n = _notification(finished, [_Ev("timed_out", {"limit_seconds": 0})])
        n.build_wake_text()
        assert n.wake_kinds == set(), n.wake_kinds
        assert getattr(n, "synth", "") == "", getattr(n, "synth", "")

        # Same event, same task, still queued for the retry: the wake stands.
        queued = kb.create_task(conn, title="queued", assignee="worker")
        _timeout_event(conn, queued)
        n = _notification(kb.get_task(conn, queued), [_Ev("timed_out", {"limit_seconds": 0})])
        n.build_wake_text()
        assert "timed_out" in n.wake_kinds, n.wake_kinds
        # The very sentence the wake must not carry for a finished card.
        assert "timed out; dispatcher will retry" in n.synth, n.synth
    finally:
        conn.close()
