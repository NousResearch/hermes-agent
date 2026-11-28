"""Staleness gate for the gateway kanban notifier's failure wakes.

A worker that was still running when its card closed can emit a
``timed_out`` / ``crashed`` / ``gave_up`` event AFTER the card reached a
terminal status. Waking the assignee then costs a full agent turn to
discover there is nothing to do.

The gate re-reads the card's LIVE status before delivering a failure
notification and drops it when the card is already terminal
(``done`` / ``archived`` / ``cancelled``).

Two traps this module deliberately avoids:

* **Do NOT suppress on "live status != reported status".** The dispatcher
  re-queues a failed card immediately, so a mismatch is the NORMAL case for
  every genuine escalation. Terminal status is the only reliable signal.
* **Do NOT stage the card's live status to MATCH the reported status.** That
  is a state the real board never produces; a wrong "status moved on" rule
  would pass against it. The dispatcher-realistic re-queued status
  (``ready`` / ``todo`` / ``blocked``) is staged instead, and those must
  STILL notify.
"""

import asyncio

from gateway.config import Platform
from gateway.run import GatewayRunner
from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_notify as kbn


class RecordingAdapter:
    """Records passive pings (``sent``) and wake turns (``handled``)."""

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


async def _run_one_notifier_tick(monkeypatch, runner):
    real_sleep = asyncio.sleep

    async def fake_sleep(delay):
        if delay == 5:
            return
        runner._running = False
        await real_sleep(0)

    monkeypatch.setattr(asyncio, "sleep", fake_sleep)
    await runner._kanban_notifier_watcher(interval=1)


def _subscribe(conn, tid):
    kbn.add_notify_sub(
        conn,
        task_id=tid,
        platform="telegram",
        chat_id="chat-1",
        chat_type="dm",
        delivery_mode="notify+wake",
    )


def _advance_past_completed(conn, tid):
    """Move the subscription cursor past the already-delivered ``completed`` event."""
    completed_id = max(ev.id for ev in kb.list_events(conn, tid) if ev.kind == "completed")
    kbn.advance_notify_cursor(
        conn, task_id=tid, platform="telegram", chat_id="chat-1",
        thread_id="", new_cursor=completed_id,
    )


def _stale_failure_card(db_path, *, live_status):
    """Stage a card that closed while its worker was still running.

    The realistic sequence of the 2026-10-25 incident:

    1. The card completes -> a ``completed`` event is emitted.
    2. That ``completed`` event is delivered and the subscription cursor
       advances past it.
    3. The worker that was STILL RUNNING when the card closed finally trips
       its timeout -> a ``timed_out`` event lands AFTER the card is done.
    4. The dispatcher re-queues (or the card stays done).

    Only step 3's event is left unseen, so the notifier's batch contains the
    stale failure ALONE — which is the batch the gate must drop.

    The card is then forced to ``live_status`` to model what the dispatcher
    did afterwards.

    Returns ``(task_id, sub)``.
    """
    conn = kbc.connect()
    try:
        tid = kb.create_task(
            conn,
            title="stale failure card",
            assignee="worker",
            session_id="agent:main:telegram:dm:chat-1",
        )
        _subscribe(conn, tid)
        # The card closes while the worker is still running...
        kb.complete_task(conn, tid, summary="done first")
        # ...the `completed` event is delivered and the cursor moves past it.
        _advance_past_completed(conn, tid)
        # ...and the worker's timeout lands afterwards.
        kb._append_event(
            conn, tid, "timed_out",
            {"pid": 4242, "elapsed_seconds": 3600, "limit_seconds": 1800,
             "sigkill": True, "retry_status": "ready"},
        )
        # Model the dispatcher's post-failure disposition. The legacy
        # ``completed`` status is a synonym for ``done``, so it KEEPS its
        # completion timestamp; a re-queued status clears it.
        if live_status != "done":
            keep_completed = live_status in ("completed", "archived")
            conn.execute(
                "UPDATE tasks SET status = ?, completed_at = ? WHERE id = ?",
                (live_status, 1788000000 if keep_completed else None, tid),
            )
        return tid
    finally:
        conn.close()


def _seen_events(tid):
    """Every event the notifier could have claimed for this card."""
    conn = kbc.connect()
    try:
        return kb.list_events(conn, tid)
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# AC1: a terminal card produces ZERO wake notifications for a failure event.
# ---------------------------------------------------------------------------


def test_stale_timed_out_on_done_card_produces_no_notification(tmp_path, monkeypatch):
    """A timeout that lands after the card closed must not ping or wake."""
    monkeypatch.setenv("HERMES_KANBAN_DB", str(tmp_path / "stale-done.db"))
    kb.init_db()
    tid = _stale_failure_card(tmp_path / "stale-done.db", live_status="done")

    adapter = RecordingAdapter()
    runner = _make_runner(adapter)
    asyncio.run(_run_one_notifier_tick(monkeypatch, runner))

    assert adapter.sent == [], (
        f"a stale failure on a done card must not ping; got {adapter.sent!r}"
    )
    assert adapter.handled == [], (
        f"a stale failure on a done card must not wake; got {adapter.handled!r}"
    )

    # AC4: assert the notifier actually SAW the card. Without this, a crash
    # or a locked DB would look exactly like a pass.
    events = _seen_events(tid)
    assert any(ev.kind == "timed_out" for ev in events), (
        "the timed_out event must exist on the card, otherwise the test is "
        "asserting nothing about the notifier"
    )


def test_stale_failure_on_archived_card_produces_no_notification(tmp_path, monkeypatch):
    """``archived`` is terminal for notification purposes too."""
    monkeypatch.setenv("HERMES_KANBAN_DB", str(tmp_path / "stale-archived.db"))
    kb.init_db()
    tid = _stale_failure_card(tmp_path / "stale-archived.db", live_status="archived")

    adapter = RecordingAdapter()
    runner = _make_runner(adapter)
    asyncio.run(_run_one_notifier_tick(monkeypatch, runner))

    assert adapter.sent == []
    assert adapter.handled == []
    assert any(ev.kind == "timed_out" for ev in _seen_events(tid))


def test_stale_failure_on_cancelled_card_produces_no_notification(tmp_path, monkeypatch):
    """``cancelled`` is terminal for notification purposes too."""
    monkeypatch.setenv("HERMES_KANBAN_DB", str(tmp_path / "stale-cancelled.db"))
    kb.init_db()
    tid = _stale_failure_card(tmp_path / "stale-cancelled.db", live_status="cancelled")

    adapter = RecordingAdapter()
    runner = _make_runner(adapter)
    asyncio.run(_run_one_notifier_tick(monkeypatch, runner))

    assert adapter.sent == []
    assert adapter.handled == []
    assert any(ev.kind == "timed_out" for ev in _seen_events(tid))


def test_stale_crashed_on_done_card_produces_no_notification(tmp_path, monkeypatch):
    """The gate covers every failure kind, not just ``timed_out``."""
    monkeypatch.setenv("HERMES_KANBAN_DB", str(tmp_path / "stale-crashed.db"))
    kb.init_db()

    conn = kbc.connect()
    try:
        tid = kb.create_task(
            conn, title="crashed card", assignee="worker",
            session_id="agent:main:telegram:dm:chat-1",
        )
        _subscribe(conn, tid)
        kb.complete_task(conn, tid, summary="done first")
        _advance_past_completed(conn, tid)
        kb._append_event(conn, tid, "crashed", {"error": "worker died"})
    finally:
        conn.close()

    adapter = RecordingAdapter()
    runner = _make_runner(adapter)
    asyncio.run(_run_one_notifier_tick(monkeypatch, runner))

    assert adapter.sent == []
    assert adapter.handled == []
    assert any(ev.kind == "crashed" for ev in _seen_events(tid))


def test_stale_gave_up_on_done_card_produces_no_notification(tmp_path, monkeypatch):
    """The gate covers every failure kind, not just ``timed_out``."""
    monkeypatch.setenv("HERMES_KANBAN_DB", str(tmp_path / "stale-gaveup.db"))
    kb.init_db()

    conn = kbc.connect()
    try:
        tid = kb.create_task(
            conn, title="gave up card", assignee="worker",
            session_id="agent:main:telegram:dm:chat-1",
        )
        _subscribe(conn, tid)
        kb.complete_task(conn, tid, summary="done first")
        _advance_past_completed(conn, tid)
        kb._append_event(conn, tid, "gave_up", {"failures": 3, "error": "boom"})
    finally:
        conn.close()

    adapter = RecordingAdapter()
    runner = _make_runner(adapter)
    asyncio.run(_run_one_notifier_tick(monkeypatch, runner))

    assert adapter.sent == []
    assert adapter.handled == []
    assert any(ev.kind == "gave_up" for ev in _seen_events(tid))


# ---------------------------------------------------------------------------
# AC2: a still-live card STILL notifies.
#
# The dispatcher re-queues a failed card immediately, so the live status is
# ``ready`` / ``todo`` / ``blocked`` — NOT the reported failure status, and
# NOT a terminal status. These must all still notify.
# ---------------------------------------------------------------------------


def test_requeued_ready_card_still_notifies(tmp_path, monkeypatch):
    """The dispatcher re-queued the card -> the assignee MUST be told."""
    monkeypatch.setenv("HERMES_KANBAN_DB", str(tmp_path / "live-ready.db"))
    kb.init_db()
    tid = _stale_failure_card(tmp_path / "live-ready.db", live_status="ready")

    adapter = RecordingAdapter()
    runner = _make_runner(adapter)
    asyncio.run(_run_one_notifier_tick(monkeypatch, runner))

    assert len(adapter.sent) == 1, (
        f"a re-queued (ready) card must still be pinged; got {adapter.sent!r}"
    )
    assert len(adapter.handled) == 1, (
        f"a re-queued (ready) card must still be woken; got {adapter.handled!r}"
    )
    assert tid in adapter.handled[0].text
    assert any(ev.kind == "timed_out" for ev in _seen_events(tid))


def test_requeued_todo_card_still_notifies(tmp_path, monkeypatch):
    """A ``todo`` card is still live work."""
    monkeypatch.setenv("HERMES_KANBAN_DB", str(tmp_path / "live-todo.db"))
    kb.init_db()
    tid = _stale_failure_card(tmp_path / "live-todo.db", live_status="todo")

    adapter = RecordingAdapter()
    runner = _make_runner(adapter)
    asyncio.run(_run_one_notifier_tick(monkeypatch, runner))

    assert len(adapter.sent) == 1
    assert len(adapter.handled) == 1
    assert tid in adapter.handled[0].text
    assert any(ev.kind == "timed_out" for ev in _seen_events(tid))


def test_requeued_blocked_card_still_notifies(tmp_path, monkeypatch):
    """A ``blocked`` card is waiting on a human -> the assignee MUST be told."""
    monkeypatch.setenv("HERMES_KANBAN_DB", str(tmp_path / "live-blocked.db"))
    kb.init_db()
    tid = _stale_failure_card(tmp_path / "live-blocked.db", live_status="blocked")

    adapter = RecordingAdapter()
    runner = _make_runner(adapter)
    asyncio.run(_run_one_notifier_tick(monkeypatch, runner))

    assert len(adapter.sent) == 1
    assert len(adapter.handled) == 1
    assert tid in adapter.handled[0].text
    assert any(ev.kind == "timed_out" for ev in _seen_events(tid))


def test_running_card_still_notifies(tmp_path, monkeypatch):
    """A ``running`` card is live work too."""
    monkeypatch.setenv("HERMES_KANBAN_DB", str(tmp_path / "live-running.db"))
    kb.init_db()
    tid = _stale_failure_card(tmp_path / "live-running.db", live_status="running")

    adapter = RecordingAdapter()
    runner = _make_runner(adapter)
    asyncio.run(_run_one_notifier_tick(monkeypatch, runner))

    assert len(adapter.sent) == 1
    assert len(adapter.handled) == 1
    assert tid in adapter.handled[0].text
    assert any(ev.kind == "timed_out" for ev in _seen_events(tid))


def test_stale_failure_on_legacy_completed_status_produces_no_notification(
    tmp_path, monkeypatch,
):
    """A legacy ``completed`` status is a synonym for ``done``.

    The live homelab board carries cards at ``completed`` — a status written
    by a path that predates ``kanban_db.VALID_STATUSES``. One of them
    (``t_616333fc``) is a real stale-failure case: ``reported`` and
    ``timed_out`` both land after ``completed_at``. Matching the status
    string alone would let that wake through.
    """
    monkeypatch.setenv("HERMES_KANBAN_DB", str(tmp_path / "legacy-completed.db"))
    kb.init_db()
    tid = _stale_failure_card(tmp_path / "legacy-completed.db", live_status="completed")

    adapter = RecordingAdapter()
    runner = _make_runner(adapter)
    asyncio.run(_run_one_notifier_tick(monkeypatch, runner))

    assert adapter.sent == []
    assert adapter.handled == []
    assert any(ev.kind == "timed_out" for ev in _seen_events(tid))


def test_stale_failure_with_completed_at_set_is_suppressed(tmp_path, monkeypatch):
    """``completed_at IS NOT NULL`` is the schema-independent finished signal."""
    monkeypatch.setenv("HERMES_KANBAN_DB", str(tmp_path / "completed-at.db"))
    kb.init_db()

    conn = kbc.connect()
    try:
        tid = kb.create_task(
            conn, title="completed-at card", assignee="worker",
            session_id="agent:main:telegram:dm:chat-1",
        )
        _subscribe(conn, tid)
        kb.complete_task(conn, tid, summary="done first")
        _advance_past_completed(conn, tid)
        kb._append_event(conn, tid, "timed_out", {"limit_seconds": 1800})
        # A legacy status the gate does not know, but the card IS finished.
        conn.execute(
            "UPDATE tasks SET status = 'some_legacy_status' WHERE id = ?", (tid,)
        )
    finally:
        conn.close()

    adapter = RecordingAdapter()
    runner = _make_runner(adapter)
    asyncio.run(_run_one_notifier_tick(monkeypatch, runner))

    assert adapter.sent == []
    assert adapter.handled == []
    assert any(ev.kind == "timed_out" for ev in _seen_events(tid))


def test_live_card_without_completed_at_still_notifies(tmp_path, monkeypatch):
    """An unknown status with NO completion timestamp is still live work."""
    monkeypatch.setenv("HERMES_KANBAN_DB", str(tmp_path / "unknown-live.db"))
    kb.init_db()

    conn = kbc.connect()
    try:
        tid = kb.create_task(
            conn, title="unknown live card", assignee="worker",
            session_id="agent:main:telegram:dm:chat-1",
        )
        _subscribe(conn, tid)
        kb._append_event(conn, tid, "timed_out", {"limit_seconds": 1800})
        conn.execute(
            "UPDATE tasks SET status = 'some_legacy_status', completed_at = NULL WHERE id = ?",
            (tid,),
        )
    finally:
        conn.close()

    adapter = RecordingAdapter()
    runner = _make_runner(adapter)
    asyncio.run(_run_one_notifier_tick(monkeypatch, runner))

    assert len(adapter.sent) == 1
    assert len(adapter.handled) == 1
    assert tid in adapter.handled[0].text
    assert any(ev.kind == "timed_out" for ev in _seen_events(tid))


# ---------------------------------------------------------------------------
# AC3: a live-status read failure fails OPEN (the notification is sent).
# ---------------------------------------------------------------------------


def test_live_status_read_failure_fails_open(tmp_path, monkeypatch):
    """An unreadable row is not evidence that the card is terminal."""
    monkeypatch.setenv("HERMES_KANBAN_DB", str(tmp_path / "read-fail.db"))
    kb.init_db()
    tid = _stale_failure_card(tmp_path / "read-fail.db", live_status="done")

    import gateway.kanban_watchers_notifier as notifier_mod

    def _boom(board, task_id):
        raise RuntimeError("simulated read failure")

    monkeypatch.setattr(notifier_mod, "_read_live_task_row", _boom)

    adapter = RecordingAdapter()
    runner = _make_runner(adapter)
    asyncio.run(_run_one_notifier_tick(monkeypatch, runner))

    assert len(adapter.sent) == 1, (
        "a live-status read failure must fail OPEN (deliver the notification); "
        f"got {adapter.sent!r}"
    )
    assert len(adapter.handled) == 1
    assert tid in adapter.handled[0].text
    assert any(ev.kind == "timed_out" for ev in _seen_events(tid))


def test_missing_live_status_row_fails_open(tmp_path, monkeypatch):
    """A row that cannot be found is not evidence that the card is terminal."""
    monkeypatch.setenv("HERMES_KANBAN_DB", str(tmp_path / "missing-row.db"))
    kb.init_db()
    tid = _stale_failure_card(tmp_path / "missing-row.db", live_status="done")

    import gateway.kanban_watchers_notifier as notifier_mod

    monkeypatch.setattr(notifier_mod, "_read_live_task_row", lambda board, task_id: (None, False))

    adapter = RecordingAdapter()
    runner = _make_runner(adapter)
    asyncio.run(_run_one_notifier_tick(monkeypatch, runner))

    assert len(adapter.sent) == 1
    assert len(adapter.handled) == 1
    assert tid in adapter.handled[0].text
    assert any(ev.kind == "timed_out" for ev in _seen_events(tid))


# ---------------------------------------------------------------------------
# The gate must not swallow a batch that carries a REAL decision.
# ---------------------------------------------------------------------------


def test_failure_batch_with_a_real_decision_is_still_delivered(tmp_path, monkeypatch):
    """A batch that also carries ``review_requested`` must go through in full."""
    monkeypatch.setenv("HERMES_KANBAN_DB", str(tmp_path / "mixed-batch.db"))
    kb.init_db()

    conn = kbc.connect()
    try:
        tid = kb.create_task(
            conn, title="mixed card", assignee="worker",
            session_id="agent:main:telegram:dm:chat-1",
        )
        _subscribe(conn, tid)
        kb.complete_task(conn, tid, summary="done first")
        _advance_past_completed(conn, tid)
        kb._append_event(conn, tid, "timed_out", {"limit_seconds": 1800})
        kb._append_event(conn, tid, "review_requested", {"summary": "needs review"})
    finally:
        conn.close()

    adapter = RecordingAdapter()
    runner = _make_runner(adapter)
    asyncio.run(_run_one_notifier_tick(monkeypatch, runner))

    assert len(adapter.handled) == 1, (
        "a batch carrying a genuine decision must still wake the assignee"
    )
    assert tid in adapter.handled[0].text


def test_non_failure_batch_on_terminal_card_is_still_delivered(tmp_path, monkeypatch):
    """A plain ``completed`` on a done card is NOT a stale failure."""
    monkeypatch.setenv("HERMES_KANBAN_DB", str(tmp_path / "plain-completed.db"))
    kb.init_db()

    conn = kbc.connect()
    try:
        tid = kb.create_task(
            conn, title="plain card", assignee="worker",
            session_id="agent:main:telegram:dm:chat-1",
        )
        _subscribe(conn, tid)
        kb.complete_task(conn, tid, summary="done")
    finally:
        conn.close()

    adapter = RecordingAdapter()
    runner = _make_runner(adapter)
    asyncio.run(_run_one_notifier_tick(monkeypatch, runner))

    assert len(adapter.sent) == 1
    assert len(adapter.handled) == 1
    assert tid in adapter.handled[0].text


# ---------------------------------------------------------------------------
# The suppressed event must not be re-claimed forever.
# ---------------------------------------------------------------------------


def test_suppressed_failure_advances_the_cursor(tmp_path, monkeypatch):
    """A suppressed event must not be re-claimed on every subsequent tick."""
    monkeypatch.setenv("HERMES_KANBAN_DB", str(tmp_path / "cursor-advance.db"))
    kb.init_db()
    tid = _stale_failure_card(tmp_path / "cursor-advance.db", live_status="done")

    adapter = RecordingAdapter()
    runner = _make_runner(adapter)
    asyncio.run(_run_one_notifier_tick(monkeypatch, runner))
    assert adapter.sent == []
    assert adapter.handled == []

    # Second tick: the stale event must not be delivered again.
    adapter2 = RecordingAdapter()
    runner2 = _make_runner(adapter2)
    asyncio.run(_run_one_notifier_tick(monkeypatch, runner2))
    assert adapter2.sent == [], "the suppressed event must not be re-delivered"
    assert adapter2.handled == []
    assert any(ev.kind == "timed_out" for ev in _seen_events(tid))
