import asyncio
from pathlib import Path

import pytest

from gateway.config import Platform
from gateway.run import GatewayRunner
from hermes_cli import kanban_db as kb


class RecordingAdapter:
    def __init__(self):
        self.sent = []

    async def send(self, chat_id, text, metadata=None):
        self.sent.append({"chat_id": chat_id, "text": text, "metadata": metadata or {}})


class DisconnectedAdapters(dict):
    """Expose a platform during collection, then simulate disconnect on get()."""

    def get(self, key, default=None):
        return None


async def _run_one_notifier_tick(monkeypatch, runner):
    real_sleep = asyncio.sleep

    async def fake_sleep(delay):
        if delay == 5:
            return None
        runner._running = False
        await real_sleep(0)

    monkeypatch.setattr(asyncio, "sleep", fake_sleep)
    await runner._kanban_notifier_watcher(interval=1)


def _make_runner(adapter):
    runner = GatewayRunner.__new__(GatewayRunner)
    runner._running = True
    runner.adapters = {Platform.TELEGRAM: adapter}
    runner._kanban_sub_fail_counts = {}
    return runner


def _create_completed_subscription(summary="done once"):
    conn = kb.connect()
    try:
        tid = kb.create_task(conn, title="notify once", assignee="worker")
        kb.add_notify_sub(conn, task_id=tid, platform="telegram", chat_id="chat-1")
        kb.complete_task(conn, tid, summary=summary)
        return tid
    finally:
        conn.close()


def _unseen_terminal_events(tid):
    conn = kb.connect()
    try:
        _, events = kb.unseen_events_for_sub(
            conn,
            task_id=tid,
            platform="telegram",
            chat_id="chat-1",
            kinds=["completed", "blocked", "gave_up", "crashed", "timed_out"],
        )
        return events
    finally:
        conn.close()


def test_kanban_notifier_dedupes_board_slugs_pointing_to_same_db(tmp_path, monkeypatch):
    db_path = tmp_path / "shared-kanban.db"
    monkeypatch.setenv("HERMES_KANBAN_DB", str(db_path))
    kb.init_db()
    kb.write_board_metadata("alias-a", name="Alias A")
    kb.write_board_metadata("alias-b", name="Alias B")

    tid = _create_completed_subscription()

    adapter = RecordingAdapter()
    runner = _make_runner(adapter)

    asyncio.run(_run_one_notifier_tick(monkeypatch, runner))

    assert len(adapter.sent) == 1
    assert "Kanban" in adapter.sent[0]["text"]
    assert tid in adapter.sent[0]["text"]


def test_kanban_notifier_claim_prevents_second_watcher_send(tmp_path, monkeypatch):
    db_path = tmp_path / "single-owner.db"
    monkeypatch.setenv("HERMES_KANBAN_DB", str(db_path))
    kb.init_db()

    tid = _create_completed_subscription()

    adapter1 = RecordingAdapter()
    adapter2 = RecordingAdapter()

    asyncio.run(_run_one_notifier_tick(monkeypatch, _make_runner(adapter1)))
    asyncio.run(_run_one_notifier_tick(monkeypatch, _make_runner(adapter2)))

    assert len(adapter1.sent) == 1
    assert adapter2.sent == []


def test_kanban_notifier_rewinds_claim_if_adapter_disconnects(tmp_path, monkeypatch):
    db_path = tmp_path / "adapter-disconnect.db"
    monkeypatch.setenv("HERMES_KANBAN_DB", str(db_path))
    kb.init_db()
    tid = _create_completed_subscription()

    runner = GatewayRunner.__new__(GatewayRunner)
    runner._running = True
    runner.adapters = DisconnectedAdapters({Platform.TELEGRAM: RecordingAdapter()})
    runner._kanban_sub_fail_counts = {}

    asyncio.run(_run_one_notifier_tick(monkeypatch, runner))

    assert [ev.kind for ev in _unseen_terminal_events(tid)] == ["completed"]


def test_kanban_db_path_is_test_isolated_from_real_home():
    hermes_home = Path(kb.kanban_home())
    production_db = Path.home() / ".hermes" / "kanban.db"
    assert kb.kanban_db_path().resolve() != production_db.resolve()

    conn = kb.connect()
    try:
        tid = kb.create_task(conn, title="x", assignee="worker")
        kb.add_notify_sub(conn, task_id=tid, platform="telegram", chat_id="chat-1")
    finally:
        conn.close()

    assert kb.kanban_db_path().resolve().is_relative_to(hermes_home.resolve())
    assert kb.kanban_db_path().resolve() != production_db.resolve()


class FailingAdapter:
    """Adapter whose send() always raises, simulating a transient send error."""

    def __init__(self):
        self.attempts = 0

    async def send(self, chat_id, text, metadata=None):
        self.attempts += 1
        raise RuntimeError("simulated send failure")


def test_kanban_notifier_rewinds_claim_on_send_exception(tmp_path, monkeypatch):
    """A raising adapter rewinds the claim so the next tick can retry.

    This is the second rewind path (distinct from the adapter-disconnect path
    in test_kanban_notifier_rewinds_claim_if_adapter_disconnects). Here the
    adapter is connected and the send call actually fires; the claim must
    still rewind so the event isn't lost when send() raises mid-tick.
    """
    db_path = tmp_path / "send-failure.db"
    monkeypatch.setenv("HERMES_KANBAN_DB", str(db_path))
    kb.init_db()
    tid = _create_completed_subscription()

    adapter = FailingAdapter()
    runner = _make_runner(adapter)

    asyncio.run(_run_one_notifier_tick(monkeypatch, runner))

    # Send was attempted (so we exercised the failure path, not just the
    # disconnect path) and the claim was rewound — the unseen-events query
    # still returns the event for retry on the next tick.
    assert adapter.attempts >= 1, "send should have been attempted at least once"
    assert [ev.kind for ev in _unseen_terminal_events(tid)] == ["completed"]


def test_notifier_redelivers_same_kind_on_dispatch_cycle(tmp_path, monkeypatch):
    """A retry cycle (crashed → reclaimed → crashed) notifies the user twice.

    Before #21398 the notifier auto-unsubscribed on any terminal event kind
    (gave_up / crashed / timed_out), so the second crash in a respawn cycle
    silently dropped — the subscription was already gone. This test pins the
    new contract: subscription survives non-final terminal events; the
    cursor handles dedup.

    Two crashes ten seconds apart on the same task — both should land on
    the adapter.
    """
    db_path = tmp_path / "redeliver-cycle.db"
    monkeypatch.setenv("HERMES_KANBAN_DB", str(db_path))
    kb.init_db()

    conn = kb.connect()
    try:
        tid = kb.create_task(conn, title="cycle test", assignee="worker")
        kb.add_notify_sub(conn, task_id=tid, platform="telegram", chat_id="chat-1")
        # First crash — fired by the dispatcher when the worker PID dies.
        kb._append_event(conn, tid, kind="crashed")
    finally:
        conn.close()

    adapter = RecordingAdapter()
    runner = _make_runner(adapter)
    asyncio.run(_run_one_notifier_tick(monkeypatch, runner))

    # First crash delivered.
    assert len(adapter.sent) == 1
    assert "crashed" in adapter.sent[0]["text"].lower()

    # Subscription survives — the cursor advanced past event #1, but the
    # row is still there.
    conn = kb.connect()
    try:
        subs = kb.list_notify_subs(conn, tid)
        assert len(subs) == 1, (
            "Subscription must survive a crashed event so a respawn-cycle "
            "second crash also notifies the user (issue #21398)."
        )

        # Second crash — same task, same dispatcher (or a respawn). Append
        # another event to simulate the dispatcher firing crashed a second
        # time during retry.
        kb._append_event(conn, tid, kind="crashed")
    finally:
        conn.close()

    # New tick: the second event has a fresh id past the cursor advance,
    # so it gets claimed and delivered.
    runner = _make_runner(adapter)
    asyncio.run(_run_one_notifier_tick(monkeypatch, runner))

    assert len(adapter.sent) == 2, (
        f"Second crashed event should also notify; got {len(adapter.sent)} "
        f"deliveries (texts: {[d['text'] for d in adapter.sent]})"
    )
    assert "crashed" in adapter.sent[1]["text"].lower()


# ---------------------------------------------------------------------------
# Comment-event delivery (t_6c3947c0): commented events are now in the
# deliverable set; per-task subs get them via the union claim; wildcard
# subs (task_id='*') get board-wide comment broadcasts with optional
# per-sub author filter.
# ---------------------------------------------------------------------------


def test_notifier_delivers_comment_event_to_pertask_sub(tmp_path, monkeypatch):
    """A regular per-task subscription receives `commented` events alongside
    terminal events. Before this change, comments were filtered out because
    TERMINAL_KINDS did not include 'commented'."""
    db_path = tmp_path / "comment-pertask.db"
    monkeypatch.setenv("HERMES_KANBAN_DB", str(db_path))
    kb.init_db()
    conn = kb.connect()
    try:
        tid = kb.create_task(conn, title="needs review", assignee="worker")
        kb.add_notify_sub(conn, task_id=tid, platform="telegram", chat_id="chat-1")
        kb.add_comment(conn, tid, author="dashboard", body="LGTM, ship it")
    finally:
        conn.close()

    adapter = RecordingAdapter()
    runner = _make_runner(adapter)
    asyncio.run(_run_one_notifier_tick(monkeypatch, runner))

    assert len(adapter.sent) == 1
    msg = adapter.sent[0]["text"]
    assert "comment" in msg.lower()
    assert tid in msg
    assert "dashboard" in msg
    assert "LGTM, ship it" in msg


def test_notifier_wildcard_sub_delivers_comments_across_tasks(tmp_path, monkeypatch):
    """A wildcard subscription (task_id='*') gets comment events for every
    task on the board, with the per-sub author_filter applied."""
    db_path = tmp_path / "wildcard.db"
    monkeypatch.setenv("HERMES_KANBAN_DB", str(db_path))
    kb.init_db()
    conn = kb.connect()
    try:
        kb.add_notify_sub(
            conn, task_id="*", platform="telegram", chat_id="chat-1",
            author_filter="dashboard",
        )
        t1 = kb.create_task(conn, title="task one", assignee="worker")
        t2 = kb.create_task(conn, title="task two", assignee="worker")
        kb.add_comment(conn, t1, author="dashboard", body="reply on task one")
        kb.add_comment(conn, t2, author="worker", body="ignored — not dashboard")
        kb.add_comment(conn, t2, author="dashboard", body="reply on task two")
    finally:
        conn.close()

    adapter = RecordingAdapter()
    runner = _make_runner(adapter)
    asyncio.run(_run_one_notifier_tick(monkeypatch, runner))

    # All 3 comment events captured in one tick → coalesced into one
    # bundled message (skipping the 'worker'-authored one via filter).
    assert len(adapter.sent) == 1, (
        f"Expected single coalesced bundle; got {len(adapter.sent)}: "
        f"{[d['text'] for d in adapter.sent]}"
    )
    msg = adapter.sent[0]["text"]
    assert "reply on task one" in msg
    assert "reply on task two" in msg
    assert "ignored" not in msg  # author filter excluded the worker-authored
    assert "dashboard" in msg
    assert t1 in msg
    assert t2 in msg


def test_notifier_wildcard_does_not_unsub_on_done_task(tmp_path, monkeypatch):
    """Wildcard subs are board-scoped; per-task lifecycle never removes
    them. A `done` task whose final comment fired must NOT drop the sub."""
    db_path = tmp_path / "wildcard-survives.db"
    monkeypatch.setenv("HERMES_KANBAN_DB", str(db_path))
    kb.init_db()
    conn = kb.connect()
    try:
        kb.add_notify_sub(
            conn, task_id="*", platform="telegram", chat_id="chat-1",
            author_filter=None,  # match any author
        )
        tid = kb.create_task(conn, title="ephemeral", assignee="worker")
        kb.add_comment(conn, tid, author="anyone", body="hello")
        kb.complete_task(conn, tid, summary="done")
    finally:
        conn.close()

    adapter = RecordingAdapter()
    runner = _make_runner(adapter)
    asyncio.run(_run_one_notifier_tick(monkeypatch, runner))

    # Confirm wildcard sub still exists after the tick.
    conn = kb.connect()
    try:
        subs = kb.list_notify_subs(conn)
        wildcard = [s for s in subs if s["task_id"] == "*"]
        assert len(wildcard) == 1, (
            "Wildcard sub must outlive any single task's terminal state"
        )
    finally:
        conn.close()


def test_notifier_pertask_author_filter_drops_non_matching_comment(tmp_path, monkeypatch):
    """A per-task sub with author_filter='dashboard' silently drops
    comments authored by anyone else (cursor still advances)."""
    db_path = tmp_path / "pertask-filtered.db"
    monkeypatch.setenv("HERMES_KANBAN_DB", str(db_path))
    kb.init_db()
    conn = kb.connect()
    try:
        tid = kb.create_task(conn, title="filtered", assignee="worker")
        kb.add_notify_sub(
            conn, task_id=tid, platform="telegram", chat_id="chat-1",
            author_filter="dashboard",
        )
        kb.add_comment(conn, tid, author="worker-bot", body="self-note")
    finally:
        conn.close()

    adapter = RecordingAdapter()
    runner = _make_runner(adapter)
    asyncio.run(_run_one_notifier_tick(monkeypatch, runner))

    # No delivery for the filtered-out comment.
    assert adapter.sent == []
    # Cursor still advanced so a later matching comment doesn't replay
    # the filtered one.
    conn = kb.connect()
    try:
        subs = kb.list_notify_subs(conn, tid)
        assert subs and subs[0]["last_event_id"] > 0
    finally:
        conn.close()


def test_notifier_terminal_event_not_delivered_to_wildcard(tmp_path, monkeypatch):
    """Wildcard subs are comment-only — they must not deliver terminal
    events (would otherwise flood the chat with every completion)."""
    db_path = tmp_path / "wildcard-no-terminal.db"
    monkeypatch.setenv("HERMES_KANBAN_DB", str(db_path))
    kb.init_db()
    conn = kb.connect()
    try:
        kb.add_notify_sub(
            conn, task_id="*", platform="telegram", chat_id="chat-1",
        )
        tid = kb.create_task(conn, title="completes silently", assignee="worker")
        kb.complete_task(conn, tid, summary="done")
    finally:
        conn.close()

    adapter = RecordingAdapter()
    runner = _make_runner(adapter)
    asyncio.run(_run_one_notifier_tick(monkeypatch, runner))

    assert adapter.sent == [], (
        "Wildcard sub must not deliver terminal completion events"
    )
