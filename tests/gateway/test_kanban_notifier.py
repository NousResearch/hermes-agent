import asyncio
import sqlite3
import time


from gateway.config import Platform
from gateway.kanban_watchers_common import (
    _acquire_singleton_lock,
    _release_singleton_lock,
)
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


class DisconnectedAdapters(dict):
    """Expose a platform during collection, then simulate disconnect on get()."""

    def get(self, key, default=None):
        return None


async def _run_one_notifier_tick(monkeypatch, runner):
    real_sleep = asyncio.sleep

    async def fake_sleep(delay):
        if delay == 5:
            return
        runner._running = False
        await real_sleep(0)

    monkeypatch.setattr(asyncio, "sleep", fake_sleep)
    await runner._kanban_notifier_watcher(interval=1)


def _make_runner(adapter):
    runner = GatewayRunner.__new__(GatewayRunner)
    runner._running = True
    runner.adapters = {Platform.TELEGRAM: adapter}
    # Most tests model the default gateway after its dispatcher acquired the
    # singleton lock. Tests for startup or non-owner gateways clear this.
    runner._kanban_dispatcher_lock_handle = object()
    return runner


def _create_completed_subscription(summary="done once"):
    conn = kbc.connect()
    try:
        tid = kb.create_task(conn, title="notify once", assignee="worker")
        kbn.add_notify_sub(conn, task_id=tid, platform="telegram", chat_id="chat-1")
        kb.complete_task(conn, tid, summary=summary)
        return tid
    finally:
        conn.close()


def _unseen_terminal_events(tid):
    conn = kbc.connect()
    try:
        _, events = kbn.unseen_events_for_sub(
            conn,
            task_id=tid,
            platform="telegram",
            chat_id="chat-1",
            kinds=["completed", "blocked", "gave_up", "crashed", "timed_out"],
        )
        return events
    finally:
        conn.close()


def test_kanban_notifier_replays_telegram_dm_topic_delivery_metadata(tmp_path, monkeypatch):
    db_path = tmp_path / "dm-topic-metadata.db"
    monkeypatch.setenv("HERMES_KANBAN_DB", str(db_path))
    kb.init_db()

    conn = kbc.connect()
    try:
        tid = kb.create_task(
            conn,
            title="dm topic task",
            assignee="worker",
            session_id="agent:main:telegram:dm:chat-1",
        )
        kbn.add_notify_sub(
            conn,
            task_id=tid,
            platform="telegram",
            chat_id="chat-1",
            thread_id="20197",
            delivery_mode="notify+wake",
            delivery_metadata={
                "chat_type": "dm",
                "direct_messages_topic_id": "20197",
                "telegram_dm_topic_reply_fallback": True,
                "telegram_reply_to_message_id": "462",
                "thread_id": "20197",
            },
        )
        kb.complete_task(conn, tid, summary="done")
    finally:
        conn.close()

    adapter = RecordingAdapter()
    runner = _make_runner(adapter)
    asyncio.run(_run_one_notifier_tick(monkeypatch, runner))

    assert len(adapter.sent) == 1
    assert adapter.sent[0]["metadata"] == {
        "chat_type": "dm",
        "direct_messages_topic_id": "20197",
        "telegram_dm_topic_reply_fallback": True,
        "telegram_reply_to_message_id": "462",
        "thread_id": "20197",
    }
    assert len(adapter.handled) == 1
    assert adapter.handled[0].source.chat_type == "dm"
    assert adapter.handled[0].source.thread_id == "20197"


def test_active_named_profile_subscription_is_delivered(tmp_path, monkeypatch):
    """A sub stamped with the gateway's own named profile uses self.adapters.

    Regression for #71340: on a standalone (non-multiplex) gateway running a
    named profile, _authorization_adapter() used to treat the active name as a
    multiplex secondary, find no _profile_adapters entry, fail closed, and
    rewind the claim forever — silent zero-delivery.
    """
    db_path = tmp_path / "actionable-block.db"
    monkeypatch.setenv("HERMES_KANBAN_DB", str(db_path))
    kb.init_db()
    reason = "AGE-39 — https://linear.example/AGE-39 — publishing verified."
    conn = kbc.connect()
    try:
        tid = kb.create_task(conn, title="approval", assignee="publisher")
        kbn.add_notify_sub(
            conn,
            task_id=tid,
            platform="telegram",
            chat_id="chat-1",
            notifier_profile="main",
        )
        kb.block_task(conn, tid, reason=reason, kind="needs_input")
    finally:
        conn.close()

    adapter = RecordingAdapter()
    runner = _make_runner(adapter)
    runner._active_profile_name = lambda: "main"

    asyncio.run(_run_one_notifier_tick(monkeypatch, runner))

    assert len(adapter.sent) == 1
    message = adapter.sent[0]["text"]
    assert tid in message
    assert "blocked" in message


def test_non_dispatch_gateway_claims_only_its_profile_subscriptions(
    tmp_path, monkeypatch,
):
    """A profile gateway delivers its events while another gateway dispatches."""
    db_path = tmp_path / "cross-profile-notifier.db"
    monkeypatch.setenv("HERMES_KANBAN_DB", str(db_path))
    kb.init_db()
    conn = kbc.connect()
    try:
        foreign_tid = kb.create_task(
            conn, title="default-owned", assignee="worker",
        )
        kbn.add_notify_sub(
            conn,
            task_id=foreign_tid,
            platform="telegram",
            chat_id="default-chat",
            notifier_profile="default",
        )
        kb.complete_task(conn, foreign_tid, summary="default done")

        owned_tid = kb.create_task(
            conn, title="writer-owned", assignee="worker",
        )
        kbn.add_notify_sub(
            conn,
            task_id=owned_tid,
            platform="telegram",
            chat_id="writer-chat",
            notifier_profile="writer",
        )
        kb.complete_task(conn, owned_tid, summary="writer done")
    finally:
        conn.close()

    adapter = RecordingAdapter()
    runner = _make_runner(adapter)
    runner._active_profile_name = lambda: "writer"
    runner._kanban_dispatcher_lock_handle = None

    asyncio.run(_run_one_notifier_tick(monkeypatch, runner))

    assert [delivery["chat_id"] for delivery in adapter.sent] == ["writer-chat"]
    assert owned_tid in adapter.sent[0]["text"]
    assert len(_unseen_terminal_events_for(foreign_tid, "default-chat")) == 1


def test_legacy_subscription_requires_confirmed_dispatcher_lock_owner(
    tmp_path, monkeypatch,
):
    """Startup and lock-losing gateways cannot claim legacy notifications."""
    db_path = tmp_path / "legacy-lock-owner.db"
    monkeypatch.setenv("HERMES_KANBAN_DB", str(db_path))
    kb.init_db()
    conn = kbc.connect()
    try:
        task_id = kb.create_task(conn, title="legacy", assignee="worker")
        kbn.add_notify_sub(
            conn,
            task_id=task_id,
            platform="telegram",
            chat_id="legacy-chat",
        )
        kb.complete_task(conn, task_id, summary="legacy done")
    finally:
        conn.close()

    startup_adapter = RecordingAdapter()
    startup_runner = _make_runner(startup_adapter)
    startup_runner._kanban_dispatcher_lock_handle = None
    asyncio.run(_run_one_notifier_tick(monkeypatch, startup_runner))
    assert startup_adapter.sent == []
    assert len(_unseen_terminal_events_for(task_id, "legacy-chat")) == 1

    lock_path = tmp_path / ".dispatcher.lock"
    winner_handle, winner_state = _acquire_singleton_lock(lock_path)
    loser_handle, loser_state = _acquire_singleton_lock(lock_path)
    try:
        assert winner_state == "held"
        assert loser_state == "contended"

        loser_adapter = RecordingAdapter()
        loser_runner = _make_runner(loser_adapter)
        loser_runner._kanban_dispatcher_lock_handle = loser_handle
        asyncio.run(_run_one_notifier_tick(monkeypatch, loser_runner))
        assert loser_adapter.sent == []
        assert len(_unseen_terminal_events_for(task_id, "legacy-chat")) == 1

        winner_adapter = RecordingAdapter()
        winner_runner = _make_runner(winner_adapter)
        winner_runner._kanban_dispatcher_lock_handle = winner_handle
        asyncio.run(_run_one_notifier_tick(monkeypatch, winner_runner))
        assert [item["chat_id"] for item in winner_adapter.sent] == ["legacy-chat"]
        assert task_id in winner_adapter.sent[0]["text"]
    finally:
        _release_singleton_lock(loser_handle)
        _release_singleton_lock(winner_handle)


class FailingAdapter:
    """Adapter whose send() always raises, simulating a transient send error."""

    def __init__(self):
        self.attempts = 0

    async def send(self, chat_id, text, metadata=None):
        self.attempts += 1
        raise RuntimeError("simulated send failure")


class ReportedFailureAdapter:
    """Adapter that REPORTS failure via SendResult(success=False) instead of
    raising — the exact contract the Telegram adapter uses for 'Not connected'
    and degraded-send paths."""

    def __init__(self):
        self.attempts = 0

    async def send(self, chat_id, text, metadata=None):
        self.attempts += 1
        from gateway.platforms.base import SendResult
        return SendResult(success=False, error="Not connected")


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

    conn = kbc.connect()
    try:
        tid = kb.create_task(conn, title="cycle test", assignee="worker")
        kbn.add_notify_sub(conn, task_id=tid, platform="telegram", chat_id="chat-1")
        # First crash — fired by the dispatcher when the worker PID dies.
        kb._append_event(conn, tid, kind="crashed")
    finally:
        conn.close()

    adapter = RecordingAdapter()
    runner = _make_runner(adapter)
    asyncio.run(_run_one_notifier_tick(monkeypatch, runner))

    # First crash delivered.
    assert len(adapter.sent) == 1
    assert "stopped unexpectedly" in adapter.sent[0]["text"].lower()

    # Subscription survives — the cursor advanced past event #1, but the
    # row is still there.
    conn = kbc.connect()
    try:
        subs = kbn.list_notify_subs(conn, tid)
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
    assert "stopped unexpectedly" in adapter.sent[1]["text"].lower()


def test_notifier_subscription_survives_done_reopen_until_archive(
    tmp_path, monkeypatch,
):
    """Done is reversible; archive alone ends notification ownership."""
    db_path = tmp_path / "done-reopen-archive.db"
    monkeypatch.setenv("HERMES_KANBAN_DB", str(db_path))
    kb.init_db()

    conn = kbc.connect()
    try:
        tid = kb.create_task(
            conn,
            title="review continuation",
            assignee="worker",
            session_id="origin-session",
        )
        kbn.add_notify_sub(
            conn,
            task_id=tid,
            platform="telegram",
            chat_id="origin-chat",
            thread_id="origin-thread",
            user_id="origin-user",
            chat_type="group",
            notifier_profile="reviewer",
            delivery_mode="notify+wake",
        )
        assert kb.complete_task(conn, tid, summary="first completion")
    finally:
        conn.close()

    adapter = RecordingAdapter()
    runner = _make_runner(adapter)
    runner._active_profile_name = lambda: "reviewer"
    asyncio.run(_run_one_notifier_tick(monkeypatch, runner))

    assert len(adapter.sent) == 1
    assert len(adapter.handled) == 1
    assert adapter.sent[0]["chat_id"] == "origin-chat"
    assert adapter.sent[0]["metadata"]["thread_id"] == "origin-thread"
    assert adapter.handled[0].source.thread_id == "origin-thread"
    assert adapter.handled[0].source.profile == "reviewer"

    conn = kbc.connect()
    try:
        subs = kbn.list_notify_subs(conn, tid)
        assert len(subs) == 1, "completion must retain the origin subscription"
        first_cursor = subs[0]["last_event_id"]
    finally:
        conn.close()

    # A quiet tick proves the completed event cannot replay after its cursor
    # was advanced, even though the subscription now remains present.
    runner = _make_runner(adapter)
    runner._active_profile_name = lambda: "reviewer"
    asyncio.run(_run_one_notifier_tick(monkeypatch, runner))
    assert len(adapter.sent) == 1
    assert len(adapter.handled) == 1

    conn = kbc.connect()
    try:
        with kb.write_txn(conn):
            conn.execute("UPDATE tasks SET status = 'ready' WHERE id = ?", (tid,))
            kb._append_event(conn, tid, "status", {"status": "ready"})
        assert kb.complete_task(conn, tid, summary="corrected completion")
    finally:
        conn.close()

    runner = _make_runner(adapter)
    runner._active_profile_name = lambda: "reviewer"
    asyncio.run(_run_one_notifier_tick(monkeypatch, runner))

    # The reopen status and second completion each deliver once, while only
    # completion wakes the exact original session/thread.
    assert len(adapter.sent) == 3
    assert len(adapter.handled) == 2
    assert all(item["chat_id"] == "origin-chat" for item in adapter.sent)
    assert adapter.handled[-1].source.thread_id == "origin-thread"
    assert adapter.handled[-1].source.profile == "reviewer"

    conn = kbc.connect()
    try:
        subs = kbn.list_notify_subs(conn, tid)
        assert len(subs) == 1
        assert subs[0]["last_event_id"] > first_cursor
        assert kb.archive_task(conn, tid)
    finally:
        conn.close()

    runner = _make_runner(adapter)
    runner._active_profile_name = lambda: "reviewer"
    asyncio.run(_run_one_notifier_tick(monkeypatch, runner))

    # Archive itself is intentionally silent, but consumes its event and
    # removes the subscription so no later historical event can replay.
    assert len(adapter.sent) == 3
    assert len(adapter.handled) == 2
    conn = kbc.connect()
    try:
        assert kbn.list_notify_subs(conn, tid) == []
    finally:
        conn.close()


def test_notifier_wakeup_uses_subscription_chat_type(tmp_path, monkeypatch):
    db_path = tmp_path / "chat-type-wakeup.db"
    monkeypatch.setenv("HERMES_KANBAN_DB", str(db_path))
    kb.init_db()

    conn = kbc.connect()
    try:
        tid = kb.create_task(
            conn,
            title="dm requester",
            assignee="worker",
            session_id="origin-session",
        )
        kbn.add_notify_sub(
            conn,
            task_id=tid,
            platform="telegram",
            chat_id="chat-dm",
            chat_type="dm",
            delivery_mode="notify+wake",
        )
        kb.complete_task(conn, tid, summary="done")
    finally:
        conn.close()

    adapter = RecordingAdapter()
    asyncio.run(_run_one_notifier_tick(monkeypatch, _make_runner(adapter)))

    assert len(adapter.sent) == 1
    assert len(adapter.handled) == 1
    assert adapter.handled[0].source.chat_type == "dm"

    # The wake must resume the creator's real DM session key — the whole bug
    # was that a hardcoded chat_type="group" made build_session_key() produce
    # a group-scoped key (a NEW session) instead of the ":dm:<chat_id>" shape
    # the original conversation runs under (#56580 / #68874).
    from gateway.session import build_session_key

    wake_key = build_session_key(adapter.handled[0].source)
    assert wake_key == "agent:main:telegram:dm:chat-dm"
    assert ":group:" not in wake_key


def _unseen_terminal_events_for(tid, chat_id):
    conn = kbc.connect()
    try:
        _, events = kbn.unseen_events_for_sub(
            conn,
            task_id=tid,
            platform="telegram",
            chat_id=chat_id,
            kinds=["completed", "blocked", "gave_up", "crashed", "timed_out"],
        )
        return events
    finally:
        conn.close()


def test_kanban_notifier_isolates_per_subscription_failure(tmp_path, monkeypatch):
    """One bad subscription must not block delivery for all others.

    Regression for #59269: when claim_unseen_events_for_sub raises for one
    subscription, the entire notifier tick used to abort — silently blocking
    delivery for every other subscription.
    """
    db_path = tmp_path / "isolation.db"
    monkeypatch.setenv("HERMES_KANBAN_DB", str(db_path))
    kb.init_db()

    # Create two tasks with subscriptions and complete both. The BAD task is
    # created first: list_notify_subs() has no ORDER BY, so SQLite's natural
    # scan returns insertion order — the failing subscription must be
    # processed BEFORE the good one or this test passes even without the
    # per-subscription isolation (the good delivery happens before the tick
    # aborts). A deterministic-order shim below removes the reliance on the
    # scan order entirely.
    conn = kbc.connect()
    try:
        tid_bad = kb.create_task(conn, title="bad task", assignee="worker")
        kbn.add_notify_sub(conn, task_id=tid_bad, platform="telegram", chat_id="chat-bad")
        kb.complete_task(conn, tid_bad, summary="done")

        tid_good = kb.create_task(conn, title="good task", assignee="worker")
        kbn.add_notify_sub(conn, task_id=tid_good, platform="telegram", chat_id="chat-good")
        kb.complete_task(conn, tid_good, summary="done")
    finally:
        conn.close()

    original_claim = kbn.claim_unseen_events_for_sub

    def selective_claim(conn, task_id, **kwargs):
        if task_id == tid_bad:
            raise RuntimeError("simulated DB corruption for bad task")
        return original_claim(conn, task_id=task_id, **kwargs)

    monkeypatch.setattr(kbn, "claim_unseen_events_for_sub", selective_claim)

    # Force the failing subscription to be iterated FIRST regardless of the
    # unordered SELECT's scan order.
    original_list = kbn.list_notify_subs

    def bad_first(conn, task_id=None, **kwargs):
        subs = original_list(conn, task_id, **kwargs)
        return sorted(subs, key=lambda s: 0 if s["task_id"] == tid_bad else 1)

    monkeypatch.setattr(kbn, "list_notify_subs", bad_first)

    adapter = RecordingAdapter()
    runner = _make_runner(adapter)

    asyncio.run(_run_one_notifier_tick(monkeypatch, runner))

    # The good task must still be delivered despite the bad task failing.
    assert len(adapter.sent) == 1
    assert tid_good in adapter.sent[0]["text"]


def test_notifier_delivers_block_loop_detected_triage_ping(tmp_path, monkeypatch):
    """A `block_loop_detected` event must reach the subscriber as a triage ping.

    Regression for the silent-triage gap (PR #62712): kanban_db routes a task
    to `triage` after BLOCK_RECURRENCE_LIMIT re-blocks for the same cause and
    emits ONLY a `block_loop_detected` event — no `blocked`/`status` event.
    Before `block_loop_detected` joined TERMINAL_KINDS with its own message
    branch, that one transition (the whole point of which is to force human
    attention) produced zero notification and the task stalled in triage
    silently.
    """
    db_path = tmp_path / "block-loop.db"
    monkeypatch.setenv("HERMES_KANBAN_DB", str(db_path))
    kb.init_db()

    conn = kbc.connect()
    try:
        tid = kb.create_task(conn, title="loops forever", assignee="worker")
        kbn.add_notify_sub(conn, task_id=tid, platform="telegram", chat_id="chat-1")
        kb._append_event(
            conn, tid, "block_loop_detected",
            {"reason": "needs credentials", "kind": "needs_input",
             "recurrences": 2, "limit": kb.BLOCK_RECURRENCE_LIMIT},
        )
    finally:
        conn.close()

    adapter = RecordingAdapter()
    runner = _make_runner(adapter)

    asyncio.run(_run_one_notifier_tick(monkeypatch, runner))

    assert len(adapter.sent) == 1, "block_loop_detected must produce a notification"
    text = adapter.sent[0]["text"]
    assert tid in text
    assert "needs credentials" in text
    # Cursor advanced: the event is claimed and not re-delivered.
    conn = kbc.connect()
    try:
        _, remaining = kbn.unseen_events_for_sub(
            conn, task_id=tid, platform="telegram", chat_id="chat-1",
            kinds=["block_loop_detected"],
        )
    finally:
        conn.close()
    assert remaining == []


# ---------------------------------------------------------------------------
# #111125 — a repeated-block circuit breaker establishes that orchestration
# attention is needed, NOT that a human decision exists. The formatter must
# use neutral wording unless the block was typed as a genuine owner-input
# request (`needs_input`, the only kind that carries a concrete question).
# ---------------------------------------------------------------------------


class _StubEvent:
    def __init__(self, payload):
        self.payload = payload


class _StubNotif:
    head = "H123"


def _fmt_block_loop(payload):
    from gateway.kanban_watchers_notifier import _EVENT_FORMATTERS

    msg, _, _ = _EVENT_FORMATTERS["block_loop_detected"](_StubEvent(payload), _StubNotif())
    return msg


def test_block_loop_technical_kind_uses_neutral_orchestration_wording():
    """A repeated technical block (transient/capability/untyped) routed to
    triage is an orchestration handoff with no question for the owner, so the
    ping must not claim a human decision (#111125)."""
    payload = {"reason": "waiting on upstream", "kind": "transient", "recurrences": 2}
    msg = _fmt_block_loop(payload)
    assert "for orchestration attention" in msg
    assert "human decision" not in msg
    # Circuit-breaker visibility is preserved.
    assert "TRIAGE" in msg
    assert "waiting on upstream" in msg


def test_block_loop_owner_input_keeps_decision_wording():
    """A `needs_input` block carries a concrete question for the owner, so the
    owner-decision wording is correct and must be retained (#111125)."""
    payload = {
        "reason": "Which API key should this use?",
        "kind": "needs_input",
        "recurrences": 2,
        "limit": kb.BLOCK_RECURRENCE_LIMIT,
    }
    msg = _fmt_block_loop(payload)
    assert "needs a human decision" in msg
    assert "for orchestration attention" not in msg
    assert "Which API key should this use?" in msg


# ---------------------------------------------------------------------------
# Handoffs that hand a decision back to the origin must wake it, not only ping
# it: `review_requested` (implementation done, waiting for a reviewer) and
# `block_loop_detected` (routed to triage) are terminal kinds just like
# `blocked`.
# ---------------------------------------------------------------------------


def _wake_text(adapter):
    """Text of the single synthetic wake turn injected into the adapter."""
    assert len(adapter.handled) == 1, (
        f"expected exactly one wake turn, got {len(adapter.handled)}"
    )
    return getattr(adapter.handled[0], "text", "") or ""


def _review_handoff_task(
    *,
    delivery_mode="notify+wake",
    summary="PR ready: https://example.invalid/pr/7\nfull details below",
):
    conn = kbc.connect()
    try:
        tid = kb.create_task(
            conn,
            title="implement the thing",
            assignee="worker",
            session_id="agent:main:telegram:dm:chat-1",
        )
        kbn.add_notify_sub(
            conn,
            task_id=tid,
            platform="telegram",
            chat_id="chat-1",
            chat_type="dm",
            delivery_mode=delivery_mode,
        )
        kb.claim_task(conn, tid)
        run_id = kb.get_task(conn, tid).current_run_id
        assert kb.request_review(
            conn, tid, summary=summary, expected_run_id=run_id,
        ) is True
        return tid
    finally:
        conn.close()


def test_review_requested_wakes_the_origin_session(tmp_path, monkeypatch):
    """A review handoff wakes the origin and carries the worker's summary."""
    monkeypatch.setenv("HERMES_KANBAN_DB", str(tmp_path / "review-wake.db"))
    kb.init_db()
    tid = _review_handoff_task()

    adapter = RecordingAdapter()
    runner = _make_runner(adapter)
    asyncio.run(_run_one_notifier_tick(monkeypatch, runner))

    assert len(adapter.sent) == 1, "the passive review ping is unchanged"

    wake = _wake_text(adapter)
    assert tid in wake
    assert "PR ready: https://example.invalid/pr/7" in wake, (
        "the worker's handoff must ride the wake turn like it does for "
        "`completed`, otherwise the woken reviewer has to re-read the board"
    )


def test_block_loop_detected_wakes_the_origin_session(tmp_path, monkeypatch):
    """A triage escalation wakes the origin so a decision gets made."""
    monkeypatch.setenv("HERMES_KANBAN_DB", str(tmp_path / "triage-wake.db"))
    kb.init_db()

    conn = kbc.connect()
    try:
        tid = kb.create_task(
            conn,
            title="loops forever",
            assignee="worker",
            session_id="agent:main:telegram:dm:chat-1",
        )
        kbn.add_notify_sub(
            conn,
            task_id=tid,
            platform="telegram",
            chat_id="chat-1",
            chat_type="dm",
            delivery_mode="notify+wake",
        )
        kb._append_event(
            conn, tid, "block_loop_detected",
            {"reason": "needs credentials", "kind": "needs_input",
             "recurrences": 2, "limit": kb.BLOCK_RECURRENCE_LIMIT},
        )
    finally:
        conn.close()

    adapter = RecordingAdapter()
    runner = _make_runner(adapter)
    asyncio.run(_run_one_notifier_tick(monkeypatch, runner))

    assert len(adapter.sent) == 1
    assert tid in _wake_text(adapter)


def test_review_requested_does_not_wake_a_notify_only_subscription(
    tmp_path, monkeypatch,
):
    """delivery_mode still decides whether a wake-worthy kind wakes at all."""
    monkeypatch.setenv("HERMES_KANBAN_DB", str(tmp_path / "review-notify.db"))
    kb.init_db()
    _review_handoff_task(delivery_mode="notify")

    adapter = RecordingAdapter()
    runner = _make_runner(adapter)
    asyncio.run(_run_one_notifier_tick(monkeypatch, runner))

    assert len(adapter.sent) == 1
    assert adapter.handled == [], (
        "notify-only subscriptions must not be woken by a review handoff"
    )


# --- Durable parking of permanently-refused deliveries ----------------------


def _park_row(tid):
    conn = kbc.connect()
    try:
        return conn.execute(
            "SELECT delivery_failures, retry_after, last_event_id, last_ping_event_id"
            " FROM kanban_notify_subs WHERE task_id = ?",
            (tid,),
        ).fetchone()
    finally:
        conn.close()


def _board_data_version():
    conn = kbc.connect()
    try:
        return conn.execute("PRAGMA data_version").fetchone()[0]
    finally:
        conn.close()


def _backdate_park(tid):
    conn = kbc.connect()
    try:
        with kb.write_txn(conn):
            conn.execute(
                "UPDATE kanban_notify_subs SET retry_after = ? WHERE task_id = ?",
                (time.time() - 1.0, tid),
            )
    finally:
        conn.close()


class PermanentRefusalAdapter(RecordingAdapter):
    """Wake leg refused with a final store code (dead destination), the same
    classification ``run_notifications`` applies via PERMANENT_REFUSAL_REASONS."""

    def __init__(self):
        super().__init__()
        self.wake_attempts = 0

    async def handle_message(self, event):
        self.wake_attempts += 1
        from gateway.wake import WakeNotAccepted
        raise WakeNotAccepted("no session behind the wake destination", reason="not_found")


def test_permanent_wake_refusal_parks_with_backoff_and_zero_writes_while_parked(
    tmp_path, monkeypatch,
):
    """A permanently-refused destination must stop re-attempting every tick:
    one counted attempt parks the subscription (durable, backoff-scheduled);
    parked ticks claim nothing and write nothing. Today the refusal rewinds
    uncounted forever — a claim/rewind write pair every poll tick."""
    monkeypatch.setenv("HERMES_KANBAN_DB", str(tmp_path / "park-permanent.db"))
    kb.init_db()
    conn = kbc.connect()
    try:
        tid = kb.create_task(conn, title="park me", assignee="worker")
        kbn.add_notify_sub(conn, task_id=tid, platform="telegram", chat_id="chat-1",
                           chat_type="dm", delivery_mode="notify+wake")
        kb.complete_task(conn, tid, summary="done once")
    finally:
        conn.close()

    adapter = PermanentRefusalAdapter()
    runner = _make_runner(adapter)
    asyncio.run(_run_one_notifier_tick(monkeypatch, runner))

    assert len(adapter.sent) == 1, "the notify leg delivered its ping first"
    row = _park_row(tid)
    assert row["delivery_failures"] == 1, "a permanent refusal is counted durably"
    assert row["retry_after"] is not None and row["retry_after"] > time.time()
    assert _unseen_terminal_events(tid), "claim must rewind so parked events redeliver later"
    assert row["last_ping_event_id"] > 0, "the sent ping is checkpointed before the wake leg"

    # Parked rows are excluded from claims: no send, no wake attempt, no DB write.
    version = _board_data_version()
    for _ in range(3):
        asyncio.run(_run_one_notifier_tick(monkeypatch, _make_runner(adapter)))
    assert adapter.wake_attempts == 1
    assert len(adapter.sent) == 1
    assert _board_data_version() == version, "parked ticks must not write to the board"
    assert _park_row(tid)["delivery_failures"] == 1

    # Backoff expiry: exactly one new attempt, count grows, delay doubles, ping
    # checkpoint still holds (at-least-once wake, exactly-once ping).
    previous_retry_after = _park_row(tid)["retry_after"]
    _backdate_park(tid)
    asyncio.run(_run_one_notifier_tick(monkeypatch, _make_runner(adapter)))
    assert adapter.wake_attempts == 2
    row = _park_row(tid)
    assert row["delivery_failures"] == 2
    assert row["retry_after"] > previous_retry_after
    assert len(adapter.sent) == 1


def test_transient_wake_refusal_keeps_uncounted_immediate_retry(tmp_path, monkeypatch):
    """Startup / full queue is not a dead destination (bf1bf7515a): draining
    refusals keep today's uncounted rewind — retried every tick, never parked."""
    monkeypatch.setenv("HERMES_KANBAN_DB", str(tmp_path / "park-transient.db"))
    kb.init_db()
    conn = kbc.connect()
    try:
        tid = kb.create_task(conn, title="transient", assignee="worker")
        kbn.add_notify_sub(conn, task_id=tid, platform="telegram", chat_id="chat-1",
                           chat_type="dm", delivery_mode="notify+wake")
        kb.complete_task(conn, tid, summary="done once")
    finally:
        conn.close()

    adapter = PermanentRefusalAdapter()

    async def draining(event):
        adapter.wake_attempts += 1
        from gateway.wake import WakeNotAccepted
        raise WakeNotAccepted("gateway is draining; retry after restart", reason="runtime_draining")

    adapter.handle_message = draining
    for _ in range(2):
        asyncio.run(_run_one_notifier_tick(monkeypatch, _make_runner(adapter)))
    assert adapter.wake_attempts == 2, "transient refusals retry on every tick"
    row = _park_row(tid)
    assert row["delivery_failures"] == 0, "transient refusals are not counted"
    assert row["retry_after"] is None, "transient refusals never park"
    assert _unseen_terminal_events(tid), "events stay claimed-pending until admission"


def test_consecutive_send_failures_park_capped_instead_of_dropping(tmp_path, monkeypatch):
    """The counted send-failure path parks (durable, backoff capped at 1h) —
    it no longer deletes the subscription after 12 in-memory strikes, and the
    count survives a restart because it lives on the subscription row."""
    monkeypatch.setenv("HERMES_KANBAN_DB", str(tmp_path / "park-send-cap.db"))
    kb.init_db()
    conn = kbc.connect()
    try:
        tid = kb.create_task(conn, title="flappy send", assignee="worker")
        kbn.add_notify_sub(conn, task_id=tid, platform="telegram", chat_id="chat-1")
        kb.complete_task(conn, tid, summary="done once")
    finally:
        conn.close()

    adapter = FailingAdapter()
    for attempt in range(1, 13):
        _backdate_park(tid)  # no-op before the first failure (columns still default)
        asyncio.run(_run_one_notifier_tick(monkeypatch, _make_runner(adapter)))
        row = _park_row(tid)
        assert row is not None, "parking must never delete the subscription row"
        assert row["delivery_failures"] == attempt
    assert adapter.attempts == 12
    # 5s * 2**11 far exceeds the 1h cap — the 12th backoff is capped.
    assert time.time() + 3599.0 <= _park_row(tid)["retry_after"] <= time.time() + 3601.0


def test_resubscribe_rearms_a_parked_subscription(tmp_path, monkeypatch):
    """Re-subscribing is the operator's re-arm gesture: the destination's own
    session asking again proves it is reachable, so the park resets and the
    next tick delivers normally."""
    monkeypatch.setenv("HERMES_KANBAN_DB", str(tmp_path / "park-rearm.db"))
    kb.init_db()
    conn = kbc.connect()
    try:
        tid = kb.create_task(conn, title="rearm", assignee="worker")
        kbn.add_notify_sub(conn, task_id=tid, platform="telegram", chat_id="chat-1",
                           chat_type="dm", delivery_mode="notify+wake")
        kb.complete_task(conn, tid, summary="done once")
    finally:
        conn.close()

    refusing = PermanentRefusalAdapter()
    asyncio.run(_run_one_notifier_tick(monkeypatch, _make_runner(refusing)))
    assert _park_row(tid)["delivery_failures"] == 1

    conn = kbc.connect()
    try:
        kbn.add_notify_sub(conn, task_id=tid, platform="telegram", chat_id="chat-1",
                           chat_type="dm", delivery_mode="notify+wake")
    finally:
        conn.close()
    row = _park_row(tid)
    assert row["delivery_failures"] == 0 and row["retry_after"] is None

    healthy = RecordingAdapter()
    asyncio.run(_run_one_notifier_tick(monkeypatch, _make_runner(healthy)))
    assert len(healthy.handled) == 1, "rearmed sub delivers on the very next tick"
    assert _unseen_terminal_events(tid) == []
    row = _park_row(tid)
    assert row["delivery_failures"] == 0 and row["retry_after"] is None


def test_ping_checkpoint_failure_is_counted_and_redelivers_at_least_once(
    tmp_path, monkeypatch,
):
    """A ping whose durable checkpoint write fails consumes the failure budget
    (park + rewind) and the ping is resent on the retry — no checkpoint, no
    dedup. A later clean delivery resets the counters."""
    monkeypatch.setenv("HERMES_KANBAN_DB", str(tmp_path / "park-pingfail.db"))
    kb.init_db()
    conn = kbc.connect()
    try:
        tid = kb.create_task(conn, title="ping checkpoint", assignee="worker")
        kbn.add_notify_sub(conn, task_id=tid, platform="telegram", chat_id="chat-1",
                           chat_type="dm", delivery_mode="notify+wake")
        kb.complete_task(conn, tid, summary="done once")
    finally:
        conn.close()

    real_record = kbn.record_notify_ping
    state = {"failed_once": False}

    def flaky_record(conn, **kwargs):
        if not state["failed_once"]:
            state["failed_once"] = True
            raise sqlite3.OperationalError("disk I/O error (simulated)")
        return real_record(conn, **kwargs)

    monkeypatch.setattr(kbn, "record_notify_ping", flaky_record)
    adapter = RecordingAdapter()
    asyncio.run(_run_one_notifier_tick(monkeypatch, _make_runner(adapter)))

    assert len(adapter.sent) == 1, "the ping itself was sent"
    row = _park_row(tid)
    assert row["delivery_failures"] == 1, "checkpoint failure consumes the budget"
    assert row["retry_after"] is not None
    assert row["last_ping_event_id"] == 0, "checkpoint never landed"
    assert _unseen_terminal_events(tid), "rewound for a later retry"

    _backdate_park(tid)
    asyncio.run(_run_one_notifier_tick(monkeypatch, _make_runner(adapter)))
    assert len(adapter.sent) == 2, "no checkpoint → the ping is resent (at-least-once)"
    assert len(adapter.handled) == 1, "wake delivered once the checkpoint holds"
    row = _park_row(tid)
    assert row["delivery_failures"] == 0 and row["retry_after"] is None, "success resets"
    assert _unseen_terminal_events(tid) == []
