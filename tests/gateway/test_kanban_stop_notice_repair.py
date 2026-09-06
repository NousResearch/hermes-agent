
from hermes_cli import kanban_db_connect, kanban_db_dispatch, kanban_db_notify
"""Silent-stop repair: every terminal stop must produce one actionable notice.

Reproduces the reefmind ``t_5e4df497`` incident (2026-09-06):

* run 10730 ended with a ``dependency_wait`` event. ``dependency_wait`` was not
  in the notifier's ``TERMINAL_KINDS``, so the stop produced **zero** Discord
  notices and the card was auto-promoted back to ``ready`` eight seconds later,
  re-running the same verification as run 10739.
* run 10739 ended with ``blocked`` carrying a legacy prose
  ``review-required:`` review-handoff prefix. The notice rendered as a bare
  ``⏸ Kanban <id> blocked: <reason truncated at 160 chars>`` — no card title,
  no Ready yes/no, no next owner or action — so the review handoff was not
  actionable and the parent had to notice the stall and hand-create the review
  card much later.

The transport is a fake in-process adapter; the notifier, the Kanban DB, the
subscription cursor, durable delivery ledger, and structured review lifecycle
are real. Only ``request_review`` establishes a review route.
"""

import asyncio
import json

import pytest

from gateway.config import Platform
from gateway.platforms.base import SendResult
from gateway.run import GatewayRunner
from hermes_cli import kanban_db as kb


class RecordingAdapter:
    """Fake Discord transport. Records sends; never touches the network."""

    def __init__(self):
        self.sent = []
        self.handled = []
        self.fail_next = 0

    async def send(self, chat_id, text, metadata=None):
        if self.fail_next > 0:
            self.fail_next -= 1
            return SendResult(
                success=False,
                error="simulated confirmed transport rejection",
                raw_response={"delivery_rejected": True},
            )
        self.sent.append({"chat_id": chat_id, "text": text, "metadata": metadata or {}})

    async def handle_message(self, event):
        self.handled.append(event)


def _make_runner(adapter):
    runner = GatewayRunner.__new__(GatewayRunner)
    runner._running = True
    runner.adapters = {Platform.DISCORD: adapter}
    runner._kanban_sub_fail_counts = {}
    runner._kanban_dispatcher_lock_handle = object()
    return runner


async def _one_tick(monkeypatch, runner):
    real_sleep = asyncio.sleep

    async def fake_sleep(delay):
        if delay == 5:
            return None
        runner._running = False
        await real_sleep(0)

    monkeypatch.setattr(asyncio, "sleep", fake_sleep)
    await runner._kanban_notifier_watcher(interval=1)


def _tick(monkeypatch, runner):
    runner._running = True
    asyncio.run(_one_tick(monkeypatch, runner))


@pytest.fixture()
def board(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_KANBAN_DB", str(tmp_path / "stop-notice.db"))
    kanban_db_connect.init_db()
    return tmp_path


def _subscribed_task(title="INT-3164 · Assess and adopt the official HYDROS API",
                     assignee="orchestrator", delivery_mode="notify+wake"):
    conn = kanban_db_connect.connect()
    try:
        tid = kb.create_task(conn, title=title, assignee=assignee)
        kanban_db_notify.add_notify_sub(
            conn,
            task_id=tid,
            platform="discord",
            chat_id="1543672226589704283",
            thread_id="1543672226589704283",
            chat_type="thread",
            delivery_mode=delivery_mode,
        )
        return tid
    finally:
        conn.close()

def _emit(tid, kind, payload=None):
    conn = kanban_db_connect.connect()
    try:
        with kanban_db_connect.write_txn(conn):
            kb._append_event(conn, tid, kind, payload or {})
    finally:
        conn.close()


def _texts(adapter):
    return [s["text"] for s in adapter.sent]


REVIEW_REASON = (
    "review-required: PR #3165 @ 1703d99b9 is delivery-complete and "
    "independently re-verified this run — non-Draft, MERGEABLE, contains "
    "current main, hosted CI 34043721559 SUCCESS 17/17 at that exact head, "
    "local core lane green, browser proof 33/33. The only remaining gate is "
    "an independent exact-head review, which this implementation lane must "
    "not self-approve."
)


# --------------------------------------------------------------------------
# 1. dependency_wait — the run-10730 silent stop
# --------------------------------------------------------------------------

def test_dependency_wait_produces_exactly_one_actionable_notice(board, monkeypatch):
    tid = _subscribed_task()
    _emit(tid, "dependency_wait", {
        "reason": "review-handoff: PR #3165 needs an independent reviewer",
        "kind": "dependency",
        "source_status": "ready",
    })

    adapter = RecordingAdapter()
    runner = _make_runner(adapter)
    _tick(monkeypatch, runner)

    assert len(adapter.sent) == 1, (
        "a dependency_wait stop must produce exactly one notice; got "
        f"{_texts(adapter)!r}"
    )
    text = adapter.sent[0]["text"]
    assert tid in text
    assert "HYDROS" in text, "notice must name the card title"
    assert "review-handoff" in text, "notice must carry the exact stop reason"
    assert "Ready:" in text, "notice must state Ready yes/no"
    assert "Next:" in text, "notice must state the next owner/action"


def test_dependency_wait_without_unsatisfied_parent_flags_the_respin(board, monkeypatch):
    """A dependency wait with no unfinished parent auto-promotes and respins."""
    tid = _subscribed_task()
    _emit(tid, "dependency_wait", {"reason": "waiting on review", "kind": "dependency"})

    adapter = RecordingAdapter()
    runner = _make_runner(adapter)
    _tick(monkeypatch, runner)

    text = adapter.sent[0]["text"]
    assert "no unsatisfied dependency" in text.lower(), (
        "the notice must say the board will re-dispatch the same work when the "
        f"dependency wait has no real parent to wait on; got {text!r}"
    )


def test_dependency_wait_with_real_parent_names_the_blocking_parent(board, monkeypatch):
    conn = kanban_db_connect.connect()
    try:
        parent = kb.create_task(conn, title="upstream migration", assignee="backend")
    finally:
        conn.close()
    tid = _subscribed_task()
    conn = kanban_db_connect.connect()
    try:
        kb.link_tasks(conn, parent_id=parent, child_id=tid)
    finally:
        conn.close()
    _emit(tid, "dependency_wait", {"reason": "waiting on upstream", "kind": "dependency"})

    adapter = RecordingAdapter()
    runner = _make_runner(adapter)
    _tick(monkeypatch, runner)

    text = adapter.sent[0]["text"]
    assert parent in text, f"notice must name the blocking parent; got {text!r}"
    assert "no unsatisfied dependency" not in text.lower()


# --------------------------------------------------------------------------
# 2. blocked — the run-10739 unactionable stop
# --------------------------------------------------------------------------

def test_blocked_notice_is_actionable(board, monkeypatch):
    tid = _subscribed_task()
    _emit(tid, "blocked", {"reason": "collector credentials are missing", "kind": "capability"})

    adapter = RecordingAdapter()
    runner = _make_runner(adapter)
    _tick(monkeypatch, runner)

    assert len(adapter.sent) == 1
    text = adapter.sent[0]["text"]
    assert "HYDROS" in text, "blocked notice must name the card title"
    assert "collector credentials are missing" in text
    assert "Ready:" in text
    assert "Next:" in text
    assert "@orchestrator" in text, "notice must name the next owner"


def test_blocked_review_prose_does_not_impersonate_structured_review(board, monkeypatch):
    tid = _subscribed_task()
    _emit(tid, "blocked", {"reason": REVIEW_REASON, "kind": "needs_input"})

    adapter = RecordingAdapter()
    runner = _make_runner(adapter)
    _tick(monkeypatch, runner)

    assert len(adapter.sent) == 1
    text = adapter.sent[0]["text"]
    assert "stopped for REVIEW HANDOFF" not in text
    assert "blocked" in text.lower()
    assert "unblock" in text.lower()
    # The reason is the operator's whole decision basis — 160 chars cut it
    # mid-sentence in the incident.
    assert "independent exact-head review" in text, (
        "the actionable part of the reason must survive truncation"
    )


def test_blocked_notice_never_leaks_the_whole_reason_unbounded(board, monkeypatch):
    tid = _subscribed_task()
    _emit(tid, "blocked", {"reason": "x" * 5000, "kind": "needs_input"})

    adapter = RecordingAdapter()
    runner = _make_runner(adapter)
    _tick(monkeypatch, runner)

    assert len(adapter.sent[0]["text"]) < 1600, "notice must stay inside one Discord message"


# --------------------------------------------------------------------------
# 3. gave_up / crashed
# --------------------------------------------------------------------------

def test_gave_up_produces_one_actionable_notice_and_no_wake(board, monkeypatch):
    tid = _subscribed_task()
    _emit(tid, "gave_up", {"error": "claude CLI exited 1: not logged in"})

    adapter = RecordingAdapter()
    runner = _make_runner(adapter)
    _tick(monkeypatch, runner)

    assert len(adapter.sent) == 1
    text = adapter.sent[0]["text"]
    assert "HYDROS" in text
    assert "not logged in" in text
    assert "Ready:" in text and "Next:" in text
    # Current operator policy: gave_up notifies, it does not wake an agent.
    assert adapter.handled == []


def test_crashed_notice_asks_for_an_operator_decision_not_an_autonomous_retry(board, monkeypatch):
    tid = _subscribed_task(delivery_mode="notify")
    _emit(tid, "crashed", {})

    adapter = RecordingAdapter()
    runner = _make_runner(adapter)
    _tick(monkeypatch, runner)

    assert len(adapter.sent) == 1
    text = adapter.sent[0]["text"]
    assert "dispatcher will retry" not in text, (
        "the crash notice must not promise an autonomous retry — it is an "
        f"operator decision; got {text!r}"
    )
    assert "operator decision" in text.lower()
    assert "Ready:" in text and "Next:" in text


def test_detected_crash_is_held_then_notified_and_wakes_for_operator_decision(
    board, monkeypatch
):
    tid = _subscribed_task(delivery_mode="notify+wake")
    conn = kanban_db_connect.connect()
    try:
        claimed = kb.claim_task(
            conn, tid,
            claimer=f"{kb._claimer_id().split(':', 1)[0]}:crash-test",
        )
        assert claimed is not None
        kanban_db_dispatch._set_worker_pid(conn, tid, 818181)
        conn.execute("UPDATE tasks SET started_at = 1 WHERE id = ?", (tid,))
        conn.commit()
        monkeypatch.setattr(kb, "_pid_alive", lambda _pid: False)
        monkeypatch.setattr(
            kanban_db_dispatch, "_classify_worker_exit", lambda _pid: ("nonzero_exit", 23),
        )
        monkeypatch.setattr(kb, "_resolve_crash_grace_seconds", lambda: 0)
        assert kanban_db_dispatch.detect_crashed_workers(conn) == [tid]
        assert kb.get_task(conn, tid).status == "blocked"
    finally:
        conn.close()

    adapter = RecordingAdapter()
    _tick(monkeypatch, _make_runner(adapter))

    assert len(adapter.sent) == 1
    assert "exited with code 23" in adapter.sent[0]["text"]
    assert "Ready: no" in adapter.sent[0]["text"]
    assert len(adapter.handled) == 1
    assert tid in adapter.handled[0].text


# --------------------------------------------------------------------------
# 4. spam containment
#
# `status` notices are deliberately left alone: an existing pin
# (tests/gateway/test_kanban_notifier.py) requires a reopen (`→ ready`) to
# notify, and the incident's board history shows only three `status` events in
# two hours — status writes are operator intent, not dispatcher churn. What has
# to hold is that the newly-notified kinds add no repeat traffic and that the
# silent kinds never wedge a stop behind them.
# --------------------------------------------------------------------------

def test_dependency_wait_is_not_repeated_on_later_ticks(board, monkeypatch):
    tid = _subscribed_task(delivery_mode="notify")
    _emit(tid, "dependency_wait", {"reason": "waiting on review", "kind": "dependency"})

    adapter = RecordingAdapter()
    runner = _make_runner(adapter)
    _tick(monkeypatch, runner)
    _tick(monkeypatch, runner)
    _tick(monkeypatch, runner)

    assert len(adapter.sent) == 1, (
        f"a dependency wait must ping once, not every tick; got {_texts(adapter)!r}"
    )


def test_dependency_wait_never_wakes_the_agent(board, monkeypatch):
    tid = _subscribed_task()
    _emit(tid, "dependency_wait", {"reason": "waiting on review", "kind": "dependency"})

    adapter = RecordingAdapter()
    _tick(monkeypatch, _make_runner(adapter))

    assert len(adapter.sent) == 1
    assert adapter.handled == [], (
        "current operator policy notifies on stops; only completion-class "
        "events wake an agent"
    )


def test_silent_bookkeeping_events_do_not_wedge_a_later_stop(board, monkeypatch):
    """archived/unblocked are claimed but silent — they must not hide a stop."""
    tid = _subscribed_task(delivery_mode="notify")
    _emit(tid, "unblocked", {})
    _emit(tid, "blocked", {"reason": "needs a human", "kind": "needs_input"})

    adapter = RecordingAdapter()
    _tick(monkeypatch, _make_runner(adapter))

    assert len(adapter.sent) == 1
    assert "needs a human" in adapter.sent[0]["text"]


# --------------------------------------------------------------------------
# 5. delivery durability: failure → recovery, cursor, replay, duplicates
# --------------------------------------------------------------------------

def test_failed_send_preserves_the_event_and_recovers_next_tick(board, monkeypatch):
    tid = _subscribed_task(delivery_mode="notify")
    _emit(tid, "blocked", {"reason": "needs a human decision", "kind": "needs_input"})

    adapter = RecordingAdapter()
    adapter.fail_next = 1
    runner = _make_runner(adapter)

    _tick(monkeypatch, runner)
    assert adapter.sent == [], "the failed send must not be recorded as delivered"

    _tick(monkeypatch, runner)
    assert len(adapter.sent) == 1, "the unacknowledged event must be re-delivered"
    assert "needs a human decision" in adapter.sent[0]["text"]

    _tick(monkeypatch, runner)
    assert len(adapter.sent) == 1, "the recovered event must not be delivered twice"


def test_cursor_persists_across_restart_and_never_replays(board, monkeypatch):
    tid = _subscribed_task(delivery_mode="notify")
    _emit(tid, "blocked", {"reason": "needs a human decision", "kind": "needs_input"})

    adapter = RecordingAdapter()
    _tick(monkeypatch, _make_runner(adapter))
    assert len(adapter.sent) == 1

    # A fresh gateway process — new runner, new in-memory failure counters —
    # must read the persisted cursor and stay silent.
    replay = RecordingAdapter()
    _tick(monkeypatch, _make_runner(replay))
    assert replay.sent == [], "restart must not replay an acknowledged event"


def test_nonreview_completion_remains_final_not_a_review_handoff(board, monkeypatch):
    tid = _subscribed_task(delivery_mode="notify")
    conn = kanban_db_connect.connect()
    try:
        assert kb.complete_task(
            conn, tid, summary="documentation cleanup complete",
        )
        payload = json.loads(conn.execute(
            "SELECT payload FROM task_events WHERE task_id = ? "
            "AND kind = 'completed' ORDER BY id DESC LIMIT 1",
            (tid,),
        ).fetchone()["payload"])
        assert payload["completion_kind"] == "final"
        assert payload["source_status"] == "ready"
    finally:
        conn.close()

    adapter = RecordingAdapter()
    runner = _make_runner(adapter)
    _tick(monkeypatch, runner)
    _tick(monkeypatch, runner)

    assert len(adapter.sent) == 1
    text = adapter.sent[0]["text"]
    assert "done" in text.lower()
    assert "review handoff" not in text.lower()


def test_review_request_uses_explicit_reviewer_and_ignores_nonreview_child(
    board, monkeypatch
):
    tid = _subscribed_task(delivery_mode="notify")
    conn = kanban_db_connect.connect()
    try:
        nonreview_id = kb.create_task(
            conn,
            title="Publish release notes",
            assignee="docs",
            parents=[tid],
            created_by="orchestrator",
        )
        assert kb.request_review(
            conn,
            tid,
            summary="implementation committed and tested",
            reviewer="independent-reviewer",
        )
    finally:
        conn.close()

    adapter = RecordingAdapter()
    _tick(monkeypatch, _make_runner(adapter))

    assert len(adapter.sent) == 1
    text = adapter.sent[0]["text"]
    assert "review handoff" in text.lower()
    assert "@independent-reviewer" in text
    assert nonreview_id not in text
    assert "@docs" not in text


def test_review_request_without_reviewer_returns_route_to_origin_owner_once(
    board, monkeypatch
):
    tid = _subscribed_task(delivery_mode="notify")
    conn = kanban_db_connect.connect()
    try:
        assert kb.request_review(
            conn, tid, summary="implementation committed and tested",
        )
    finally:
        conn.close()

    adapter = RecordingAdapter()
    runner = _make_runner(adapter)
    _tick(monkeypatch, runner)
    _tick(monkeypatch, runner)

    assert len(adapter.sent) == 1
    text = adapter.sent[0]["text"]
    assert "review handoff" in text.lower()
    assert "origin owner" in text.lower()
    assert "assign exactly one independent reviewer" in text.lower()


def test_same_card_review_approval_completion_is_final(board, monkeypatch):
    tid = _subscribed_task(delivery_mode="notify")
    conn = kanban_db_connect.connect()
    try:
        assert kb.request_review(
            conn,
            tid,
            summary="implementation committed and tested",
            reviewer="independent-reviewer",
        )
        review = kb.claim_review_task(conn, tid, claimer="reviewer:1")
        assert review is not None
        assert kb.complete_task(
            conn,
            tid,
            summary="independent review approved",
            expected_run_id=review.current_run_id,
        )
        payload = json.loads(conn.execute(
            "SELECT payload FROM task_events WHERE task_id = ? "
            "AND kind = 'completed' ORDER BY id DESC LIMIT 1",
            (tid,),
        ).fetchone()["payload"])
        assert payload["completion_kind"] == "review_approved"
        assert payload["source_status"] == "review"
    finally:
        conn.close()

    adapter = RecordingAdapter()
    _tick(monkeypatch, _make_runner(adapter))

    assert len(adapter.sent) == 2
    assert "review handoff" in adapter.sent[0]["text"].lower()
    assert "done" in adapter.sent[1]["text"].lower()
    assert "review handoff" not in adapter.sent[1]["text"].lower()


def test_duplicate_events_each_deliver_exactly_once(board, monkeypatch):
    tid = _subscribed_task(delivery_mode="notify")
    _emit(tid, "blocked", {"reason": "same cause", "kind": "needs_input"})
    _emit(tid, "blocked", {"reason": "same cause", "kind": "needs_input"})

    adapter = RecordingAdapter()
    runner = _make_runner(adapter)
    _tick(monkeypatch, runner)
    assert len(adapter.sent) == 2, "two distinct board events → two notices"

    _tick(monkeypatch, runner)
    assert len(adapter.sent) == 2, "no re-delivery on the next tick"


def test_missing_subscription_delivers_nothing_and_does_not_raise(board, monkeypatch):
    conn = kanban_db_connect.connect()
    try:
        tid = kb.create_task(conn, title="unsubscribed", assignee="worker")
    finally:
        conn.close()
    _emit(tid, "blocked", {"reason": "nobody is listening", "kind": "needs_input"})

    adapter = RecordingAdapter()
    _tick(monkeypatch, _make_runner(adapter))
    assert adapter.sent == []


def test_notice_routes_to_the_originating_thread(board, monkeypatch):
    tid = _subscribed_task()
    _emit(tid, "blocked", {"reason": "needs a human", "kind": "needs_input"})

    adapter = RecordingAdapter()
    _tick(monkeypatch, _make_runner(adapter))

    assert adapter.sent[0]["chat_id"] == "1543672226589704283"
    assert adapter.sent[0]["metadata"].get("thread_id") == "1543672226589704283"


def test_undeliverable_routing_is_surfaced_distinctly(board, monkeypatch, caplog):
    """An unroutable platform must be logged loudly, not dropped in silence."""
    conn = kanban_db_connect.connect()
    try:
        tid = kb.create_task(conn, title="mars task", assignee="worker")
        kanban_db_notify.add_notify_sub(
            conn, task_id=tid, platform="discord", chat_id="c-1",
        )
        conn.execute(
            "UPDATE kanban_notify_subs SET platform = 'martian' WHERE task_id = ?",
            (tid,),
        )
        conn.commit()
    finally:
        conn.close()
    _emit(tid, "blocked", {"reason": "unroutable", "kind": "needs_input"})

    adapter = RecordingAdapter()
    runner = _make_runner(adapter)
    runner.adapters = {Platform.DISCORD: adapter}
    monkeypatch.setattr(
        "gateway.kanban_watchers.GatewayKanbanWatchersMixin._owns_kanban_dispatcher_lock",
        lambda self: True,
    )
    with caplog.at_level("WARNING", logger="gateway.run"):
        _tick(monkeypatch, runner)

    assert any(
        "undeliverable" in rec.message.lower() and tid in rec.getMessage()
        for rec in caplog.records
    ), (
        "an unroutable subscription must surface as a distinct WARNING; got "
        f"{[r.getMessage() for r in caplog.records]!r}"
    )
