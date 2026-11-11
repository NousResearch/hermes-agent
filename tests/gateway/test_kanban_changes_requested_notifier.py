import asyncio

from gateway.config import Platform
from gateway.run import GatewayRunner
from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_notify as kbn


class RecordingAdapter:
    def __init__(self, *, fail_send=False):
        self.sent = []
        self.handled = []
        self.fail_send = fail_send

    async def send(self, chat_id, text, metadata=None):
        self.sent.append({"chat_id": chat_id, "text": text, "metadata": metadata or {}})
        if self.fail_send:
            raise RuntimeError("transient send failure")

    async def handle_message(self, event):
        self.handled.append(event)
        event._gateway_accepted = True


async def _run_one_tick(monkeypatch, runner):
    real_sleep = asyncio.sleep

    async def fake_sleep(delay):
        if delay == 5:
            return
        runner._running = False
        await real_sleep(0)

    monkeypatch.setattr(asyncio, "sleep", fake_sleep)
    await runner._kanban_notifier_watcher(interval=1)


def _runner(adapter):
    runner = GatewayRunner.__new__(GatewayRunner)
    runner._running = True
    runner.adapters = {Platform.TELEGRAM: adapter}
    runner._kanban_sub_fail_counts = {}
    runner._kanban_dispatcher_lock_handle = object()
    return runner


def _create_review_block(delivery_mode, *, reason="Tests need updates"):
    conn = kbc.connect()
    try:
        task_id = kb.create_task(
            conn,
            title="existing implementation card",
            assignee="implementer",
            session_id="agent:main:telegram:thread:chat-1:topic-7",
        )
        kbn.add_notify_sub(
            conn,
            task_id=task_id,
            platform="telegram",
            chat_id="chat-1",
            thread_id="topic-7",
            chat_type="thread",
            delivery_mode=delivery_mode,
            delivery_metadata={"thread_id": "topic-7", "chat_type": "thread"},
        )
        kb._append_event(
            conn,
            task_id,
            kind="changes_requested",
            payload={
                "reason": reason,
                "reviewer": "claude-qa",
                "implementer": "codex-cua",
                "status": "ready",
            },
        )
        return task_id
    finally:
        conn.close()


def _unseen(task_id):
    conn = kbc.connect()
    try:
        _, events = kbn.unseen_events_for_sub(
            conn,
            task_id=task_id,
            platform="telegram",
            chat_id="chat-1",
            thread_id="topic-7",
            kinds=["changes_requested"],
        )
        return events
    finally:
        conn.close()


def test_changes_requested_notify_wake_is_actionable_and_exactly_routed(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_KANBAN_DB", str(tmp_path / "review-block.db"))
    kb.init_db()
    task_id = _create_review_block("notify+wake")
    adapter = RecordingAdapter()

    asyncio.run(_run_one_tick(monkeypatch, _runner(adapter)))

    assert len(adapter.sent) == 1
    text = adapter.sent[0]["text"]
    assert text.startswith(f"🛑 [default] Kanban {task_id} review requested changes/BLOCK: Tests need updates")
    assert "reviewer @claude-qa → implementer @codex-cua" in text
    assert adapter.sent[0]["metadata"]["thread_id"] == "topic-7"
    assert len(adapter.handled) == 1
    wake = adapter.handled[0]
    assert wake.source.chat_id == "chat-1"
    assert wake.source.chat_type == "thread"
    assert wake.source.thread_id == "topic-7"
    assert "implementation is not approved" in wake.text
    assert "Inspect the existing card and its current review run" in wake.text
    assert "do not create a duplicate task" in wake.text
    assert wake.text.count("do not create a duplicate task") == 1
    assert _unseen(task_id) == []

    # A fresh watcher after restart cannot replay an event whose cursor advanced.
    asyncio.run(_run_one_tick(monkeypatch, _runner(adapter)))
    assert len(adapter.sent) == 1
    assert len(adapter.handled) == 1


def test_changes_requested_notify_is_passive_only(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_KANBAN_DB", str(tmp_path / "notify.db"))
    kb.init_db()
    task_id = _create_review_block("notify")
    adapter = RecordingAdapter()

    asyncio.run(_run_one_tick(monkeypatch, _runner(adapter)))

    assert len(adapter.sent) == 1
    assert adapter.handled == []
    assert _unseen(task_id) == []


def test_changes_requested_wake_only_has_no_passive_post(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_KANBAN_DB", str(tmp_path / "wake.db"))
    kb.init_db()
    task_id = _create_review_block("wake")
    adapter = RecordingAdapter()

    asyncio.run(_run_one_tick(monkeypatch, _runner(adapter)))

    assert adapter.sent == []
    assert len(adapter.handled) == 1
    assert _unseen(task_id) == []


def test_changes_requested_send_failure_retries_without_event_loss(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_KANBAN_DB", str(tmp_path / "retry.db"))
    kb.init_db()
    task_id = _create_review_block("notify")
    failing = RecordingAdapter(fail_send=True)

    asyncio.run(_run_one_tick(monkeypatch, _runner(failing)))
    assert len(failing.sent) == 1
    assert len(_unseen(task_id)) == 1

    healthy = RecordingAdapter()
    asyncio.run(_run_one_tick(monkeypatch, _runner(healthy)))
    assert len(healthy.sent) == 1
    assert _unseen(task_id) == []


def test_changes_requested_reason_is_redacted_path_safe_and_truncated(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_KANBAN_DB", str(tmp_path / "redact.db"))
    kb.init_db()
    secret = "sk-abcdefghijklmnopqrstuvwxyz123456"
    reason = f"See /Users/alice/private/review.log token={secret} " + ("x" * 300)
    _create_review_block("notify", reason=reason)
    adapter = RecordingAdapter()

    asyncio.run(_run_one_tick(monkeypatch, _runner(adapter)))

    text = adapter.sent[0]["text"]
    assert "/Users/alice" not in text
    assert "abcdefghijklmnopqrstuvwxyz" not in text
    assert "[local path]" in text
    assert "… — reviewer @claude-qa" in text


# --- Batched delivery: the wake must report the CURRENT state, not the batch ---


def _create_batched_card(delivery_mode):
    """A card whose events arrive in one undelivered batch (client restart)."""
    conn = kbc.connect()
    try:
        task_id = kb.create_task(
            conn,
            title="batched review cycle card",
            assignee="implementer",
            session_id="agent:main:telegram:dm:chat-1",
        )
        kbn.add_notify_sub(
            conn,
            task_id=task_id,
            platform="telegram",
            chat_id="chat-1",
            chat_type="dm",
            delivery_mode=delivery_mode,
        )
        return task_id
    finally:
        conn.close()


def _append_event(task_id, kind, payload):
    conn = kbc.connect()
    try:
        kb._append_event(conn, task_id, kind=kind, payload=payload)
    finally:
        conn.close()


def _complete_card(task_id, summary):
    conn = kbc.connect()
    try:
        assert kb.complete_task(conn, task_id, summary=summary) is True
    finally:
        conn.close()


def _wake_status_line(text):
    """``[kanban] Task <id> <status>.`` — the composed current-state line."""
    return text.splitlines()[0]


def test_batched_review_cycle_wake_names_only_the_current_state(tmp_path, monkeypatch):
    """review_requested + changes_requested + completed in ONE batch.

    That is exactly what a desktop/client restart produces: everything queued
    while delivery was down arrives at once. The wake must name the newest
    state (completed) and carry only that event's detail. Before the fix it
    joined every kind present in ``_WAKE_KINDS`` tuple order and paired the
    round-3 completion handoff with the round-1 review verdict, so an APPROVED
    card woke the creator with "review requested changes (BLOCK); implementation
    is not approved".
    """
    monkeypatch.setenv("HERMES_KANBAN_DB", str(tmp_path / "batched.db"))
    kb.init_db()
    task_id = _create_batched_card("notify+wake")
    _append_event(task_id, "review_requested", {"summary": "round 3 handoff for review"})
    _append_event(task_id, "changes_requested", {
        "reason": "round 1: four precise corrections",
        "reviewer": "claude-qa",
        "implementer": "codex-cua",
        "status": "ready",
    })
    _complete_card(task_id, "Reviewed and APPROVED (round 3)")
    adapter = RecordingAdapter()

    asyncio.run(_run_one_tick(monkeypatch, _runner(adapter)))

    assert len(adapter.handled) == 1, "one wake for the whole batch"
    wake = adapter.handled[0].text
    assert _wake_status_line(wake) == f"[kanban] Task {task_id} completed."
    assert "handed off for review" not in wake
    assert "review requested changes" not in wake
    assert "implementation is not approved" not in wake
    # Only the current (completed) event's detail rides along.
    assert "Result: Reviewed and APPROVED (round 3)" in wake
    assert "Review feedback" not in wake
    assert "round 1: four precise corrections" not in wake
    # Pings stay per-event: every claimed event is still reported on its own.
    assert len(adapter.sent) == 3


def test_closed_card_never_wakes_with_a_superseded_block(tmp_path, monkeypatch):
    """A card that is already ``done`` must never emit BLOCK wording later.

    The completion is delivered first, then a verdict that reached the event log
    after the card closed (queued while delivery was down) arrives on its own.
    The card is done at send time, so that verdict is history: report the card's
    real state, never "review requested changes (BLOCK); implementation is not
    approved".
    """
    monkeypatch.setenv("HERMES_KANBAN_DB", str(tmp_path / "closed.db"))
    kb.init_db()
    task_id = _create_batched_card("notify+wake")
    _complete_card(task_id, "Reviewed and APPROVED (round 3)")
    adapter = RecordingAdapter()

    asyncio.run(_run_one_tick(monkeypatch, _runner(adapter)))

    # completed-only batch: unchanged single-event rendering.
    assert len(adapter.handled) == 1
    assert _wake_status_line(adapter.handled[0].text) == f"[kanban] Task {task_id} completed."
    assert "Result: Reviewed and APPROVED (round 3)" in adapter.handled[0].text

    _append_event(task_id, "changes_requested", {
        "reason": "round 1: needs work",
        "reviewer": "claude-qa",
        "status": "ready",
    })
    asyncio.run(_run_one_tick(monkeypatch, _runner(adapter)))

    assert len(adapter.handled) == 2, "the late verdict still wakes, as the card's real state"
    late = adapter.handled[1].text
    assert _wake_status_line(late) == f"[kanban] Task {task_id} completed."
    assert "review requested changes" not in late
    assert "implementation is not approved" not in late
    assert "round 1: needs work" not in late
