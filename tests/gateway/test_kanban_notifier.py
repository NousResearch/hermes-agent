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


def _create_blocked_subscription(reason: str):
    conn = kb.connect()
    try:
        tid = kb.create_task(conn, title="needs human input", assignee="worker")
        kb.add_notify_sub(conn, task_id=tid, platform="telegram", chat_id="chat-1")
        kb._append_event(conn, tid, kind="blocked", payload={"reason": reason})
        return tid
    finally:
        conn.close()


def _create_gave_up_subscription(error: str):
    conn = kb.connect()
    try:
        tid = kb.create_task(conn, title="dead worker", assignee="worker")
        kb.add_notify_sub(conn, task_id=tid, platform="telegram", chat_id="chat-1")
        kb._append_event(conn, tid, kind="gave_up", payload={"error": error})
        return tid
    finally:
        conn.close()


def test_blocked_reason_carries_full_actionable_context(tmp_path, monkeypatch):
    """Blocked reasons must survive up to NOTIFY_BLOCKED_REASON_MAX chars.

    Regression for the 160-char truncation that made blocked notifications
    routinely unactionable — users couldn't see the question without
    opening the dashboard. Caller-supplied 800-char reason must land in
    the chat message intact.
    """
    from gateway.run import NOTIFY_BLOCKED_REASON_MAX

    db_path = tmp_path / "blocked-cap.db"
    monkeypatch.setenv("HERMES_KANBAN_DB", str(db_path))
    kb.init_db()

    reason = (
        "Rate limit key choice needs a human call. "
        "Option A: IP from Cloudflare headers — simple, but NAT-unsafe "
        "(false positives for users behind shared egress). "
        "Option B: user_id — requires auth, skips anonymous endpoints "
        "(login, signup, forgot-password). "
        "Option C: hybrid — IP for anonymous endpoints, user_id for "
        "authenticated, with a per-IP burst budget to soften shared-NAT "
        "false-positives. Each option has a different blast radius and a "
        "different impl cost; I need a decision before wiring anything in."
    )
    assert 160 < len(reason) <= NOTIFY_BLOCKED_REASON_MAX
    _create_blocked_subscription(reason)

    adapter = RecordingAdapter()
    runner = _make_runner(adapter)
    asyncio.run(_run_one_notifier_tick(monkeypatch, runner))

    assert len(adapter.sent) == 1
    text = adapter.sent[0]["text"]
    # Full reason should be present — every distinctive substring lands.
    assert "Cloudflare headers" in text
    assert "forgot-password" in text
    assert "blast radius" in text


def test_blocked_reason_is_truncated_at_documented_cap(tmp_path, monkeypatch):
    """A reason longer than the cap is truncated at NOTIFY_BLOCKED_REASON_MAX."""
    from gateway.run import NOTIFY_BLOCKED_REASON_MAX

    db_path = tmp_path / "blocked-overflow.db"
    monkeypatch.setenv("HERMES_KANBAN_DB", str(db_path))
    kb.init_db()

    reason = "x" * (NOTIFY_BLOCKED_REASON_MAX + 500)
    _create_blocked_subscription(reason)

    adapter = RecordingAdapter()
    runner = _make_runner(adapter)
    asyncio.run(_run_one_notifier_tick(monkeypatch, runner))

    assert len(adapter.sent) == 1
    text = adapter.sent[0]["text"]
    # The reason substring (xs) is capped; assert exact cap length.
    x_run = text.count("x")
    assert x_run == NOTIFY_BLOCKED_REASON_MAX


def test_done_summary_carries_extended_handoff(tmp_path, monkeypatch):
    """Done summaries support up to NOTIFY_DONE_SUMMARY_MAX chars (first line)."""
    from gateway.run import NOTIFY_DONE_SUMMARY_MAX

    db_path = tmp_path / "done-cap.db"
    monkeypatch.setenv("HERMES_KANBAN_DB", str(db_path))
    kb.init_db()

    summary = (
        "shipped token-bucket rate limiter on /api/* — keyed primarily on "
        "user_id with IP fallback for anonymous endpoints; per-IP burst "
        "budget of 60 rpm to soften shared-NAT false positives; 14/14 "
        "tests pass including a NAT-collision regression and a "
        "Cloudflare-header-stripping test; redis-backed counters with "
        "60s TTL; metrics emitted to statsd under rate_limit.{hit,miss}."
    )
    assert 200 < len(summary) <= NOTIFY_DONE_SUMMARY_MAX
    conn = kb.connect()
    try:
        tid = kb.create_task(conn, title="rate limit", assignee="worker")
        kb.add_notify_sub(conn, task_id=tid, platform="telegram", chat_id="chat-1")
        kb.complete_task(conn, tid, summary=summary)
    finally:
        conn.close()

    adapter = RecordingAdapter()
    runner = _make_runner(adapter)
    asyncio.run(_run_one_notifier_tick(monkeypatch, runner))

    assert len(adapter.sent) == 1
    text = adapter.sent[0]["text"]
    assert "token-bucket" in text
    assert "rate_limit.{hit,miss}" in text


def test_gave_up_error_carries_extended_traceback_snippet(tmp_path, monkeypatch):
    """gave_up errors carry up to NOTIFY_GAVE_UP_ERROR_MAX chars."""
    from gateway.run import NOTIFY_GAVE_UP_ERROR_MAX

    db_path = tmp_path / "gaveup-cap.db"
    monkeypatch.setenv("HERMES_KANBAN_DB", str(db_path))
    kb.init_db()

    err = (
        "spawn_failed after 5 attempts: profile 'worker' missing "
        "HERMES_ANTHROPIC_API_KEY in profile env; subprocess exited 2 "
        "with stderr 'authentication failed: no credentials configured'. "
        "Check ~/.hermes/profiles/worker/.env or set the key in the "
        "global ~/.hermes/.env. See logs at ~/.hermes/logs/agent.log "
        "around 2026-05-23T14:30:00Z for the full traceback."
    )
    assert 200 < len(err) <= NOTIFY_GAVE_UP_ERROR_MAX
    _create_gave_up_subscription(err)

    adapter = RecordingAdapter()
    runner = _make_runner(adapter)
    asyncio.run(_run_one_notifier_tick(monkeypatch, runner))

    assert len(adapter.sent) == 1
    text = adapter.sent[0]["text"]
    assert "HERMES_ANTHROPIC_API_KEY" in text
    assert "agent.log" in text


def test_notification_caps_are_ordered_so_blocked_wins(tmp_path, monkeypatch):
    """Sanity-check the cap budget: blocked > done > error > legacy result.

    The whole point of per-kind caps (rather than one shared limit) is
    that ``blocked`` — the one users actually answer — should never lose
    budget to a chatty ``done`` summary. Pin the ordering so future
    tuning can't accidentally invert it.
    """
    from gateway.run import (
        NOTIFY_BLOCKED_REASON_MAX,
        NOTIFY_DONE_RESULT_LEGACY_MAX,
        NOTIFY_DONE_SUMMARY_MAX,
        NOTIFY_GAVE_UP_ERROR_MAX,
    )

    assert NOTIFY_BLOCKED_REASON_MAX > NOTIFY_DONE_SUMMARY_MAX
    assert NOTIFY_DONE_SUMMARY_MAX > NOTIFY_GAVE_UP_ERROR_MAX
    assert NOTIFY_GAVE_UP_ERROR_MAX > NOTIFY_DONE_RESULT_LEGACY_MAX
    # Stay under Discord/Slack's ~2000-char single-message ceiling so
    # even the largest cap fits in one chat message on the tightest
    # platform we currently target.
    assert NOTIFY_BLOCKED_REASON_MAX < 2000
