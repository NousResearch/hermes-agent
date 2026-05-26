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


def test_kanban_notifier_delivers_verifier_result_events(tmp_path, monkeypatch):
    """Verifier PASS/FAIL should be visible through native Kanban subscriptions."""
    db_path = tmp_path / "verifier-result.db"
    monkeypatch.setenv("HERMES_KANBAN_DB", str(db_path))
    kb.init_db()

    conn = kb.connect()
    try:
        tid = kb.create_task(conn, title="verify me", assignee="worker")
        kb.add_notify_sub(conn, task_id=tid, platform="telegram", chat_id="chat-1")
        kb._append_event(
            conn,
            tid,
            kind="verifier_result",
            payload={
                "verdict": "FAIL",
                "reason_codes": ["missing_criterion_dc_01"],
                "retry_decision": "retry_worker",
                "criterion_results": [
                    {"criterion_id": "dc-01", "status": "FAIL", "evidence": "missing"}
                ],
            },
        )
    finally:
        conn.close()

    adapter = RecordingAdapter()
    asyncio.run(_run_one_notifier_tick(monkeypatch, _make_runner(adapter)))

    assert len(adapter.sent) == 1
    text = adapter.sent[0]["text"]
    assert "verifier" in text.lower()
    assert "FAIL" in text
    assert "missing_criterion_dc_01" in text
    assert "retry_worker" in text


def test_kanban_notifier_delivers_review_ready_closeout_transition(tmp_path, monkeypatch):
    """Review-ready closeout transitions should surface as review package pings."""
    db_path = tmp_path / "review-ready-transition.db"
    monkeypatch.setenv("HERMES_KANBAN_DB", str(db_path))
    kb.init_db()

    conn = kb.connect()
    try:
        tid = kb.create_task(conn, title="review package", assignee="worker")
        kb.add_notify_sub(conn, task_id=tid, platform="telegram", chat_id="chat-1")
        kb._append_event(
            conn,
            tid,
            kind="closeout_transition",
            payload={
                "review_phase": "review_ready",
                "allowed": True,
                "pr": {"url": "https://github.com/chriskim12/hermes-agent/pull/123"},
                "verifier_verdict": {"verdict": "PASS"},
            },
        )
    finally:
        conn.close()

    adapter = RecordingAdapter()
    asyncio.run(_run_one_notifier_tick(monkeypatch, _make_runner(adapter)))

    assert len(adapter.sent) == 1
    text = adapter.sent[0]["text"]
    assert "review_ready" in text
    assert "PR" in text
    assert "PASS" in text


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
# MVP fallback routing tests — verifier / review_ready events → Discord
# MVP channel 1500713192765132912 without a per-task subscription.
# ---------------------------------------------------------------------------

def _make_runner_with_discord(adapter):
    """GatewayRunner wired with a Discord adapter for MVP fallback testing."""
    runner = GatewayRunner.__new__(GatewayRunner)
    runner._running = True
    runner.adapters = {Platform.DISCORD: adapter}
    runner._kanban_sub_fail_counts = {}
    return runner


def test_kanban_notifier_mvp_chat_id_is_correct():
    """MVP destination is explicitly Discord channel 1500713192765132912."""
    # This is a pure documentation test — the constant is embedded in the
    # notifier code, so we assert the *intent* by reading the source.
    import gateway.run as grun
    import inspect
    src = inspect.getsource(grun.GatewayRunner._kanban_notifier_watcher)
    assert '"1500713192765132912"' in src, (
        "MVP Discord channel must be hardcoded as `1500713192765132912`"
    )


def test_kanban_notifier_mvp_fallback_delivers_verifier_result(tmp_path, monkeypatch):
    """Verifier result events route to MVP Discord channel without per-task sub."""
    db_path = tmp_path / "mvp-verifier.db"
    monkeypatch.setenv("HERMES_KANBAN_DB", str(db_path))
    kb.init_db()

    conn = kb.connect()
    try:
        # Create a task with NO per-task subscription.
        tid = kb.create_task(conn, title="orphan task", assignee="worker")
        kb._append_event(
            conn, tid,
            kind="verifier_result",
            payload={
                "verdict": "FAIL",
                "reason_codes": ["missing_criterion_dc_01"],
                "retry_decision": "retry_worker",
            },
        )
    finally:
        conn.close()

    adapter = RecordingAdapter()
    runner = _make_runner_with_discord(adapter)
    asyncio.run(_run_one_notifier_tick(monkeypatch, runner))

    assert len(adapter.sent) == 1
    assert adapter.sent[0]["chat_id"] == "1500713192765132912"
    text = adapter.sent[0]["text"]
    assert "verifier" in text.lower()
    assert "FAIL" in text
    assert tid in text


def test_kanban_notifier_mvp_fallback_delivers_closeout_transition(tmp_path, monkeypatch):
    """Review-ready closeout events route to MVP Discord channel without per-task sub."""
    db_path = tmp_path / "mvp-closeout.db"
    monkeypatch.setenv("HERMES_KANBAN_DB", str(db_path))
    kb.init_db()

    conn = kb.connect()
    try:
        tid = kb.create_task(conn, title="review me", assignee="worker")
        kb._append_event(
            conn, tid,
            kind="closeout_transition",
            payload={
                "review_phase": "review_ready",
                "allowed": True,
                "pr": {"url": "https://github.com/chriskim12/hermes-agent/pull/123"},
                "verifier_verdict": {"verdict": "PASS"},
            },
        )
    finally:
        conn.close()

    adapter = RecordingAdapter()
    runner = _make_runner_with_discord(adapter)
    asyncio.run(_run_one_notifier_tick(monkeypatch, runner))

    assert len(adapter.sent) == 1
    assert adapter.sent[0]["chat_id"] == "1500713192765132912"
    text = adapter.sent[0]["text"]
    assert "review_ready" in text
    assert "PASS" in text
    assert tid in text


def test_kanban_notifier_mvp_does_not_duplicate_per_task_sub(tmp_path, monkeypatch):
    """MVP fallback MUST NOT deliver events that a per-task subscription already covers."""
    db_path = tmp_path / "mvp-no-dupe.db"
    monkeypatch.setenv("HERMES_KANBAN_DB", str(db_path))
    kb.init_db()

    conn = kb.connect()
    try:
        # Task WITH its own subscription.
        tid = kb.create_task(conn, title="has own sub", assignee="worker")
        kb.add_notify_sub(conn, task_id=tid, platform="discord", chat_id="specific-chat")
        kb._append_event(
            conn, tid,
            kind="verifier_result",
            payload={"verdict": "PASS"},
        )

        # Orphan task WITHOUT subscription.
        tid2 = kb.create_task(conn, title="orphan", assignee="worker2")
        kb._append_event(
            conn, tid2,
            kind="verifier_result",
            payload={"verdict": "FAIL", "reason_codes": ["missing_evidence"]},
        )
    finally:
        conn.close()

    adapter = RecordingAdapter()
    runner = _make_runner_with_discord(adapter)
    asyncio.run(_run_one_notifier_tick(monkeypatch, runner))

    # tid has per-task sub → delivered to "specific-chat"
    # tid2 has NO per-task sub → delivered via MVP fallback
    # Total deliveries: 2 (one per task, not duplicated)
    assert len(adapter.sent) == 2

    specific = [d for d in adapter.sent if d["chat_id"] == "specific-chat"]
    mvp = [d for d in adapter.sent if d["chat_id"] == "1500713192765132912"]

    assert len(specific) == 1
    assert tid in specific[0]["text"]
    assert len(mvp) == 1
    assert tid2 in mvp[0]["text"]


def test_kanban_notifier_mvp_fallback_delivers_when_only_non_discord_sub_exists(tmp_path, monkeypatch):
    """Non-Discord subscriptions must not suppress the MVP Discord fallback."""
    db_path = tmp_path / "mvp-non-discord-sub.db"
    monkeypatch.setenv("HERMES_KANBAN_DB", str(db_path))
    kb.init_db()

    conn = kb.connect()
    try:
        tid = kb.create_task(conn, title="telegram-only", assignee="worker")
        kb.add_notify_sub(conn, task_id=tid, platform="telegram", chat_id="telegram-chat")
        kb._append_event(
            conn,
            tid,
            kind="verifier_result",
            payload={"verdict": "PASS"},
        )
    finally:
        conn.close()

    adapter = RecordingAdapter()
    runner = _make_runner_with_discord(adapter)
    asyncio.run(_run_one_notifier_tick(monkeypatch, runner))

    assert len(adapter.sent) == 1
    assert adapter.sent[0]["chat_id"] == "1500713192765132912"
    assert tid in adapter.sent[0]["text"]


def test_kanban_notifier_mvp_drop_does_not_replay_old_event_after_send_failures(tmp_path, monkeypatch):
    """After repeated MVP send failures, old events must not replay from cursor 0."""
    db_path = tmp_path / "mvp-failure-no-replay.db"
    monkeypatch.setenv("HERMES_KANBAN_DB", str(db_path))
    kb.init_db()

    conn = kb.connect()
    try:
        tid = kb.create_task(conn, title="orphan", assignee="worker")
        kb._append_event(conn, tid, kind="verifier_result", payload={"verdict": "PASS"})
    finally:
        conn.close()

    failing = FailingAdapter()
    runner = _make_runner_with_discord(failing)
    for _ in range(3):
        runner._running = True
        asyncio.run(_run_one_notifier_tick(monkeypatch, runner))

    assert failing.attempts == 3

    recovered = RecordingAdapter()
    runner.adapters = {Platform.DISCORD: recovered}
    runner._running = True
    asyncio.run(_run_one_notifier_tick(monkeypatch, runner))

    assert recovered.sent == []


def test_kanban_notifier_mvp_does_not_activate_without_discord_adapter(tmp_path, monkeypatch):
    """MVP fallback must stay silent when Discord adapter is not connected."""
    db_path = tmp_path / "mvp-no-discord.db"
    monkeypatch.setenv("HERMES_KANBAN_DB", str(db_path))
    kb.init_db()

    conn = kb.connect()
    try:
        tid = kb.create_task(conn, title="orphan", assignee="worker")
        kb._append_event(
            conn, tid,
            kind="verifier_result",
            payload={"verdict": "PASS"},
        )
    finally:
        conn.close()

    adapter = RecordingAdapter()
    # Wire as Telegram, NOT Discord — MVP fallback should not fire.
    runner = _make_runner(adapter)
    asyncio.run(_run_one_notifier_tick(monkeypatch, runner))

    assert len(adapter.sent) == 0, (
        "MVP fallback must not deliver when Discord adapter is not connected"
    )


def test_kanban_notifier_mvp_preserves_existing_terminal_pings(tmp_path, monkeypatch):
    """Existing explicit subscriptions still deliver all normal event kinds."""
    db_path = tmp_path / "mvp-preserve.db"
    monkeypatch.setenv("HERMES_KANBAN_DB", str(db_path))
    kb.init_db()

    tid = _create_completed_subscription()

    adapter = RecordingAdapter()
    runner = _make_runner(adapter)
    asyncio.run(_run_one_notifier_tick(monkeypatch, runner))

    assert len(adapter.sent) == 1
    assert "Kanban" in adapter.sent[0]["text"]
    assert tid in adapter.sent[0]["text"]
    # Original chat_id is preserved for explicit subscriptions.
    assert adapter.sent[0]["chat_id"] == "chat-1"
