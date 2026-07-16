import asyncio
import multiprocessing
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock


from gateway.config import Platform
from gateway.kanban_status_card import (
    render_kanban_active_task_index,
    render_kanban_status_card,
)
from gateway.kanban_watchers import _active_index_link_label
from gateway.platforms.base import SendResult
from gateway.run import GatewayRunner
from hermes_cli import kanban_db as kb


class RecordingAdapter:
    def __init__(self):
        self.sent = []

    async def send(self, chat_id, text, metadata=None):
        self.sent.append({"chat_id": chat_id, "text": text, "metadata": metadata or {}})
        return SendResult(success=True, message_id=f"message-{len(self.sent)}")

    async def edit_message(self, chat_id, message_id, content, *, finalize=False):
        self.sent.append({"chat_id": chat_id, "text": content, "message_id": message_id, "finalize": finalize})
        return SendResult(success=True, message_id=message_id)


class StatusCardAdapter:
    def __init__(self):
        self.sent = []
        self.edits = []

    async def send(self, chat_id, text, metadata=None):
        self.sent.append({"chat_id": chat_id, "text": text, "metadata": metadata or {}})
        return SendResult(success=True, message_id="status-card-1")

    async def edit_message(self, chat_id, message_id, content, *, finalize=False):
        self.edits.append({"chat_id": chat_id, "message_id": message_id, "content": content, "finalize": finalize})
        return SendResult(success=True, message_id=message_id)


class WakeRecordingStatusCardAdapter(StatusCardAdapter):
    """Captures synthetic agent wakes without involving a real gateway session."""

    def __init__(self):
        super().__init__()
        self.wakes = []

    async def handle_message(self, event):
        self.wakes.append(event)


class MissingStatusCardAdapter(StatusCardAdapter):
    """A Telegram edit failure must never turn into a replacement send."""

    def __init__(self):
        super().__init__()
        self.fail_edits = False

    async def edit_message(self, chat_id, message_id, content, *, finalize=False):
        self.edits.append({"chat_id": chat_id, "message_id": message_id, "content": content, "finalize": finalize})
        if self.fail_edits:
            return SendResult(success=False, error="Message not found")
        return SendResult(success=True, message_id=message_id)


class YieldingStatusCardAdapter(StatusCardAdapter):
    """Yield during edits so competing refreshes overlap at the lease boundary."""

    async def edit_message(self, chat_id, message_id, content, *, finalize=False):
        self.edits.append({"chat_id": chat_id, "message_id": message_id, "content": content, "finalize": finalize})
        await asyncio.sleep(0)
        return SendResult(success=True, message_id=message_id)


class CustomEmojiStatusCardAdapter(StatusCardAdapter):
    def kanban_status_metadata(self, status):
        return {"telegram_custom_emoji": {"🔄": "custom-emoji-id"}}

    async def edit_message(self, chat_id, message_id, content, *, finalize=False, metadata=None):
        self.edits.append({
            "chat_id": chat_id,
            "message_id": message_id,
            "content": content,
            "finalize": finalize,
            "metadata": metadata,
        })
        return SendResult(success=True, message_id=message_id)


class TopicAwareStatusCardAdapter(StatusCardAdapter):
    """Current adapters use edit metadata to preserve the origin topic route."""

    async def edit_message(self, chat_id, message_id, content, *, finalize=False, metadata=None):
        self.edits.append({
            "chat_id": chat_id,
            "message_id": message_id,
            "content": content,
            "finalize": finalize,
            "metadata": metadata or {},
        })
        return SendResult(success=True, message_id=message_id)


class ActiveIndexAdapter(StatusCardAdapter):
    def __init__(self):
        super().__init__()
        self.pins = []

    async def pin_message(self, chat_id, message_id, *, disable_notification=True):
        self.pins.append({
            "chat_id": chat_id,
            "message_id": message_id,
            "disable_notification": disable_notification,
        })
        return SendResult(success=True, message_id=message_id)


class CustomEmojiActiveIndexAdapter(ActiveIndexAdapter):
    def kanban_status_metadata(self, status):
        icons = {
            "running": {"🔫": "running-emoji-id"},
            "todo": {"⏳": "queue-emoji-id"},
            "ready": {"⏳": "queue-emoji-id"},
        }
        return {"telegram_custom_emoji": icons.get(status, {})}

    async def edit_message(self, chat_id, message_id, content, *, finalize=False, metadata=None):
        self.edits.append({
            "chat_id": chat_id,
            "message_id": message_id,
            "content": content,
            "finalize": finalize,
            "metadata": metadata or {},
        })
        return SendResult(success=True, message_id=message_id)


def _claim_surface_in_child(db_path, owner, queue):
    """Separate interpreter/process for the cross-process SQLite lease test."""
    import os
    from hermes_cli import kanban_db as child_kb

    os.environ["HERMES_KANBAN_DB"] = db_path
    conn = child_kb.connect()
    try:
        claimed = child_kb.claim_status_surface(
            conn, task_id="task", platform="telegram", chat_id="chat", thread_id="1",
            owner=owner, event_id=7, lease_seconds=30,
        )
        queue.put(bool(claimed))
    finally:
        conn.close()


def _enqueue_terminal_notification_in_child(db_path, task_id, event_id, queue):
    """Separate interpreter/process for intent-final notification dedupe."""
    import os
    from hermes_cli import kanban_db as child_kb

    os.environ["HERMES_KANBAN_DB"] = db_path
    conn = child_kb.connect()
    try:
        queue.put(child_kb.enqueue_terminal_notification(
            conn,
            task_id=task_id,
            platform="telegram",
            chat_id="114874376",
            thread_id="1515141",
            notifier_profile="reviewer",
            event_id=event_id,
            title="intent final",
            outcome_key="subprocess-final",
            outcome_summary=(
                "What changed: проверили итог.\n"
                "What works: итог принят.\n"
                "How to verify: открыть чат."
            ),
        ))
    finally:
        conn.close()


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


def test_notifier_tolerates_malformed_timeout_limit(tmp_path, monkeypatch):
    """One legacy/malformed event must not abort every notifier delivery."""
    db_path = tmp_path / "malformed-timeout.db"
    monkeypatch.setenv("HERMES_KANBAN_DB", str(db_path))
    kb.init_db()
    conn = kb.connect()
    try:
        tid = kb.create_task(conn, title="timeout payload", assignee="worker")
        kb.add_notify_sub(conn, task_id=tid, platform="telegram", chat_id="chat-timeout")
        kb._append_event(conn, tid, kind="timed_out", payload={"limit_seconds": "reviewer"})
    finally:
        conn.close()

    adapter = RecordingAdapter()
    asyncio.run(_run_one_notifier_tick(monkeypatch, _make_runner(adapter)))

    assert len(adapter.sent) == 1
    assert _unseen_terminal_events(tid) == []


def test_kanban_notifier_private_dm_topic_uses_recorded_reply_anchor(tmp_path, monkeypatch):
    """Private DM topic subs (chat_id > 0, real thread_id) must send with a
    reply anchor — Telegram hard-refuses a bare send into a numeric private
    topic lane. The anchor comes from gateway/topic_anchors.py, populated by
    a live inbound message in that lane."""
    db_path = tmp_path / "dm-topic.db"
    monkeypatch.setenv("HERMES_KANBAN_DB", str(db_path))
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    kb.init_db()

    conn = kb.connect()
    try:
        tid = kb.create_task(conn, title="dm topic task", assignee="worker")
        kb.add_notify_sub(
            conn, task_id=tid, platform="telegram", chat_id="114874376",
            thread_id="1508033",
        )
        kb.complete_task(conn, tid, summary="done")
    finally:
        conn.close()

    from gateway.topic_anchors import record_topic_anchor
    record_topic_anchor("telegram", "114874376", "1508033", "7810")

    adapter = RecordingAdapter()
    runner = _make_runner(adapter)
    asyncio.run(_run_one_notifier_tick(monkeypatch, runner))

    assert len(adapter.sent) == 1
    meta = adapter.sent[0]["metadata"]
    assert meta["telegram_dm_topic_reply_fallback"] is True
    assert meta["telegram_reply_to_message_id"] == "7810"
    assert meta["thread_id"] == "1508033"


def test_kanban_notifier_private_dm_topic_without_anchor_still_marks_fallback(tmp_path, monkeypatch):
    """No recorded anchor yet (lane never saw an inbound message) — the send
    still declares the DM-topic-reply-fallback intent so the adapter can
    fail closed with a clear error instead of silently landing in General."""
    db_path = tmp_path / "dm-topic-no-anchor.db"
    monkeypatch.setenv("HERMES_KANBAN_DB", str(db_path))
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    kb.init_db()

    conn = kb.connect()
    try:
        tid = kb.create_task(conn, title="dm topic task no anchor", assignee="worker")
        kb.add_notify_sub(
            conn, task_id=tid, platform="telegram", chat_id="114874376",
            thread_id="1509999",
        )
        kb.complete_task(conn, tid, summary="done")
    finally:
        conn.close()

    adapter = RecordingAdapter()
    runner = _make_runner(adapter)
    asyncio.run(_run_one_notifier_tick(monkeypatch, runner))

    assert len(adapter.sent) == 1
    meta = adapter.sent[0]["metadata"]
    assert meta["telegram_dm_topic_reply_fallback"] is True
    assert "telegram_reply_to_message_id" not in meta


def test_kanban_notifier_missing_anchor_error_is_transient_not_dropped(tmp_path, monkeypatch):
    """A 'requires a reply anchor' failure must not burn a drop-strike — the
    subscription should still exist after 3 ticks, unlike a real dead chat."""
    from gateway.kanban_watchers import _is_transient_kanban_send_error

    assert _is_transient_kanban_send_error(
        RuntimeError("Telegram DM topic delivery requires a reply anchor")
    )
    assert not _is_transient_kanban_send_error(RuntimeError("Forbidden: bot was blocked"))


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
    assert adapter.sent[0]["text"].startswith("🗂 notify once · " + tid)


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


def test_kanban_notifier_coalesces_a_backlog_to_one_status_write(tmp_path, monkeypatch):
    """A lifecycle backlog is one current card snapshot, not many Telegram edits."""
    db_path = tmp_path / "coalesced-backlog.db"
    monkeypatch.setenv("HERMES_KANBAN_DB", str(db_path))
    kb.init_db()
    with kb.connect() as conn:
        task_id = kb.create_task(conn, title="one card for a backlog", assignee="worker")
        kb.add_notify_sub(conn, task_id=task_id, platform="telegram", chat_id="chat-1")
        claimed = kb.claim_task(conn, task_id)
        assert claimed is not None
        assert kb.complete_task(conn, task_id, summary="done", expected_run_id=claimed.current_run_id)

    adapter = StatusCardAdapter()
    asyncio.run(_run_one_notifier_tick(monkeypatch, _make_runner(adapter)))

    assert len(adapter.sent) == 1
    assert adapter.edits == []
    with kb.connect() as conn:
        _, events = kb.unseen_events_for_sub(
            conn, task_id=task_id, platform="telegram", chat_id="chat-1",
            kinds=["created", "claimed", "completed"],
        )
    assert events == []


def test_kanban_notifier_renders_reviewer_claimed_review_status(tmp_path, monkeypatch):
    """An reviewer claim must refresh the same status card, not remain invisible."""
    db_path = tmp_path / "reviewer-claimed-review.db"
    monkeypatch.setenv("HERMES_KANBAN_DB", str(db_path))
    kb.init_db()
    with kb.connect() as conn:
        task_id = kb.create_task(conn, title="reviewer claim visible", assignee="heavy")
        kb.add_notify_sub(conn, task_id=task_id, platform="telegram", chat_id="chat-1")
        assert kb.request_review(conn, task_id, summary="ready for audit")
        kb._append_event(conn, task_id, kind="reviewer_review_claimed", payload={"profile": "reviewer"})

    adapter = StatusCardAdapter()
    asyncio.run(_run_one_notifier_tick(monkeypatch, _make_runner(adapter)))

    assert len(adapter.sent) == 1
    assert "Reviewer started review" in adapter.sent[0]["text"]
    with kb.connect() as conn:
        _, events = kb.unseen_events_for_sub(
            conn, task_id=task_id, platform="telegram", chat_id="chat-1",
            kinds=["review_requested", "reviewer_review_claimed"],
        )
    assert events == []


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
    assert "⚠️ Worker is restarting" in adapter.sent[0]["text"]

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
    assert "⚠️ Worker is restarting" in adapter.sent[1]["text"]


def test_notifier_owning_profile_adapter_no_default_fallback(tmp_path, monkeypatch):
    """A subscription owned by a secondary profile whose profile-adapter
    registry entry EXISTS but lacks this platform must NOT fall back to the
    default profile's same-platform adapter — the notifier must route through
    the shared ``_authorization_adapter`` chokepoint, which forbids that
    fallback (gateway/authz_mixin.py). Delivering via the default profile's bot
    is the exact cross-profile mis-delivery this whole change exists to fix
    (`[230002] Bot can NOT be out of the chat`).

    Mutation check: reverting kanban_watchers.py's adapter selection to the old
    inline ``if adapter is None: adapter = self.adapters.get(plat)`` fallback
    makes this test FAIL (the default adapter receives the delivery).
    """
    db_path = tmp_path / "profile-no-fallback.db"
    monkeypatch.setenv("HERMES_KANBAN_DB", str(db_path))
    kb.init_db()

    conn = kb.connect()
    try:
        tid = kb.create_task(conn, title="owned by beta", assignee="worker")
        # Subscription is owned by profile "beta".
        kb.add_notify_sub(
            conn, task_id=tid, platform="telegram", chat_id="chat-beta",
            notifier_profile="beta",
        )
        kb.complete_task(conn, tid, summary="done")
    finally:
        conn.close()

    default_adapter = RecordingAdapter()
    other_adapter = RecordingAdapter()
    runner = GatewayRunner.__new__(GatewayRunner)
    runner._running = True
    # Default profile has a telegram adapter …
    runner.adapters = {Platform.TELEGRAM: default_adapter}
    # … and profile "beta" HAS a non-empty registry entry (so it passes the
    # notifier's upstream skip-filter, which only skips owning profiles with NO
    # adapter at all), but that entry does NOT contain a telegram adapter — beta
    # connected a different platform (discord). The telegram sub owned by beta
    # must therefore resolve to NO adapter, not silently borrow the default
    # profile's telegram bot.
    runner._profile_adapters = {"beta": {Platform.DISCORD: other_adapter}}
    runner._kanban_sub_fail_counts = {}

    asyncio.run(_run_one_notifier_tick(monkeypatch, runner))

    # The default profile's adapter must never receive beta's notification.
    assert default_adapter.sent == [], (
        "Owning-profile subscription must not fall back to the default "
        f"profile's adapter; got {default_adapter.sent!r}"
    )
    assert other_adapter.sent == [], (
        f"beta's discord adapter must not receive a telegram sub; got {other_adapter.sent!r}"
    )
    # The claim is rewound (adapter resolved to None → treated as disconnected),
    # so the event is still unseen and will deliver once beta's adapter connects.
    assert [ev.kind for ev in _unseen_terminal_events_for(tid, "chat-beta")] == ["completed"]


def test_notifier_standalone_owner_uses_its_own_adapter(tmp_path, monkeypatch):
    """A standalone profile gateway keeps its own adapter in ``self.adapters``.

    The stamped origin must be allowed to resolve there, but only when it
    matches the gateway's active profile; unrelated secondary profiles remain
    fail-closed in ``test_notifier_owning_profile_adapter_no_default_fallback``.
    """
    db_path = tmp_path / "standalone-owner.db"
    monkeypatch.setenv("HERMES_KANBAN_DB", str(db_path))
    kb.init_db()
    conn = kb.connect()
    try:
        tid = kb.create_task(conn, title="owned by reviewer", assignee="worker")
        kb.add_notify_sub(
            conn, task_id=tid, platform="telegram", chat_id="chat-reviewer",
            notifier_profile="reviewer",
        )
        kb.complete_task(conn, tid, summary="done")
    finally:
        conn.close()

    reviewer_adapter = RecordingAdapter()
    runner = GatewayRunner.__new__(GatewayRunner)
    runner._running = True
    runner.adapters = {Platform.TELEGRAM: reviewer_adapter}  # type: ignore[assignment]
    runner._profile_adapters = {}
    runner._active_profile_name = lambda: "reviewer"
    runner._kanban_sub_fail_counts = {}

    asyncio.run(_run_one_notifier_tick(monkeypatch, runner))

    assert len(reviewer_adapter.sent) == 1
    assert reviewer_adapter.sent[0]["chat_id"] == "chat-reviewer"


def test_notifier_lease_owner_can_use_stamped_profile_adapter(tmp_path, monkeypatch):
    """Lease ownership is independent of the stamped notifier profile.

    The active gateway may run a different profile while hosting the stamped
    profile's adapter. It must use that exact adapter, never its default bot.
    """
    db_path = tmp_path / "non-owner-registry.db"
    monkeypatch.setenv("HERMES_KANBAN_DB", str(db_path))
    monkeypatch.setattr(
        "hermes_cli.config.load_config",
        lambda: {"kanban": {"dispatch_in_gateway": False}},
    )
    kb.init_db()

    conn = kb.connect()
    try:
        tid = kb.create_task(conn, title="owned by reviewer", assignee="worker")
        kb.add_notify_sub(
            conn, task_id=tid, platform="telegram", chat_id="chat-reviewer",
            notifier_profile="reviewer",
        )
        kb.complete_task(conn, tid, summary="done")
    finally:
        conn.close()

    wrong_gateway_adapter = RecordingAdapter()
    owner_adapter = RecordingAdapter()
    runner = GatewayRunner.__new__(GatewayRunner)
    runner._running = True
    runner._kanban_notifier_profile = "gym-health-bro"
    runner.adapters = {Platform.TELEGRAM: wrong_gateway_adapter}
    runner._profile_adapters = {"reviewer": {Platform.TELEGRAM: owner_adapter}}
    runner._kanban_sub_fail_counts = {}

    asyncio.run(_run_one_notifier_tick(monkeypatch, runner))

    assert wrong_gateway_adapter.sent == []
    assert len(owner_adapter.sent) == 1
    assert owner_adapter.sent[0]["chat_id"] == "chat-reviewer"
    assert _unseen_terminal_events_for(tid, "chat-reviewer") == []


def test_status_card_reuses_verified_message_id(tmp_path, monkeypatch):
    db_path = tmp_path / "status-card.db"
    monkeypatch.setenv("HERMES_KANBAN_DB", str(db_path))
    kb.init_db()
    conn = kb.connect()
    try:
        tid = kb.create_task(conn, title="status card", assignee="worker")
        kb.add_notify_sub(conn, task_id=tid, platform="telegram", chat_id="-100", thread_id="1")
        kb._append_event(conn, tid, kind="blocked", payload={"reason": "need input"})
    finally:
        conn.close()

    adapter = StatusCardAdapter()
    asyncio.run(_run_one_notifier_tick(monkeypatch, _make_runner(adapter)))
    assert len(adapter.sent) == 1
    assert adapter.edits == []

    conn = kb.connect()
    try:
        kb._append_event(conn, tid, kind="status", payload={"status": "running"})
        surface = conn.execute("SELECT message_id, delivered_event_id FROM kanban_status_surfaces").fetchone()
        assert surface["message_id"] == "status-card-1"
        assert surface["delivered_event_id"] > 0
    finally:
        conn.close()

    asyncio.run(_run_one_notifier_tick(monkeypatch, _make_runner(adapter)))
    assert len(adapter.sent) == 1, "updates must edit the one origin-lane card"
    assert [edit["message_id"] for edit in adapter.edits] == ["status-card-1"]


def test_status_card_preserves_custom_emoji_metadata_on_edit(tmp_path, monkeypatch):
    """Telegram custom-emoji status presentation survives each card edit."""
    db_path = tmp_path / "custom-emoji-card.db"
    monkeypatch.setenv("HERMES_KANBAN_DB", str(db_path))
    kb.init_db()
    conn = kb.connect()
    try:
        tid = kb.create_task(conn, title="custom emoji card", assignee="worker")
        kb.add_notify_sub(conn, task_id=tid, platform="telegram", chat_id="-100", thread_id="1")
        kb._append_event(conn, tid, kind="claimed", payload={"run_id": 1})
        kb._append_event(conn, tid, kind="heartbeat", payload={"note": "working"})
    finally:
        conn.close()

    adapter = CustomEmojiStatusCardAdapter()
    asyncio.run(_run_one_notifier_tick(monkeypatch, _make_runner(adapter)))

    assert adapter.sent[0]["metadata"]["telegram_custom_emoji"] == {"🔄": "custom-emoji-id"}
    with kb.connect() as conn:
        kb._append_event(conn, tid, kind="blocked", payload={"reason": "review"})
    asyncio.run(_run_one_notifier_tick(monkeypatch, _make_runner(adapter)))
    assert adapter.edits[0]["metadata"]["telegram_custom_emoji"] == {"🔄": "custom-emoji-id"}


def test_status_surface_lease_is_single_owner_across_processes(tmp_path, monkeypatch):
    """A real second process cannot take an unexpired status-card lease."""
    db_path = tmp_path / "surface-lease.db"
    monkeypatch.setenv("HERMES_KANBAN_DB", str(db_path))
    kb.init_db()
    queue = multiprocessing.get_context("spawn").Queue()
    first = multiprocessing.get_context("spawn").Process(
        target=_claim_surface_in_child, args=(str(db_path), "gateway-a:1", queue),
    )
    second = multiprocessing.get_context("spawn").Process(
        target=_claim_surface_in_child, args=(str(db_path), "gateway-b:2", queue),
    )
    first.start()
    first.join(20)
    second.start()
    second.join(20)
    assert first.exitcode == 0
    assert second.exitcode == 0
    assert sorted([queue.get(timeout=5), queue.get(timeout=5)]) == [False, True]


def test_status_surface_failure_retains_lane_and_recovery_claim(tmp_path, monkeypatch):
    db_path = tmp_path / "surface-recovery.db"
    monkeypatch.setenv("HERMES_KANBAN_DB", str(db_path))
    kb.init_db()
    conn = kb.connect()
    try:
        claimed = kb.claim_status_surface(
            conn, task_id="task", platform="telegram", chat_id="114874376", thread_id="1",
            owner="gateway-a:1", event_id=4,
        )
        assert claimed is not None
        assert kb.record_status_surface_failure(
            conn, task_id="task", platform="telegram", chat_id="114874376", thread_id="1",
            owner="gateway-a:1", generation=claimed["lease_generation"], error="timeout",
        )
        assert kb.claim_status_surface(
            conn, task_id="task", platform="telegram", chat_id="114874376", thread_id="1",
            owner="gateway-b:2", event_id=4,
        ) is None
        recovered = kb.claim_status_surface(
            conn, task_id="task", platform="telegram", chat_id="114874376", thread_id="1",
            owner="gateway-b:2", event_id=4, recover_parked=True,
        )
        assert recovered is not None
        assert recovered["attempts"] == 1
        assert recovered["last_error"] == "timeout"
    finally:
        conn.close()


def test_flood_control_cooldown_is_durable_and_board_wide(tmp_path, monkeypatch):
    db_path = tmp_path / "surface-flood-control.db"
    monkeypatch.setenv("HERMES_KANBAN_DB", str(db_path))
    kb.init_db()
    conn = kb.connect()
    try:
        claimed = kb.claim_status_surface(
            conn, task_id="task", platform="telegram", chat_id="114874376", thread_id="1515141",
            owner="gateway-a:1", event_id=4,
        )
        assert claimed is not None
        state = kb.record_status_surface_failure(
            conn, task_id="task", platform="telegram", chat_id="114874376", thread_id="1515141",
            owner="gateway-a:1", generation=claimed["lease_generation"],
            error="flood_control: retry after 3600",
        )
        assert state is not None
        assert kb.notification_flood_cooldown_until(conn) == state["next_retry_at"]
    finally:
        conn.close()


def test_flood_control_does_not_claim_events_before_remote_delivery(tmp_path, monkeypatch):
    db_path = tmp_path / "notifier-flood-control.db"
    monkeypatch.setenv("HERMES_KANBAN_DB", str(db_path))
    kb.init_db()
    conn = kb.connect()
    try:
        tid = kb.create_task(conn, title="preserve event during flood")
        kb.add_notify_sub(conn, task_id=tid, platform="telegram", chat_id="chat-1")
        assert kb.complete_task(conn, tid, result="done")
        claimed = kb.claim_status_surface(
            conn, task_id=tid, platform="telegram", chat_id="chat-1", thread_id="",
            owner="previous-gateway", event_id=0,
        )
        assert claimed is not None
        assert kb.record_status_surface_failure(
            conn, task_id=tid, platform="telegram", chat_id="chat-1", thread_id="",
            owner="previous-gateway", generation=claimed["lease_generation"],
            error="flood_control: retry after 3600",
        ) is not None
    finally:
        conn.close()

    adapter = StatusCardAdapter()
    asyncio.run(_run_one_notifier_tick(monkeypatch, _make_runner(adapter)))

    conn = kb.connect()
    try:
        assert adapter.sent == []
        assert kb.list_notify_subs(conn, tid)[0]["last_event_id"] == 0
    finally:
        conn.close()


def test_status_card_lifecycle_keeps_one_exact_group_topic_message(tmp_path, monkeypatch):
    """Claim/progress/block/unblock/complete all edit the one group-topic card."""
    db_path = tmp_path / "lifecycle.db"
    monkeypatch.setenv("HERMES_KANBAN_DB", str(db_path))
    kb.init_db()
    conn = kb.connect()
    try:
        tid = kb.create_task(conn, title="lifecycle", assignee="worker")
        kb.add_notify_sub(conn, task_id=tid, platform="telegram", chat_id="-100123", thread_id="1")
        kb._append_event(conn, tid, kind="claimed", payload={"run_id": 1})
        kb._append_event(conn, tid, kind="heartbeat", payload={"note": "indexing 1/2"})
        kb._append_event(conn, tid, kind="blocked", payload={"reason": "input"})
        kb._append_event(conn, tid, kind="unblocked", payload={})
        kb._append_event(conn, tid, kind="completed", payload={"summary": "done"})
    finally:
        conn.close()

    adapter = StatusCardAdapter()
    asyncio.run(_run_one_notifier_tick(monkeypatch, _make_runner(adapter)))

    assert len(adapter.sent) == 1
    assert adapter.sent[0]["chat_id"] == "-100123"
    assert adapter.sent[0]["metadata"]["thread_id"] == "1"
    assert adapter.edits == []
    assert all(edit["message_id"] == "status-card-1" for edit in adapter.edits)


def test_status_card_edit_preserves_exact_group_general_topic_route(tmp_path, monkeypatch):
    """A current adapter receives the same exact route on create and edit.

    ``StatusCardAdapter`` above intentionally has the legacy edit signature
    without metadata. This current-adapter fixture protects the compatibility
    bridge while preserving the exact Workout group General-topic lane.
    """
    db_path = tmp_path / "workout-general-topic-edit.db"
    monkeypatch.setenv("HERMES_KANBAN_DB", str(db_path))
    kb.init_db()
    conn = kb.connect()
    try:
        tid = kb.create_task(conn, title="Workout general topic", assignee="worker")
        kb.add_notify_sub(
            conn,
            task_id=tid,
            platform="telegram",
            chat_id="-1003943225553",
            thread_id="1",
        )
        kb._append_event(conn, tid, kind="claimed", payload={"run_id": 1})
        kb._append_event(conn, tid, kind="heartbeat", payload={"note": "working"})
    finally:
        conn.close()

    adapter = TopicAwareStatusCardAdapter()
    asyncio.run(_run_one_notifier_tick(monkeypatch, _make_runner(adapter)))

    assert len(adapter.sent) == 1
    assert adapter.sent[0]["chat_id"] == "-1003943225553"
    assert adapter.sent[0]["metadata"] == {"thread_id": "1"}
    with kb.connect() as conn:
        kb._append_event(conn, tid, kind="blocked", payload={"reason": "review"})
    asyncio.run(_run_one_notifier_tick(monkeypatch, _make_runner(adapter)))
    assert len(adapter.edits) == 1
    assert adapter.edits[0]["chat_id"] == "-1003943225553"
    assert adapter.edits[0]["message_id"] == "status-card-1"
    assert adapter.edits[0]["metadata"] == {"thread_id": "1"}
    conn = kb.connect()
    try:
        surface = conn.execute(
            "SELECT chat_id, thread_id, message_id FROM kanban_status_surfaces WHERE task_id=?",
            (tid,),
        ).fetchone()
    finally:
        conn.close()
    assert tuple(surface) == ("-1003943225553", "1", "status-card-1")


def test_status_card_no_event_refresh_preserves_exact_group_topic_route(tmp_path, monkeypatch):
    """Pulse and renderer migration edits retain the original group topic."""
    db_path = tmp_path / "workout-general-topic-refresh.db"
    monkeypatch.setenv("HERMES_KANBAN_DB", str(db_path))
    kb.init_db()
    conn = kb.connect()
    try:
        tid = kb.create_task(conn, title="Workout general topic refresh", assignee="worker")
        kb.add_notify_sub(
            conn,
            task_id=tid,
            platform="telegram",
            chat_id="-1003943225553",
            thread_id="1",
        )
        claimed = kb.claim_task(conn, tid)
        assert claimed is not None
        assert kb.heartbeat_worker(conn, tid, note="working", expected_run_id=claimed.current_run_id)
    finally:
        conn.close()

    baseline = int(time.time())
    for module in (
        "gateway.kanban_status_card.time.time",
        "gateway.kanban_watchers.time.time",
        "hermes_cli.kanban_db.time.time",
    ):
        monkeypatch.setattr(module, lambda: baseline)
    adapter = TopicAwareStatusCardAdapter()
    asyncio.run(_run_one_notifier_tick(monkeypatch, _make_runner(adapter)))
    assert adapter.sent[0]["metadata"] == {"thread_id": "1"}

    for module in (
        "gateway.kanban_status_card.time.time",
        "gateway.kanban_watchers.time.time",
        "hermes_cli.kanban_db.time.time",
    ):
        monkeypatch.setattr(module, lambda: baseline + 15)
    asyncio.run(_run_one_notifier_tick(monkeypatch, _make_runner(adapter)))
    assert adapter.edits[-1]["message_id"] == "status-card-1"
    assert adapter.edits[-1]["metadata"] == {"thread_id": "1"}

    conn = kb.connect()
    try:
        conn.execute("UPDATE kanban_status_surfaces SET renderer_version='old'")
        conn.commit()
    finally:
        conn.close()
    edits_before_migration = len(adapter.edits)
    asyncio.run(_run_one_notifier_tick(monkeypatch, _make_runner(adapter)))
    assert len(adapter.edits) == edits_before_migration + 1
    assert adapter.edits[-1]["message_id"] == "status-card-1"
    assert adapter.edits[-1]["metadata"] == {"thread_id": "1"}


def test_completed_lifecycle_uses_one_card_and_one_final_ping(tmp_path, monkeypatch):
    """A completion edits the origin card, then sends one separately receipted ping."""
    db_path = tmp_path / "final-ready-lifecycle.db"
    monkeypatch.setenv("HERMES_KANBAN_DB", str(db_path))
    monkeypatch.setenv("HERMES_MEDIA_ALLOW_DIRS", str(tmp_path))
    kb.init_db()

    report = tmp_path / "REPORT.md"
    manifest = tmp_path / "internal.json"
    text = tmp_path / "notes.txt"
    archive = tmp_path / "artifacts.tar.gz"
    pdf = tmp_path / "deliverable.pdf"
    image = tmp_path / "preview.png"
    for path in (report, manifest, text, archive, pdf, image):
        path.write_bytes(b"artifact")

    conn = kb.connect()
    try:
        tid = kb.create_task(
            conn,
            title="internal title",
            assignee="heavy",
            origin_intent_id="notifier-lifecycle-intent",
        )
        kb.add_notify_sub(
            conn, task_id=tid, platform="telegram", chat_id="-100123", thread_id="1515141",
            notifier_profile="reviewer",
        )
        kb._append_event(conn, tid, kind="claimed", payload={})
        kb._append_event(conn, tid, kind="heartbeat", payload={"note": "Проверяю доставку"})
        claimed = kb.claim_task(conn, tid)
        assert claimed is not None
        assert kb.request_review(
            conn, tid, summary="ready for review: notifier plumbing",
            metadata={
                "user_facing_title": "Проверить готовность карточки",
                "notification_key": "notifier-final-ready",
                "notification_summary": (
                    "What changed: настроили доставку уведомлений.\n"
                    "What works: финальное сообщение приходит после аудита.\n"
                    "How to verify: завершить тестовую карточку."
                ),
                    "artifacts": [str(report), str(manifest), str(text), str(archive), str(pdf), str(image)],
            },
            expected_run_id=kb.get_task(conn, tid).current_run_id,
        )
        assert kb.accept_review(conn, tid, summary="verified by reviewer")
    finally:
        conn.close()

    class ExactLifecycleAdapter(StatusCardAdapter):
        def __init__(self):
            super().__init__()
            self.documents = []
            self.images = []

        async def send(self, chat_id, text, metadata=None):
            self.sent.append({"chat_id": chat_id, "text": text, "metadata": metadata or {}})
            return SendResult(success=True, message_id=f"message-{len(self.sent)}")

        async def send_document(self, *, chat_id, file_path, metadata):
            self.documents.append((chat_id, file_path, metadata))

        async def send_multiple_images(self, *, chat_id, images, metadata):
            self.images.extend((chat_id, image, metadata) for image in images)

    adapter = ExactLifecycleAdapter()
    first = _make_runner(adapter)
    first._active_profile_name = lambda: "reviewer"
    asyncio.run(_run_one_notifier_tick(monkeypatch, first))
    second = _make_runner(adapter)
    second._active_profile_name = lambda: "reviewer"
    asyncio.run(_run_one_notifier_tick(monkeypatch, second))

    assert len(adapter.sent) == 2
    assert adapter.sent[0]["chat_id"] == "-100123"
    assert adapter.sent[0]["metadata"]["thread_id"] == "1515141"
    assert adapter.sent[1]["text"] == (
        "Results are ready:\n"
        "What changed: настроили доставку уведомлений.\n"
        "What works: финальное сообщение приходит после аудита.\n"
        "How to verify: завершить тестовую карточку."
    )
    assert adapter.sent[1]["metadata"] == {"thread_id": "1515141"}
    assert adapter.edits == []
    assert adapter.documents == []
    assert adapter.images == []
    assert all(edit["message_id"] == "message-1" for edit in adapter.edits)

    conn = kb.connect()
    try:
        row = conn.execute(
            "SELECT notifier_profile, message_id FROM kanban_terminal_notifications WHERE task_id=?",
            (tid,),
        ).fetchone()
        assert dict(row) == {"notifier_profile": "reviewer", "message_id": "message-2"}
    finally:
        conn.close()


def test_terminal_notification_intent_dedupe_survives_subprocess_replay(tmp_path, monkeypatch):
    """Two independent notifier processes can enqueue one final intent handoff."""
    db_path = tmp_path / "intent-subprocess.db"
    monkeypatch.setenv("HERMES_KANBAN_DB", str(db_path))
    kb.init_db()
    with kb.connect() as conn:
        task_id = kb.create_task(
            conn, title="intent final", origin_intent_id="intent-subprocess"
        )
        assert kb.request_review(conn, task_id, summary="ready for review")
        assert kb.accept_review(conn, task_id, summary="accepted")
        event_id = conn.execute(
            "SELECT id FROM task_events WHERE task_id=? AND kind='review_accepted' "
            "ORDER BY id DESC LIMIT 1",
            (task_id,),
        ).fetchone()[0]
        conn.execute(
            "INSERT INTO kanban_status_surfaces "
            "(task_id, platform, chat_id, thread_id, message_id, delivered_event_id, created_at, updated_at) "
            "VALUES (?, 'telegram', '114874376', '1515141', 'card', ?, 1, 1)",
            (task_id, event_id),
        )

    context = multiprocessing.get_context("spawn")
    queue = context.Queue()
    workers = [
        context.Process(
            target=_enqueue_terminal_notification_in_child,
            args=(str(db_path), task_id, event_id, queue),
        )
        for _ in range(2)
    ]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join(20)

    assert [worker.exitcode for worker in workers] == [0, 0]
    assert sorted([queue.get(timeout=5), queue.get(timeout=5)]) == [False, True]
    with kb.connect() as conn:
        rows = conn.execute(
            "SELECT task_id, outcome_key FROM kanban_terminal_notifications"
        ).fetchall()
    assert [tuple(row) for row in rows] == [
        (task_id, "origin-intent:intent-subprocess")
    ]


def test_missing_send_receipt_is_retryable_and_does_not_advance_cursor(tmp_path, monkeypatch):
    class MissingReceiptAdapter:
        async def send(self, chat_id, text, metadata=None):
            return SendResult(success=True, message_id=None)

    db_path = tmp_path / "missing-receipt.db"
    monkeypatch.setenv("HERMES_KANBAN_DB", str(db_path))
    kb.init_db()
    conn = kb.connect()
    try:
        tid = kb.create_task(conn, title="receipt", assignee="worker")
        kb.add_notify_sub(conn, task_id=tid, platform="telegram", chat_id="chat-receipt")
        kb._append_event(conn, tid, kind="claimed", payload={})
    finally:
        conn.close()

    asyncio.run(_run_one_notifier_tick(monkeypatch, _make_runner(MissingReceiptAdapter())))

    conn = kb.connect()
    try:
        sub = kb.list_notify_subs(conn, tid)[0]
        surface = conn.execute("SELECT message_id, attempts, last_error FROM kanban_status_surfaces").fetchone()
        assert sub["last_event_id"] == 0
        assert surface["message_id"] is None
        assert surface["attempts"] == 1
        assert "receipt" in surface["last_error"]
    finally:
        conn.close()


def test_status_card_pulse_refresh_and_version_migration_are_same_message(tmp_path, monkeypatch):
    """No-event ticks edit the live card, never cursor/event rows or message id."""
    db_path = tmp_path / "pulse-and-migration.db"
    monkeypatch.setenv("HERMES_KANBAN_DB", str(db_path))
    kb.init_db()
    conn = kb.connect()
    try:
        tid = kb.create_task(conn, title="pulse", assignee="worker")
        kb.add_notify_sub(conn, task_id=tid, platform="telegram", chat_id="chat-pulse")
        claimed = kb.claim_task(conn, tid)
        assert claimed is not None
        assert kb.heartbeat_worker(conn, tid, note="working", expected_run_id=claimed.current_run_id)
    finally:
        conn.close()

    baseline = int(time.time())
    monkeypatch.setattr("gateway.kanban_status_card.time.time", lambda: baseline)
    monkeypatch.setattr("gateway.kanban_watchers.time.time", lambda: baseline)
    monkeypatch.setattr("hermes_cli.kanban_db.time.time", lambda: baseline)
    adapter = StatusCardAdapter()
    asyncio.run(_run_one_notifier_tick(monkeypatch, _make_runner(adapter)))
    assert len(adapter.sent) == 1

    for offset, pulse in ((15, "🟢"), (120, "🟡"), (915, "🔴")):
        monkeypatch.setattr("gateway.kanban_status_card.time.time", lambda: baseline + offset)
        monkeypatch.setattr("gateway.kanban_watchers.time.time", lambda: baseline + offset)
        monkeypatch.setattr("hermes_cli.kanban_db.time.time", lambda: baseline + offset)
        asyncio.run(_run_one_notifier_tick(monkeypatch, _make_runner(adapter)))
        assert adapter.edits[-1]["message_id"] == "status-card-1"
        assert pulse in adapter.edits[-1]["content"]

    conn = kb.connect()
    try:
        cursor = kb.list_notify_subs(conn, tid)[0]["last_event_id"]
        conn.execute("UPDATE kanban_status_surfaces SET renderer_version='old'")
        conn.commit()
    finally:
        conn.close()
    migration_edits = len(adapter.edits)
    asyncio.run(_run_one_notifier_tick(monkeypatch, _make_runner(adapter)))
    assert len(adapter.sent) == 1
    assert len(adapter.edits) == migration_edits + 1
    asyncio.run(_run_one_notifier_tick(monkeypatch, _make_runner(adapter)))
    assert len(adapter.edits) == migration_edits + 1
    conn = kb.connect()
    try:
        surface = conn.execute(
            "SELECT message_id, renderer_version FROM kanban_status_surfaces"
        ).fetchone()
        assert surface["message_id"] == "status-card-1"
        assert surface["renderer_version"] == "2026-07-15.5"
        assert kb.list_notify_subs(conn, tid)[0]["last_event_id"] == cursor
    finally:
        conn.close()


def test_status_card_failed_edit_retries_existing_card_then_parks_without_replacement_send(tmp_path, monkeypatch):
    """15s/60s pulses and a deleted Telegram card never create a replacement.

    A bad edit keeps the durable message id intact through the bounded retry
    budget. Once parked, repeated notifier ticks are silent until a deliberate
    recovery action, rather than posting one status message per tick.
    """
    db_path = tmp_path / "failed-edit-no-replacement.db"
    monkeypatch.setenv("HERMES_KANBAN_DB", str(db_path))
    kb.init_db()
    conn = kb.connect()
    try:
        tid = kb.create_task(conn, title="missing Telegram card", assignee="worker")
        kb.add_notify_sub(conn, task_id=tid, platform="telegram", chat_id="chat-missing")
        assert kb.claim_task(conn, tid) is not None
        kb._append_event(conn, tid, kind="heartbeat", payload={"note": "working"})
    finally:
        conn.close()

    baseline = int(time.time())
    for module in ("gateway.kanban_status_card.time.time", "gateway.kanban_watchers.time.time", "hermes_cli.kanban_db.time.time"):
        monkeypatch.setattr(module, lambda: baseline)
    adapter = MissingStatusCardAdapter()
    asyncio.run(_run_one_notifier_tick(monkeypatch, _make_runner(adapter)))
    assert len(adapter.sent) == 1

    edits_before_pulse = len(adapter.edits)
    for module in ("gateway.kanban_status_card.time.time", "gateway.kanban_watchers.time.time", "hermes_cli.kanban_db.time.time"):
        monkeypatch.setattr(module, lambda: baseline + 15)
    asyncio.run(_run_one_notifier_tick(monkeypatch, _make_runner(adapter)))
    assert len(adapter.sent) == 1
    assert len(adapter.edits) == edits_before_pulse + 1
    assert all(edit["message_id"] == "status-card-1" for edit in adapter.edits)

    adapter.fail_edits = True
    for offset in (60, 90, 150, 300):
        for module in ("gateway.kanban_status_card.time.time", "gateway.kanban_watchers.time.time", "hermes_cli.kanban_db.time.time"):
            monkeypatch.setattr(module, lambda offset=offset: baseline + offset)
        asyncio.run(_run_one_notifier_tick(monkeypatch, _make_runner(adapter)))

    assert len(adapter.sent) == 1, "a failed edit must never fall back to adapter.send()"
    assert all(edit["message_id"] == "status-card-1" for edit in adapter.edits)
    conn = kb.connect()
    try:
        surface = conn.execute(
            "SELECT message_id, attempts, dead_lettered_at FROM kanban_status_surfaces WHERE task_id=?", (tid,)
        ).fetchone()
        assert surface["message_id"] == "status-card-1"
        assert surface["attempts"] == 3
        assert surface["dead_lettered_at"] is not None
        conn.execute("DELETE FROM kanban_notify_subs WHERE task_id=?", (tid,))
        kb.add_notify_sub(conn, task_id=tid, platform="telegram", chat_id="chat-missing")
        recovered = conn.execute(
            "SELECT message_id, attempts, dead_lettered_at FROM kanban_status_surfaces WHERE task_id=?", (tid,)
        ).fetchone()
        assert recovered["message_id"] == "status-card-1"
        assert recovered["attempts"] == 0
        assert recovered["dead_lettered_at"] is None
    finally:
        conn.close()


def test_status_card_recreates_once_after_edit_target_lost_then_resumes_edit_only(tmp_path, monkeypatch):
    """A card already parked with ``Message to edit not found`` self-heals.

    This models a card that reached ``dead_lettered_at`` before the live
    intra-refresh fallback (edit fails with ``error_kind=edit_message_not_found``
    -> immediate ``adapter.send()`` in the same tick) ever got a chance to
    recreate it — e.g. state left over from an older deploy, exactly the
    shape found in production. The next notifier tick must pick it up via
    the recover-parked path, recreate it exactly once, and resume normal
    edit-only refresh on the new message id. A second, distinct loss of
    that same recreated message id must not be auto-recovered again.
    """

    class DeletedCardAdapter(StatusCardAdapter):
        def __init__(self):
            super().__init__()
            self.fail_edits = False

        async def send(self, chat_id, text, metadata=None):
            self.sent.append({"chat_id": chat_id, "text": text, "metadata": metadata or {}})
            return SendResult(success=True, message_id=f"status-card-{len(self.sent) + 1}")

        async def edit_message(self, chat_id, message_id, content, *, finalize=False):
            self.edits.append({"chat_id": chat_id, "message_id": message_id, "content": content, "finalize": finalize})
            if self.fail_edits:
                return SendResult(
                    success=False, error="Message to edit not found",
                    error_kind="edit_message_not_found",
                )
            return SendResult(success=True, message_id=message_id)

    db_path = tmp_path / "recreate-once.db"
    monkeypatch.setenv("HERMES_KANBAN_DB", str(db_path))
    kb.init_db()
    now = int(time.time())
    conn = kb.connect()
    try:
        tid = kb.create_task(conn, title="lost Telegram card", assignee="worker")
        kb.add_notify_sub(conn, task_id=tid, platform="telegram", chat_id="chat-lost")
        assert kb.claim_task(conn, tid) is not None
        kb._append_event(conn, tid, kind="heartbeat", payload={"note": "working"})
        conn.execute(
            "UPDATE kanban_status_surfaces SET message_id='status-card-1', attempts=3, "
            "last_error='Message to edit not found', dead_lettered_at=?, updated_at=? "
            "WHERE task_id=?",
            (now, now, tid),
        )
        conn.commit()
    finally:
        conn.close()

    time_modules = (
        "gateway.kanban_status_card.time.time",
        "gateway.kanban_watchers.time.time",
        "hermes_cli.kanban_db.time.time",
    )
    for module in time_modules:
        monkeypatch.setattr(module, lambda: now)

    # First tick: parked lane is recovered — recreated exactly once.
    adapter = DeletedCardAdapter()
    asyncio.run(_run_one_notifier_tick(monkeypatch, _make_runner(adapter)))

    assert len(adapter.sent) == 1, "the parked card must be recreated exactly once"
    conn = kb.connect()
    try:
        recovered = conn.execute(
            "SELECT message_id, attempts, dead_lettered_at, recovered_message_id "
            "FROM kanban_status_surfaces WHERE task_id=?", (tid,),
        ).fetchone()
        assert recovered["message_id"] == "status-card-2"
        assert recovered["attempts"] == 0
        assert recovered["dead_lettered_at"] is None
    finally:
        conn.close()

    # Resumes edit-only behavior: a later pulse edits the recreated message,
    # it never sends a second replacement.
    for module in time_modules:
        monkeypatch.setattr(module, lambda: now + 360)
    asyncio.run(_run_one_notifier_tick(monkeypatch, _make_runner(adapter)))
    assert len(adapter.sent) == 1
    assert adapter.edits[-1]["message_id"] == "status-card-2"


def test_concurrent_status_refresh_claims_only_one_edit(tmp_path, monkeypatch):
    """Two same-process refresh ticks contend on the durable lease exactly once."""
    db_path = tmp_path / "concurrent-refresh.db"
    monkeypatch.setenv("HERMES_KANBAN_DB", str(db_path))
    kb.init_db()
    conn = kb.connect()
    try:
        tid = kb.create_task(conn, title="concurrent refresh", assignee="worker")
        kb.add_notify_sub(conn, task_id=tid, platform="telegram", chat_id="chat-concurrent")
        assert kb.claim_task(conn, tid) is not None
        kb._append_event(conn, tid, kind="heartbeat", payload={"note": "working"})
    finally:
        conn.close()

    baseline = int(time.time())
    for module in ("gateway.kanban_status_card.time.time", "gateway.kanban_watchers.time.time", "hermes_cli.kanban_db.time.time"):
        monkeypatch.setattr(module, lambda: baseline)
    adapter = YieldingStatusCardAdapter()
    asyncio.run(_run_one_notifier_tick(monkeypatch, _make_runner(adapter)))
    assert len(adapter.sent) == 1
    edits_before_concurrent_refresh = len(adapter.edits)

    for module in ("gateway.kanban_status_card.time.time", "gateway.kanban_watchers.time.time", "hermes_cli.kanban_db.time.time"):
        monkeypatch.setattr(module, lambda: baseline + 15)
    conn = kb.connect()
    try:
        refresh = {
            "sub": kb.list_notify_subs(conn, tid)[0],
            "task": kb.get_task(conn, tid),
            "timeline": kb.list_events(conn, tid),
            "parents": [],
        }
    finally:
        conn.close()
    runner = _make_runner(adapter)

    async def run_concurrent_refreshes():
        await asyncio.gather(
            runner._refresh_kanban_status_surface(refresh, notifier_profile="default"),
            runner._refresh_kanban_status_surface(refresh, notifier_profile="default"),
        )

    asyncio.run(run_concurrent_refreshes())

    assert len(adapter.edits) == edits_before_concurrent_refresh + 1
    assert adapter.edits[-1]["message_id"] == "status-card-1"


def test_terminal_helpers_are_silent_until_root_outcome_is_accepted(tmp_path, monkeypatch):
    db_path = tmp_path / "outcome-dedup.db"
    monkeypatch.setenv("HERMES_KANBAN_DB", str(db_path))
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    kb.init_db()
    conn = kb.connect()
    try:
        root = kb.create_task(conn, title="user request", assignee="worker")
        kb.add_notify_sub(
            conn, task_id=root, platform="telegram", chat_id="114874376", thread_id="1515141",
        )
        assert kb.claim_task(conn, root) is not None
        for title in ("first helper", "second helper"):
            tid = kb.create_task(conn, title=title, assignee="worker")
            kb.add_notify_sub(
                conn, task_id=tid, platform="telegram", chat_id="114874376", thread_id="1515141",
            )
            assert kb.claim_task(conn, tid) is not None
            assert kb.request_review(
                conn,
                tid,
                summary="ready for review: helper completed",
                metadata={
                    "notification_key": "reviewer-gateway-restart",
                    "notification_summary": "Reviewer gateway restarted.",
                },
                expected_run_id=kb.get_task(conn, tid).current_run_id,
            )
            kb.link_tasks(conn, root, tid)
            assert kb.accept_review(conn, tid, summary="verified")
    finally:
        conn.close()

    from gateway.topic_anchors import record_topic_anchor
    record_topic_anchor("telegram", "114874376", "1515141", "10813")

    adapter = RecordingAdapter()
    asyncio.run(_run_one_notifier_tick(monkeypatch, _make_runner(adapter)))
    asyncio.run(_run_one_notifier_tick(monkeypatch, _make_runner(adapter)))

    final_messages = [item for item in adapter.sent if item["text"].startswith("Results are ready:")]
    assert final_messages == []

    conn = kb.connect()
    try:
        assert conn.execute("SELECT COUNT(*) FROM kanban_terminal_notifications").fetchone()[0] == 0
        technical_root = kb.create_task(conn, title="restart helper without parent", assignee="worker")
        kb.add_notify_sub(
            conn, task_id=technical_root, platform="telegram", chat_id="114874376", thread_id="1515141",
        )
        assert kb.claim_task(conn, technical_root) is not None
        technical_task = kb.get_task(conn, technical_root)
        assert technical_task is not None
        assert kb.request_review(
            conn,
            technical_root,
            summary="ready for review: restart helper completed",
            metadata={
                "notification_key": "technical-helper",
                "notification_summary": "Reviewer gateway restarted.",
            },
            expected_run_id=technical_task.current_run_id,
        )
        assert kb.accept_review(conn, technical_root, summary="verified")
    finally:
        conn.close()

    asyncio.run(_run_one_notifier_tick(monkeypatch, _make_runner(adapter)))
    asyncio.run(_run_one_notifier_tick(monkeypatch, _make_runner(adapter)))

    final_messages = [item for item in adapter.sent if item["text"].startswith("Results are ready:")]
    assert final_messages == []

    conn = kb.connect()
    try:
        assert conn.execute("SELECT COUNT(*) FROM kanban_terminal_notifications").fetchone()[0] == 0
        root_task = kb.get_task(conn, root)
        assert root_task is not None
        assert kb.request_review(
            conn,
            root,
            summary="ready for review: user outcome completed",
            metadata={
                "notification_key": "user-request-complete",
                "notification_summary": (
                    "What changed: завершили пользовательский запрос.\n"
                    "What works: итог проверен аудитором.\n"
                    "How to verify: открыть результат в исходном чате."
                ),
            },
            expected_run_id=root_task.current_run_id,
        )
        assert kb.accept_review(conn, root, summary="verified")
    finally:
        conn.close()

    asyncio.run(_run_one_notifier_tick(monkeypatch, _make_runner(adapter)))
    asyncio.run(_run_one_notifier_tick(monkeypatch, _make_runner(adapter)))

    final_messages = [item for item in adapter.sent if item["text"].startswith("Results are ready:")]
    assert final_messages == [{
        "chat_id": "114874376",
        "text": (
            "Results are ready:\n"
            "What changed: завершили пользовательский запрос.\n"
            "What works: итог проверен аудитором.\n"
            "How to verify: открыть результат в исходном чате."
        ),
        "metadata": {
            "thread_id": "1515141",
            "telegram_dm_topic_reply_fallback": True,
            "telegram_reply_to_message_id": "10813",
        },
    }]
    assert all(item["metadata"].get("telegram_reply_to_message_id") == "10813" for item in final_messages)

    conn = kb.connect()
    try:
        rows = conn.execute(
            "SELECT outcome_key, outcome_summary, message_id FROM kanban_terminal_notifications"
        ).fetchall()
        assert len(rows) == 1
        assert rows[0]["outcome_key"] == "user-request-complete"
        assert rows[0]["outcome_summary"] == (
            "What changed: завершили пользовательский запрос.\n"
            "What works: итог проверен аудитором.\n"
            "How to verify: открыть результат в исходном чате."
        )
        assert rows[0]["message_id"]
    finally:
        conn.close()


def test_completed_session_is_not_woken_after_status_card_delivery(tmp_path, monkeypatch):
    """Auto-recovery lifecycle events must not become a second chat producer."""
    db_path = tmp_path / "completed-no-wake.db"
    monkeypatch.setenv("HERMES_KANBAN_DB", str(db_path))
    kb.init_db()
    with kb.connect() as conn:
        task_id = kb.create_task(
            conn, title="routine completion", assignee="worker", session_id="origin-session"
        )
        kb.add_notify_sub(conn, task_id=task_id, platform="telegram", chat_id="114874376")
        assert kb.claim_task(conn, task_id) is not None
        assert kb.complete_task(conn, task_id, summary="completed")

    adapter = WakeRecordingStatusCardAdapter()
    asyncio.run(_run_one_notifier_tick(monkeypatch, _make_runner(adapter)))

    assert len(adapter.sent) == 1
    assert adapter.wakes == []


def _unseen_terminal_events_for(tid, chat_id):
    conn = kb.connect()
    try:
        _, events = kb.unseen_events_for_sub(
            conn,
            task_id=tid,
            platform="telegram",
            chat_id=chat_id,
            kinds=["completed", "blocked", "gave_up", "crashed", "timed_out"],
        )
        return events
    finally:
        conn.close()


def test_current_run_projection_keeps_card_and_overview_in_sync_across_retry_and_buckets(tmp_path, monkeypatch):
    """Only durable events for ``tasks.current_run_id`` feed both live views."""
    db_path = tmp_path / "current-run-progress.db"
    monkeypatch.setenv("HERMES_KANBAN_DB", str(db_path))
    kb.init_db()

    def set_db_time(value):
        monkeypatch.setattr(kb.time, "time", lambda: value)

    with kb.connect() as conn:
        task_id = kb.create_task(conn, title="Текущий запуск", assignee="heavy")
        kb.add_notify_sub(conn, task_id=task_id, platform="telegram", chat_id="chat", thread_id="topic")

        set_db_time(1_000)
        first = kb.claim_task(conn, task_id)
        assert first is not None
        first_run_id = first.current_run_id
        set_db_time(1_005)
        assert kb.heartbeat_worker(conn, task_id, note="старый checkpoint", expected_run_id=first_run_id)
        set_db_time(1_010)
        assert kb.block_task(conn, task_id, reason="повторный запуск", kind="transient")
        set_db_time(1_015)
        assert kb.unblock_task(conn, task_id)
        set_db_time(1_020)
        second = kb.claim_task(conn, task_id)
        assert second is not None
        second_run_id = second.current_run_id
        assert second_run_id != first_run_id
        set_db_time(1_025)
        assert kb.heartbeat_worker(conn, task_id, note="текущий checkpoint", expected_run_id=second_run_id)

    # A new DB connection models notifier restart: no process-local run state.
    with kb.connect() as conn:
        task = kb.get_task(conn, task_id)
        timeline = kb.list_events(conn, task_id)
        current_run = kb.current_run_progress(conn, task_id)

    assert current_run.run_id == second_run_id
    assert [event.run_id for event in current_run.events] == [second_run_id, second_run_id, second_run_id]
    assert all("старый" not in str(event.payload) for event in current_run.events)

    def render(now):
        card = render_kanban_status_card(
            sub={"task_id": task_id}, task=task, timeline=timeline, current_run=current_run, now=now,
        )
        overview = render_kanban_active_task_index(
            [("Planner", task, timeline, [], current_run)], now=now,
        )
        return card, overview

    before_boundary = render(1_034)
    at_boundary = render(1_040)
    card_before, overview_before = before_boundary
    card_at_boundary, overview_at_boundary = at_boundary
    assert "текущий checkpoint" in card_before
    for text in before_boundary:
        assert "старый checkpoint" not in text
        assert "under 15s" in text
    assert "текущий checkpoint" in card_at_boundary
    for text in (card_at_boundary, overview_at_boundary):
        assert "15s" in text

    with kb.connect() as conn:
        set_db_time(1_045)
        assert kb.complete_task(conn, task_id, summary="готово", expected_run_id=second_run_id)
        terminal_task = kb.get_task(conn, task_id)
        terminal_timeline = kb.list_events(conn, task_id)
        assert kb.current_run_progress(conn, task_id).run_id is None

    terminal_card = render_kanban_status_card(
        sub={"task_id": task_id}, task=terminal_task, timeline=terminal_timeline,
        current_run=kb.CurrentRunProgress(run_id=None, started_at=None), now=1_045,
    )
    assert "текущий checkpoint" not in terminal_card
    assert "Прогресс по задаче" not in terminal_card


def test_active_index_keeps_terminal_lifecycle_after_current_run_closes(tmp_path, monkeypatch):
    """An empty live-run projection must not erase a task-wide restart warning."""
    db_path = tmp_path / "terminal-current-run.db"
    monkeypatch.setenv("HERMES_KANBAN_DB", str(db_path))
    kb.init_db()

    with kb.connect() as conn:
        tasks = []
        for offset, kind in enumerate(("crashed", "timed_out", "gave_up")):
            task_id = kb.create_task(conn, title=f"После {kind}", assignee="heavy")
            claimed = kb.claim_task(conn, task_id)
            assert claimed is not None
            run_id = claimed.current_run_id
            now = 2_000 + offset
            with kb.write_txn(conn):
                conn.execute(
                    "UPDATE tasks SET status = 'ready', current_run_id = NULL, claim_lock = NULL, "
                    "claim_expires = NULL, worker_pid = NULL WHERE id = ?",
                    (task_id,),
                )
                conn.execute(
                    "UPDATE task_runs SET status = ?, outcome = ?, ended_at = ? WHERE id = ?",
                    (kind, kind, now, run_id),
                )
                kb._append_event(conn, task_id, kind, run_id=run_id)
            task = kb.get_task(conn, task_id)
            timeline = kb.list_events(conn, task_id)
            current_run = kb.current_run_progress(conn, task_id)
            assert current_run.run_id is None
            tasks.append(("Planner", task, timeline, [], current_run))

    overview = render_kanban_active_task_index(tasks, now=2_010)
    assert overview.count("⚠️ После") == 3
    assert "⏳ " not in overview
    assert all(f"После {kind}" in overview for kind in ("crashed", "timed_out", "gave_up"))


def test_active_index_link_label_matches_the_visible_truncated_title():
    task = SimpleNamespace(
        id="t_long",
        title="Enable verified press-hold-drag capability in the Workout Logger Maestro harness",
    )

    label = _active_index_link_label(("Planner", task, []))

    assert label == "Enable verified press-hold-drag capability in the Workout Logger…"


def test_active_task_index_pins_one_message_for_three_tasks_in_one_topic(tmp_path, monkeypatch):
    db_path = tmp_path / "active-index.db"
    monkeypatch.setenv("HERMES_KANBAN_DB", str(db_path))
    kb.init_db()
    conn = kb.connect()
    try:
        task_ids = []
        for title in ("Первая", "Вторая", "Третья"):
            task_id = kb.create_task(conn, title=title, assignee="heavy")
            kb.add_notify_sub(conn, task_id=task_id, platform="telegram", chat_id="-100", thread_id="42")
            task_ids.append(task_id)
    finally:
        conn.close()

    adapter = ActiveIndexAdapter()
    asyncio.run(_run_one_notifier_tick(monkeypatch, _make_runner(adapter)))

    indexes = [item for item in adapter.sent if item["text"].startswith("📌")]
    assert len(indexes) == 2
    assert all(title in indexes[0]["text"] for title in ("Первая", "Вторая", "Третья"))
    assert indexes[0]["metadata"].get("telegram_custom_emoji") == {}
    assert len(adapter.pins) == 2

    asyncio.run(_run_one_notifier_tick(monkeypatch, _make_runner(adapter)))
    assert len([item for item in adapter.sent if item["text"].startswith("📌")]) == 2
    assert len(adapter.pins) == 2


def test_active_task_index_uses_the_same_custom_emoji_as_status_cards(tmp_path, monkeypatch):
    db_path = tmp_path / "active-index-custom-emoji.db"
    monkeypatch.setenv("HERMES_KANBAN_DB", str(db_path))
    kb.init_db()
    with kb.connect() as conn:
        running = kb.create_task(conn, title="В работе", assignee="heavy")
        queued = kb.create_task(conn, title="Следом", assignee="heavy")
        kb.add_notify_sub(conn, task_id=running, platform="telegram", chat_id="-100", thread_id="42")
        kb.add_notify_sub(conn, task_id=queued, platform="telegram", chat_id="-100", thread_id="42")
        assert kb.claim_task(conn, running) is not None

    adapter = CustomEmojiActiveIndexAdapter()
    asyncio.run(_run_one_notifier_tick(monkeypatch, _make_runner(adapter)))

    indexes = [item for item in adapter.sent if item["text"].startswith("📌")]
    assert len(indexes) == 2
    assert indexes[0]["metadata"]["telegram_custom_emoji"] == {
        "🔫": "running-emoji-id", "⏳": "queue-emoji-id",
    }
    assert "📌 Default · 2 tasks" in indexes[0]["text"]
    assert "🔫 В работе" in indexes[0]["text"]
    assert "⏳ Следом" in indexes[0]["text"]

    with kb.connect() as conn:
        assert kb.heartbeat_worker(conn, running, note="есть прогресс")
    asyncio.run(_run_one_notifier_tick(monkeypatch, _make_runner(adapter)))

    index_edits = [item for item in adapter.edits if item["content"].startswith("📌")]
    assert index_edits[-1]["metadata"]["telegram_custom_emoji"] == {
        "🔫": "running-emoji-id", "⏳": "queue-emoji-id",
    }


def test_active_task_index_private_dm_topic_never_pins_and_reuses_one_message_after_restart(tmp_path, monkeypatch):
    db_path = tmp_path / "active-index-dm-topic.db"
    monkeypatch.setenv("HERMES_KANBAN_DB", str(db_path))
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    kb.init_db()
    with kb.connect() as conn:
        task_id = kb.create_task(conn, title="Личный топик без pin", assignee="heavy")
        kb.add_notify_sub(
            conn, task_id=task_id, platform="telegram", chat_id="114874376", thread_id="1515141",
        )

    adapter = ActiveIndexAdapter()
    asyncio.run(_run_one_notifier_tick(monkeypatch, _make_runner(adapter)))
    indexes = [item for item in adapter.sent if item["text"].startswith("📌")]
    assert len(indexes) == 2
    assert any(item["metadata"].get("thread_id") == "1515141" for item in indexes)
    assert len(adapter.pins) == 2

    with kb.connect_active_task_index_registry() as conn:
        assert conn.execute("SELECT COUNT(*) FROM kanban_active_task_indexes").fetchone()[0] == 2
    with kb.connect() as conn:
        assert kb.block_task(conn, task_id, reason="нужен ответ", kind="needs_input")

    asyncio.run(_run_one_notifier_tick(monkeypatch, _make_runner(adapter)))
    asyncio.run(_run_one_notifier_tick(monkeypatch, _make_runner(adapter)))

    index_edits = [edit for edit in adapter.edits if edit["content"].startswith("📌")]
    assert len(index_edits) >= 2
    assert len(adapter.pins) == 2
    assert len([item for item in adapter.sent if item["text"].startswith("📌")]) == 2


def test_active_task_indexes_keep_private_dm_topics_separate_without_pins(tmp_path, monkeypatch):
    db_path = tmp_path / "active-index-dm-topics.db"
    monkeypatch.setenv("HERMES_KANBAN_DB", str(db_path))
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    kb.init_db()
    with kb.connect() as conn:
        first = kb.create_task(conn, title="Первый личный топик", assignee="heavy")
        second = kb.create_task(conn, title="Второй личный топик", assignee="heavy")
        kb.add_notify_sub(conn, task_id=first, platform="telegram", chat_id="114874376", thread_id="1515141")
        kb.add_notify_sub(conn, task_id=second, platform="telegram", chat_id="114874376", thread_id="1515142")

    adapter = ActiveIndexAdapter()
    asyncio.run(_run_one_notifier_tick(monkeypatch, _make_runner(adapter)))

    indexes = [item for item in adapter.sent if item["text"].startswith("📌")]
    assert len(indexes) == 3
    topic_indexes = [item for item in indexes if item["metadata"].get("thread_id")]
    assert len(topic_indexes) == 2
    assert all(("Первый личный топик" in item["text"]) != ("Второй личный топик" in item["text"]) for item in topic_indexes)
    assert len(adapter.pins) == 3


def test_private_dm_topic_false_pinned_receipt_is_normalized_without_recreating_index(tmp_path, monkeypatch):
    db_path = tmp_path / "active-index-dm-repair.db"
    monkeypatch.setenv("HERMES_KANBAN_DB", str(db_path))
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    kb.init_db()
    with kb.connect() as conn:
        task_id = kb.create_task(conn, title="Исправить старый receipt", assignee="heavy")
        kb.add_notify_sub(
            conn, task_id=task_id, platform="telegram", chat_id="114874376", thread_id="1515141",
        )
        now = int(time.time())
        conn.execute(
            """
            INSERT INTO kanban_active_task_indexes
                (platform, chat_id, thread_id, notifier_profile, message_id, pinned, created_at, updated_at)
            VALUES ('telegram', '114874376', '1515141', '', '9999', 1, ?, ?)
            """,
            (now, now),
        )

    adapter = ActiveIndexAdapter()
    asyncio.run(_run_one_notifier_tick(monkeypatch, _make_runner(adapter)))

    assert len([item for item in adapter.sent if item["text"].startswith("📌")]) == 2
    assert len(adapter.pins) == 2
    with kb.connect_active_task_index_registry() as conn:
        receipt = conn.execute(
            "SELECT message_id, pinned FROM kanban_active_task_indexes WHERE thread_id=''"
        ).fetchone()
        assert receipt is not None
        assert receipt["message_id"] == "status-card-1"
        assert receipt["pinned"] == 1


def test_active_task_index_is_exact_topic_projection_and_removes_done_task(tmp_path, monkeypatch):
    db_path = tmp_path / "active-index-topics.db"
    monkeypatch.setenv("HERMES_KANBAN_DB", str(db_path))
    kb.init_db()
    conn = kb.connect()
    try:
        first = kb.create_task(conn, title="Только первый топик", assignee="heavy")
        second = kb.create_task(conn, title="Только второй топик", assignee="heavy")
        kb.add_notify_sub(conn, task_id=first, platform="telegram", chat_id="-100", thread_id="1")
        kb.add_notify_sub(conn, task_id=second, platform="telegram", chat_id="-100", thread_id="2")
    finally:
        conn.close()

    adapter = ActiveIndexAdapter()
    asyncio.run(_run_one_notifier_tick(monkeypatch, _make_runner(adapter)))
    indexes = [item for item in adapter.sent if item["text"].startswith("📌")]
    assert len(indexes) == 3
    assert "Только первый топик" in indexes[0]["text"]
    assert "Только второй топик" in indexes[0]["text"]

    conn = kb.connect()
    try:
        assert kb.complete_task(conn, first, summary="готово")
    finally:
        conn.close()
    asyncio.run(_run_one_notifier_tick(monkeypatch, _make_runner(adapter)))
    index_edits = [edit for edit in adapter.edits if "📌" in edit["content"]]
    assert any("Только второй топик" in edit["content"] and "Только первый топик" not in edit["content"] for edit in index_edits)


def test_telegram_adapter_pins_index_message_with_quiet_exact_args():
    from plugins.platforms.telegram.adapter import TelegramAdapter

    adapter = object.__new__(TelegramAdapter)
    adapter._bot = AsyncMock()
    result = asyncio.run(adapter.pin_message("-100123", "777", disable_notification=True))

    assert result.success is True
    assert result.message_id == "777"
    adapter._bot.pin_chat_message.assert_awaited_once_with(
        chat_id=-100123, message_id=777, disable_notification=True,
    )


def test_active_task_index_recovers_once_after_deleted_message(tmp_path, monkeypatch):
    class DeletedIndexAdapter(ActiveIndexAdapter):
        fail_index_edit = False

        async def send(self, chat_id, text, metadata=None):
            await super().send(chat_id, text, metadata)
            return SendResult(success=True, message_id=f"index-{len(self.sent)}")

        async def edit_message(self, chat_id, message_id, content, *, finalize=False):
            if self.fail_index_edit and content.startswith("📌"):
                return SendResult(success=False, error_kind="edit_message_not_found")
            return await super().edit_message(chat_id, message_id, content, finalize=finalize)

    db_path = tmp_path / "active-index-deleted.db"
    monkeypatch.setenv("HERMES_KANBAN_DB", str(db_path))
    kb.init_db()
    conn = kb.connect()
    try:
        task_id = kb.create_task(conn, title="Восстановить индекс", assignee="heavy")
        kb.add_notify_sub(conn, task_id=task_id, platform="telegram", chat_id="-100", thread_id="7")
    finally:
        conn.close()

    adapter = DeletedIndexAdapter()
    asyncio.run(_run_one_notifier_tick(monkeypatch, _make_runner(adapter)))
    adapter.fail_index_edit = True
    conn = kb.connect()
    try:
        assert kb.block_task(conn, task_id, reason="нужен ответ", kind="needs_input")
    finally:
        conn.close()
    asyncio.run(_run_one_notifier_tick(monkeypatch, _make_runner(adapter)))

    baseline = int(time.time())
    monkeypatch.setattr("gateway.kanban_watchers.time.time", lambda: baseline + 31)
    monkeypatch.setattr("hermes_cli.kanban_db.time.time", lambda: baseline + 31)
    adapter.fail_index_edit = False
    asyncio.run(_run_one_notifier_tick(monkeypatch, _make_runner(adapter)))
    indexes = [item for item in adapter.sent if item["text"].startswith("📌")]
    assert len(indexes) == 4
    assert len(adapter.pins) == 4
    with kb.connect_active_task_index_registry() as conn:
        receipt = conn.execute(
            "SELECT message_id FROM kanban_active_task_indexes WHERE platform='telegram' AND chat_id='-100' AND thread_id=''"
        ).fetchone()
        assert receipt["message_id"] == "index-4"
        conn.execute(
            "UPDATE kanban_active_task_indexes SET render_hash='stale' WHERE platform='telegram' AND chat_id='-100' AND thread_id=''"
        )

    asyncio.run(_make_runner(adapter)._refresh_kanban_active_task_index(
        {"platform": "telegram", "chat_id": "-100", "thread_id": "7"}, notifier_profile="default",
    ))
    assert len([item for item in adapter.sent if item["text"].startswith("📌")]) == 4


def test_active_task_index_pin_permission_fallback_is_bounded(tmp_path, monkeypatch):
    class UnpinnedAdapter(ActiveIndexAdapter):
        async def pin_message(self, chat_id, message_id, *, disable_notification=True):
            self.pins.append({"chat_id": chat_id, "message_id": message_id})
            return SendResult(success=False, error="not enough rights to pin")

    db_path = tmp_path / "active-index-unpinned.db"
    monkeypatch.setenv("HERMES_KANBAN_DB", str(db_path))
    kb.init_db()
    conn = kb.connect()
    try:
        task_id = kb.create_task(conn, title="Без права закрепления", assignee="heavy")
        kb.add_notify_sub(conn, task_id=task_id, platform="telegram", chat_id="-100", thread_id="8")
    finally:
        conn.close()

    adapter = UnpinnedAdapter()
    asyncio.run(_run_one_notifier_tick(monkeypatch, _make_runner(adapter)))
    baseline = int(time.time())
    monkeypatch.setattr("gateway.kanban_watchers.time.time", lambda: baseline + 61)
    monkeypatch.setattr("hermes_cli.kanban_db.time.time", lambda: baseline + 61)
    asyncio.run(_run_one_notifier_tick(monkeypatch, _make_runner(adapter)))
    assert len(adapter.pins) == 2
    assert len([item for item in adapter.sent if item["text"].startswith("📌")]) == 2


def test_active_task_index_is_never_created_for_discord_subscription(tmp_path, monkeypatch):
    """The Telegram-only overview must not add a second Discord message or receipt."""
    db_path = tmp_path / "active-index-discord.db"
    monkeypatch.setenv("HERMES_KANBAN_DB", str(db_path))
    kb.init_db()
    with kb.connect() as conn:
        task_id = kb.create_task(conn, title="Discord status only", assignee="heavy")
        kb.add_notify_sub(conn, task_id=task_id, platform="discord", chat_id="discord-channel")

    adapter = ActiveIndexAdapter()
    runner = _make_runner(adapter)
    runner.adapters = {Platform.DISCORD: adapter}  # type: ignore[assignment]
    asyncio.run(_run_one_notifier_tick(monkeypatch, runner))

    assert len(adapter.sent) == 1
    assert not any(item["text"].startswith("📌") for item in adapter.sent)
    assert adapter.edits == []
    assert adapter.pins == []
    with kb.connect() as conn:
        assert conn.execute("SELECT COUNT(*) FROM kanban_active_task_indexes").fetchone()[0] == 0


def test_active_task_overviews_are_one_per_telegram_chat_with_durable_topic_sections(tmp_path, monkeypatch):
    """One registry receipt aggregates same-chat subscriptions across board DBs."""
    from gateway.topic_anchors import record_topic_name

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("HERMES_KANBAN_HOME", str(tmp_path))
    monkeypatch.delenv("HERMES_KANBAN_DB", raising=False)
    task_ids_by_board = {}
    for board, chat_id, profile, topics in (
        ("reviewer-a", "114874376", "reviewer", (("1515141", "Reviewer"), ("1515142", "Research"))),
        ("reviewer-b", "114874376", "reviewer", (("1515143", "Planning"), ("1515144", "Review"))),
        ("workout", "-100222", "workout-logger", (("42", "Planner"), ("43", "Training"))),
    ):
        kb.create_board(board)
        task_ids_by_board[board] = []
        with kb.connect(board=board) as conn:
            for topic_id, topic_name in topics:
                record_topic_name("telegram", chat_id, topic_id, topic_name)
                task_id = kb.create_task(conn, title=f"{topic_name} task", assignee="heavy")
                task_ids_by_board[board].append(task_id)
                kb.add_notify_sub(
                    conn, task_id=task_id, platform="telegram", chat_id=chat_id,
                    thread_id=topic_id, notifier_profile=profile,
                )

    adapter = ActiveIndexAdapter()
    runner = _make_runner(adapter)
    runner._profile_adapters = {
        "reviewer": {Platform.TELEGRAM: adapter},
        "workout-logger": {Platform.TELEGRAM: adapter},
    }
    asyncio.run(_run_one_notifier_tick(monkeypatch, runner))

    indexes = [item for item in adapter.sent if item["text"].startswith("📌")]
    assert len(indexes) == 8
    assert {item["chat_id"] for item in indexes} == {"114874376", "-100222"}
    topic_indexes = [item for item in indexes if item["metadata"].get("thread_id")]
    assert len(topic_indexes) == 6
    assert all(sum(title in item["text"] for title in ("Reviewer task", "Research task", "Planning task", "Review task", "Planner task", "Training task")) == 1 for item in topic_indexes)
    assert len(adapter.pins) == 8

    with kb.connect_active_task_index_registry() as conn:
        rows = conn.execute(
            "SELECT chat_id, thread_id, message_id FROM kanban_active_task_indexes"
        ).fetchall()
        assert len(rows) == 8
        reviewer_message_id = next(row["message_id"] for row in rows if row["chat_id"] == "114874376")

    for board in ("reviewer-a", "reviewer-b"):
        with kb.connect(board=board) as conn:
            for task_id in task_ids_by_board[board]:
                assert kb.complete_task(conn, task_id, summary="готово")
    baseline = int(time.time())
    monkeypatch.setattr("gateway.kanban_watchers.time.time", lambda: baseline + 61)
    monkeypatch.setattr("hermes_cli.kanban_db.time.time", lambda: baseline + 61)
    refresh_runner = _make_runner(adapter)
    refresh_runner._profile_adapters = {
        "reviewer": {Platform.TELEGRAM: adapter},
        "workout-logger": {Platform.TELEGRAM: adapter},
    }
    asyncio.run(_run_one_notifier_tick(monkeypatch, refresh_runner))
    assert len([item for item in adapter.sent if item["text"].startswith("📌")]) == 8
    assert len(adapter.pins) == 8
    reviewer_edits = [edit for edit in adapter.edits if edit["chat_id"] == "114874376"]
    assert reviewer_edits[-1]["message_id"] == reviewer_message_id
    assert reviewer_edits[-1]["content"] == "📌 Active tasks\n\n✅ No active tasks"

    restarted_runner = _make_runner(adapter)
    restarted_runner._profile_adapters = {
        "reviewer": {Platform.TELEGRAM: adapter},
        "workout-logger": {Platform.TELEGRAM: adapter},
    }
    asyncio.run(_run_one_notifier_tick(monkeypatch, restarted_runner))
    assert len([item for item in adapter.sent if item["text"].startswith("📌")]) == 8


def test_active_task_overview_adopts_one_legacy_receipt_before_registry_send(tmp_path, monkeypatch):
    """A pre-upgrade board receipt remains the overview identity after restart."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("HERMES_KANBAN_HOME", str(tmp_path))
    monkeypatch.delenv("HERMES_KANBAN_DB", raising=False)
    kb.create_board("legacy")
    now = int(time.time())
    with kb.connect(board="legacy") as conn:
        task_id = kb.create_task(conn, title="legacy task", assignee="heavy")
        kb.add_notify_sub(
            conn, task_id=task_id, platform="telegram", chat_id="same-chat",
            thread_id="1515141", notifier_profile="reviewer",
        )
        conn.execute(
            """
            INSERT INTO kanban_active_task_indexes
                (platform, chat_id, thread_id, notifier_profile, message_id, pinned,
                 pin_attempted_at, created_at, updated_at)
            VALUES ('telegram', 'same-chat', '', 'reviewer', 'existing-777', 1, ?, ?, ?)
            """,
            (now, now, now),
        )

    adapter = ActiveIndexAdapter()
    runner = _make_runner(adapter)
    runner._profile_adapters = {"reviewer": {Platform.TELEGRAM: adapter}}
    asyncio.run(_run_one_notifier_tick(monkeypatch, runner))

    assert len([item for item in adapter.sent if item["text"].startswith("📌")]) == 1
    overview_edits = [item for item in adapter.edits if item["content"].startswith("📌")]
    assert [item["message_id"] for item in overview_edits] == ["existing-777"]
    assert len(adapter.pins) == 1
    with kb.connect_active_task_index_registry() as conn:
        receipt = conn.execute(
            "SELECT message_id, pinned FROM kanban_active_task_indexes "
            "WHERE platform='telegram' AND chat_id='same-chat' AND thread_id='' AND notifier_profile='reviewer'"
        ).fetchone()
        assert dict(receipt) == {"message_id": "existing-777", "pinned": 1}


def test_active_task_overview_adopts_empty_legacy_board_receipt_for_active_sibling(tmp_path, monkeypatch):
    """An empty old board still owns the chat-wide receipt for an active sibling board."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("HERMES_KANBAN_HOME", str(tmp_path))
    monkeypatch.delenv("HERMES_KANBAN_DB", raising=False)
    now = int(time.time())
    kb.create_board("legacy-empty")
    with kb.connect(board="legacy-empty") as conn:
        task_id = kb.create_task(conn, title="completed legacy task", assignee="heavy")
        kb.add_notify_sub(
            conn, task_id=task_id, platform="telegram", chat_id="same-chat",
            thread_id="1515141", notifier_profile="reviewer",
        )
        assert kb.complete_task(conn, task_id, summary="готово")
        conn.execute(
            """
            INSERT INTO kanban_active_task_indexes
                (platform, chat_id, thread_id, notifier_profile, message_id, pinned,
                 pin_attempted_at, created_at, updated_at)
            VALUES ('telegram', 'same-chat', '', 'reviewer', 'existing-777', 1, ?, ?, ?)
            """,
            (now, now, now),
        )
    kb.create_board("active-sibling")
    with kb.connect(board="active-sibling") as conn:
        task_id = kb.create_task(conn, title="active sibling task", assignee="heavy")
        kb.add_notify_sub(
            conn, task_id=task_id, platform="telegram", chat_id="same-chat",
            thread_id="1515142", notifier_profile="reviewer",
        )

    adapter = ActiveIndexAdapter()
    runner = _make_runner(adapter)
    runner._profile_adapters = {"reviewer": {Platform.TELEGRAM: adapter}}
    asyncio.run(_run_one_notifier_tick(monkeypatch, runner))

    overview_sends = [item for item in adapter.sent if item["text"].startswith("📌")]
    overview_edits = [item for item in adapter.edits if item["content"].startswith("📌")]
    assert len(overview_sends) == 2
    assert len(adapter.pins) == 2
    assert [item["message_id"] for item in overview_edits] == ["existing-777"]
    with kb.connect_active_task_index_registry() as conn:
        receipt = conn.execute(
            "SELECT message_id, pinned FROM kanban_active_task_indexes "
            "WHERE platform='telegram' AND chat_id='same-chat' AND thread_id='' AND notifier_profile='reviewer'"
        ).fetchone()
        assert dict(receipt) == {"message_id": "existing-777", "pinned": 1}


def test_active_task_overview_adopts_conflicting_legacy_receipts_without_replacement_send(tmp_path, monkeypatch):
    """Conflicting old receipts deterministically edit an existing message only."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("HERMES_KANBAN_HOME", str(tmp_path))
    monkeypatch.delenv("HERMES_KANBAN_DB", raising=False)
    now = int(time.time())
    for board, topic_id, message_id in (
        ("legacy-a", "1515141", "existing-777"),
        ("legacy-b", "1515142", "existing-888"),
    ):
        kb.create_board(board)
        with kb.connect(board=board) as conn:
            task_id = kb.create_task(conn, title=f"{board} task", assignee="heavy")
            kb.add_notify_sub(
                conn, task_id=task_id, platform="telegram", chat_id="same-chat",
                thread_id=topic_id, notifier_profile="reviewer",
            )
            conn.execute(
                """
                INSERT INTO kanban_active_task_indexes
                    (platform, chat_id, thread_id, notifier_profile, message_id, pinned,
                     pin_attempted_at, created_at, updated_at)
                VALUES ('telegram', 'same-chat', '', 'reviewer', ?, 1, ?, ?, ?)
                """,
                (message_id, now, now, now),
            )

    adapter = ActiveIndexAdapter()
    runner = _make_runner(adapter)
    runner._profile_adapters = {"reviewer": {Platform.TELEGRAM: adapter}}
    asyncio.run(_run_one_notifier_tick(monkeypatch, runner))

    overview_sends = [item for item in adapter.sent if item["text"].startswith("📌")]
    overview_edits = [item for item in adapter.edits if item["content"].startswith("📌")]
    assert len(overview_sends) == 2
    assert len(adapter.pins) == 2
    assert [item["message_id"] for item in overview_edits] == ["existing-777"]
    with kb.connect_active_task_index_registry() as conn:
        receipt = conn.execute(
            "SELECT message_id, pinned FROM kanban_active_task_indexes "
            "WHERE platform='telegram' AND chat_id='same-chat' AND thread_id='' AND notifier_profile='reviewer'"
        ).fetchone()
        assert dict(receipt) == {"message_id": "existing-777", "pinned": 1}
