import asyncio
from pathlib import Path


from gateway.config import Platform
from gateway.platforms.base import SendResult
from gateway.run import GatewayRunner
from hermes_cli import kanban_db as kb

_REAL_ASYNCIO_SLEEP = asyncio.sleep


class RecordingAdapter:
    def __init__(self):
        self.sent = []

    async def send(self, chat_id, text, metadata=None):
        self.sent.append({"chat_id": chat_id, "text": text, "metadata": metadata or {}})
        return SendResult(success=True, message_id="m-1")


class DisconnectedAdapters(dict):
    """Expose a platform during collection, then simulate disconnect on get()."""

    def get(self, key, default=None):
        return None


async def _run_one_notifier_tick(monkeypatch, runner):
    async def fake_sleep(delay):
        if delay == 5:
            return None
        runner._running = False
        await _REAL_ASYNCIO_SLEEP(0)

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


def test_kanban_notifier_leaves_events_unstaged_until_an_adapter_reconnects(
    tmp_path, monkeypatch,
):
    db_path = tmp_path / "adapter-disconnect.db"
    monkeypatch.setenv("HERMES_KANBAN_DB", str(db_path))
    kb.init_db()
    tid = _create_completed_subscription()

    runner = GatewayRunner.__new__(GatewayRunner)
    runner._running = True
    runner.adapters = DisconnectedAdapters({Platform.TELEGRAM: RecordingAdapter()})
    runner._kanban_sub_fail_counts = {}

    asyncio.run(_run_one_notifier_tick(monkeypatch, runner))

    conn = kb.connect()
    try:
        sub = kb.list_notify_subs(conn, tid)[0]
        row = conn.execute(
            "SELECT 1 FROM kanban_notification_outbox "
            "WHERE subscription_id = ?",
            (sub["subscription_id"],),
        ).fetchone()
        assert int(sub["last_event_id"]) == 0
        assert row is None
    finally:
        conn.close()

    reconnected_adapter = RecordingAdapter()
    runner.adapters = {Platform.TELEGRAM: reconnected_adapter}
    runner._running = True
    asyncio.run(_run_one_notifier_tick(monkeypatch, runner))

    assert len(reconnected_adapter.sent) == 1


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


def test_kanban_notifier_persists_retry_on_send_exception(tmp_path, monkeypatch):
    """A raising adapter leaves a pending outbox item for the next tick.

    Unlike the adapter-disconnect path, the send call actually starts, so this
    consumes one durable attempt before returning the item to pending.
    """
    db_path = tmp_path / "send-failure.db"
    monkeypatch.setenv("HERMES_KANBAN_DB", str(db_path))
    kb.init_db()
    tid = _create_completed_subscription()

    adapter = FailingAdapter()
    runner = _make_runner(adapter)

    asyncio.run(_run_one_notifier_tick(monkeypatch, runner))

    # The cursor stays advanced because the outbox, not a cursor rewind, owns
    # the durable retry obligation.
    assert adapter.attempts >= 1, "send should have been attempted at least once"
    conn = kb.connect()
    try:
        sub = kb.list_notify_subs(conn, tid)[0]
        row = conn.execute(
            "SELECT state, attempts FROM kanban_notification_outbox "
            "WHERE subscription_id = ?",
            (sub["subscription_id"],),
        ).fetchone()
        assert int(sub["last_event_id"]) > 0
        assert row["state"] == "pending"
        assert row["attempts"] == 1
    finally:
        conn.close()


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


def test_notifier_owning_profile_adapter_never_falls_back_to_default(
    tmp_path, monkeypatch,
):
    """An unavailable owning route remains unstaged until it reconnects."""
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
    runner._kanban_notifier_profile = "default"
    # Default profile has a telegram adapter …
    runner.adapters = {Platform.TELEGRAM: default_adapter}
    # Profile "beta" has a different adapter but no Telegram route. Its
    # Telegram subscription must not borrow the default profile's bot.
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
    conn = kb.connect()
    try:
        sub = kb.list_notify_subs(conn, tid)[0]
        row = conn.execute(
            "SELECT 1 FROM kanban_notification_outbox "
            "WHERE subscription_id = ?",
            (sub["subscription_id"],),
        ).fetchone()
        assert int(sub["last_event_id"]) == 0
        assert row is None
    finally:
        conn.close()

    beta_telegram_adapter = RecordingAdapter()
    runner._profile_adapters["beta"][Platform.TELEGRAM] = beta_telegram_adapter
    runner._running = True
    asyncio.run(_run_one_notifier_tick(monkeypatch, runner))

    assert default_adapter.sent == []
    assert len(beta_telegram_adapter.sent) == 1


def test_notifier_delivers_through_a_secondary_profile_only_adapter(tmp_path, monkeypatch):
    db_path = tmp_path / "secondary-only-adapter.db"
    monkeypatch.setenv("HERMES_KANBAN_DB", str(db_path))
    kb.init_db()

    conn = kb.connect()
    try:
        task_id = kb.create_task(conn, title="owned by beta", assignee="worker")
        kb.add_notify_sub(
            conn,
            task_id=task_id,
            platform="discord",
            chat_id="chat-beta",
            notifier_profile="beta",
        )
        kb.complete_task(conn, task_id, summary="done")
    finally:
        conn.close()

    default_adapter = RecordingAdapter()
    beta_adapter = RecordingAdapter()
    runner = GatewayRunner.__new__(GatewayRunner)
    runner._running = True
    runner._kanban_notifier_profile = "default"
    runner.adapters = {Platform.TELEGRAM: default_adapter}
    runner._profile_adapters = {"beta": {Platform.DISCORD: beta_adapter}}

    asyncio.run(_run_one_notifier_tick(monkeypatch, runner))

    assert default_adapter.sent == []
    assert len(beta_adapter.sent) == 1
    assert beta_adapter.sent[0]["chat_id"] == "chat-beta"


def test_notifier_routes_secondary_default_profile_when_active_profile_is_named(
    tmp_path, monkeypatch,
):
    db_path = tmp_path / "secondary-default-adapter.db"
    monkeypatch.setenv("HERMES_KANBAN_DB", str(db_path))
    kb.init_db()

    conn = kb.connect()
    try:
        task_id = kb.create_task(conn, title="owned by default", assignee="worker")
        kb.add_notify_sub(
            conn,
            task_id=task_id,
            platform="telegram",
            chat_id="chat-default",
            notifier_profile="default",
        )
        kb.complete_task(conn, task_id, summary="done")
    finally:
        conn.close()

    active_beta_adapter = RecordingAdapter()
    secondary_default_adapter = RecordingAdapter()
    runner = GatewayRunner.__new__(GatewayRunner)
    runner._running = True
    runner._kanban_notifier_profile = "beta"
    runner.adapters = {Platform.DISCORD: active_beta_adapter}
    runner._profile_adapters = {
        "default": {Platform.TELEGRAM: secondary_default_adapter},
    }

    asyncio.run(_run_one_notifier_tick(monkeypatch, runner))

    assert active_beta_adapter.sent == []
    assert len(secondary_default_adapter.sent) == 1
    assert secondary_default_adapter.sent[0]["chat_id"] == "chat-default"


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
