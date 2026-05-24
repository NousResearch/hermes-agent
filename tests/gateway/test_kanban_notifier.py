import asyncio
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from gateway.config import Platform
from gateway.run import GatewayRunner
from hermes_cli import kanban_db as kb


class RecordingAdapter:
    def __init__(self):
        self.sent = []
        self.documents = []
        self.images = []
        self.videos = []

    async def send(self, chat_id, text, metadata=None):
        self.sent.append({"chat_id": chat_id, "text": text, "metadata": metadata or {}})

    async def send_document(self, chat_id, file_path, metadata=None):
        self.documents.append({"chat_id": chat_id, "file_path": file_path, "metadata": metadata or {}})

    async def send_multiple_images(self, chat_id, images, metadata=None):
        self.images.append({"chat_id": chat_id, "images": images, "metadata": metadata or {}})

    async def send_video(self, chat_id, video_path, metadata=None):
        self.videos.append({"chat_id": chat_id, "video_path": video_path, "metadata": metadata or {}})


class DisconnectedAdapters(dict):
    """Expose a platform during collection, then simulate disconnect on get()."""

    def get(self, key, default=None):
        return None


async def _run_one_notifier_tick(monkeypatch, runner):
    monkeypatch.setenv("HERMES_KANBAN_NOTIFY_IN_GATEWAY", "1")
    runner._running = True
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


def _create_completed_subscription_with_artifacts(artifacts, summary="done with artifact"):
    conn = kb.connect()
    try:
        tid = kb.create_task(conn, title="notify artifact", assignee="worker")
        kb.add_notify_sub(conn, task_id=tid, platform="telegram", chat_id="chat-1")
        kb.complete_task(
            conn,
            tid,
            summary=summary,
            metadata={
                "artifacts": artifacts,
                "review_required": False,
                "review_skip_reason": "notifier artifact delivery regression test",
            },
        )
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
    assert adapter.sent[0]["text"] == "✅ Done: notify once"
    assert "done once" not in adapter.sent[0]["text"]
    assert tid not in adapter.sent[0]["text"]


def test_kanban_notifier_watcher_respects_env_override(monkeypatch):
    """HERMES_KANBAN_NOTIFY_IN_GATEWAY=0 disables notifier DB polling."""
    monkeypatch.setenv("HERMES_KANBAN_NOTIFY_IN_GATEWAY", "0")
    monkeypatch.setattr(kb, "list_boards", lambda **_: pytest.fail("notifier touched kanban DB"))

    runner = _make_runner(RecordingAdapter())

    asyncio.run(
        asyncio.wait_for(
            runner._kanban_notifier_watcher(interval=1),
            timeout=3.0,
        )
    )


def test_kanban_notifier_watcher_respects_config_flag_off(monkeypatch):
    """kanban.notify_in_gateway=false disables notifier DB polling."""
    monkeypatch.setattr(
        "hermes_cli.config.load_config",
        lambda: {"kanban": {"notify_in_gateway": False}},
    )
    monkeypatch.setattr(kb, "list_boards", lambda **_: pytest.fail("notifier touched kanban DB"))

    runner = _make_runner(RecordingAdapter())

    asyncio.run(
        asyncio.wait_for(
            runner._kanban_notifier_watcher(interval=1),
            timeout=3.0,
        )
    )


def test_kanban_notifier_completion_message_rejects_extra_prose(tmp_path, monkeypatch):
    db_path = tmp_path / "terse-completion-message.db"
    monkeypatch.setenv("HERMES_KANBAN_DB", str(db_path))
    kb.init_db()

    tid = _create_completed_subscription(
        summary="Good info: deployment finished cleanly with no follow-up needed"
    )

    adapter = RecordingAdapter()
    runner = _make_runner(adapter)

    asyncio.run(_run_one_notifier_tick(monkeypatch, runner))

    assert len(adapter.sent) == 1
    message = adapter.sent[0]["text"]
    assert message == "✅ Done: notify once"
    assert message.count("\n") == 0
    assert "Good info" not in message
    assert "deployment finished" not in message
    assert "worker" not in message
    assert tid not in message


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


def test_kanban_notifier_home_channel_reports_completion_without_task_subscription(tmp_path, monkeypatch):
    """Home-channel completion pings must not depend on per-task subscriptions.

    A paused/absent broad fallback cron should not make Kanban completions go
    silent: the gateway watcher itself owns a durable board-level cursor for
    configured home channels.
    """
    db_path = tmp_path / "home-channel-completion.db"
    monkeypatch.setenv("HERMES_KANBAN_DB", str(db_path))
    kb.init_db()

    adapter = RecordingAdapter()
    runner = _make_runner(adapter)
    runner.config = SimpleNamespace(
        platforms={
            Platform.TELEGRAM: SimpleNamespace(enabled=True, token="bot-token", extra={})
        },
        get_home_channel=lambda platform: SimpleNamespace(
            chat_id="home-chat", thread_id=None
        ) if platform == Platform.TELEGRAM else None,
    )

    # First tick initializes the durable board-level cursor without replaying
    # historical task events.
    asyncio.run(_run_one_notifier_tick(monkeypatch, runner))

    conn = kb.connect()
    try:
        tid = kb.create_task(conn, title="unsubscribed completion", assignee="worker")
        kb.complete_task(conn, tid, summary="internal handoff text stays out of the ping")
    finally:
        conn.close()

    asyncio.run(_run_one_notifier_tick(monkeypatch, runner))
    asyncio.run(_run_one_notifier_tick(monkeypatch, runner))

    assert [msg["text"] for msg in adapter.sent] == ["✅ Done: unsubscribed completion"]
    assert adapter.sent[0]["chat_id"] == "home-chat"
    assert tid not in adapter.sent[0]["text"]
    assert "internal handoff" not in adapter.sent[0]["text"]


def test_kanban_notifier_home_channel_skips_completion_with_matching_task_subscription(tmp_path, monkeypatch):
    """Board-level home pings must not duplicate matching per-task subscriptions."""
    db_path = tmp_path / "home-channel-matching-task-sub.db"
    monkeypatch.setenv("HERMES_KANBAN_DB", str(db_path))
    kb.init_db()

    adapter = RecordingAdapter()
    runner = _make_runner(adapter)
    runner.config = SimpleNamespace(
        platforms={
            Platform.TELEGRAM: SimpleNamespace(enabled=True, token="bot-token", extra={})
        },
        get_home_channel=lambda platform: SimpleNamespace(
            chat_id="home-chat", thread_id=None
        ) if platform == Platform.TELEGRAM else None,
    )

    # Initialize the board-level cursor before the task exists.
    asyncio.run(_run_one_notifier_tick(monkeypatch, runner))

    conn = kb.connect()
    try:
        tid = kb.create_task(conn, title="subscribed completion", assignee="worker")
        kb.add_notify_sub(conn, task_id=tid, platform="telegram", chat_id="home-chat")
        kb.complete_task(conn, tid, summary="task-scoped sub owns this ping")
    finally:
        conn.close()

    asyncio.run(_run_one_notifier_tick(monkeypatch, runner))
    asyncio.run(_run_one_notifier_tick(monkeypatch, runner))

    assert [msg["text"] for msg in adapter.sent] == ["✅ Done: subscribed completion"]
    assert adapter.sent[0]["chat_id"] == "home-chat"


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


def test_kanban_notifier_uses_notifications_bot_for_telegram(tmp_path, monkeypatch):
    """Telegram kanban notifications should use the dedicated notifier token when configured."""
    db_path = tmp_path / "notifications-bot.db"
    monkeypatch.setenv("HERMES_KANBAN_DB", str(db_path))
    monkeypatch.setenv("NOTIFICATIONS_TELEGRAM_BOT_TOKEN", "noti-token")
    kb.init_db()

    tid = _create_completed_subscription()
    adapter = RecordingAdapter()
    runner = _make_runner(adapter)
    runner.config = SimpleNamespace(
        platforms={
            Platform.TELEGRAM: SimpleNamespace(
                enabled=True,
                token="ava-token",
                extra={},
            )
        }
    )

    with patch("tools.send_message_tool._send_telegram", new=AsyncMock(return_value={"success": True})) as send_mock:
        asyncio.run(_run_one_notifier_tick(monkeypatch, runner))

    assert adapter.sent == []
    send_mock.assert_called_once()
    assert send_mock.call_args.args[0] == "noti-token"
    assert send_mock.call_args.args[1] == "chat-1"
    assert send_mock.call_args.args[2] == "✅ Done: notify once"
    assert tid not in send_mock.call_args.args[2]


class FailingAdapter:
    """Adapter whose send() always raises, simulating a transient send error."""

    def __init__(self):
        self.attempts = 0

    async def send(self, chat_id, text, metadata=None):
        self.attempts += 1
        raise RuntimeError("simulated send failure")


class ErrorResultAdapter:
    """Adapter that reports a downstream API error without raising."""

    def __init__(self, error="downstream API stack trace: token=secret-token"):
        self.error = error
        self.sent = []

    async def send(self, chat_id, text, metadata=None):
        self.sent.append({"chat_id": chat_id, "text": text, "metadata": metadata or {}})
        return {"error": self.error}


class FailOnceOnSecondSendAdapter:
    """Adapter that accepts the first send, fails the second once, then recovers."""

    def __init__(self):
        self.sent = []
        self._failed_second_send = False

    async def send(self, chat_id, text, metadata=None):
        self.sent.append({"chat_id": chat_id, "text": text, "metadata": metadata or {}})
        if len(self.sent) == 2 and not self._failed_second_send:
            self._failed_second_send = True
            raise RuntimeError("transient failure on second board ping")


def test_kanban_notifier_board_home_partial_failure_does_not_duplicate_prior_success(tmp_path, monkeypatch):
    """A failed later board-level ping must not redeliver already-sent earlier pings."""
    db_path = tmp_path / "board-partial-failure.db"
    monkeypatch.setenv("HERMES_KANBAN_DB", str(db_path))
    kb.init_db()

    adapter = FailOnceOnSecondSendAdapter()
    runner = _make_runner(adapter)
    runner.config = SimpleNamespace(
        platforms={
            Platform.TELEGRAM: SimpleNamespace(enabled=True, token="bot-token", extra={})
        },
        get_home_channel=lambda platform: SimpleNamespace(
            chat_id="home-chat", thread_id=None
        ) if platform == Platform.TELEGRAM else None,
    )

    # Initialize the board-level cursor without replaying historical events.
    asyncio.run(_run_one_notifier_tick(monkeypatch, runner))

    conn = kb.connect()
    try:
        first = kb.create_task(conn, title="one", assignee="worker")
        second = kb.create_task(conn, title="two", assignee="worker")
        kb.complete_task(conn, first, summary="first done")
        kb.complete_task(conn, second, summary="second done")
    finally:
        conn.close()

    asyncio.run(_run_one_notifier_tick(monkeypatch, runner))
    assert [msg["text"] for msg in adapter.sent] == [
        "✅ Done: one",
        "✅ Done: two",
    ]

    asyncio.run(_run_one_notifier_tick(monkeypatch, runner))
    assert [msg["text"] for msg in adapter.sent] == [
        "✅ Done: one",
        "✅ Done: two",
        "✅ Done: two",
    ]


def test_kanban_notifier_artifacts_use_notifications_bot_for_telegram(tmp_path, monkeypatch):
    """Artifact uploads for Telegram kanban notifications should also use the dedicated notifier token."""
    db_path = tmp_path / "notifications-bot-artifacts.db"
    artifact = tmp_path / "handoff.txt"
    artifact.write_text("artifact payload", encoding="utf-8")
    monkeypatch.setenv("HERMES_KANBAN_DB", str(db_path))
    monkeypatch.setenv("NOTIFICATIONS_TELEGRAM_BOT_TOKEN", "notifier-token")
    monkeypatch.setenv("HERMES_MEDIA_ALLOW_DIRS", str(tmp_path))
    kb.init_db()

    _create_completed_subscription_with_artifacts([str(artifact)])
    adapter = RecordingAdapter()
    runner = _make_runner(adapter)
    runner.config = SimpleNamespace(
        platforms={
            Platform.TELEGRAM: SimpleNamespace(
                enabled=True,
                token="interactive-token",
                extra={},
            )
        }
    )

    with patch("tools.send_message_tool._send_telegram", new=AsyncMock(return_value={"success": True})) as send_mock:
        asyncio.run(_run_one_notifier_tick(monkeypatch, runner))

    assert adapter.sent == []
    assert adapter.documents == []
    assert adapter.images == []
    assert adapter.videos == []
    assert send_mock.call_count == 2
    artifact_call = send_mock.call_args_list[-1]
    assert artifact_call.args[0] == "notifier-token"
    assert artifact_call.args[1] == "chat-1"
    assert artifact_call.kwargs["media_files"] == [(str(artifact), False)]


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


def test_kanban_notifier_rewinds_claim_on_adapter_error_result(tmp_path, monkeypatch, caplog):
    """API-style error results are logged internally and retried, not silently advanced."""
    db_path = tmp_path / "send-error-result.db"
    monkeypatch.setenv("HERMES_KANBAN_DB", str(db_path))
    kb.init_db()
    tid = _create_completed_subscription()

    adapter = ErrorResultAdapter(error="telegram 500: raw traceback payload")
    runner = _make_runner(adapter)

    with caplog.at_level("WARNING", logger="gateway.run"):
        asyncio.run(_run_one_notifier_tick(monkeypatch, runner))

    assert [msg["text"] for msg in adapter.sent] == ["✅ Done: notify once"]
    assert "telegram 500" not in adapter.sent[0]["text"]
    assert [ev.kind for ev in _unseen_terminal_events(tid)] == ["completed"]
    assert "kanban notifier: send failed" in caplog.text
    assert "telegram 500" in caplog.text


def test_kanban_notifier_drops_dead_subscription_after_repeated_send_errors(tmp_path, monkeypatch, caplog):
    """Repeated downstream send errors stop retry spam by removing task subscriptions."""
    db_path = tmp_path / "send-error-drop-sub.db"
    monkeypatch.setenv("HERMES_KANBAN_DB", str(db_path))
    kb.init_db()
    tid = _create_completed_subscription()

    adapter = ErrorResultAdapter(error="telegram 403: bot was kicked")
    runner = _make_runner(adapter)

    with caplog.at_level("WARNING", logger="gateway.run"):
        for _ in range(3):
            asyncio.run(_run_one_notifier_tick(monkeypatch, runner))

    assert len(adapter.sent) == 3
    assert all(msg["text"] == "✅ Done: notify once" for msg in adapter.sent)
    assert all("telegram 403" not in msg["text"] for msg in adapter.sent)
    conn = kb.connect()
    try:
        assert kb.list_notify_subs(conn, tid) == []
    finally:
        conn.close()
    assert "dropping subscription" in caplog.text
    assert "telegram 403" in caplog.text


@pytest.mark.parametrize(
    "event_kind,event_payload",
    [
        ("created", {"status": "running"}),
        ("assigned", {"assignee": "new-worker"}),
        ("status", {"from": "running", "to": "ready"}),
        ("commented", {"author": "reviewer", "len": 12}),
        ("blocked", {"reason": "needs review"}),
        ("crashed", None),
        ("timed_out", {"limit_seconds": 60}),
    ],
)
def test_notifier_ignores_non_completion_events(
    tmp_path, monkeypatch, event_kind, event_payload,
):
    """The dedicated notification bot is completion-only.

    Creation, assignment, non-done status movement, comments, blocked/stuck,
    and retry-loop worker failures should not send noti-bot messages.
    """
    db_path = tmp_path / f"non-completion-{event_kind}.db"
    monkeypatch.setenv("HERMES_KANBAN_DB", str(db_path))
    kb.init_db()

    conn = kb.connect()
    try:
        tid = kb.create_task(conn, title=f"{event_kind} test", assignee="worker")
        kb.add_notify_sub(conn, task_id=tid, platform="telegram", chat_id="chat-1")
        kb._append_event(conn, tid, kind=event_kind, payload=event_payload)
    finally:
        conn.close()

    adapter = RecordingAdapter()
    runner = _make_runner(adapter)
    asyncio.run(_run_one_notifier_tick(monkeypatch, runner))

    assert adapter.sent == []

    conn = kb.connect()
    try:
        assert len(kb.list_notify_subs(conn, tid)) == 1
    finally:
        conn.close()


def test_notifier_ignores_non_completion_events_before_later_completion(
    tmp_path, monkeypatch,
):
    """Earlier non-completion events must not block a later completion ping."""
    db_path = tmp_path / "non-completion-before-completion.db"
    monkeypatch.setenv("HERMES_KANBAN_DB", str(db_path))
    kb.init_db()

    conn = kb.connect()
    try:
        tid = kb.create_task(conn, title="eventual completion", assignee="worker")
        kb.add_notify_sub(conn, task_id=tid, platform="telegram", chat_id="chat-1")
        kb.add_comment(conn, tid, "reviewer", "please revise")
        kb._append_event(conn, tid, kind="status", payload={"from": "running", "to": "ready"})
        kb.complete_task(conn, tid, summary="finished after intermediate noise")
    finally:
        conn.close()

    adapter = RecordingAdapter()
    runner = _make_runner(adapter)
    asyncio.run(_run_one_notifier_tick(monkeypatch, runner))

    assert len(adapter.sent) == 1
    assert adapter.sent[0]["text"] == "✅ Done: eventual completion"
    assert "finished after intermediate noise" not in adapter.sent[0]["text"]
