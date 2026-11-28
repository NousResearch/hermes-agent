"""A wake the session authority refuses for good must not wedge its subscription.

``WakeNotAccepted`` normally means "retry later" (startup, full queue, drain), so the notifier
rewinds the claim and tries again without charging the chat. A refusal that is proven permanent,
or whose reason is ``not_found`` / ``storage_unavailable``, can never succeed: rewinding it
forever left the subscription claiming the same events every tick. It now counts toward the
consecutive-failure limit like any other wake failure, and the subscription is dropped past it.
"""

import asyncio

import pytest

from gateway import wake
from gateway.config import Platform
from gateway.kanban_watchers_notifier import MAX_SEND_FAILURES
from gateway.run import GatewayRunner
from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_notify as kbn

CHAT = "chat-dm"


class RecordingAdapter:
    def __init__(self):
        self.sent = []

    async def send(self, chat_id, text, metadata=None):
        self.sent.append(text)

    async def handle_message(self, event):
        event._gateway_accepted = True


@pytest.fixture
def subscribed_task(tmp_path, monkeypatch):
    """A completed task whose requester subscribed with ``notify+wake``."""
    monkeypatch.setenv("HERMES_KANBAN_DB", str(tmp_path / "wake-refusal.db"))
    kb.init_db()
    conn = kbc.connect()
    try:
        tid = kb.create_task(conn, title="requester", assignee="worker", session_id="origin-session")
        kbn.add_notify_sub(
            conn, task_id=tid, platform="telegram", chat_id=CHAT, chat_type="dm",
            delivery_mode="notify+wake",
        )
        kb.complete_task(conn, tid, summary="done")
        return tid
    finally:
        conn.close()


def _refuse_wakes(monkeypatch, *, permanent=False, reason=None):
    async def refuse(*_args, **_kwargs):
        exc = wake.WakeNotAccepted(reason=reason)
        exc.permanent = permanent
        raise exc

    monkeypatch.setattr(wake, "deliver_wake", refuse)


def _make_runner():
    runner = GatewayRunner.__new__(GatewayRunner)
    runner._running = True
    runner.adapters = {Platform.TELEGRAM: RecordingAdapter()}
    runner._kanban_sub_fail_counts = {}
    runner._kanban_dispatcher_lock_handle = object()
    return runner


def _tick(monkeypatch, runner):
    """Run exactly one notifier pass."""
    real_sleep = asyncio.sleep

    async def fake_sleep(delay):
        if delay == 5:
            return
        runner._running = False
        await real_sleep(0)

    runner._running = True
    monkeypatch.setattr(asyncio, "sleep", fake_sleep)
    asyncio.run(runner._kanban_notifier_watcher(interval=1))


def _subscriptions(tid):
    conn = kbc.connect()
    try:
        return kbn.list_notify_subs(conn, tid)
    finally:
        conn.close()


def _unseen_terminal_events(tid):
    conn = kbc.connect()
    try:
        _, events = kbn.unseen_events_for_sub(
            conn, task_id=tid, platform="telegram", chat_id=CHAT,
            kinds=["completed", "blocked", "gave_up", "crashed", "timed_out"],
        )
        return events
    finally:
        conn.close()


@pytest.mark.parametrize(
    "refusal",
    [{"permanent": True}, {"reason": "not_found"}, {"reason": "storage_unavailable"}],
    ids=["permanent", "not_found", "storage_unavailable"],
)
def test_final_refusal_counts_toward_dropping_the_subscription(subscribed_task, monkeypatch, refusal):
    _refuse_wakes(monkeypatch, **refusal)
    runner = _make_runner()

    _tick(monkeypatch, runner)

    assert list(runner._kanban_sub_fail_counts.values()) == [1]
    assert len(_subscriptions(subscribed_task)) == 1
    # The claim is rewound, so the next tick retries and counts again.
    assert _unseen_terminal_events(subscribed_task)


def test_transient_refusal_is_retried_without_being_charged(subscribed_task, monkeypatch):
    _refuse_wakes(monkeypatch, reason="runtime_draining")
    runner = _make_runner()

    for _ in range(MAX_SEND_FAILURES + 2):
        _tick(monkeypatch, runner)

    assert runner._kanban_sub_fail_counts == {}
    assert len(_subscriptions(subscribed_task)) == 1
    assert _unseen_terminal_events(subscribed_task)


def test_permanently_refused_subscription_is_dropped_at_the_failure_limit(subscribed_task, monkeypatch):
    _refuse_wakes(monkeypatch, permanent=True)
    runner = _make_runner()

    for _ in range(MAX_SEND_FAILURES - 1):
        _tick(monkeypatch, runner)
    assert len(_subscriptions(subscribed_task)) == 1

    _tick(monkeypatch, runner)

    assert _subscriptions(subscribed_task) == []
    assert runner._kanban_sub_fail_counts == {}
