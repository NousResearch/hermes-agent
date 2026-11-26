"""A leftover background notice must not strand on the session FIFO.

A notice the turn could not deliver at a tool boundary is handed back by the finalizer
(``result["pending_notifications"]``) and re-queued at the post-turn drain. The drain returns
``(pending_event, pending)`` and the follow-up recursion only runs when one of them is set:
FIFO-appending the notice without touching the return value strands it there until the user's
next message — the notification arrives minutes (or never) after the process finished. With
nothing else pending the notice IS the next turn (the idle-session completion behavior); with
queued work pending it appends behind it.
"""

from types import SimpleNamespace

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.event import MessageEvent
from gateway.run_notifications import INTERNAL_NOTIFICATION_FOOTER
from gateway.session import SessionSource

NOTICE = "[IMPORTANT: Background process proc_1 exited (exit code 0).]"


def _source() -> SessionSource:
    return SessionSource(platform=Platform.TELEGRAM, user_id="u1", chat_id="c1",
                         user_name="tester", chat_type="dm")


def _runner(fifo):
    """A busy mixin bound to nothing but the FIFO recorder: the requeue only needs
    ``_queue_or_replace_pending_event``."""
    from gateway.run_busy import GatewayBusySessionMixin

    runner = SimpleNamespace()

    def _queue(session_key, event):
        fifo.append(event)

    runner._queue_or_replace_pending_event = _queue
    return GatewayBusySessionMixin._requeue_leftover_notifications.__get__(runner)


def _result(*notices):
    return {"pending_notifications": list(notices)}


def test_nothing_pending_notice_becomes_the_next_turn():
    fifo = []
    requeue = _runner(fifo)

    event, pending = requeue(_result(NOTICE), None, _source(), "agent:main:telegram:dm:c1")

    assert fifo == [], "with an empty FIFO the notice must run directly, not strand queued"
    assert event is not None and pending == event.text
    assert isinstance(event, MessageEvent)
    assert event.internal is True
    assert NOTICE in event.text
    assert event.text.endswith(INTERNAL_NOTIFICATION_FOOTER), "the turn path needs the provenance footer"


def test_nothing_pending_two_notices_first_runs_rest_queue():
    fifo = []
    requeue = _runner(fifo)
    second = "[IMPORTANT: Background process proc_2 exited (exit code 7).]"

    event, _pending = requeue(_result(NOTICE, second), None, _source(), "agent:main:telegram:dm:c1")

    assert event is not None and NOTICE in event.text
    assert [e.text for e in fifo] == [
        _mark(second)
    ], "the second notice keeps its FIFO turn behind the first"


def _mark(text: str) -> str:
    from gateway.run_notifications import _mark_internal_notification

    return _mark_internal_notification(text)


def test_queued_followup_pending_notices_append_behind_it():
    fifo = []
    requeue = _runner(fifo)
    queued = MessageEvent(text="user follow-up", source=_source())

    event, pending = requeue(_result(NOTICE), None, _source(), "agent:main:telegram:dm:c1", queued, None)

    assert (event, pending) == (queued, None), \
        "the user's queued follow-up must stay the next turn"
    assert [e.text for e in fifo] == [_mark(NOTICE)], "the notice queues behind the user's turn"
    assert fifo[0].internal is True


def test_pending_text_only_notices_append_behind_it():
    fifo = []
    requeue = _runner(fifo)

    event, pending = requeue(_result(NOTICE), None, _source(), "agent:main:telegram:dm:c1", None, "voice reply")

    assert event is None and pending == "voice reply"
    assert [e.text for e in fifo] == [_mark(NOTICE)]


def test_no_leftovers_is_a_noop():
    fifo = []
    requeue = _runner(fifo)
    queued = MessageEvent(text="user follow-up", source=_source())

    assert requeue({}, None, _source(), "k", None, None) == (None, None)
    assert requeue({"pending_notifications": []}, None, _source(), "k", None, None) == (None, None)
    assert requeue({"pending_notifications": ["", "   "]}, None, _source(), "k", queued, "wait") \
        == (queued, "wait")
    assert fifo == []


def test_notice_at_busy_queue_cap_parks_on_overflow_not_dropped():
    """The 32-item cap drops user follow-ups on purpose; a notice must not be dropped twice."""
    overflow, dropped = [], []
    adapter = SimpleNamespace(_pending_messages={})

    from gateway.run_busy import GatewayBusySessionMixin

    runner = SimpleNamespace()
    runner._BUSY_QUEUE_MAX_PENDING = 32
    runner._delivery_adapter_for = lambda source: adapter
    runner._queue_depth = lambda session_key, adapter=None: 32
    runner._enqueue_fifo = lambda session_key, event, target: overflow.append(event)
    runner._queue_or_replace_pending_event = lambda session_key, event: dropped.append(event)
    requeue = GatewayBusySessionMixin._requeue_leftover_notifications.__get__(runner)

    event, pending = requeue(_result(NOTICE), None, _source(), "agent:main:telegram:dm:c1", None, "queued work")

    assert (event, pending) == (None, "queued work"), "queued work stays the next turn"
    assert dropped == [], "a notice must never take the drop-at-cap path"
    assert [e.text for e in overflow] == [_mark(NOTICE)], "the notice waits on the overflow list"
