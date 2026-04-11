"""
Tests for ``gateway.media_queue`` — the per-session pending-media queue
used by tools like ``browser_screenshot`` for direct-enqueue delivery.

The queue lives in the gateway process and is drained by
``gateway.platforms.base`` after each agent response, so its invariants
matter more than its surface area:

1.  enqueue + drain (happy path, queue cleared after drain)
2.  multiple enqueues in one session preserve FIFO order, drain is atomic
3.  cross-session isolation — draining one session must not touch another
4.  ContextVar resolution — ``enqueue_media()`` with no explicit
    ``session_key`` picks up the active session set via
    ``tools.approval.set_current_session_key``, and state does not leak
    across context scopes
5.  draining an unknown session is a no-op, not an error
"""

import pytest

from gateway.media_queue import (
    _pending,
    drain_media,
    enqueue_media,
    peek_media,
)
from tools.approval import (
    reset_current_session_key,
    set_current_session_key,
)


@pytest.fixture(autouse=True)
def _reset_queue_between_tests():
    """Reset the module-level queue dict so tests don't bleed into each other."""
    _pending.clear()
    yield
    _pending.clear()


def test_enqueue_and_drain_single_item():
    """A single enqueued path comes back from drain, and the queue is empty after."""
    enqueue_media("/tmp/a.png", session_key="session-A")

    drained = drain_media("session-A")

    assert drained == ["/tmp/a.png"]
    # Drain is atomic — a second drain returns nothing.
    assert drain_media("session-A") == []


def test_multiple_enqueue_preserves_fifo_order_and_atomic_drain():
    """Multiple enqueues for the same session come back in insertion order, all at once."""
    enqueue_media("/tmp/a.png", session_key="session-A")
    enqueue_media("/tmp/b.png", session_key="session-A")
    enqueue_media("/tmp/c.png", session_key="session-A")

    drained = drain_media("session-A")

    assert drained == ["/tmp/a.png", "/tmp/b.png", "/tmp/c.png"]
    # Single drain takes everything; nothing left behind for a duplicate send.
    assert drain_media("session-A") == []


def test_session_isolation_across_drains():
    """Draining one session must not touch a different session's queue."""
    enqueue_media("/tmp/a.png", session_key="session-A")
    enqueue_media("/tmp/b.png", session_key="session-B")
    enqueue_media("/tmp/a2.png", session_key="session-A")

    # Drain A — only A's items come out, B is left alone.
    assert drain_media("session-A") == ["/tmp/a.png", "/tmp/a2.png"]
    assert peek_media("session-B") == ["/tmp/b.png"]

    # B can still be drained independently.
    assert drain_media("session-B") == ["/tmp/b.png"]
    assert drain_media("session-B") == []


def test_contextvar_resolution_does_not_leak_across_sessions():
    """``enqueue_media()`` without an explicit ``session_key`` resolves via the
    ``tools.approval`` ContextVar, and state from one session does not bleed
    into another. This is the production call path used by the browser
    screenshot tool — tools call ``enqueue_media(path)`` with no arguments.
    """
    token_a = set_current_session_key("session-A")
    try:
        enqueue_media("/tmp/a.png")  # no explicit session_key
    finally:
        reset_current_session_key(token_a)

    token_b = set_current_session_key("session-B")
    try:
        enqueue_media("/tmp/b.png")  # no explicit session_key
    finally:
        reset_current_session_key(token_b)

    # Each path landed in its own session's queue — no cross-contamination.
    assert drain_media("session-A") == ["/tmp/a.png"]
    assert drain_media("session-B") == ["/tmp/b.png"]


def test_drain_unknown_session_returns_empty_list():
    """Draining a session that was never touched is a no-op, not an error."""
    assert drain_media("never-touched") == []
    assert peek_media("never-touched") == []
