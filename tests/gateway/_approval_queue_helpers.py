"""Shared helpers for the approval request_id threading regression tests (#124974).

Every test here drives the REAL ``tools.approval`` queue and the REAL
``resolve_gateway_approval`` (never a mock): two pending entries in one
``session_key``, the card carrying the NEWEST entry's ``approval_request_id``,
and the tap must resolve THAT entry — not the FIFO-oldest one the two-argument
call would have picked. The no-id variant must keep FIFO exactly as before.
"""

from tools.approval import _gateway_queues, _lock
from tools.approval_gateway_wait import _ApprovalEntry


def enqueue_approvals(session_key, *payloads):
    """Queue one real :class:`_ApprovalEntry` per payload dict (FIFO order), under the lock.

    Returns the entries in queue order (oldest first). Each entry's
    ``data["request_id"]`` is auto-assigned by ``_ApprovalEntry.__init__`` —
    the id the card metadata forwards for THAT card.
    """
    entries = [_ApprovalEntry(dict(p)) for p in payloads]
    with _lock:
        _gateway_queues[session_key] = entries
    return entries


def clear_approvals(session_key):
    """Remove the session's queue (idempotent) so tests never leak state."""
    with _lock:
        _gateway_queues.pop(session_key, None)


def assert_resolved(entry, choice):
    """The waiting thread was unblocked and the choice committed to ``entry``."""
    assert entry.event.is_set(), "waiting thread was not unblocked"
    assert entry.result == choice, f"resolved with {entry.result!r}, expected {choice!r}"


def assert_still_pending(entry):
    """The entry must be untouched: still waiting, no choice committed."""
    assert not entry.event.is_set(), "entry was resolved but should still be pending"
    assert entry.result is None, f"entry got a committed choice {entry.result!r} but should be pending"
