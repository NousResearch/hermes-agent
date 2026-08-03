"""Verify that compression-lock timeout errors are not misclassified as
disk-full failures (#77386).

The desktop's ``isDiskFullErrorMessage()`` regex matches any error string
containing "disk full" or "full disk". When a compression lock timeout was
surfaced as ``session_persistence_failed``, the generic message in
``run_agent.py`` contained "full disk" and triggered the misleading toast.
The fix introduces a distinct ``compression_lock_timeout`` exit reason whose
message avoids those phrases entirely.
"""

from __future__ import annotations

import re


# Replicate the desktop's isDiskFullErrorMessage regex (notifications.ts).
_DISK_FULL_PATTERNS = [
    re.compile(p, re.IGNORECASE)
    for p in (
        r"no space left on device",
        r"not enough space",
        r"database or disk is full",
        r"\bENOSPC\b",
        r"disk full",
        r"full disk",
    )
]


def _is_disk_full_message(message: str) -> bool:
    """Mirror of the desktop's isDiskFullErrorMessage()."""
    return any(p.search(message) for p in _DISK_FULL_PATTERNS)


def test_compression_lock_timeout_message_is_not_disk_full() -> None:
    """The compression_lock_timeout user-facing message must NOT match the
    desktop's disk-full regex, otherwise the user sees a misleading
    'Disk full — free some space' toast when the real issue was a
    long-running context compression.
    """
    # This is the message format from run_agent.py's _exit_reason_explanation.
    prefix = "⚠️ No reply: "
    message = (
        prefix
        + "the turn was stopped because a context compression was "
        "still running when this message arrived. The compression "
        "has now finished — send your message again and it will "
        "persist normally."
    )
    assert not _is_disk_full_message(message), (
        f"compression_lock_timeout message matches disk-full regex: {message!r}"
    )


def test_compression_lock_timeout_turn_finalizer_fallback_is_not_disk_full() -> None:
    """The turn_finalizer fallback for compression_lock_timeout must also
    avoid disk-full language so the desktop doesn't toast incorrectly.
    """
    message = (
        "session storage was temporarily locked by a context "
        "compression — the compression has finished; send your "
        "message again"
    )
    assert not _is_disk_full_message(message), (
        f"turn_finalizer fallback matches disk-full regex: {message!r}"
    )


def test_session_persistence_failed_message_still_mentions_disk() -> None:
    """Sanity check: the genuine session_persistence_failed path should
    still mention disk space, since that IS the likely cause for
    non-compression persistence failures.
    """
    message = (
        "⚠️ No reply: the turn was stopped because session storage could "
        "not be written (the transcript would have been lost on restart). "
        "This is often a full disk — free some space (or fix state.db "
        "permissions), then send your message again."
    )
    assert _is_disk_full_message(message), (
        "session_persistence_failed message should match disk-full regex"
    )