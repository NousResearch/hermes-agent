"""After ``session.replay_gap``: re-attach, settle once, never splice the reply across the hole.

On overflow the owner retires a viewer's subscription and sends one ``session.replay_gap``; the turns run
on. The viewer re-attaches with ``session.resume``. Its snapshot is taken under the event-stream lock, so
an admission it still lists publishes its completion on the new subscription, while one it no longer
lists ended in the hole and only its receipt holds the reply. Live text stops at the gap.
"""
from __future__ import annotations

RESUMED = "[output resumed]"


def pending_status(pending, admission):
    """``queued``/``started`` (its completion follows live), or None (it ended in the hole)."""
    return next((row.get("status") for row in pending or () if row.get("admission_id") == admission), None)


def settled(receipt):
    """True: terminal with its reply (cancelled/failed need none); False: still running; None: unreadable."""
    if receipt.get("status") != "terminal":
        return False
    return True if receipt.get("outcome") in {"cancelled", "interrupted", "failed"} or "result" in receipt else None


def owed_reply(shown, final):
    """What is owed of ``final`` after ``shown`` of its segment was painted: the rest of an exact prefix
    (ignoring the owner's leading ``"\\n\\n"`` and trailing whitespace), else a marker and the whole."""
    head = shown.removeprefix("\n\n")
    for prefix in (head, head.rstrip()):
        if final.startswith(prefix):
            return final[len(prefix):]
    return f"\n{RESUMED}\n{final}"
