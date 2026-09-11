"""Persistent INBOX resume point for the email adapter.

``EmailAdapter._seen_uids`` lives only in memory, so every start re-baselines
against whatever happens to be in the mailbox at that moment: ``connect()``
marks every existing UID as already seen, and mail that arrived while the
gateway was down is swallowed into that skip set and never dispatched (#80925).

What this module persists is one integer per mailbox — the highest UID the
adapter has committed to — plus the UIDVALIDITY generation it was measured in.
IMAP UIDs are strictly ascending within a generation (RFC 3501 2.3.1.1), so
``uid > cursor`` is a complete test for "not handled yet".  A persisted *set* of
UIDs would carry the same information plus a trimming policy, and trimming a set
is exactly what makes old mail replay (#60637).  A cursor cannot forget.

The cursor records that the adapter took responsibility for a UID, not that a
reply was delivered: ``BasePlatformAdapter.handle_message`` is documented
fire-and-forget, so there is no delivery signal to wait for.  That is precisely
what the in-memory skip set already means, which is what keeps the opt-in path
from ever being worse than the default one.

Reads and writes are best-effort, in the style of ``gateway/dead_targets.py``:
a corrupt or unwritable file degrades to establishing a fresh baseline from the
mailbox, logs at debug, and never raises on the polling path.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Optional

from hermes_constants import get_hermes_home

logger = logging.getLogger(__name__)


def cursor_path(address: str) -> Path:
    """Profile-scoped, per-mailbox path of the cursor file.

    ``get_hermes_home()`` is already per-profile.  The address goes into the
    filename as well because profile multiplexing (#50094) can run two email
    adapters whose homes resolve to the same directory; a single shared file
    would make each mailbox invalidate the other's cursor on every connect and
    silently re-baseline, which is the failure this module exists to prevent.
    """
    safe = re.sub(r"[^a-z0-9._-]", "_", (address or "").strip().lower())
    return get_hermes_home() / "gateway" / f"email_uid_cursor_{safe or 'default'}.json"


class EmailUidCursor:
    """Highest committed INBOX UID for one mailbox, persisted as small JSON."""

    def __init__(self, address: str, path: Optional[Path] = None) -> None:
        self._address = (address or "").strip().lower()
        self._path = path or cursor_path(self._address)
        self.uidvalidity: str = ""
        self.uid: int = 0
        self._written_uid: Optional[int] = None
        self._load()

    def _load(self) -> None:
        """Read the stored cursor.  Anything unreadable means "no cursor"."""
        try:
            raw = json.loads(self._path.read_text(encoding="utf-8"))
            self.uidvalidity = str(raw["uidvalidity"])
            self.uid = int(raw["uid"])
            self._written_uid = self.uid
        except (OSError, ValueError, TypeError, KeyError) as exc:
            logger.debug("[Email] No usable UID cursor at %s (%s)", self._path, exc)
            self.uidvalidity = ""
            self.uid = 0
            self._written_uid = None

    def resume_from(self, uidvalidity: str) -> Optional[int]:
        """Return the UID to resume after, or None when a baseline is needed.

        A cursor is only meaningful inside the UIDVALIDITY generation it was
        measured in.  A different (or unreadable) generation returns None so the
        caller re-baselines instead of comparing UIDs across namespaces.

        A stored 0 is a real resume point — the mailbox was empty when the
        baseline was taken — not "no cursor": treating it as absent would
        re-baseline past mail that arrived during the outage.
        """
        if uidvalidity and self.uidvalidity == uidvalidity:
            return self.uid
        return None

    def baseline(self, uidvalidity: str, uid: int) -> None:
        """Adopt a fresh starting point and persist it immediately."""
        self.uidvalidity = uidvalidity
        self.uid = int(uid)
        self._written_uid = None
        self.flush()

    def advance(self, uid: int) -> None:
        """Record *uid* as committed.  Monotonic; older UIDs are ignored."""
        if uid > self.uid:
            self.uid = uid

    def flush(self) -> None:
        """Persist the cursor if it moved.  Never raises on the polling path."""
        if self.uid == self._written_uid:
            return
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self._path.with_suffix(self._path.suffix + ".tmp")
            tmp.write_text(
                json.dumps(
                    {
                        "address": self._address,
                        "uidvalidity": self.uidvalidity,
                        "uid": self.uid,
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
            tmp.replace(self._path)
            self._written_uid = self.uid
        except OSError as exc:
            logger.debug(
                "[Email] Could not persist UID cursor to %s (%s)", self._path, exc
            )
