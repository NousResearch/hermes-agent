"""Interactive allow-list: the dynamic, owner-approved sender list.

Deny-by-default access control needs a way to grow the allow-list at
runtime without restarting the gateway. Rather than invent our own store,
we reuse Hermes core's :class:`PairingStore` — the same persisted,
per-platform approved-user file the core authorization check already
consults for *every* platform (``gateway/run.py``:
``pairing_store.is_approved(platform, user_id)`` is "always checked,
regardless of allowlists"). So approving a user here authorizes them in
Hermes core automatically, with no core changes.

This module is import-safe without Hermes core (CI imports the plugin
standalone): the ``PairingStore`` import is lazy and every method degrades
to a safe default when core isn't present.

Two pieces live here:

  - :class:`ApprovalStore` — thin wrapper over ``PairingStore`` scoped to
    the ``carbonvoice`` platform (is_approved / approve / revoke / list).
  - :func:`parse_admin_command` — parses the operator's ``/cv-allow`` /
    ``/cv-deny`` / ``/cv-list`` replies in the home channel.
"""

from __future__ import annotations

import logging
import re
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)

# Platform key under which approved users are stored
# (``~/.hermes/platforms/pairing/carbonvoice-approved.json``). Must match
# the ``Platform("carbonvoice")`` value the adapter sends on SessionSource,
# so a user approved here is authorized by Hermes core's own check too.
PLATFORM = "carbonvoice"


class ApprovalStore:
    """Dynamic allow-list backed by Hermes core's ``PairingStore``.

    All methods are no-ops returning safe defaults when ``PairingStore``
    can't be imported (e.g. CI, or a core too old to have it) — the plugin
    still runs, just without the dynamic list.
    """

    def __init__(self) -> None:
        self._store = None  # None = not tried, False = unavailable

    def _pairing(self):
        if self._store is None:
            try:
                from gateway.pairing import PairingStore

                self._store = PairingStore()
            except Exception as exc:  # core absent / incompatible
                logger.debug("carbonvoice: PairingStore unavailable: %s", exc)
                self._store = False
        return self._store or None

    @property
    def available(self) -> bool:
        return self._pairing() is not None

    def is_approved(self, user_id: Optional[str]) -> bool:
        store = self._pairing()
        if not store or not user_id:
            return False
        try:
            return bool(store.is_approved(PLATFORM, user_id))
        except Exception as exc:
            logger.debug("carbonvoice: is_approved(%s) failed: %s", user_id, exc)
            return False

    def approve(self, user_id: str, user_name: str = "") -> bool:
        """Add *user_id* to the dynamic allow-list. Returns success."""
        store = self._pairing()
        if not store or not user_id:
            return False
        try:
            # ``_approve_user`` is documented "must be called under
            # self._lock"; it doesn't take the lock itself, so we do.
            with store._lock:
                store._approve_user(PLATFORM, user_id, user_name)
            return True
        except Exception as exc:
            logger.warning("carbonvoice: approve(%s) failed: %s", user_id, exc)
            return False

    def revoke(self, user_id: str) -> bool:
        """Remove *user_id* from the dynamic allow-list. Returns whether a
        row was actually removed."""
        store = self._pairing()
        if not store or not user_id:
            return False
        try:
            return bool(store.revoke(PLATFORM, user_id))
        except Exception as exc:
            logger.warning("carbonvoice: revoke(%s) failed: %s", user_id, exc)
            return False

    def list_approved(self) -> List[dict]:
        """Return ``[{user_id, user_name, approved_at, ...}, ...]``."""
        store = self._pairing()
        if not store:
            return []
        try:
            return list(store.list_approved(PLATFORM) or [])
        except Exception as exc:
            logger.debug("carbonvoice: list_approved failed: %s", exc)
            return []


# Operator commands (owner-only, in the home channel). Explicit verb-object
# names so they're self-documenting:
#   /cv-allow-user <id>     /cv-deny-user <id>     /cv-list-allow-users
# Case-insensitive; leading/trailing space ok. Order the longest alternative
# first so ``list-allow-users`` isn't shadowed by ``allow-user``.
_CMD_RE = re.compile(
    r"^\s*/cv-(list-allow-users|allow-user|deny-user)\b\s*(\S+)?\s*$",
    re.IGNORECASE,
)
# Map the spoken command to the canonical action the handler switches on.
_ACTION = {
    "allow-user": "allow",
    "deny-user": "deny",
    "list-allow-users": "list",
}


def parse_admin_command(text: Optional[str]) -> Optional[Tuple[str, Optional[str]]]:
    """Parse an operator allow-list command.

    Returns ``(action, arg)`` — ``action`` in ``{"allow","deny","list"}``,
    ``arg`` the target user_guid (or ``None`` for ``list``) — or ``None``
    when *text* is not one of our commands. ``allow``/``deny`` with no arg
    return ``(action, None)`` so the caller can reply with usage help.
    """
    if not text:
        return None
    m = _CMD_RE.match(text)
    if not m:
        return None
    action = _ACTION[m.group(1).lower()]
    arg = m.group(2)
    return (action, arg)
