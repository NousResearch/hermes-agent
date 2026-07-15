"""Allowlist gating + ignored-sender audit log.

Hermes core already enforces ``CARBONVOICE_ALLOWED_USERS`` /
``CARBONVOICE_ALLOW_ALL_USERS`` *after* the adapter dispatches. We
replicate the check inside the adapter so we can:

  1. Short-circuit before the agent ever sees the message (cheaper).
  2. Record the rejection in an append-only audit log with the resolved
     username, so the operator can see who's trying to reach the bot.

Log path defaults to ``$HERMES_HOME/logs/carbonvoice-ignored-senders.log``
and is one JSON object per line: ``{"time", "user_id", "username", "channel_id"}``.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Optional, Set

if TYPE_CHECKING:
    from .channels import ChannelCache
    from .permits import ApprovalStore

logger = logging.getLogger(__name__)


class AllowlistGate:
    """**Deny-by-default** access control for inbound Carbon Voice messages.

    A user may reach the agent if ANY of these holds:

      1. **allow-all** is explicitly enabled (``CARBONVOICE_ALLOW_ALL_USERS=true``)
         — the escape hatch back to the old open behavior.
      2. they are the **owner** — ``whoami.created_by``, the user who created
         the bot account. Auto-detected at connect; always allowed. This is
         what makes deny-by-default usable without any manual setup.
      3. they are in ``CARBONVOICE_ALLOWED_USERS`` (static env list).
      4. they were **approved at runtime** via ``/cv-allow`` — the
         :class:`~permits.ApprovalStore` (Hermes core's ``PairingStore``).

    Default (no config) → **only the owner**. This closes the security hole
    where anyone on a shared channel could ask the agent to read or run
    things on the host. The owner grows the list interactively from the
    home channel (``/cv-allow <id>``) without restarting.

    History: the default used to be allow-all (``CARBONVOICE_ALLOW_ALL_USERS``
    defaulted true / was an opt-out). It is now an opt-IN. Existing
    deployments with an empty allow-list will, after this change, only
    answer the owner until they approve others — see README/CHANGELOG.
    """

    def __init__(
        self,
        allow_all: bool,
        allowed_ids: Set[str],
        approvals: Optional["ApprovalStore"] = None,
    ):
        self._allow_all = allow_all
        self._allowed_ids = allowed_ids
        self._approvals = approvals
        self._owner_id: Optional[str] = None  # set at connect via set_owner()

    @classmethod
    def from_env(
        cls, approvals: Optional["ApprovalStore"] = None
    ) -> "AllowlistGate":
        raw = os.getenv("CARBONVOICE_ALLOWED_USERS", "")
        allowed = {u.strip() for u in raw.split(",") if u.strip()}
        # allow-all is now an explicit opt-IN (truthy enables it); deny is
        # the default.
        allow_all = os.getenv("CARBONVOICE_ALLOW_ALL_USERS", "").strip().lower() in (
            "true", "1", "yes", "on",
        )
        return cls(allow_all=allow_all, allowed_ids=allowed, approvals=approvals)

    def set_owner(self, owner_id: Optional[str]) -> None:
        """Record the bot owner (``whoami.created_by``). Always allowed."""
        self._owner_id = (owner_id or "").strip() or None

    @property
    def owner_id(self) -> Optional[str]:
        return self._owner_id

    def is_owner(self, user_id: Optional[str]) -> bool:
        return bool(self._owner_id and user_id and user_id == self._owner_id)

    @property
    def has_any_authorizer(self) -> bool:
        """True if *anyone* can be allowed (owner / env list / allow-all).

        When this is False after connect, deny-by-default would mute the bot
        for everyone — the adapter logs a loud bootstrap warning.
        """
        return bool(self._allow_all or self._owner_id or self._allowed_ids)

    def is_allowed(self, user_id: Optional[str]) -> bool:
        if self._allow_all:
            return True
        if not user_id:
            return False
        if self._owner_id and user_id == self._owner_id:
            return True
        if user_id in self._allowed_ids:
            return True
        if self._approvals is not None and self._approvals.is_approved(user_id):
            return True
        return False


class IgnoredSenderLog:
    """Append-only JSON-lines log of rejected inbound senders."""

    def __init__(self, path: Path, channels: "ChannelCache"):
        self._path = path
        self._channels = channels

    @property
    def path(self) -> Path:
        return self._path

    def record(self, user_id: str, channel_id: Optional[str] = None) -> None:
        """Fire-and-forget — never blocks the inbound path."""
        asyncio.create_task(self._record(user_id, channel_id))

    async def _record(self, user_id: str, channel_id: Optional[str]) -> None:
        try:
            # Resolve the name from the channel roster when we have a
            # channel; an unauthorized sender may not be a collaborator,
            # in which case this is None and we log just the guid.
            username = ""
            if user_id and channel_id:
                username = (
                    await self._channels.resolve_name(channel_id, user_id) or ""
                )
            entry = {
                "time": datetime.now(timezone.utc).isoformat(),
                "user_id": user_id,
                "username": username,
            }
            if channel_id:
                entry["channel_id"] = channel_id
            self._path.parent.mkdir(parents=True, exist_ok=True)
            with self._path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(entry) + "\n")
        except Exception as exc:
            logger.debug("carbonvoice: ignored-sender log failed: %s", exc)


def default_ignored_log_path() -> Path:
    """``$HERMES_HOME/logs/carbonvoice-ignored-senders.log`` with safe fallback."""
    try:
        from hermes_constants import get_hermes_home
        home = get_hermes_home()
    except Exception:
        home = Path.home() / ".hermes"
    return home / "logs" / "carbonvoice-ignored-senders.log"
