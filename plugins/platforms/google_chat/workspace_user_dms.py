"""Google Chat: answer person-to-person DMs with the impersonated Workspace user, as that user.

A Chat app cannot be added to a DM between two people, so a message someone sends to the
agent's Workspace user (rather than to the app) never reaches the agent. With
``workspace_user.answer_dms: true`` the adapter finds the user's DMs with each
``GOOGLE_CHAT_ALLOWED_USERS`` member, holds a Workspace Events subscription on each one it has
joined, admits every message the peer sends there, and does everything in those spaces as the
user: replies, edits, deletes, uploads and downloads (scope ``chat.messages``). Users cannot post
cards, so cards and clarify buttons become text there. DMs with anyone else are ignored: no
subscription, no reply, no reaction. App spaces and the app's own DMs are unchanged.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import re
from typing import Any, Optional

from .workspace_events import SubscriptionKeeper

logger = logging.getLogger("gateway.platforms.google_chat")


def space_of(name: Any) -> str:
    """``spaces/X`` for a space, message, reaction or attachment resource name."""
    return "/".join(str(name or "").split("/")[:2])


def card_text(card: Any) -> str:
    """Plain-text rendering of a ``cardsV2`` entry: titles and texts in order, markup dropped."""
    lines: list[str] = []

    def walk(node: Any) -> None:
        if isinstance(node, dict):
            for key in ("title", "subtitle", "topLabel", "text", "bottomLabel"):
                value = node.get(key)
                if isinstance(value, str) and value.strip() and value.strip() not in lines:
                    lines.append(value.strip())
            for key, value in node.items():
                if key not in ("onClick", "icon", "image", "imageUrl") and isinstance(value, (dict, list)):
                    walk(value)
        elif isinstance(node, list):
            for item in node:
                walk(item)
    walk(card)
    return re.sub(r"<[^>]+>", "", "\n".join(lines)) or "(card)"


class DirectMessages:
    """The Workspace user's DMs with allowed people, kept in step every ``INTERVAL`` seconds."""

    INTERVAL = 300.0
    FIRST_PASS_TIMEOUT = 15.0

    def __init__(self, user: Any, adapter: Any) -> None:
        self._user, self._adapter = user, adapter
        self.spaces: dict[str, dict[str, Any]] = {}  # space -> {"email", "peer", "joined"}
        self.user_id: Optional[str] = None  # users/{id} of the Workspace user
        self.keeper: Optional[SubscriptionKeeper] = None
        self._reported: dict[str, str] = {}
        self._task: Optional[asyncio.Task] = None
        self._wake: Optional[asyncio.Event] = None

    def owns(self, name: Any) -> bool:
        """True when ``name`` (space, message or attachment) lives in one of these DMs."""
        return space_of(name) in self.spaces

    # -- background upkeep -------------------------------------------------------
    async def start(self) -> None:
        """First pass before the adapter pulls (so a DM message is never mistaken for a
        stranger's), then the periodic loop."""
        with contextlib.suppress(TimeoutError):
            await asyncio.wait_for(self._pass(), timeout=self.FIRST_PASS_TIMEOUT)
        self._wake = asyncio.Event()
        self._task = asyncio.create_task(self._run())

    async def stop(self) -> None:
        if (task := self._task) is not None:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError, TimeoutError):
                await asyncio.wait_for(task, timeout=5.0)
            self._task = None

    def wake(self) -> None:
        loop, wake = self._adapter._loop, self._wake
        if wake is not None and self._adapter._loop_accepts_callbacks(loop):
            with contextlib.suppress(RuntimeError):
                loop.call_soon_threadsafe(wake.set)

    async def _pass(self, last_err: Optional[str] = None) -> Optional[str]:
        """One upkeep pass in a thread; the error text (logged once while it repeats) or None."""
        from . import adapter as gc
        try:
            await asyncio.to_thread(self.upkeep)
        except Exception as exc:  # retried on the next pass; logged with its traceback
            msg = gc._redact_sensitive(str(exc))[:300]
            if msg != last_err:
                logger.warning("[GoogleChat] answer_dms: DM discovery failed (retrying every %ds): %s",
                               int(self.INTERVAL), msg, exc_info=True)
            return msg
        return None

    async def _run(self) -> None:
        last_err: Optional[str] = None
        while not self._adapter._shutting_down:
            self._wake.clear()
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(self._wake.wait(), timeout=self.INTERVAL)
            err = await self._pass(last_err)
            if last_err is not None and err is None:
                logger.info("[GoogleChat] answer_dms: DM discovery recovered")
            last_err = err

    def _topic(self) -> str:
        sub = self._adapter._subscriber.get_subscription(request={"subscription": self._adapter._subscription_path})
        return sub.topic

    def upkeep(self) -> None:
        """Discover the DMs, subscribe the joined ones, drop the ones that went away (blocking)."""
        if self.keeper is None:
            self.keeper = SubscriptionKeeper(self._user.events, self._user.http, self._topic())
        found = self.discover()
        for space in [s for s in self.spaces if s not in found]:
            self.spaces.pop(space, None)
            logger.info("[GoogleChat] answer_dms: %s is no longer a DM with an allowed user", space)
            self._drop(space)
        self.spaces.update(found)
        for space, info in found.items():
            self._report(space, info)
            if info["joined"]:
                self.keeper.ensure(space)

    def _drop(self, space: str) -> None:
        from . import adapter as gc
        try:
            self.keeper.drop(space)
        except gc.HttpError as exc:  # it expires on its own; nothing is admitted from that space meanwhile
            logger.debug("[GoogleChat] answer_dms: deleting the subscription on %s failed: %s", space,
                         gc._redact_sensitive(str(exc)))

    def _report(self, space: str, info: dict[str, Any]) -> None:
        state = "joined" if info["joined"] else "invited"
        if self._reported.get(space) == state:
            return
        self._reported[space] = state
        if info["joined"]:
            logger.info("[GoogleChat] answer_dms: answering %s in %s as %s", info["email"], space,
                        self._user.settings.email)
        else:
            logger.warning("[GoogleChat] answer_dms: the DM %s from %s is still an invitation to %s, and the API "
                           "cannot accept it: sign in as that user and open it once, or turn on automatic "
                           "acceptance of invitations between users (Admin console: Google Chat → Chat "
                           "invitations); rechecking every %ds", space, info["email"], self._user.settings.email,
                           int(self.INTERVAL))

    def _member_id(self, chat: Any, space: str, email: str) -> Optional[str]:
        from . import adapter as gc
        try:
            found = chat.spaces().members().get(name=f"{space}/members/{email}").execute(http=self._user.http())
        except gc.HttpError as exc:
            if gc._http_status(exc) not in (403, 404):
                raise
            return None
        return str(((found or {}).get("member") or {}).get("name") or "") or None

    def discover(self) -> dict[str, dict[str, Any]]:
        """``findDirectMessage`` per allowed user; ``spaces.get`` says whether the user joined
        (an invitation answers 403); ``members.get`` by email gives both members' ids."""
        from . import adapter as gc
        chat, settings = self._user.chat(), self._user.settings
        found: dict[str, dict[str, Any]] = {}
        for email in sorted(settings.allowed_users - {settings.email}):
            try:
                dm = chat.spaces().findDirectMessage(name=f"users/{email}").execute(http=self._user.http())
            except gc.HttpError as exc:
                if gc._http_status(exc) in (403, 404):  # no DM with that person (yet)
                    continue
                raise
            space = str((dm or {}).get("name") or "")
            if not space or (dm or {}).get("singleUserBotDm"):
                continue
            try:
                chat.spaces().get(name=space).execute(http=self._user.http())
                joined = True
            except gc.HttpError as exc:
                if gc._http_status(exc) not in (403, 404):
                    raise
                joined = False
            peer = self._member_id(chat, space, email) if joined else None
            if joined and self.user_id is None:
                self.user_id = self._member_id(chat, space, settings.email)
            found[space] = {"email": email, "peer": peer, "joined": joined}
        return found

    # -- inbound (Pub/Sub thread) ------------------------------------------------
    def admit(self, m: dict[str, Any]) -> Optional[tuple[dict[str, Any], dict[str, Any]]]:
        """``(msg, envelope)`` for every message the DM's allowed peer sends; never the user's
        own, a bot's, or anyone else's."""
        name = str(m.get("name") or "")
        space_name = space_of(name)
        info = self.spaces.get(space_name)
        sender = m.get("sender") or {}
        sender_id = str(sender.get("name") or "")
        if not (name and info and info.get("peer")) or sender.get("type") == "BOT":
            return None
        if sender_id != info["peer"] or sender_id == self.user_id:
            logger.info("[GoogleChat] answer_dms: %s from %s is not from the DM's allowed peer; ignored",
                        name, sender_id or "?")
            return None
        if self._adapter._dedup.is_duplicate(name):
            return None
        email, text = info["email"], str(m.get("text") or "")
        space = {**(m.get("space") or {}), "name": space_name, "spaceType": "DIRECT_MESSAGE"}
        msg = {**m, "space": space, "argumentText": text,
               "sender": {**sender, "email": email, "displayName": sender.get("displayName") or email}}
        logger.info("[GoogleChat] answer_dms: admitted %s from %s (%d chars)", name, email, len(text))
        return msg, {"space": space, "message": msg}

    def refused(self, chat_id: str) -> None:
        """A send as the user was refused (403): left the DM? Recheck soon."""
        logger.warning("[GoogleChat] answer_dms: sending as %s was refused in %s (403); rechecking the DM",
                       self._user.settings.email, chat_id)
        self.wake()

