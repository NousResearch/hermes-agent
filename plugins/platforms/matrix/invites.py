"""Matrix invite admission, reconciliation and joining."""

from __future__ import annotations

import asyncio
import logging
from contextlib import suppress
from typing import TYPE_CHECKING, Any, Dict

if TYPE_CHECKING:
    from plugins.platforms.matrix.adapter import MatrixAdapter

logger = logging.getLogger("plugins.platforms.matrix.adapter")


def _is_invited_room_source(source: Any) -> bool:
    """True when *source* (the mautrix ``SyncStream`` flag ``Client.dispatch_event`` stamps on an
    event) says it came from the ``rooms.invite`` section. Anything unclassifiable answers True so
    an unusual caller keeps the old act-on-it behaviour rather than silently dropping an invite."""
    try:
        from mautrix.client import SyncStream

        return bool(source & SyncStream.INVITED_ROOM)
    except Exception:
        return True



class MatrixInvitesMixin:
    async def _on_invite(self: MatrixAdapter, event: Any) -> None:
        """Auto-join rooms when invited, recording DM rooms in m.direct.

        mautrix's ``MembershipEventDispatcher`` fans out every ``m.room.member`` event whose
        membership is ``invite`` as ``InternalEventType.INVITE``, including historic ones
        re-read on each (re)connect: we sync with ``MemorySyncStore``, so every connect is a
        full-state initial sync that dispatches each joined room's state and recent timeline
        again. The only invite we can act on is the one the homeserver delivers in
        ``rooms.invite``; everything else is history and stays quiet.
        """
        room_id = str(getattr(event, "room_id", ""))
        source = getattr(event, "source", None)
        if source is not None and not _is_invited_room_source(source):
            logger.debug("Matrix: ignoring replayed membership invite in %s (source=%s)", room_id, source)
            return
        if room_id and room_id in self._joined_rooms:
            logger.debug("Matrix: ignoring invite to %s — already joined", room_id)
            return
        # Skip invites addressed to someone else (bridged rooms carry other users'
        # invites via state_key). An unresolved target keeps the old path (#76292).
        target = str(getattr(event, "state_key", "") or "")
        if self._user_id and target and not self._is_self_sender(target):
            logger.debug("Matrix: ignoring invite to %s addressed to %s", room_id, target)
            return
        is_direct = bool(getattr(getattr(event, "content", None), "is_direct", False))
        inviter = str(getattr(event, "sender", ""))
        # Only authorized inviters — otherwise any federated user could pull the bot into rooms.
        if not self._is_authorized_user(inviter, str(room_id)):
            logger.warning("Matrix: rejecting invite to %s from unauthorized user %s", room_id, inviter)
            return
        logger.info("Matrix: invited to %s — joining (is_direct=%s)", room_id, is_direct)
        # Join off the sync path; a declared DM is recorded in m.direct once the join lands.
        self._schedule_invite_join(room_id, is_direct=is_direct and bool(inviter), inviter=inviter)

    async def _join_room_by_id(self: MatrixAdapter, room_id: str) -> bool:
        from plugins.platforms.matrix.adapter import RoomID

        if not room_id or room_id in self._joined_rooms:
            return bool(room_id)
        try:
            await self._client.join_room(RoomID(room_id))
            self._joined_rooms.add(room_id)
            self._invalidate_room_identities(room_id)
            logger.info("Matrix: joined %s", room_id)
            await self._refresh_dm_cache()
            return True
        except Exception as exc:
            logger.warning("Matrix: error joining %s: %s", room_id, exc)
            # Abandoned rooms ("no servers ..." / "room not found") would retry every startup
            # unless we leave the invite; the match is narrow so transient errors keep retrying.
            msg = str(exc).lower()
            if ("no servers" in msg) or ("room not found" in msg):
                with suppress(Exception):
                    await self._client.leave_room(RoomID(room_id))
                    logger.info("Matrix: declined dead invite to %s", room_id)
            return False

    def _schedule_invite_join(self: MatrixAdapter, room_id: str, *, is_direct: bool = False, inviter: str = "") -> None:
        """Schedule an invite join without blocking sync or gateway readiness."""
        existing = self._invite_join_tasks.get(room_id)
        if not room_id or room_id in self._joined_rooms or (existing and not existing.done()):
            return

        async def _join_invite() -> None:
            try:
                joined = await asyncio.wait_for(self._join_room_by_id(room_id), timeout=45.0)
                if joined and is_direct and inviter:
                    await self._record_dm_room(room_id, inviter)
            except TimeoutError:
                logger.warning("Matrix: timed out joining invite %s", room_id)
            finally:
                self._invite_join_tasks.pop(room_id, None)
        self._invite_join_tasks[room_id] = asyncio.create_task(_join_invite())

    def _schedule_pending_invite_joins(self: MatrixAdapter, sync_data: Dict[str, Any]) -> None:
        """Join rooms still present in rooms.invite after sync processing."""
        invites = (sync_data.get("rooms", {}) if isinstance(sync_data, dict) else {}).get("invite", {})
        if not isinstance(invites, dict):
            return
        for room_id, invited_room in invites.items():
            if room_id in self._joined_rooms:
                continue
            # This reconcile pass runs after _dispatch_sync and sees every
            # rooms.invite entry, whether _on_invite joined it, rejected
            # it, or (for invites that arrived while the gateway was down)
            # is only now seeing it. The invite event object is gone by
            # this point, so the DM signal must be read from the stripped
            # invite state; without it a direct invite joined here is never
            # recorded in m.direct and gets misclassified as a group.
            is_direct, inviter = self._extract_invite_dm_signal(invited_room)
            # The inviter allowlist gate from _on_invite must apply here
            # too: an unconditional join would re-admit a live invite that
            # _on_invite just rejected milliseconds earlier, and would
            # auto-join any invite from an arbitrary federated user on
            # restart. An inviter missing from the stripped invite state
            # fails closed, like an empty sender in _on_invite.
            if not self._is_authorized_user(inviter, str(room_id)):
                logger.warning(
                    "Matrix: rejecting invite to %s from unauthorized user %s",
                    room_id,
                    inviter,
                )
                continue
            logger.info(
                "Matrix: reconciling pending invite for %s (is_direct=%s)",
                room_id,
                is_direct,
            )
            self._schedule_invite_join(str(room_id), is_direct=is_direct, inviter=inviter)

    def _extract_invite_dm_signal(self: MatrixAdapter, invited_room: Any) -> tuple[bool, str]:
        """Read the is_direct flag and inviter from a room's invite_state.

        The stripped ``m.room.member`` event for our own user carries the
        ``is_direct`` flag from the original invite; its sender is the
        inviter. Returns ``(False, "")`` when the signal is absent.
        """
        if not self._user_id:
            return False, ""

        if not isinstance(invited_room, dict):
            return False, ""

        invite_state = invited_room.get("invite_state", {})
        if not isinstance(invite_state, dict):
            return False, ""

        events = invite_state.get("events", [])
        if not isinstance(events, list):
            return False, ""

        members = [e for e in events if isinstance(e, dict) and e.get("type") == "m.room.member"]
        # Our event is the one addressed to our exact id (the one mautrix dispatches to
        # _on_invite). Only when there is none, fall back to _on_invite's case-insensitive
        # identity check, so a look-alike member event can never stand in for ours.
        own = [e for e in members if e.get("state_key") == self._user_id]
        if not own:
            own = [e for e in members if self._is_self_sender(str(e.get("state_key") or ""))]

        for event in own:
            content = event.get("content", {})
            if not isinstance(content, dict):
                continue
            if content.get("membership") != "invite":
                continue

            return bool(content.get("is_direct")), str(event.get("sender", ""))

        return False, ""
