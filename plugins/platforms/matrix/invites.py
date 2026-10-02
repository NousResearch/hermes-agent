"""Matrix invite admission, reconciliation and joining."""

from __future__ import annotations

import asyncio
import logging
from contextlib import suppress
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from plugins.platforms.matrix.adapter import MatrixAdapter

logger = logging.getLogger("plugins.platforms.matrix.adapter")


class MatrixInvitesMixin:
    async def _on_invite(self: MatrixAdapter, event: Any) -> None:
        """Auto-join rooms when invited, recording DM rooms in m.direct."""
        room_id = str(getattr(event, "room_id", ""))
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


    def _schedule_pending_invite_joins(self: MatrixAdapter, sync_data: dict[str, Any]) -> None:
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
            # invite state. Without it, a direct invite joined here is never
            # recorded in m.direct.
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

        for event in events:
            if not isinstance(event, dict):
                continue
            if event.get("type") != "m.room.member":
                continue
            if event.get("state_key") != self._user_id:
                continue

            content = event.get("content", {})
            if not isinstance(content, dict):
                continue
            if content.get("membership") != "invite":
                continue

            return bool(content.get("is_direct")), str(event.get("sender", ""))

        return False, ""
