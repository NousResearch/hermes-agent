"""Matrix room identity, membership and room policy."""

from __future__ import annotations

import asyncio
import hashlib
import json
import time
from contextlib import suppress
from typing import TYPE_CHECKING, Any, Dict, Optional

from plugins.platforms.matrix.room_context import MatrixRoomIdentity, MatrixRoomState

if TYPE_CHECKING:
    from plugins.platforms.matrix.adapter import MatrixAdapter

_ROOM_STATE_READ_TIMEOUT_SECONDS = 10.0
_ROOM_NAME_STATE_KEYS = {"m.room.name": "name", "m.room.topic": "topic", "m.room.canonical_alias": "alias"}


class MatrixContextMixin:
    @staticmethod
    def _state_event_value(event: Any, key: str) -> Optional[str]:
        """Extract a simple value from a Matrix state event object or dict (top-level, then .content)."""
        if event is None:
            return None
        for obj in (event, event.get("content") if isinstance(event, dict) else getattr(event, "content", None)):
            value = obj.get(key) if isinstance(obj, dict) else getattr(obj, key, None)
            if value:
                return str(value)
        return None

    async def _get_room_members(
        self: MatrixAdapter, room_id: str
    ) -> Optional[set[str]]:
        """Read the complete joined member list from the store or homeserver."""
        from plugins.platforms.matrix.adapter import RoomID, Membership

        client = getattr(self, "_client", None)
        if client is None:
            return None

        state_store = getattr(client, "state_store", None)
        if state_store is not None:
            with suppress(Exception):
                if await state_store.has_full_member_list(RoomID(room_id)):
                    members = await state_store.get_members(
                        RoomID(room_id), memberships=(Membership.JOIN,)
                    )
                    if members is not None:
                        return {str(member) for member in members}

        with suppress(Exception):
            members = await asyncio.wait_for(client.get_joined_members(RoomID(room_id)), timeout=10)
            if isinstance(members, dict):
                return {str(member) for member in members}
        return None

    async def _get_room_member_profiles(
        self: MatrixAdapter, room_id: str
    ) -> Optional[Dict[Any, Any]]:
        from plugins.platforms.matrix.adapter import RoomID, Membership

        state_store = getattr(self._client, "state_store", None) if self._client else None
        if state_store:
            with suppress(Exception):
                profiles = await state_store.get_member_profiles(
                    RoomID(room_id), memberships=(Membership.JOIN,)
                )
                if profiles:
                    return dict(profiles)

        client = getattr(self, "_client", None)
        if client is not None and hasattr(client, "get_joined_members"):
            with suppress(Exception):
                profiles = await asyncio.wait_for(
                    client.get_joined_members(RoomID(room_id)), _ROOM_STATE_READ_TIMEOUT_SECONDS,
                )
                if profiles:
                    return dict(profiles)
        return None

    def _compute_room_display_name(
        self: MatrixAdapter, profiles: Optional[Dict[Any, Any]]
    ) -> Optional[str]:
        if not profiles:
            return None

        own_user_id = (self._user_id or "").strip().lower()
        names = []
        for user_id, member in profiles.items():
            if str(user_id).strip().lower() == own_user_id:
                continue
            display_name = getattr(member, "displayname", None)
            if display_name and display_name.strip():
                names.append(display_name.strip())
            elif str(user_id).startswith("@") and ":" in str(user_id):
                names.append(str(user_id)[1:].split(":", 1)[0])
            else:
                names.append(str(user_id))

        if not names:
            return None

        names.sort()
        if len(names) == 1:
            return names[0]
        if len(names) <= 3:
            return f"{', '.join(names[:-1])} and {names[-1]}"
        remaining = len(names) - 3
        noun = "other" if remaining == 1 else "others"
        return f"{', '.join(names[:3])} and {remaining} {noun}"

    async def _read_room_state_event(
        self: MatrixAdapter, room_id: str, event_type: str
    ) -> Any:
        """The content of a room state event, or None when the room has no such event (``M_NOT_FOUND``).
        Any other failure, including the read deadline, raises."""
        from plugins.platforms.matrix.adapter import RoomID, MNotFound

        if not self._client or not hasattr(self._client, "get_state_event"):
            return None
        try:
            return await asyncio.wait_for(
                self._client.get_state_event(RoomID(room_id), event_type), _ROOM_STATE_READ_TIMEOUT_SECONDS,
            )
        except Exception as exc:
            if isinstance(exc, MNotFound) or getattr(exc, "errcode", None) == "M_NOT_FOUND":
                return None
            raise

    async def _read_room_member_profiles(
        self: MatrixAdapter, room_id: str
    ) -> tuple[Optional[set[str]], Optional[Dict[Any, Any]]]:
        members = await self._get_room_members(room_id)
        profiles = await self._get_room_member_profiles(room_id) if members is not None else None
        return members, profiles

    def _remember_room_names(
        self: MatrixAdapter,
        room_id: str,
        reads: Dict[str, Any],
        profiles: Optional[Dict[Any, Any]],
    ) -> tuple[Optional[str], Optional[str], Optional[str], Optional[str]]:
        """The room's name, topic, canonical alias and member-derived name. *reads* maps each event type
        in ``_ROOM_NAME_STATE_KEYS`` to its content or to the exception that its read raised, and
        *profiles* is None when the member read failed. A failed read gives the last value read for the
        room, so a timeout does not rename the room; a successful read replaces that value, so a removed
        name or topic applies."""
        values = self._room_state_values.pop(room_id, {})
        for event_type, key in _ROOM_NAME_STATE_KEYS.items():
            event = reads[event_type]
            if not isinstance(event, Exception):
                values[event_type] = (self._state_event_value(event, key) or "").strip() or None
        if profiles is not None:
            values["m.room.member"] = self._compute_room_display_name(profiles)
        if len(self._room_state_values) >= self._room_identity_cache_max:
            del self._room_state_values[next(iter(self._room_state_values))]
        self._room_state_values[room_id] = values
        return (
            values.get("m.room.name"), values.get("m.room.topic"), values.get("m.room.canonical_alias"),
            values.get("m.room.member"),
        )

    def _invalidate_room_identities(
        self: MatrixAdapter, room_id: str | None = None
    ) -> None:
        """Drop one cached room identity (or all when *room_id* is None)."""
        if room_id is None:
            self._room_identities.clear()
            self._room_identity_cached_at.clear()
        else:
            self._room_identities.pop(room_id, None)
            self._room_identity_cached_at.pop(room_id, None)

    async def _resolve_room_identity(
        self: MatrixAdapter, room_id: str, *, force_refresh: bool = False
    ) -> MatrixRoomIdentity:
        """Resolve room identity from joined membership and room metadata."""
        from plugins.platforms.matrix.adapter import logger

        cached = self._room_identities.get(room_id)
        ttl = self._room_identity_ttl_seconds
        cache_fresh = ttl <= 0 or time.monotonic() - self._room_identity_cached_at.get(room_id, 0.0) <= ttl
        if cached is not None and cache_fresh and not force_refresh:
            return cached
        (
            name_event, topic_event, alias_event, join_rules_event, history_event, encryption_event,
            tombstone_event, member_read,
        ) = reads = await asyncio.gather(
            *(
                self._read_room_state_event(room_id, event_type) for event_type in (
                    "m.room.name", "m.room.topic", "m.room.canonical_alias", "m.room.join_rules",
                    "m.room.history_visibility", "m.room.encryption", "m.room.tombstone",
                )
            ),
            self._read_room_member_profiles(room_id),
            return_exceptions=True,
        )
        for result in reads:
            if isinstance(result, BaseException) and not isinstance(result, Exception):
                raise result
        failed_reads = [result for result in reads if isinstance(result, Exception)]
        members, profiles = (
            (None, None) if isinstance(member_read, BaseException) else member_read
        )
        if failed_reads:
            logger.debug("Matrix: room state read failed for %s: %r", room_id, failed_reads[0])

        def state_value(event: Any, key: str) -> Optional[str]:
            if isinstance(event, Exception):
                return None
            return (self._state_event_value(event, key) or "").strip() or None

        room_name, room_topic, canonical_alias, member_name = self._remember_room_names(
            room_id, dict(zip(_ROOM_NAME_STATE_KEYS, (name_event, topic_event, alias_event))), profiles,
        )
        member_count = len(members) if members is not None else None
        members_digest = None
        if members is not None and profiles is not None:
            profile_names = {
                str(user_id): str(getattr(profile, "displayname", None) or "")
                for user_id, profile in profiles.items()
            }
            member_rows = [(user_id, profile_names.get(user_id, "")) for user_id in sorted(members)]
            members_digest = hashlib.sha256(
                json.dumps(member_rows, ensure_ascii=False).encode("utf-8")
            ).hexdigest()
        has_explicit_name = bool(room_name)
        is_direct = bool(self._dm_rooms.get(room_id, False))
        is_likely_dm = bool(members is not None and len(members) == 2 and self._user_id in members)
        display_name = room_name or canonical_alias or member_name or room_id
        room_state = (
            None if failed_reads or members_digest is None
            else MatrixRoomState(
                display_name, room_topic, members_digest,
                join_rule=state_value(join_rules_event, "join_rule"),
                history_visibility=state_value(history_event, "history_visibility"),
                encrypted=encryption_event is not None, tombstoned=tombstone_event is not None,
            )
        )
        identity = MatrixRoomIdentity(
            room_id=room_id, room_name=room_name, room_topic=room_topic, canonical_alias=canonical_alias,
            server_name=(room_id.rsplit(":", 1)[-1].strip() or None) if ":" in room_id else None,
            joined_member_count=member_count, room_state=room_state,
            is_direct_account_data=is_direct, display_name=display_name,
            has_explicit_name=has_explicit_name, chat_type="dm" if is_likely_dm else "room",
            conflict=bool(is_direct and not is_likely_dm))
        if len(self._room_identities) >= self._room_identity_cache_max:
            oldest = min(self._room_identity_cached_at, key=self._room_identity_cached_at.get, default=None)
            if oldest:
                self._invalidate_room_identities(oldest)
        self._room_identities[room_id] = identity
        self._room_identity_cached_at[room_id] = time.monotonic()
        return identity

    async def _is_dm_room(self: MatrixAdapter, room_id: str) -> bool:
        return (await self._resolve_room_identity(room_id)).chat_type == "dm"

    def _is_allowed_matrix_room(
        self: MatrixAdapter, room_id: str, chat_type: str = "group"
    ) -> bool:
        return (
            not self._allowed_room_ids
            or room_id in self._allowed_room_ids
            or chat_type == "dm"
        )

    async def _is_allowed_matrix_room_event(self: MatrixAdapter, room_id: str) -> bool:
        """MATRIX_ALLOWED_ROOMS gate; DMs are exempt so personal chats survive a project allowlist."""
        from plugins.platforms.matrix.adapter import logger

        if self._is_allowed_matrix_room(room_id):
            return True
        try:
            return await self._is_dm_room(room_id)
        except Exception as exc:
            logger.debug("Matrix: could not resolve room identity for allowlist check in %s: %s", room_id, exc)
            return False
