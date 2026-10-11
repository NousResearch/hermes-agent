"""Resolve explicit Matrix destinations and decide the encryption of each send."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any, Protocol, TYPE_CHECKING

if TYPE_CHECKING:
    from gateway.session import SessionSource
    from plugins.platforms.matrix.adapter import MatrixRoomIdentity


def split_thread_target(target: str) -> tuple[str, str | None]:
    """Split a delivery target into its room reference and optional thread root event.
    ``<room>/<event_id>`` follows matrix.to, and ``<room>:$<event_id>`` is the older form."""
    target = target.strip()
    room, slash, thread = target.partition("/")
    if slash:
        return room, thread or None
    room, colon, event = target.rpartition(":$")
    if colon and room:
        return room, f"${event}"
    return target, None


@dataclass(frozen=True)
class MatrixSendTarget:
    """A resolved send target. ``delivery`` marks an explicit delivery (cron, ``send_message``,
    a configured home), whose encryption state is checked with the homeserver."""

    room_id: str
    metadata: dict[str, Any] | None
    encrypted: bool
    delivery: bool


class _RoomIdentityInvalidator(Protocol):
    def __call__(self, room_id: str | None = None) -> None: ...


class _RoomIdentityResolver(Protocol):
    def __call__(
        self, room_id: str, *, force_refresh: bool = False
    ) -> Awaitable[MatrixRoomIdentity]: ...


class MatrixDeliveryMixin:
    _client: Any
    _user_id: str
    _joined_rooms: set[str]
    _encryption: bool
    _e2ee_mode: str
    _invalidate_room_identities: _RoomIdentityInvalidator
    _refresh_dm_cache: Callable[[], Awaitable[None]]
    _resolve_room_identity: _RoomIdentityResolver

    async def resolve_delivery_target(
        self, source: SessionSource, *, refresh: bool = False
    ) -> SessionSource:
        """Resolve the room, thread and reply session shape through this authenticated client.
        ``refresh`` rereads room membership instead of using the synced identity."""
        try:
            return await asyncio.wait_for(
                self._resolve_delivery_source(source, refresh=refresh), timeout=90
            )
        except asyncio.TimeoutError as exc:
            raise ValueError(
                f"Matrix target '{source.chat_id}': destination resolution timeout (90s)"
            ) from exc

    async def _resolve_delivery_source(
        self, source: SessionSource, *, refresh: bool
    ) -> SessionSource:
        from gateway.session_identity import replace_source

        room_id, suffix_thread = await self._resolve_send_target(source.chat_id)
        identity = await self._resolve_room_identity(room_id, force_refresh=refresh)
        chat_type = "unknown"
        if identity.joined_member_count is not None and (
            identity.joined_member_count != 2 or self._user_id
        ):
            chat_type = "dm" if identity.chat_type == "dm" else "group"
        else:
            self._invalidate_room_identities(room_id)
        return replace_source(
            source,
            chat_id=room_id,
            thread_id=source.thread_id or suffix_thread,
            chat_type=chat_type,
        )

    async def _resolve_send_destination(
        self, chat_id: str, metadata: dict[str, Any] | None, *, upload: bool
    ) -> MatrixSendTarget:
        """Resolve ``chat_id`` for a send and decide whether the send is encrypted.
        A delivery caller passes ``_original_target`` in ``metadata``; a target that needs
        resolving (an alias or a thread suffix) is a delivery as well. A thread suffix applies
        when the caller passed no thread."""
        room_id, suffix_thread = await self._resolve_send_target(chat_id)
        delivery = bool((metadata or {}).get("_original_target")) or room_id != chat_id
        if suffix_thread and not (metadata or {}).get("thread_id"):
            metadata = {**(metadata or {}), "thread_id": suffix_thread}
        encrypted = await self._room_encryption(room_id, verify=delivery, upload=upload)
        return MatrixSendTarget(room_id, metadata, encrypted, delivery)

    async def _resolve_send_target(self, chat_id: str) -> tuple[str, str | None]:
        from plugins.platforms.matrix.adapter import RoomID

        target, thread_id = split_thread_target(chat_id)
        if target.startswith("@"):
            raise ValueError(
                f"Matrix target '{chat_id}' is an MXID. Use the DM's room ID or a room alias."
            )
        if target.startswith("!"):
            return target, thread_id
        if not target.startswith("#"):
            raise ValueError(
                f"Invalid Matrix target '{chat_id}': use a room ID or alias"
            )
        try:
            info = await asyncio.wait_for(
                self._client.resolve_room_alias(target), timeout=15
            )
            room_id = str(info.room_id or "")
            if not room_id.startswith("!"):
                raise ValueError(
                    "alias did not resolve to a room ID; publish the alias as a Local Address "
                    "on the room or use its room ID"
                )
            if room_id in self._joined_rooms:
                return room_id, thread_id
            joined = await asyncio.wait_for(
                self._client.join_room(
                    RoomID(room_id), servers=info.servers, max_retries=0
                ),
                timeout=45,
            )
            if str(joined) != room_id:
                raise ValueError(
                    f"join returned '{joined}', expected room ID '{room_id}'"
                )
            self._joined_rooms.add(room_id)
            self._invalidate_room_identities(room_id)
            await asyncio.wait_for(self._refresh_dm_cache(), timeout=15)
            return room_id, thread_id
        except asyncio.TimeoutError as exc:
            raise ValueError(
                f"Matrix target '{chat_id}': alias resolution, join or account-data refresh timed out"
            ) from exc
        except Exception as exc:
            raise ValueError(f"Matrix target '{chat_id}': {exc}") from exc

    def _e2ee_ready(self) -> bool:
        return bool(self._encryption and getattr(self._client, "crypto", None))

    async def _synced_room_encryption(self, room_id: str) -> bool | None:
        """The room's encryption as recorded by sync, or None when the state store has no
        answer for the room."""
        from plugins.platforms.matrix.adapter import RoomID, logger

        state_store = getattr(self._client, "state_store", None)
        if state_store is None:
            return None
        try:
            return await state_store.is_encrypted(RoomID(room_id))
        except Exception as exc:
            logger.debug(
                "Matrix: state store encryption read failed for %s: %s", room_id, exc
            )
            return None

    async def _room_encryption(
        self, room_id: str, *, verify: bool, upload: bool
    ) -> bool:
        """Whether a send to ``room_id`` is encrypted. With ``verify`` the homeserver answers,
        because a delivery may target a room that sync has not described yet. Otherwise the
        synced state answers. Before an upload, missing state is fetched when E2EE is on, as
        the SDK does before it encrypts an event."""
        if verify:
            return await self._check_room_encryption(room_id)
        if self._e2ee_mode == "required" and not self._e2ee_ready():
            raise ValueError(
                "Matrix E2EE is required but the encryption client is unavailable"
            )
        encrypted = await self._synced_room_encryption(room_id)
        if encrypted is None and upload and self._e2ee_ready():
            return await self._check_room_encryption(room_id)
        if encrypted and not self._e2ee_ready():
            raise ValueError(
                f"Room '{room_id}' is encrypted but Matrix E2EE is unavailable"
            )
        return bool(encrypted)

    async def _check_room_encryption(self, room_id: str) -> bool:
        from plugins.platforms.matrix.adapter import EventType, RoomID

        if self._e2ee_mode == "required" and not self._e2ee_ready():
            raise ValueError(
                "Matrix E2EE is required but the encryption client is unavailable"
            )
        try:
            await asyncio.wait_for(
                self._client.get_state_event(
                    RoomID(room_id), EventType.ROOM_ENCRYPTION
                ),
                timeout=15,
            )
        except asyncio.TimeoutError as exc:
            raise ValueError(
                f"Cannot check encryption state for '{room_id}': timeout (15s)"
            ) from exc
        except Exception as exc:
            if getattr(exc, "errcode", None) == "M_NOT_FOUND":
                return False
            raise ValueError(
                f"Cannot check encryption state for '{room_id}': {exc}"
            ) from exc
        if not self._e2ee_ready():
            raise ValueError(
                f"Room '{room_id}' is encrypted but Matrix E2EE is unavailable"
            )
        # mautrix sends plaintext if its state store reports an unencrypted room.
        if not await self._client.state_store.is_encrypted(RoomID(room_id)):
            raise ValueError(
                f"Room '{room_id}' is encrypted but the client state is unsynchronised"
            )
        return True
