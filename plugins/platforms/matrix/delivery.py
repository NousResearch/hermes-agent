"""Resolve explicit Matrix destinations and check encryption before delivery."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import Any, Protocol, TYPE_CHECKING

if TYPE_CHECKING:
    from gateway.session import SessionSource
    from plugins.platforms.matrix.adapter import MatrixRoomIdentity


class _RoomIdentityInvalidator(Protocol):
    def __call__(self, room_id: str | None = None) -> None: ...


class _RoomIdentityResolver(Protocol):
    def __call__(
        self, room_id: str, *, force_refresh: bool = False
    ) -> Awaitable[MatrixRoomIdentity]: ...


class MatrixDeliveryMixin:
    _client: Any
    _user_id: str | None
    _joined_rooms: set[str]
    _encryption: bool
    _e2ee_mode: str
    _invalidate_room_identities: _RoomIdentityInvalidator
    _refresh_dm_cache: Callable[[], Awaitable[None]]
    _resolve_room_identity: _RoomIdentityResolver

    async def resolve_delivery_target(self, source: SessionSource) -> SessionSource:
        """Resolve the room and reply session shape through this authenticated client."""
        try:
            return await asyncio.wait_for(
                self._resolve_delivery_source(source), timeout=90
            )
        except asyncio.TimeoutError as exc:
            raise ValueError(
                f"Matrix target '{source.chat_id}': destination resolution timeout (90s)"
            ) from exc

    async def _resolve_delivery_source(self, source: SessionSource) -> SessionSource:
        from gateway.session_identity import replace_source

        room_id = await self._resolve_send_target(source.chat_id)
        identity = await self._resolve_room_identity(room_id, force_refresh=True)
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
            chat_type=chat_type,
        )

    async def _resolve_send_target(self, chat_id: str) -> str:
        from plugins.platforms.matrix.adapter import RoomID

        target = chat_id.split("/", 1)[0]
        if target.startswith("@"):
            raise ValueError(
                f"Matrix target '{chat_id}' is an MXID. Use the DM's room ID or a room alias."
            )
        if target.startswith("!"):
            return target
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
                return room_id
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
            return room_id
        except asyncio.TimeoutError as exc:
            raise ValueError(
                f"Matrix target '{chat_id}': alias resolution, join or account-data refresh timed out"
            ) from exc
        except Exception as exc:
            raise ValueError(f"Matrix target '{chat_id}': {exc}") from exc

    async def _check_room_encryption(self, room_id: str) -> bool:
        from plugins.platforms.matrix.adapter import EventType, RoomID

        crypto = getattr(self._client, "crypto", None)
        if self._e2ee_mode == "required" and not (self._encryption and crypto):
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
        if not (self._encryption and crypto):
            raise ValueError(
                f"Room '{room_id}' is encrypted but Matrix E2EE is unavailable"
            )
        # mautrix sends plaintext if its state store reports an unencrypted room.
        if not await self._client.state_store.is_encrypted(RoomID(room_id)):
            raise ValueError(
                f"Room '{room_id}' is encrypted but the client state is unsynchronised"
            )
        return True
