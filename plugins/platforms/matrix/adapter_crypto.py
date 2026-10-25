"""Matrix E2EE crypto state-store shim handed to mautrix's OlmMachine."""

from __future__ import annotations

import logging
from contextlib import suppress
from typing import Any

logger = logging.getLogger(__name__)


class _CryptoStateStore:
    """StateStore shim for OlmMachine (MemoryStateStore lacks is_encrypted/get_encryption_info/
    find_shared_rooms); falls back to a homeserver state query when the store has no info."""

    def __init__(self, client_state_store: Any, joined_rooms: set, client=None):
        self._ss = client_state_store
        self._joined_rooms = joined_rooms
        self._client = client
        # MemoryStateStore has no set_encryption_info, so cache homeserver answers here.
        self._enc_info_cache: dict = {}

    async def is_encrypted(self, room_id: str) -> bool:
        return (await self.get_encryption_info(room_id)) is not None

    async def get_encryption_info(self, room_id: str):
        info = await self._ss.get_encryption_info(room_id) if hasattr(self._ss, "get_encryption_info") else None
        if info is not None:
            return info
        if room_id in self._enc_info_cache:
            return self._enc_info_cache[room_id]
        if self._client is None:
            return None
        try:
            from mautrix.types import EventType as _ET, RoomEncryptionStateEventContent as _Enc, RoomID as _RID
            raw = await self._client.get_state_event(_RID(room_id), _ET.ROOM_ENCRYPTION)
        except Exception as exc:
            logger.debug("Matrix: homeserver encryption-info query failed for %s: %s", room_id, exc)
            return None
        if not raw:
            return None
        content = raw if isinstance(raw, _Enc) else _Enc.deserialize(
            raw.serialize() if hasattr(raw, "serialize") else raw)
        if hasattr(self._ss, "set_encryption_info"):
            with suppress(Exception):
                await self._ss.set_encryption_info(_RID(room_id), content)
        self._enc_info_cache[room_id] = content
        return content

    async def find_shared_rooms(self, user_id: str) -> list:
        return list(self._joined_rooms)  # all joined rooms: correct for a single-user bot
