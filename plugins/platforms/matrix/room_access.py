"""Client ownership and unavailable classification during Matrix room admission."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from plugins.platforms.matrix.adapter import MatrixAdapter


class RoomClientChanged(Exception):
    """The adapter no longer owns the client that began a room read."""


class RoomClassificationUnavailable(Exception):
    """Joined membership is unavailable for the room's admission policy."""


@dataclass(frozen=True)
class RoomClientOwner:
    adapter: MatrixAdapter
    client: Any

    def check(self) -> None:
        if self.adapter._client is not self.client:
            raise RoomClientChanged


@dataclass(frozen=True)
class LiveRoomClient:
    """Follows the adapter's current client and accepts a reconnect."""

    adapter: MatrixAdapter

    @property
    def client(self) -> Any:
        return self.adapter._client

    def check(self) -> None:
        return None


RoomClientScope = RoomClientOwner | LiveRoomClient
