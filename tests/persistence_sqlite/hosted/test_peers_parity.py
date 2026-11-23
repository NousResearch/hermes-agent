"""The SQLite ``HostedRoomPeerRepository``'s reservation probe, which reads a placement without creating its file."""

from __future__ import annotations

import pytest

from persistence.sqlite.gateway.hosted_room_peers import SQLiteHostedRoomPeerRepository
from tests.persistence_sqlite.hosted.conftest import NOW


pytestmark = pytest.mark.usefixtures("clock")


def test_probe_never_creates_the_file(backend, scopes, root):
    peers = SQLiteHostedRoomPeerRepository(backend.open(scopes["a"]))
    assert peers.probe_peer_room_reservation(placement="shared", room_id="room", target_profile="p1", now=NOW) is False
    assert not (root / "shared-state.db").exists()
