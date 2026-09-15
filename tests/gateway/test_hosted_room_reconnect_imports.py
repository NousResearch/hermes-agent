"""The defining route owners initialize independently without compatibility aliases."""

import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.parametrize(
    "first",
    [
        "gateway.hosted_room_link_records",
        "gateway.hosted_room_safety",
        "gateway.hosted_rooms",
        "tui_gateway.hosted_room_peer_status",
    ],
)
def test_reconnect_owners_import_independently_and_share_storage(tmp_path, first):
    script = """
import importlib
import sys

importlib.import_module(sys.argv[1])
from gateway import hosted_room_link_records, hosted_rooms
from tui_gateway import hosted_room_peer_status, hosted_room_service

assert hosted_room_service._RouteStatusPeerClient is hosted_room_peer_status._RouteStatusPeerClient
hosted_rooms.create_room(sys.argv[2], room_id="room", name="Room", members=[], authority_gateway_id="gateway")
assert hosted_room_link_records.room_link_record(sys.argv[2], room_id="room", member_id="member") is None
hosted_room_link_records.begin_room_link_retirement(sys.argv[2], room_id="room", authority_gateway_id="gateway", authority_epoch=1)
assert hosted_room_link_records.room_link_retirement_started(sys.argv[2], room_id="room")
"""
    result = subprocess.run(
        [sys.executable, "-c", script, first, str(tmp_path / "state.db")],
        cwd=Path(__file__).parents[2],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
