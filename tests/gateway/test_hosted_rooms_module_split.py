import subprocess
import sys
from pathlib import Path


def test_storage_can_initialize_before_public_room_facade(tmp_path):
    repo_root = Path(__file__).parents[2]
    script = """
import sys
from pathlib import Path

from gateway import hosted_room_contract as contract
from gateway import hosted_room_storage as storage

assert "gateway.hosted_rooms" not in sys.modules
conn = storage._read_connection(Path(sys.argv[1]))
conn.close()

from gateway import hosted_rooms

assert storage._public_api() is hosted_rooms
assert hosted_rooms.HostedRoomError is contract.HostedRoomError
assert hosted_rooms._initialize_schema is storage._initialize_schema

hosted_rooms.MAX_ROOM_ID_CHARS = 4
try:
    storage.reserve_peer_room(
        Path(sys.argv[1]),
        claims={
            "room_id": "room-too-long",
            "member_id": "member",
            "target_profile": "default",
            "authority_gateway_id": "gateway",
            "authority_epoch": 1,
        },
        expires_at=2,
        now=1,
    )
except hosted_rooms.HostedRoomError:
    pass
else:
    raise AssertionError("storage bypassed the facade-owned room ID limit")
"""
    result = subprocess.run(
        [sys.executable, "-c", script, str(tmp_path / "state.db")],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
