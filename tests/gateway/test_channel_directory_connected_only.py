"""Session-based channel discovery must not resurrect disconnected platforms.

Surgical reapply of the directory portion of PR #25959: historical session
origins for platforms with no connected adapter must not become active
send_message targets."""

import asyncio
from unittest.mock import patch

from gateway.channel_directory import build_channel_directory
from gateway.platforms.base import Platform


def test_does_not_resurrect_disconnected_platforms_from_session_history(tmp_path, monkeypatch):
    cache_file = tmp_path / "channel_directory.json"

    calls = []

    def fake_build_from_sessions(plat_name):
        calls.append(plat_name)
        return {"channels": [{"id": "1", "name": "old"}]}

    with patch("gateway.channel_directory._build_from_sessions", side_effect=fake_build_from_sessions), \
         patch("gateway.channel_directory.DIRECTORY_PATH", cache_file):
        # Only telegram is connected; no discord/slack/whatsapp adapters.
        directory = asyncio.run(build_channel_directory({Platform.TELEGRAM: object()}))

    plats = directory["platforms"]
    assert "telegram" in plats
    # Disconnected platforms must not appear via session discovery.
    for stale in ("whatsapp", "signal", "matrix"):
        assert stale not in plats, f"{stale} resurrected from session history"
    assert set(calls) <= {"telegram"}


def test_connected_platform_still_uses_session_discovery(tmp_path):
    cache_file = tmp_path / "channel_directory.json"

    with patch(
        "gateway.channel_directory._build_from_sessions",
        return_value={"channels": []},
    ) as mock_sessions, patch("gateway.channel_directory.DIRECTORY_PATH", cache_file):
        directory = asyncio.run(build_channel_directory({Platform.TELEGRAM: object()}))

    assert "telegram" in directory["platforms"]
    mock_sessions.assert_any_call("telegram")


def test_connected_homeassistant_still_uses_session_discovery(tmp_path):
    cache_file = tmp_path / "channel_directory.json"

    with patch(
        "gateway.channel_directory._build_from_sessions",
        return_value={"channels": []},
    ) as mock_sessions, patch("gateway.channel_directory.DIRECTORY_PATH", cache_file):
        directory = asyncio.run(build_channel_directory({Platform.HOMEASSISTANT: object()}))

    assert "homeassistant" in directory["platforms"]
    mock_sessions.assert_any_call("homeassistant")


def test_matrix_uses_curated_lane_labels_and_suppresses_thread_ghosts(tmp_path):
    cache_file = tmp_path / "channel_directory.json"
    labels_file = tmp_path / "matrix_lane_labels.json"
    labels_file.write_text(
        """
        {
          "rooms": {
            "!room:example.org": {
              "name": "Ops Room",
              "threads": {"$thread": "Restart Notices"}
            }
          }
        }
        """.strip()
    )

    ghosts = [
        {
            "id": "!room:example.org:$old",
            "name": "dm / topic $old",
            "type": "group",
            "thread_id": "$old",
        }
    ]

    with patch("gateway.channel_directory.MATRIX_LANE_LABELS_PATH", labels_file, create=True), \
         patch("gateway.channel_directory._build_from_sessions", return_value=ghosts), \
         patch("gateway.channel_directory.DIRECTORY_PATH", cache_file):
        directory = asyncio.run(build_channel_directory({Platform.MATRIX: object()}))

    entries = directory["platforms"]["matrix"]
    assert entries == [
        {
            "id": "!room:example.org",
            "name": "Ops Room",
            "type": "group",
            "thread_id": None,
        },
        {
            "id": "!room:example.org:$thread",
            "name": "Ops Room / Restart Notices",
            "type": "group",
            "thread_id": "$thread",
        },
    ]


def test_empty_matrix_lane_labels_suppress_session_fallback(tmp_path):
    cache_file = tmp_path / "channel_directory.json"
    labels_file = tmp_path / "matrix_lane_labels.json"
    labels_file.write_text('{"rooms": {}}')
    ghosts = [
        {
            "id": "!room:example.org:$old",
            "name": "dm / topic $old",
            "type": "group",
            "thread_id": "$old",
        }
    ]

    with patch("gateway.channel_directory.MATRIX_LANE_LABELS_PATH", labels_file, create=True), \
         patch("gateway.channel_directory._build_from_sessions", return_value=ghosts) as mock_sessions, \
         patch("gateway.channel_directory.DIRECTORY_PATH", cache_file):
        directory = asyncio.run(build_channel_directory({Platform.MATRIX: object()}))

    assert directory["platforms"]["matrix"] == []
    mock_sessions.assert_not_called()


def test_matrix_lane_labels_work_with_string_adapter_keys(tmp_path):
    cache_file = tmp_path / "channel_directory.json"
    labels_file = tmp_path / "matrix_lane_labels.json"
    labels_file.write_text(
        '{"rooms": {"!room:example.org": {"name": "Ops Room", "threads": {}}}}'
    )

    with patch("gateway.channel_directory.MATRIX_LANE_LABELS_PATH", labels_file, create=True), \
         patch("gateway.channel_directory._build_from_sessions", return_value=[]), \
         patch("gateway.channel_directory.DIRECTORY_PATH", cache_file):
        directory = asyncio.run(build_channel_directory({"matrix": object()}))

    assert directory["platforms"]["matrix"] == [
        {
            "id": "!room:example.org",
            "name": "Ops Room",
            "type": "group",
            "thread_id": None,
        }
    ]
