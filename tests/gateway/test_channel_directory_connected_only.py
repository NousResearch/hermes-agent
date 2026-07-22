"""Session-based channel discovery must not resurrect disconnected platforms.

Surgical reapply of the directory portion of PR #25959: historical session
origins for platforms with no connected adapter must not become active
send_message targets."""

import asyncio
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from gateway.channel_directory import _build_telegram, build_channel_directory
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


def test_telegram_configured_topics_are_listed_without_seen_session(tmp_path):
    cache_file = tmp_path / "channel_directory.json"
    adapter = SimpleNamespace(
        config=SimpleNamespace(
            extra={
                "group_topics": [
                    {
                        "chat_id": "-100123",
                        "name": "Build Room",
                        "topics": [{"thread_id": 42, "name": "Release"}],
                    }
                ]
            }
        )
    )

    with patch("gateway.channel_directory._build_from_sessions", return_value=[]), \
         patch("gateway.channel_directory.DIRECTORY_PATH", cache_file):
        directory = asyncio.run(build_channel_directory({Platform.TELEGRAM: adapter}))

    assert directory["platforms"]["telegram"] == [
        {
            "id": "-100123:42",
            "name": "Build Room / Release",
            "type": "group",
            "thread_id": "42",
        }
    ]


@pytest.mark.parametrize(
    ("group_topics", "expected_id", "expected_name"),
    [
        (
            [
                "not-a-dict",
                {"chat_id": "-100list", "topics": "not-a-list"},
                {
                    "chat_id": "-100list",
                    "topics": ["not-a-dict", {"thread_id": 5, "name": "List Topic"}],
                },
            ],
            "-100list:5",
            "-100list / List Topic",
        ),
        (
            {
                "-100map": ["not-a-dict", {"thread_id": "7", "name": "Map Topic"}],
                "-100bad": "not-a-list",
            },
            "-100map:7",
            "-100map / Map Topic",
        ),
    ],
)
def test_telegram_group_topics_supported_shapes_skip_malformed_values(
    group_topics,
    expected_id,
    expected_name,
):
    adapter = SimpleNamespace(config=SimpleNamespace(extra={"group_topics": group_topics}))

    with patch("gateway.channel_directory._build_from_sessions", return_value=[]):
        channels = _build_telegram(adapter)

    assert channels == [
        {
            "id": expected_id,
            "name": expected_name,
            "type": "group",
            "thread_id": expected_id.rsplit(":", 1)[1],
        }
    ]
