"""Matrix reply references and thread fallbacks."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig, _apply_env_overrides
from plugins.platforms.matrix.adapter import MatrixAdapter, _apply_yaml_config


@pytest.mark.asyncio
async def test_reply_modes_control_split_chunks_without_losing_thread_relations():
    room_id = "!room:example.org"
    expected = {
        "off": [None, None, None],
        "first": [
            {"m.in_reply_to": {"event_id": "$request"}},
            None,
            None,
        ],
        "all": [
            {"m.in_reply_to": {"event_id": "$request"}},
            {"m.in_reply_to": {"event_id": "$request"}},
            {"m.in_reply_to": {"event_id": "$request"}},
        ],
    }
    expected_thread = {
        "off": [
            {
                "rel_type": "m.thread", "event_id": "$root",
                "m.in_reply_to": {"event_id": "$known"}, "is_falling_back": True,
            },
        ] * 3,
        "first": [
            {
                "rel_type": "m.thread", "event_id": "$root",
                "m.in_reply_to": {"event_id": "$request"}, "is_falling_back": False,
            },
            *[
                {
                    "rel_type": "m.thread", "event_id": "$root",
                    "m.in_reply_to": {"event_id": "$known"}, "is_falling_back": True,
                },
            ] * 2,
        ],
        "all": [
            {
                "rel_type": "m.thread", "event_id": "$root",
                "m.in_reply_to": {"event_id": "$request"}, "is_falling_back": False,
            },
        ] * 3,
    }

    for mode in ("off", "first", "all"):
        config = PlatformConfig(
            enabled=True, token="syt_test", reply_to_mode=mode,
            extra={"homeserver": "https://matrix.example.org", "user_id": "@bot:example.org"},
        )
        adapter = MatrixAdapter(config)
        client = MagicMock()
        client.send_message_event = AsyncMock(return_value="$sent")
        adapter._client = client

        with patch.object(adapter, "truncate_message", return_value=["one", "two", "three"]):
            await adapter.send(room_id, "split answer", reply_to="$request")
            plain = [call.args[2].get("m.relates_to") for call in client.send_message_event.await_args_list]

            client.send_message_event.reset_mock()
            await adapter.send(
                room_id, "split thread answer", reply_to="$request",
                metadata={"thread_id": "$root", "matrix_thread_fallback_event_id": "$known"},
            )
            threaded = [call.args[2]["m.relates_to"] for call in client.send_message_event.await_args_list]

        assert (plain, threaded) == (expected[mode], expected_thread[mode])


def test_yaml_off_mode_keeps_a_valid_fallback_on_threaded_media(monkeypatch):
    monkeypatch.delenv("MATRIX_REPLY_TO_MODE", raising=False)
    seeded = _apply_yaml_config({}, {"reply_to_mode": False})
    config = GatewayConfig()
    config.platforms[Platform.MATRIX] = PlatformConfig(
        enabled=True, token="syt_test",
        extra={"homeserver": "https://matrix.example.org", "user_id": "@bot:example.org"},
    )
    _apply_env_overrides(config)
    adapter = MatrixAdapter(config.platforms[Platform.MATRIX])
    content = {"msgtype": "m.image", "body": "photo"}

    adapter._apply_relation_metadata(
        "!room:example.org", content,
        reply_to="$specific", metadata={"thread_id": "$root"},
    )

    assert (seeded, adapter._reply_to_mode, content) == (
        {"reply_to_mode": "off"},
        "off",
        {
            "msgtype": "m.image", "body": "photo",
            "m.relates_to": {
                "rel_type": "m.thread", "event_id": "$root",
                "m.in_reply_to": {"event_id": "$root"}, "is_falling_back": True,
            },
        },
    )
