"""Tests for Matrix mention-time room history backfill."""

import time
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from gateway.config import PlatformConfig


def _make_adapter():
    from plugins.platforms.matrix.adapter import MatrixAdapter, MatrixRoomIdentity

    config = PlatformConfig(
        enabled=True,
        token="syt_test_token",
        extra={
            "homeserver": "https://matrix.example.org",
            "user_id": "@hermes:example.org",
            "history_backfill": True,
            "history_backfill_limit": 50,
        },
    )
    adapter = MatrixAdapter(config)
    adapter._text_batch_delay_seconds = 0
    adapter.handle_message = AsyncMock()
    adapter._startup_ts = time.time() - 10
    adapter._client = object()
    adapter._resolve_room_identity = AsyncMock(
        return_value=MatrixRoomIdentity(
            room_id="!room1:example.org",
            room_name="Test Room",
            room_topic="",
            canonical_alias=None,
            server_name="example.org",
            joined_member_count=5,
            is_direct_account_data=False,
            display_name="Test Room",
            has_explicit_name=True,
            chat_type="group",
        )
    )
    adapter._get_display_name = AsyncMock(side_effect=lambda _room, user: user.split(":")[0][1:])
    return adapter


def _make_event(body, *, event_id="$evt1", mention_user_ids=None):
    content = {"body": body, "msgtype": "m.text"}
    if mention_user_ids is not None:
        content["m.mentions"] = {"user_ids": mention_user_ids}
    return SimpleNamespace(
        sender="@alice:example.org",
        event_id=event_id,
        room_id="!room1:example.org",
        timestamp=int(time.time() * 1000),
        content=content,
    )


@pytest.mark.asyncio
async def test_backfill_prepended_on_mention(monkeypatch):
    monkeypatch.delenv("MATRIX_FREE_RESPONSE_ROOMS", raising=False)
    monkeypatch.setenv("MATRIX_AUTO_THREAD", "false")

    adapter = _make_adapter()
    adapter._fetch_room_context = AsyncMock(
        return_value="[Recent room messages]\n[alice] earlier context"
    )

    await adapter._on_room_message(_make_event("@hermes:example.org hello"))

    adapter._fetch_room_context.assert_awaited_once_with("!room1:example.org", "$evt1")
    event = adapter.handle_message.await_args.args[0]
    assert event.text == "hello"
    assert event.channel_context == "[Recent room messages]\n[alice] earlier context"


@pytest.mark.asyncio
async def test_backfill_not_called_when_unmentioned_dropped(monkeypatch):
    monkeypatch.delenv("MATRIX_FREE_RESPONSE_ROOMS", raising=False)
    monkeypatch.setenv("MATRIX_AUTO_THREAD", "false")

    adapter = _make_adapter()
    adapter._fetch_room_context = AsyncMock(return_value="should not run")

    await adapter._on_room_message(_make_event("hello everyone"))

    adapter._fetch_room_context.assert_not_awaited()
    adapter.handle_message.assert_not_awaited()


@pytest.mark.asyncio
async def test_fetch_room_context_honors_limit():
    adapter = _make_adapter()
    adapter.config.extra["history_backfill_limit"] = 2
    adapter.fetch_history = AsyncMock(
        return_value=[
            {"event_id": "$3", "sender": "@alice:example.org", "msgtype": "m.text", "body": "c"},
            {"event_id": "$2", "sender": "@bob:example.org", "msgtype": "m.text", "body": "b"},
            {"event_id": "$1", "sender": "@carol:example.org", "msgtype": "m.text", "body": "a"},
        ]
    )

    result = await adapter._fetch_room_context("!room1:example.org", "$trigger")

    adapter.fetch_history.assert_awaited_once_with("!room1:example.org", limit=2)
    assert result == "[Recent room messages]\n[carol] a\n[bob] b\n[alice] c"


@pytest.mark.asyncio
async def test_fetch_room_context_stops_at_bot_message():
    adapter = _make_adapter()
    adapter.fetch_history = AsyncMock(
        return_value=[
            {"event_id": "$3", "sender": "@alice:example.org", "msgtype": "m.text", "body": "new"},
            {"event_id": "$2", "sender": "@hermes:example.org", "msgtype": "m.text", "body": "bot"},
            {"event_id": "$1", "sender": "@bob:example.org", "msgtype": "m.text", "body": "old"},
        ]
    )

    result = await adapter._fetch_room_context("!room1:example.org", "$trigger")

    assert result == "[Recent room messages]\n[alice] new"


@pytest.mark.asyncio
async def test_fetch_room_context_missing_client_returns_empty():
    adapter = _make_adapter()
    adapter._client = None

    result = await adapter._fetch_room_context("!room1:example.org", "$trigger")

    assert result == ""


@pytest.mark.asyncio
async def test_fetch_room_context_skips_empty_e2ee_bodies():
    adapter = _make_adapter()
    adapter.fetch_history = AsyncMock(
        return_value=[
            {"event_id": "$2", "sender": "@alice:example.org", "msgtype": "m.text", "body": ""},
            {"event_id": "$1", "sender": "@bob:example.org", "msgtype": "m.text", "body": "visible"},
        ]
    )

    result = await adapter._fetch_room_context("!room1:example.org", "$trigger")

    assert result == "[Recent room messages]\n[bob] visible"


class TestMatrixHistoryBackfillConfigBridge:
    def test_yaml_bridge_sets_history_backfill_env(self, monkeypatch, tmp_path):
        monkeypatch.delenv("MATRIX_HISTORY_BACKFILL", raising=False)
        monkeypatch.delenv("MATRIX_HISTORY_BACKFILL_LIMIT", raising=False)

        import os
        import yaml

        config_file = tmp_path / "config.yaml"
        config_file.write_text(
            yaml.dump(
                {
                    "matrix": {
                        "history_backfill": False,
                        "history_backfill_limit": 25,
                    }
                }
            )
        )

        yaml_cfg = yaml.safe_load(config_file.read_text())
        matrix_cfg = yaml_cfg.get("matrix", {})
        if "history_backfill" in matrix_cfg and not os.getenv("MATRIX_HISTORY_BACKFILL"):
            monkeypatch.setenv(
                "MATRIX_HISTORY_BACKFILL", str(matrix_cfg["history_backfill"]).lower()
            )
        hbl = matrix_cfg.get("history_backfill_limit")
        if hbl is not None and not os.getenv("MATRIX_HISTORY_BACKFILL_LIMIT"):
            monkeypatch.setenv("MATRIX_HISTORY_BACKFILL_LIMIT", str(hbl))

        assert os.getenv("MATRIX_HISTORY_BACKFILL") == "false"
        assert os.getenv("MATRIX_HISTORY_BACKFILL_LIMIT") == "25"
