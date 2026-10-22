"""Tests for /restart idempotency against update/message re-delivery.

Telegram: when PTB's graceful-shutdown ACK call (the final `get_updates` on exit)
fails with a network error, Telegram re-delivers the `/restart` message to the
new gateway process.  Adapters without update ids (Discord/Slack) retry the
identical webhook payload, so dedup keys on the message id instead (#121325).
Without a dedup guard, the new gateway would process `/restart` again and
immediately restart — a self-perpetuating loop.
"""
import json
import time
from unittest.mock import MagicMock

import pytest

import gateway.run as gateway_run
from gateway.platforms.event import MessageEvent, MessageType
from tests.gateway.restart_test_helpers import make_restart_runner, make_restart_source


def _make_restart_event(update_id: int | None = 100) -> MessageEvent:
    return MessageEvent(
        text="/restart",
        message_type=MessageType.TEXT,
        source=make_restart_source(),
        message_id="m1",
        platform_update_id=update_id,
    )


@pytest.mark.asyncio
async def test_redelivered_restart_with_older_update_id_is_ignored(tmp_path, monkeypatch):
    """update_id strictly LESS than the recorded one is also a redelivery."""
    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
    monkeypatch.delenv("INVOCATION_ID", raising=False)

    marker = tmp_path / ".restart_last_processed.json"
    marker.write_text(json.dumps({
        "platform": "telegram",
        "update_id": 12345,
        "requested_at": time.time() - 5,
    }))

    runner, _adapter = make_restart_runner()
    runner.request_restart = MagicMock()

    event = _make_restart_event(update_id=12344)  # older update — shouldn't happen,
                                                  # but if Telegram does re-deliver
                                                  # something older, treat as stale
    result = await runner._handle_restart_command(event)

    assert result == ""
    runner.request_restart.assert_not_called()


@pytest.mark.asyncio
async def test_stale_marker_older_than_5min_does_not_block(tmp_path, monkeypatch):
    """A marker older than the 5-minute window is ignored — fresh /restart proceeds."""
    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
    monkeypatch.delenv("INVOCATION_ID", raising=False)

    marker = tmp_path / ".restart_last_processed.json"
    marker.write_text(json.dumps({
        "platform": "telegram",
        "update_id": 12345,
        "requested_at": time.time() - 600,  # 10 minutes ago
    }))

    runner, _adapter = make_restart_runner()
    runner.request_restart = MagicMock(return_value=True)

    # Same update_id as the stale marker, but the marker is too old to trust
    event = _make_restart_event(update_id=12345)
    await runner._handle_restart_command(event)

    runner.request_restart.assert_called_once()


@pytest.mark.asyncio
async def test_event_without_update_id_bypasses_dedup(tmp_path, monkeypatch):
    """Events with no platform_update_id (non-Telegram, CLI fallback) aren't gated."""
    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
    monkeypatch.delenv("INVOCATION_ID", raising=False)

    marker = tmp_path / ".restart_last_processed.json"
    marker.write_text(json.dumps({
        "platform": "telegram",
        "update_id": 999999,
        "requested_at": time.time(),
    }))

    runner, _adapter = make_restart_runner()
    runner.request_restart = MagicMock(return_value=True)

    # No update_id — the dedup check should NOT kick in
    event = _make_restart_event(update_id=None)
    await runner._handle_restart_command(event)

    runner.request_restart.assert_called_once()


@pytest.mark.asyncio
async def test_different_platform_bypasses_dedup(tmp_path, monkeypatch):
    """Marker from Telegram doesn't block a /restart from another platform."""
    from gateway.config import Platform
    from gateway.session import SessionSource

    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
    monkeypatch.delenv("INVOCATION_ID", raising=False)

    marker = tmp_path / ".restart_last_processed.json"
    marker.write_text(json.dumps({
        "platform": "telegram",
        "update_id": 12345,
        "requested_at": time.time(),
    }))

    runner, _adapter = make_restart_runner()
    runner.request_restart = MagicMock(return_value=True)

    # /restart from Discord — not a redelivery candidate
    discord_source = SessionSource(
        platform=Platform.DISCORD,
        chat_id="discord-chan",
        chat_type="dm",
        user_id="u1",
    )
    event = MessageEvent(
        text="/restart",
        message_type=MessageType.TEXT,
        source=discord_source,
        message_id="m1",
        platform_update_id=12345,
    )
    await runner._handle_restart_command(event)

    runner.request_restart.assert_called_once()


def _make_discord_restart_event() -> MessageEvent:
    """Discord /restart: adapters other than Telegram don't stamp platform_update_id."""
    from gateway.config import Platform
    from gateway.session import SessionSource

    return MessageEvent(
        text="/restart",
        message_type=MessageType.TEXT,
        source=SessionSource(
            platform=Platform.DISCORD,
            chat_id="discord-chan",
            chat_type="dm",
            user_id="u1",
        ),
        message_id="1234567890123456789",
        platform_update_id=None,
    )


@pytest.mark.asyncio
async def test_discord_redelivered_restart_with_same_message_id_is_ignored(tmp_path, monkeypatch):
    """Adapters without update ids (Discord/Slack) dedup on the raw message id (#121325)."""
    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
    monkeypatch.delenv("INVOCATION_ID", raising=False)

    marker = tmp_path / ".restart_last_processed.json"
    marker.write_text(json.dumps({
        "platform": "discord",
        "message_id": "1234567890123456789",
        "requested_at": time.time(),
    }))

    runner, _adapter = make_restart_runner()
    runner.request_restart = MagicMock()

    event = _make_discord_restart_event()
    result = await runner._handle_restart_command(event)

    assert result == ""
    runner.request_restart.assert_not_called()


@pytest.mark.asyncio
async def test_discord_restart_with_different_message_id_is_honored(tmp_path, monkeypatch):
    """A fresh Discord /restart carries a new snowflake — not a redelivery."""
    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
    monkeypatch.delenv("INVOCATION_ID", raising=False)

    marker = tmp_path / ".restart_last_processed.json"
    marker.write_text(json.dumps({
        "platform": "discord",
        "message_id": "1234567890123456789",
        "requested_at": time.time(),
    }))

    runner, _adapter = make_restart_runner()
    runner.request_restart = MagicMock(return_value=True)

    event = _make_discord_restart_event()
    event = MessageEvent(
        text=event.text,
        message_type=event.message_type,
        source=event.source,
        message_id="9876543210987654321",  # different message — genuinely new /restart
        platform_update_id=None,
    )
    await runner._handle_restart_command(event)

    runner.request_restart.assert_called_once()


@pytest.mark.asyncio
async def test_cross_platform_message_id_marker_does_not_block(tmp_path, monkeypatch):
    """A Slack marker with the same message id never blocks a Discord /restart."""
    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
    monkeypatch.delenv("INVOCATION_ID", raising=False)

    marker = tmp_path / ".restart_last_processed.json"
    marker.write_text(json.dumps({
        "platform": "slack",
        "message_id": "1234567890123456789",
        "requested_at": time.time(),
    }))

    runner, _adapter = make_restart_runner()
    runner.request_restart = MagicMock(return_value=True)

    await runner._handle_restart_command(_make_discord_restart_event())

    runner.request_restart.assert_called_once()


@pytest.mark.asyncio
async def test_discord_stale_marker_older_than_5min_does_not_block(tmp_path, monkeypatch):
    """The 5-minute trust window applies to the message-id path too."""
    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
    monkeypatch.delenv("INVOCATION_ID", raising=False)

    marker = tmp_path / ".restart_last_processed.json"
    marker.write_text(json.dumps({
        "platform": "discord",
        "message_id": "1234567890123456789",
        "requested_at": time.time() - 600,  # 10 minutes ago
    }))

    runner, _adapter = make_restart_runner()
    runner.request_restart = MagicMock(return_value=True)

    event = _make_discord_restart_event()
    await runner._handle_restart_command(event)

    runner.request_restart.assert_called_once()


@pytest.mark.asyncio
async def test_dedup_marker_records_message_id_for_adapters_without_update_ids(tmp_path, monkeypatch):
    """The dedup marker records message_id for events without platform_update_id."""
    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
    monkeypatch.delenv("INVOCATION_ID", raising=False)

    runner, _adapter = make_restart_runner()
    runner.request_restart = MagicMock(return_value=True)

    event = _make_discord_restart_event()
    event = MessageEvent(
        text="/restart",
        message_type=MessageType.TEXT,
        source=event.source,
        message_id="dm-42",
        platform_update_id=None,
    )
    await runner._handle_restart_command(event)

    data = json.loads((tmp_path / ".restart_last_processed.json").read_text(encoding="utf-8"))
    assert data["platform"] == "discord"
    assert data["message_id"] == "dm-42"
    assert "update_id" not in data


@pytest.mark.asyncio
async def test_marker_missing_but_booted_from_restart_ignores_redelivery(tmp_path, monkeypatch):
    """Missing marker + just booted from a /restart + young process → treat as stale.

    Reproduces the infinite-loop scenario (issue #18528): the dedup marker went
    missing, so the update_id comparison can't run. Because this process booted
    from a chat-originated /restart and is still within the post-boot window,
    the redelivered /restart is suppressed instead of re-restarting the gateway.
    """
    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
    monkeypatch.delenv("INVOCATION_ID", raising=False)

    runner, _adapter = make_restart_runner()
    runner.request_restart = MagicMock(return_value=True)
    runner._booted_from_restart = True
    runner._startup_time = time.time()

    event = _make_restart_event(update_id=100)
    result = await runner._handle_restart_command(event)

    assert result == ""  # silently ignored
    runner.request_restart.assert_not_called()
    # One-shot: the flag is consumed so a later legitimate /restart is honored.
    assert runner._booted_from_restart is False


@pytest.mark.asyncio
async def test_int_message_id_coerced_to_str_matches_on_replay(tmp_path, monkeypatch):
    """An int message_id is str()-coerced in the marker and matches an int replay (#121325)."""
    from gateway.config import Platform
    from gateway.session import SessionSource

    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
    monkeypatch.delenv("INVOCATION_ID", raising=False)

    runner, _adapter = make_restart_runner()
    runner.request_restart = MagicMock(return_value=True)

    discord_source = SessionSource(
        platform=Platform.DISCORD,
        chat_id="discord-chan",
        chat_type="dm",
        user_id="u1",
    )
    event = MessageEvent(
        text="/restart",
        message_type=MessageType.TEXT,
        source=discord_source,
        message_id=9876543210123,  # int, not str — adapters may pass raw ints
        platform_update_id=None,
    )
    await runner._handle_restart_command(event)

    data = json.loads((tmp_path / ".restart_last_processed.json").read_text(encoding="utf-8"))
    assert data["message_id"] == "9876543210123"
    assert isinstance(data["message_id"], str)

    # Replay with the same int message_id — guard matches across int→str coercion.
    runner.request_restart = MagicMock()
    replay = MessageEvent(
        text="/restart",
        message_type=MessageType.TEXT,
        source=discord_source,
        message_id=9876543210123,
        platform_update_id=None,
    )
    result = await runner._handle_restart_command(replay)

    assert result == ""
    runner.request_restart.assert_not_called()
