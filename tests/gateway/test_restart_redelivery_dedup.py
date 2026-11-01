"""Tests for /restart idempotency guard against Telegram update re-delivery.

When PTB's graceful-shutdown ACK call (the final `get_updates` on exit) fails
with a network error, Telegram re-delivers the `/restart` message to the new
gateway process.  Without a dedup guard, the new gateway would process
`/restart` again and immediately restart — a self-perpetuating loop.
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
@pytest.mark.parametrize("shared_runtime", [False, True])
async def test_restart_update_ids_are_scoped_to_receiving_bot(tmp_path, monkeypatch, shared_runtime):
    """Independent bot counters cannot suppress one another, even with one routed runtime."""
    from pathlib import Path

    from gateway.config import Platform
    from gateway.session_identity import resolve_identity
    from tests.gateway.restart_test_helpers import RestartTestAdapter

    home = tmp_path / ".hermes"
    for name in ("alpha", "beta", "shared"):
        (home / "profiles" / name).mkdir(parents=True)
        (home / "profiles" / name / "config.yaml").write_text("{}\n")
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setattr(gateway_run, "_hermes_home", home)
    runner, _ = make_restart_runner()
    runner.config.multiplex_profiles = True
    runner._primary_profile_name = "default"
    runner.request_restart = MagicMock(return_value=True)
    runner._profile_adapters = {}
    events = {}
    for name, update_id in (("alpha", 900), ("beta", 100)):
        adapter = RestartTestAdapter()
        adapter.set_owner_profile(name)
        runner._profile_adapters[name] = {Platform.TELEGRAM: adapter}
        source = adapter.build_source(chat_id="123456", chat_type="dm", user_id="123456")
        source.profile = "shared" if shared_runtime else name
        identity = resolve_identity(source, runner=runner, adapter=adapter, transport_profile=name)
        assert identity.runtime_home == home / "profiles" / source.profile
        events[name] = MessageEvent(text="/restart", message_type=MessageType.TEXT,
                                   source=source, message_id="m1", platform_update_id=update_id)

    # Write the real marker, then emulate B's fresh command arriving before A's replay.
    await runner._handle_restart_command(events["alpha"])
    runner.request_restart.assert_called_once()
    runner.request_restart.reset_mock()
    runner._booted_from_restart = True
    assert runner._is_stale_restart_redelivery(events["beta"]) is False
    assert runner._booted_from_restart is True  # B must not consume A's boot signal.
    assert runner._is_stale_restart_redelivery(events["alpha"]) is True
    await runner._handle_restart_command(events["beta"])
    runner.request_restart.assert_called_once()
    runner.request_restart.reset_mock()
    # A -> B -> A: A's next id is independent of B's new marker too.
    events["alpha"].platform_update_id = 901
    await runner._handle_restart_command(events["alpha"])
    runner.request_restart.assert_called_once()
