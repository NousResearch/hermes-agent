"""Telegram /restart dedup must not span the profiles of a multiplexed gateway.

Telegram numbers ``update_id`` per bot, but one gateway process serves several
profiles — each with its own bot — and writes a single ``.restart_last_processed.json``.
The id streams of two bots are unrelated, so comparing raw ids across profiles can
swallow a legitimate restart: a secondary bot's ``/restart`` looks like a redelivery of
the primary's last one, and vice versa.

The fix records the owning profile in the dedup marker and compares profiles before
ids; markers written before profiles were persisted belong to the primary profile,
which was the only writer before multiplexing.
"""
import json
import time
from unittest.mock import MagicMock

import pytest

import gateway.run as gateway_run
from gateway.platforms.event import MessageEvent, MessageType
from tests.gateway.restart_test_helpers import make_restart_runner, make_restart_source


def _restart_event(update_id: int, profile=None) -> MessageEvent:
    source = make_restart_source()
    source.profile = profile
    return MessageEvent(
        text="/restart",
        message_type=MessageType.TEXT,
        source=source,
        message_id="m1",
        platform_update_id=update_id,
    )


def _marker(tmp_path, **overrides) -> None:
    data = {"platform": "telegram", "update_id": 500, "requested_at": time.time() - 5}
    data.update(overrides)
    (tmp_path / ".restart_last_processed.json").write_text(json.dumps(data))


def _runner(primary="default"):
    runner, _adapter = make_restart_runner()
    runner._primary_profile_name = primary
    runner.request_restart = MagicMock(return_value=True)
    return runner


@pytest.mark.asyncio
async def test_older_update_id_from_another_profile_is_not_a_redelivery(tmp_path, monkeypatch):
    """A secondary bot's /restart must survive a bigger id recorded by the primary bot."""
    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
    monkeypatch.delenv("INVOCATION_ID", raising=False)
    _marker(tmp_path, profile="default", update_id=500)
    runner = _runner()

    await runner._handle_restart_command(_restart_event(update_id=400, profile="marketing"))

    runner.request_restart.assert_called_once()


@pytest.mark.asyncio
async def test_same_profile_still_dedups_redelivery(tmp_path, monkeypatch):
    """Inside one bot the id comparison still suppresses a re-delivered /restart."""
    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
    monkeypatch.delenv("INVOCATION_ID", raising=False)
    _marker(tmp_path, profile="default", update_id=500)
    runner = _runner()

    result = await runner._handle_restart_command(_restart_event(update_id=400, profile="default"))

    assert result == ""
    runner.request_restart.assert_not_called()


@pytest.mark.asyncio
async def test_legacy_marker_without_profile_belongs_to_primary(tmp_path, monkeypatch):
    """Markers written before profiles were persisted keep protecting the primary bot."""
    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
    monkeypatch.delenv("INVOCATION_ID", raising=False)
    _marker(tmp_path, update_id=500)
    runner = _runner()

    result = await runner._handle_restart_command(_restart_event(update_id=400, profile="default"))

    assert result == ""
    runner.request_restart.assert_not_called()


@pytest.mark.asyncio
async def test_legacy_marker_does_not_block_secondary_bot(tmp_path, monkeypatch):
    """A legacy marker must not swallow a secondary bot's first /restart."""
    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
    monkeypatch.delenv("INVOCATION_ID", raising=False)
    _marker(tmp_path, update_id=500)
    runner = _runner()

    await runner._handle_restart_command(_restart_event(update_id=400, profile="marketing"))

    runner.request_restart.assert_called_once()


@pytest.mark.asyncio
async def test_session_key_namespace_recovers_owning_profile(tmp_path, monkeypatch):
    """A profile-less marker still names its profile through the session-key namespace."""
    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
    monkeypatch.delenv("INVOCATION_ID", raising=False)
    _marker(tmp_path, update_id=500, session_key="agent:marketing:telegram:dm:42")
    runner = _runner()

    result = await runner._handle_restart_command(_restart_event(update_id=400, profile="marketing"))

    assert result == ""
    runner.request_restart.assert_not_called()


@pytest.mark.asyncio
async def test_dedup_marker_records_requesting_profile(tmp_path, monkeypatch):
    """The marker written on /restart carries the profile so the next boot can scope ids."""
    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
    monkeypatch.delenv("INVOCATION_ID", raising=False)
    runner = _runner()

    await runner._handle_restart_command(_restart_event(update_id=777, profile="marketing"))

    marker = json.loads((tmp_path / ".restart_last_processed.json").read_text())
    assert marker["profile"] == "marketing"
    assert marker["update_id"] == 777
