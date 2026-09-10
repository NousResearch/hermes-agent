"""Tests for update-aware gateway lifecycle messaging.

A restart driven by ``hermes update`` / ``/update`` was asked for, so it must
not be announced with the same ⚠️ wording a crash restart uses — otherwise a
user watching the home channel reads a planned upgrade as the gateway falling
over, and learns to ignore the warning that matters when it really is a crash.
"""

import os
import time
from unittest.mock import AsyncMock, MagicMock

import pytest

import gateway.run as gateway_run
from gateway.config import HomeChannel, Platform
from gateway.platforms.base import SendResult
from gateway.session import build_session_key
from tests.gateway.restart_test_helpers import make_restart_runner, make_restart_source

_PLANNED_RESTART = "⬆️ Updating Hermes"
_PLANNED_ONLINE = "⬆️ Update complete — Hermes is back on the new version and ready."
_GENERIC_RESTART = "⚠️ Gateway restarting — Your current task will be interrupted. "
_GENERIC_ONLINE = "♻️ Gateway online — Hermes is back and ready."


# ── marker names stay pinned to their writers ──────────────────────────────


def test_fleet_restart_marker_name_matches_its_writer():
    """The CLI owns this filename; a rename there must not silently mute us."""
    from hermes_cli.update_cmd_fleet import _FLEET_RESTART_PENDING_NAME

    assert gateway_run._UPDATE_MARKER_FLEET_RESTART == _FLEET_RESTART_PENDING_NAME


def test_chat_update_marker_name_matches_its_writer():
    from gateway.run import GatewayRunner

    assert GatewayRunner._update_paths().pending.name == gateway_run._UPDATE_MARKER_CHAT_PENDING


# ── marker detection ───────────────────────────────────────────────────────


def test_planned_update_marker_absent(tmp_path, monkeypatch):
    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)

    assert gateway_run._planned_update_marker() is None


@pytest.mark.parametrize(
    "name",
    [
        gateway_run._UPDATE_MARKER_FLEET_RESTART,
        gateway_run._UPDATE_MARKER_CHAT_PENDING,
    ],
)
def test_planned_update_marker_detects_either_writer(tmp_path, monkeypatch, name):
    """Both ``hermes update`` (CLI fleet restart) and in-chat ``/update`` count."""
    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
    (tmp_path / name).write_text("started=1\n")

    marker = gateway_run._planned_update_marker()
    assert marker is not None and marker.name == name


def test_planned_update_marker_ignores_stale_marker(tmp_path, monkeypatch):
    """An abandoned marker must not relabel a genuine crash restart later."""
    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
    stale = tmp_path / gateway_run._UPDATE_MARKER_FLEET_RESTART
    stale.write_text("started=1\n")
    old = time.time() - gateway_run._UPDATE_MARKER_MAX_AGE_SECONDS - 60
    os.utime(stale, (old, old))

    assert gateway_run._planned_update_marker() is None


# ── shutdown notification ──────────────────────────────────────────────────


async def _shutdown_message(runner, adapter) -> str:
    source = make_restart_source(chat_id="chat-1")
    session_key = build_session_key(source)
    runner._running_agents[session_key] = object()
    runner.session_store._entries[session_key] = MagicMock(origin=source)
    adapter.send = AsyncMock(return_value=SendResult(success=True, message_id="m"))

    await runner._notify_active_sessions_of_shutdown()

    return adapter.send.await_args.args[1]


@pytest.mark.asyncio
async def test_restart_notification_says_update_when_marker_present(tmp_path, monkeypatch):
    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
    (tmp_path / gateway_run._UPDATE_MARKER_FLEET_RESTART).write_text("started=1\n")
    runner, adapter = make_restart_runner()
    runner._restart_requested = True

    message = await _shutdown_message(runner, adapter)

    assert message.startswith(_PLANNED_RESTART)
    assert "⚠️" not in message
    # The resume hint has to survive the reword — it is the actionable half.
    assert "resume where you left off" in message


@pytest.mark.asyncio
async def test_restart_notification_unchanged_without_marker(tmp_path, monkeypatch):
    """Baseline: an unexplained restart keeps the warning wording."""
    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
    runner, adapter = make_restart_runner()
    runner._restart_requested = True

    message = await _shutdown_message(runner, adapter)

    assert message.startswith(_GENERIC_RESTART)
    assert _PLANNED_RESTART not in message


@pytest.mark.asyncio
async def test_shutdown_without_restart_ignores_update_marker(tmp_path, monkeypatch):
    """A terminal shutdown is not coming back — never call it an update."""
    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
    (tmp_path / gateway_run._UPDATE_MARKER_FLEET_RESTART).write_text("started=1\n")
    runner, adapter = make_restart_runner()

    message = await _shutdown_message(runner, adapter)

    assert message == "⚠️ Gateway shutting down — Your current task will be interrupted."


# ── startup notification ───────────────────────────────────────────────────


def _with_home(runner):
    runner.config.platforms[Platform.TELEGRAM].home_channel = HomeChannel(
        platform=Platform.TELEGRAM,
        chat_id="home-42",
        name="Ops Home",
    )


@pytest.mark.asyncio
async def test_startup_notification_reports_update_completion(tmp_path, monkeypatch):
    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
    (tmp_path / gateway_run._UPDATE_MARKER_FLEET_RESTART).write_text("started=1\n")
    runner, adapter = make_restart_runner()
    _with_home(runner)
    adapter.send = AsyncMock(return_value=SendResult(success=True, message_id="home"))

    delivered = await runner._send_home_channel_startup_notifications()

    assert delivered == {("telegram", "home-42", None)}
    assert adapter.send.await_args.args[1] == _PLANNED_ONLINE


@pytest.mark.asyncio
async def test_startup_notification_unchanged_without_marker(tmp_path, monkeypatch):
    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
    runner, adapter = make_restart_runner()
    _with_home(runner)
    adapter.send = AsyncMock(return_value=SendResult(success=True, message_id="home"))

    delivered = await runner._send_home_channel_startup_notifications()

    assert delivered == {("telegram", "home-42", None)}
    assert adapter.send.await_args.args[1] == _GENERIC_ONLINE
