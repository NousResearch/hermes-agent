"""Tests for update-aware restart messaging.

A restart triggered by ``hermes update`` / ``/update`` is expected, so it must
not be announced with the same ⚠️ wording a crash restart uses (issue: users
reading the home channel could not tell a planned update from the gateway
falling over).
"""

import os
import time
from unittest.mock import AsyncMock, MagicMock

import pytest

import gateway.run as gateway_run
from gateway.config import HomeChannel, Platform
from gateway.platforms.base import SendResult
from gateway.session import build_session_key
from tests.gateway.restart_test_helpers import (
    make_restart_runner,
    make_restart_source,
)

_PLANNED = "⬆️ Updating Hermes"
_GENERIC = "⚠️ Gateway restarting"


# ── marker detection ───────────────────────────────────────────────────────


def test_planned_update_marker_absent(tmp_path, monkeypatch):
    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)

    assert gateway_run._planned_update_marker() is None


@pytest.mark.parametrize(
    "name",
    [
        gateway_run._UPDATE_MARKER_IN_PROGRESS,
        gateway_run._UPDATE_MARKER_CHAT_PENDING,
    ],
)
def test_planned_update_marker_detects_either_writer(tmp_path, monkeypatch, name):
    """Both the CLI (``hermes update``) and in-chat ``/update`` count."""
    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
    (tmp_path / name).write_text("{}")

    marker = gateway_run._planned_update_marker()
    assert marker is not None and marker.name == name


def test_planned_update_marker_ignores_stale_marker(tmp_path, monkeypatch):
    """An abandoned marker must not relabel a genuine crash restart later."""
    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
    stale = tmp_path / gateway_run._UPDATE_MARKER_IN_PROGRESS
    stale.write_text("{}")
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
async def test_restart_shutdown_notification_says_update_when_marker_present(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
    (tmp_path / gateway_run._UPDATE_MARKER_IN_PROGRESS).write_text("{}")
    runner, adapter = make_restart_runner()
    runner._restart_requested = True

    message = await _shutdown_message(runner, adapter)

    assert message.startswith(_PLANNED)
    assert _GENERIC not in message
    # The resume hint still has to survive the reword.
    assert "resume where you left off" in message


@pytest.mark.asyncio
async def test_restart_shutdown_notification_unchanged_without_marker(
    tmp_path, monkeypatch
):
    """Baseline: an unexplained restart keeps the warning wording."""
    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
    runner, adapter = make_restart_runner()
    runner._restart_requested = True

    message = await _shutdown_message(runner, adapter)

    assert message.startswith(_GENERIC)
    assert _PLANNED not in message


@pytest.mark.asyncio
async def test_shutdown_without_restart_ignores_update_marker(tmp_path, monkeypatch):
    """A terminal shutdown is not coming back — never call it an update."""
    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
    (tmp_path / gateway_run._UPDATE_MARKER_IN_PROGRESS).write_text("{}")
    runner, adapter = make_restart_runner()

    message = await _shutdown_message(runner, adapter)

    assert message == (
        "⚠️ Gateway shutting down — Your current task will be interrupted."
    )


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
    (tmp_path / gateway_run._UPDATE_MARKER_IN_PROGRESS).write_text("{}")
    runner, adapter = make_restart_runner()
    _with_home(runner)
    adapter.send = AsyncMock()

    await runner._send_home_channel_startup_notifications()

    adapter.send.assert_called_once_with(
        "home-42",
        "⬆️ Update complete — Hermes is back on the new version and ready.",
    )


@pytest.mark.asyncio
async def test_startup_notification_unchanged_without_marker(tmp_path, monkeypatch):
    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
    runner, adapter = make_restart_runner()
    _with_home(runner)
    adapter.send = AsyncMock()

    await runner._send_home_channel_startup_notifications()

    adapter.send.assert_called_once_with(
        "home-42",
        "♻️ Gateway online — Hermes is back and ready.",
    )
