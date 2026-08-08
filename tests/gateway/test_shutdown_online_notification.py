"""Tests for the back-online notification after a plain gateway shutdown.

A non-restart shutdown pings active chats and home channels with
"⚠️ Gateway shutting down" but nothing announced the comeback: the
home-channel startup broadcast only runs for planned restarts
(.restart_pending.json) and .restart_notify.json only covers chat /restart.
These tests cover the marker persisted at shutdown and the startup pass
that pings the same chats once the gateway is back online.
"""

import json
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

ONLINE_MESSAGE = "♻️ Gateway online — Hermes is back and ready."


def _write_marker(tmp_path, targets):
    (tmp_path / ".shutdown_notified.json").write_text(
        json.dumps({"written_at": 0, "targets": targets}), encoding="utf-8"
    )


# ── shutdown side: marker persistence ────────────────────────────────────


@pytest.mark.asyncio
async def test_plain_shutdown_persists_notified_targets(tmp_path, monkeypatch):
    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)

    runner, adapter = make_restart_runner()
    source = make_restart_source(chat_id="active-42", chat_type="group", thread_id="topic-7")
    session_key = build_session_key(source)
    runner._running_agents[session_key] = object()
    runner.session_store._entries[session_key] = MagicMock(origin=None)
    runner._cache_session_source(session_key, source)
    runner.config.platforms[Platform.TELEGRAM].home_channel = HomeChannel(
        platform=Platform.TELEGRAM,
        chat_id="home-42",
        name="Ops Home",
    )
    adapter.send = AsyncMock(return_value=SendResult(success=True, message_id="1"))

    await runner._notify_active_sessions_of_shutdown()

    marker = tmp_path / ".shutdown_notified.json"
    assert marker.exists()
    data = json.loads(marker.read_text(encoding="utf-8"))
    got = {(t["platform"], t["chat_id"], t["thread_id"]) for t in data["targets"]}
    assert got == {
        ("telegram", "active-42", "topic-7"),
        ("telegram", "home-42", None),
    }


@pytest.mark.asyncio
async def test_restart_shutdown_does_not_persist_marker(tmp_path, monkeypatch):
    """Restart flows have their own comeback notifications — no marker."""
    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)

    runner, adapter = make_restart_runner()
    runner._restart_requested = True
    runner.config.platforms[Platform.TELEGRAM].home_channel = HomeChannel(
        platform=Platform.TELEGRAM,
        chat_id="home-42",
        name="Ops Home",
    )
    adapter.send = AsyncMock(return_value=SendResult(success=True, message_id="1"))

    await runner._notify_active_sessions_of_shutdown()

    adapter.send.assert_awaited()  # "restarting" ping still goes out
    assert not (tmp_path / ".shutdown_notified.json").exists()


@pytest.mark.asyncio
async def test_muted_shutdown_persists_no_marker(tmp_path, monkeypatch):
    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)

    runner, adapter = make_restart_runner()
    runner.config.platforms[Platform.TELEGRAM].gateway_restart_notification = False
    runner.config.platforms[Platform.TELEGRAM].home_channel = HomeChannel(
        platform=Platform.TELEGRAM,
        chat_id="home-42",
        name="Ops Home",
    )
    adapter.send = AsyncMock()

    await runner._notify_active_sessions_of_shutdown()

    adapter.send.assert_not_awaited()
    assert not (tmp_path / ".shutdown_notified.json").exists()


# ── startup side: back-online delivery ───────────────────────────────────


@pytest.mark.asyncio
async def test_startup_pings_persisted_targets_and_clears_marker(tmp_path, monkeypatch):
    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)

    runner, adapter = make_restart_runner()
    _write_marker(
        tmp_path,
        [
            {"platform": "telegram", "chat_id": "active-42", "thread_id": "topic-7"},
            {"platform": "telegram", "chat_id": "home-42", "thread_id": None},
        ],
    )

    await runner._send_shutdown_recovery_notifications()

    assert not (tmp_path / ".shutdown_notified.json").exists()
    assert [(chat_id, content) for chat_id, content, _meta in adapter.sent_calls] == [
        ("active-42", ONLINE_MESSAGE),
        ("home-42", ONLINE_MESSAGE),
    ]
    thread_metadata = adapter.sent_calls[0][2]
    assert thread_metadata is not None and thread_metadata.get("thread_id") == "topic-7"
    assert adapter.sent_calls[1][2] is None


@pytest.mark.asyncio
async def test_startup_skips_targets_already_notified(tmp_path, monkeypatch):
    """A /restart or planned-restart notification wins over the marker ping."""
    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)

    runner, adapter = make_restart_runner()
    adapter.send = AsyncMock(return_value=SendResult(success=True, message_id="1"))
    _write_marker(
        tmp_path,
        [{"platform": "telegram", "chat_id": "home-42", "thread_id": None}],
    )

    await runner._send_shutdown_recovery_notifications(
        skip_targets={("telegram", "home-42", None)}
    )

    adapter.send.assert_not_awaited()
    assert not (tmp_path / ".shutdown_notified.json").exists()


@pytest.mark.asyncio
async def test_startup_respects_notification_flag(tmp_path, monkeypatch):
    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)

    runner, adapter = make_restart_runner()
    runner.config.platforms[Platform.TELEGRAM].gateway_restart_notification = False
    adapter.send = AsyncMock()
    _write_marker(
        tmp_path,
        [{"platform": "telegram", "chat_id": "home-42", "thread_id": None}],
    )

    await runner._send_shutdown_recovery_notifications()

    adapter.send.assert_not_awaited()
    assert not (tmp_path / ".shutdown_notified.json").exists()


@pytest.mark.asyncio
async def test_startup_noop_without_marker(tmp_path, monkeypatch):
    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)

    runner, adapter = make_restart_runner()
    adapter.send = AsyncMock()

    await runner._send_shutdown_recovery_notifications()

    adapter.send.assert_not_awaited()


@pytest.mark.asyncio
async def test_startup_survives_corrupt_marker(tmp_path, monkeypatch):
    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)

    runner, adapter = make_restart_runner()
    adapter.send = AsyncMock()
    (tmp_path / ".shutdown_notified.json").write_text("not json", encoding="utf-8")

    await runner._send_shutdown_recovery_notifications()

    adapter.send.assert_not_awaited()
    assert not (tmp_path / ".shutdown_notified.json").exists()
