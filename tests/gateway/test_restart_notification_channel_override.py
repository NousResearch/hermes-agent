"""Per-platform lifecycle-override channel (PlatformConfig.gateway_restart_notification_channel).

When a platform sets ``gateway_restart_notification_channel``, ALL lifecycle
notifications for that platform — shutdown/restart drain (both the active-session
pings and the home-channel broadcast), the post-restart "restarted" ping, and the
startup "online" ping — are routed to that single chat instead of the active
sessions and the home channel.

These are behavioral tests against the real notification paths (not snapshots),
mirroring the style in ``test_restart_drain.py`` / ``test_restart_notification.py``.
"""

import json

import pytest

import gateway.run as gateway_run
from gateway.config import HomeChannel, Platform, PlatformConfig
from gateway.platforms.base import BasePlatformAdapter, SendResult
from tests.gateway.restart_test_helpers import (
    RestartTestAdapter,
    make_restart_runner,
    make_restart_source,
)

OVERRIDE_CHANNEL = "C0OPS999"


class _MatrixLikeAdapter(RestartTestAdapter):
    """Records sends exactly like RestartTestAdapter; used for a Matrix room id."""


class _FailingOverrideAdapter(RestartTestAdapter):
    """Returns success=False for the override target, success=True otherwise.

    Lets us prove that a failed override send falls back to the normal
    per-session / home-channel delivery instead of being silently claimed.
    """

    def __init__(self, failing_target: str):
        super().__init__()
        self._failing_target = failing_target

    async def send(self, chat_id, content, reply_to=None, metadata=None):
        self.sent.append(content)
        self.sent_calls.append((chat_id, content, metadata))
        if chat_id == self._failing_target:
            return SendResult(success=False, error="simulated override failure")
        return SendResult(success=True, message_id="1")


# ── shutdown: active-session ping routed to override ─────────────────────


@pytest.mark.asyncio
async def test_shutdown_active_session_routed_to_override_channel():
    """With an override configured, an active session's shutdown ping lands
    in the override channel ONLY — not the session's own chat."""
    runner, adapter = make_restart_runner()
    runner.config.platforms[Platform.TELEGRAM].gateway_restart_notification_channel = OVERRIDE_CHANNEL
    session_key = "agent:main:telegram:dm:999"
    runner._running_agents[session_key] = object()

    await runner._notify_active_sessions_of_shutdown()

    targets = [c[0] for c in adapter.sent_calls]
    assert targets == [OVERRIDE_CHANNEL], targets
    assert "999" not in targets


# ── shutdown: home-channel broadcast routed to override ──────────────────


@pytest.mark.asyncio
async def test_shutdown_home_channel_routed_to_override_channel():
    """With an override configured and no active sessions, the home-channel
    shutdown broadcast is redirected to the override channel."""
    runner, adapter = make_restart_runner()
    runner.config.platforms[Platform.TELEGRAM].gateway_restart_notification_channel = OVERRIDE_CHANNEL
    runner.config.platforms[Platform.TELEGRAM].home_channel = HomeChannel(
        platform=Platform.TELEGRAM,
        chat_id="home-42",
        name="Ops Home",
    )

    await runner._notify_active_sessions_of_shutdown()

    targets = [c[0] for c in adapter.sent_calls]
    assert targets == [OVERRIDE_CHANNEL], targets
    assert "home-42" not in targets


# ── ask 1: success=False falls back, never silently claims delivery ──────


@pytest.mark.asyncio
async def test_override_send_failure_falls_back_to_active_session():
    """If the override send returns success=False, the platform is NOT marked
    delivered — the per-session ping still fires as a fallback."""
    adapter = _FailingOverrideAdapter(failing_target=OVERRIDE_CHANNEL)
    runner, adapter = make_restart_runner(adapter=adapter)
    runner.config.platforms[Platform.TELEGRAM].gateway_restart_notification_channel = OVERRIDE_CHANNEL
    session_key = "agent:main:telegram:dm:999"
    runner._running_agents[session_key] = object()

    await runner._notify_active_sessions_of_shutdown()

    targets = [c[0] for c in adapter.sent_calls]
    # Override attempted (and failed), THEN the session chat received the
    # fallback ping — the failure was not silently swallowed as "delivered".
    assert OVERRIDE_CHANNEL in targets
    assert "999" in targets


# ── ask 4: Matrix room id is used verbatim, never split on ':' ───────────


@pytest.mark.asyncio
async def test_override_channel_matrix_room_id_not_split_on_colon():
    """A Matrix-style room id ("!room123:example.org") must reach the adapter
    intact — the override value is the whole chat id, never split on ':'."""
    matrix_room = "!room123:example.org"
    adapter = _MatrixLikeAdapter()
    runner, _telegram = make_restart_runner()
    runner.config.platforms[Platform.MATRIX] = PlatformConfig(
        enabled=True,
        token="***",
        gateway_restart_notification_channel=matrix_room,
    )
    runner.adapters = {Platform.MATRIX: adapter}

    await runner._notify_active_sessions_of_shutdown()

    targets = [c[0] for c in adapter.sent_calls]
    assert targets == [matrix_room], targets


# ── startup: "online" ping routed to override ────────────────────────────


@pytest.mark.asyncio
async def test_startup_notification_routed_to_override_channel():
    """The "gateway online" startup ping goes to the override channel instead
    of the configured home channel when an override is set."""
    runner, adapter = make_restart_runner()
    runner.config.platforms[Platform.TELEGRAM].gateway_restart_notification_channel = OVERRIDE_CHANNEL
    runner.config.platforms[Platform.TELEGRAM].home_channel = HomeChannel(
        platform=Platform.TELEGRAM,
        chat_id="home-42",
        name="Ops Home",
    )

    await runner._send_home_channel_startup_notifications()

    assert len(adapter.sent_calls) == 1
    chat_id, content, _ = adapter.sent_calls[0]
    assert chat_id == OVERRIDE_CHANNEL
    assert "online" in content.lower()


# ── restart: "restarted" ping routed to override ─────────────────────────


@pytest.mark.asyncio
async def test_restart_notification_routed_to_override_channel(tmp_path, monkeypatch):
    """The post-restart "gateway restarted" ping goes to the override channel
    instead of the chat that ran /restart."""
    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
    (tmp_path / ".restart_notify.json").write_text(
        json.dumps({"platform": "telegram", "chat_id": "42", "chat_type": "dm"})
    )

    runner, adapter = make_restart_runner()
    runner.config.platforms[Platform.TELEGRAM].gateway_restart_notification_channel = OVERRIDE_CHANNEL

    await runner._send_restart_notification()

    targets = [c[0] for c in adapter.sent_calls]
    assert targets == [OVERRIDE_CHANNEL], targets
    assert "42" not in targets


# ── suppression flag still wins over the override ────────────────────────


@pytest.mark.asyncio
async def test_override_suppressed_when_restart_notification_flag_false():
    """gateway_restart_notification=False fully suppresses lifecycle pings,
    even when an override channel is configured."""
    runner, adapter = make_restart_runner()
    runner.config.platforms[Platform.TELEGRAM].gateway_restart_notification = False
    runner.config.platforms[Platform.TELEGRAM].gateway_restart_notification_channel = OVERRIDE_CHANNEL
    session_key = "agent:main:telegram:dm:999"
    runner._running_agents[session_key] = object()

    await runner._notify_active_sessions_of_shutdown()

    assert adapter.sent_calls == []
