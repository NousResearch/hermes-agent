"""``home_channel_startup_notification``: when the home channel hears "Gateway online" (#27870).

``restart_only`` (default) keeps the planned-restart-marker gate; ``always`` also notifies after an
unplanned boot (crash, OOM, ``Restart=on-failure``); ``off`` never does. ``gateway_restart_notification:
false`` silences everything. Every test runs the real boot pass (``_await_startup_boot_sends``), the
requester notice, the home-channel sender and the reconnect replay; only the adapters are fakes.
"""

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

import gateway.run as gateway_run
from gateway.config import GatewayConfig, HomeChannel, Platform, PlatformConfig
from gateway.platforms.base import SendResult

ONLINE_NOTICE = "♻️ Gateway online — Hermes is back and ready."
RESTARTED_NOTICE = "♻ Gateway restarted successfully. Your session continues."


def _adapter():
    return SimpleNamespace(
        send_path_degraded=False,
        send=AsyncMock(return_value=SendResult(success=True, message_id="unit-test-notice")),
    )


def _home_cfg(chat_id="home-chat", thread_id=None, **kwargs):
    return PlatformConfig(
        enabled=True,
        home_channel=HomeChannel(platform=Platform.TELEGRAM, chat_id=chat_id, thread_id=thread_id, name="Home"),
        **kwargs,
    )


@pytest.fixture
def boot(tmp_path, monkeypatch):
    """A runner on Telegram with a home channel and no markers on disk (a cold / crash boot)."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
    monkeypatch.setattr(gateway_run, "_startup_restore_drain_timeout_secs", lambda: 0)
    monkeypatch.setattr("gateway.channel_directory.build_channel_directory", AsyncMock())
    runner = object.__new__(gateway_run.GatewayRunner)
    runner.config = GatewayConfig(platforms={Platform.TELEGRAM: _home_cfg()}, sessions_dir=tmp_path / "sessions")
    runner.adapters = {}
    runner.delivery_router = SimpleNamespace(adapters=runner.adapters)
    runner._failed_platforms = {}
    runner._sync_voice_mode_state_to_adapter = Mock()
    runner._bind_voice_input_callback = Mock()
    runner._update_platform_runtime_status = Mock()
    runner._redeliver_failed_obligations_for_platform = AsyncMock()
    runner._schedule_resume_pending_sessions = Mock()
    runner._claim_pending_obligations = AsyncMock(return_value=[])
    runner._redeliver_claimed_obligations = AsyncMock(return_value=0)
    return runner, tmp_path


async def _boot(runner):
    await runner._await_startup_boot_sends(
        planned_restart_notification_pending=gateway_run._planned_restart_notification_pending()
    )


async def _reconnect(runner, adapter, platform=Platform.TELEGRAM):
    runner._failed_platforms[platform] = {}
    await runner._install_reconnected_adapter(platform, adapter)
    await asyncio.gather(*getattr(runner, "_background_tasks", ()))  # set once a replay was scheduled


def _sent_texts(adapter):
    return [call.args[1] for call in adapter.send.await_args_list]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("settings", "expected"),
    [
        ({}, []),
        ({"home_channel_startup_notification": "restart_only"}, []),
        ({"home_channel_startup_notification": "always"}, [ONLINE_NOTICE]),
        ({"home_channel_startup_notification": "off"}, []),
        ({"home_channel_startup_notification": "always", "gateway_restart_notification": False}, []),
    ],
    ids=["default", "restart_only", "always", "off", "master-switch-wins"],
)
async def test_unplanned_boot(boot, settings, expected):
    runner, _home = boot
    runner.config.platforms[Platform.TELEGRAM] = _home_cfg(**settings)
    adapter = _adapter()
    runner.adapters[Platform.TELEGRAM] = adapter

    await _boot(runner)
    # A later reconnect of the same process never repeats the boot's notice.
    await _reconnect(runner, adapter)

    assert _sent_texts(adapter) == expected


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("mode", "expected"),
    [("restart_only", [ONLINE_NOTICE]), ("always", [ONLINE_NOTICE]), ("off", [])],
)
async def test_planned_restart_boot(boot, mode, expected):
    """With the planned-restart marker, ``always`` is not a second notice and ``off`` is not owed."""
    runner, home = boot
    runner.config.platforms[Platform.TELEGRAM] = _home_cfg(home_channel_startup_notification=mode)
    marker = home / ".restart_pending.json"
    marker.write_text("{}", encoding="utf-8")
    adapter = _adapter()
    runner.adapters[Platform.TELEGRAM] = adapter

    await _boot(runner)
    await _reconnect(runner, adapter)

    assert _sent_texts(adapter) == expected
    assert not marker.exists(), "an `off` home is not owed a notice, so it cannot keep the marker alive"


@pytest.mark.asyncio
async def test_always_home_down_at_boot_is_notified_once_on_reconnect(boot):
    runner, _home = boot
    runner.config.platforms[Platform.TELEGRAM] = _home_cfg(home_channel_startup_notification="always")
    adapter = _adapter()

    await _boot(runner)
    adapter.send.assert_not_called()

    await _reconnect(runner, adapter)
    await _reconnect(runner, adapter)

    assert _sent_texts(adapter) == [ONLINE_NOTICE]


@pytest.mark.asyncio
@pytest.mark.parametrize("live_at_boot", [True, False], ids=["live-at-boot", "replayed-on-reconnect"])
@pytest.mark.parametrize("thread_id", [None, "42"], ids=["chat", "topic"])
async def test_restart_from_home_chat_sends_one_message(boot, live_at_boot, thread_id):
    """``/restart`` sent from the home chat/topic: the requester reply is the only message there."""
    runner, home = boot
    runner.config.platforms[Platform.TELEGRAM] = _home_cfg(
        chat_id="-100123", thread_id=thread_id, home_channel_startup_notification="always")
    # Chat /restart writes the requester marker only (no planned-restart marker).
    (home / ".restart_notify.json").write_text(
        json.dumps({"platform": "telegram", "chat_id": "-100123", "thread_id": thread_id}), encoding="utf-8")
    adapter = _adapter()
    if live_at_boot:
        runner.adapters[Platform.TELEGRAM] = adapter

    await _boot(runner)
    if not live_at_boot:
        adapter.send.assert_not_called()
    await _reconnect(runner, adapter)

    assert _sent_texts(adapter) == [RESTARTED_NOTICE]
    assert not (home / ".restart_notify.json").exists()
    assert not runner._unplanned_startup_notice_owed, "the home chat got its notice; nothing stays owed"


@pytest.mark.asyncio
async def test_restart_from_another_chat_still_notifies_always_home(boot):
    runner, home = boot
    runner.config.platforms[Platform.TELEGRAM] = _home_cfg(home_channel_startup_notification="always")
    (home / ".restart_notify.json").write_text(
        json.dumps({"platform": "telegram", "chat_id": "requester-dm"}), encoding="utf-8")
    adapter = _adapter()
    runner.adapters[Platform.TELEGRAM] = adapter

    await _boot(runner)

    assert sorted((call.args[0], call.args[1]) for call in adapter.send.await_args_list) == [
        ("home-chat", ONLINE_NOTICE), ("requester-dm", RESTARTED_NOTICE)]


@pytest.mark.asyncio
async def test_profiles_sharing_a_home_chat_get_one_notice(boot):
    runner, _home = boot
    runner.config.platforms[Platform.TELEGRAM] = _home_cfg(
        chat_id="-100shared", home_channel_startup_notification="always")
    runner._profile_configs = {"coder": GatewayConfig(platforms={
        Platform.TELEGRAM: _home_cfg(chat_id="-100shared", home_channel_startup_notification="always")})}
    launch, coder = _adapter(), _adapter()
    runner.adapters[Platform.TELEGRAM] = launch
    runner._profile_adapters = {"coder": {Platform.TELEGRAM: coder}}

    await _boot(runner)

    assert launch.send.await_count + coder.send.await_count == 1
    assert not runner._unplanned_startup_notice_owed
