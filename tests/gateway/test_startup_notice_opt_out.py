"""The startup notice switch is the existing per-platform flag, default on (#27870).

The ordinary-start notice deliberately adds no config key: it is gated by the same
``gateway_restart_notification`` flag that already covers the restart notice, which defaults to
``true`` (``gateway/config.py``). These pin both ends — opted out there is no send at all, and a
config that omits the key still announces — because the docs state exactly that pair.
"""

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

import gateway.run as gateway_run
from agent.i18n import t
from gateway.config import GatewayConfig, HomeChannel, Platform, PlatformConfig
from gateway.platforms.base import SendResult
from gateway.run import GatewayRunner

ONLINE_NOTICE = t("gateway.startup.online")
HOME = HomeChannel(platform=Platform.TELEGRAM, chat_id="home-1", name="home-1")


def _runner(tmp_path, monkeypatch, platform_cfg):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    # gateway.run snapshots the home at import; the marker writer reads it live.
    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
    config = GatewayConfig(platforms={Platform.TELEGRAM: platform_cfg})
    config.sessions_dir = tmp_path / "sessions"
    runner = object.__new__(GatewayRunner)
    runner.config = config
    adapter = SimpleNamespace(send=AsyncMock(return_value=SendResult(success=True, message_id="m1")))
    runner.adapters = {Platform.TELEGRAM: adapter}
    runner._profile_configs = {}
    runner._profile_adapters = {}
    runner._planned_restart_notice_lock = None
    # Not under test: the thread-target resolver reads live session state.
    runner._thread_metadata_for_target = lambda *a, **k: None
    transport = SimpleNamespace(
        send=AsyncMock(return_value=SendResult(success=True, message_id="m2")),
        adapter=adapter,
        is_relay=False,
    )
    # A live home channel, so "nothing was sent" is the switch at work and not a missing transport.
    monkeypatch.setattr(
        runner,
        "_served_home_channel_transports",
        lambda: iter([(None, Platform.TELEGRAM, platform_cfg, HOME, transport)]),
    )
    return runner, adapter


@pytest.mark.asyncio
async def test_the_opted_out_platform_is_not_notified(tmp_path, monkeypatch):
    cfg = PlatformConfig(enabled=True, gateway_restart_notification=False, home_channel=HOME)
    runner, adapter = _runner(tmp_path, monkeypatch, cfg)

    await runner._send_ordinary_start_notice()

    adapter.send.assert_not_awaited()
    assert not (tmp_path / ".restart_pending.json").exists(), "an opt-out must not leave a debt owed"
    assert json.loads((tmp_path / ".startup_notice.json").read_text(encoding="utf-8"))["sent_at"] > 0


@pytest.mark.asyncio
async def test_an_omitted_flag_defaults_to_on(tmp_path, monkeypatch):
    cfg = PlatformConfig.from_dict({"enabled": True, "home_channel": HOME.to_dict()})
    assert cfg.gateway_restart_notification is True, "the default the docs promise"
    runner, adapter = _runner(tmp_path, monkeypatch, cfg)

    await runner._send_ordinary_start_notice()

    adapter.send.assert_awaited_once()
    assert adapter.send.await_args.args[:2] == ("home-1", ONLINE_NOTICE)
