"""Restart notices refused while Telegram's send path is still degraded are kept and replayed (#66589).

Telegram refuses sends with ``send_path_degraded`` until the polling generation's first getUpdates
succeeds. The boot pass used to consume ``.restart_notify.json`` either way, so the /restart
confirmation was lost, and the planned-restart "online" notice was replayed only on an adapter
REPLACEMENT, never when the same adapter healed in place. Drives the real boot pass, the real
``TelegramAdapter`` send gate and its in-place recovery edge.
"""
import asyncio
import json
import os
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, Mock

import pytest

import gateway.run as gateway_run
from gateway.config import GatewayConfig, HomeChannel, Platform, PlatformConfig
from plugins.platforms.telegram import telegram_send_path_recovery as recovery
from plugins.platforms.telegram.adapter import TelegramAdapter

RESTARTED = "Gateway restarted successfully"
ONLINE = "Gateway online"


@pytest.fixture
def rig(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
    monkeypatch.setattr(gateway_run, "_startup_restore_drain_timeout_secs", lambda: 0)
    monkeypatch.setattr(recovery, "_SWEEP_SETTLE_SECONDS", 0)
    runner = object.__new__(gateway_run.GatewayRunner)
    runner.config = GatewayConfig(
        platforms={Platform.TELEGRAM: PlatformConfig(
            enabled=True, token="7001:AAA-test", gateway_restart_notification=True,
            home_channel=HomeChannel(platform=Platform.TELEGRAM, chat_id="999", name="home"))},
        sessions_dir=tmp_path / "sessions")
    adapter = TelegramAdapter(runner.config.platforms[Platform.TELEGRAM])
    adapter._rich_send_disabled = True
    adapter._bot = MagicMock()
    adapter._bot.send_message = AsyncMock(return_value=MagicMock(message_id=7))
    adapter.gateway_runner = runner
    adapter._running = True
    generation, _ = adapter._begin_polling_generation()  # connected, first getUpdates still pending
    runner.adapters = {Platform.TELEGRAM: adapter}
    runner.delivery_router = SimpleNamespace(adapters=runner.adapters)
    runner._failed_platforms = {}
    runner._background_tasks = set()
    runner._running = True
    runner._redeliver_failed_obligations_for_platform = AsyncMock(return_value=0)
    runner._claim_pending_obligations = AsyncMock(return_value=[])
    runner._redeliver_claimed_obligations = AsyncMock(return_value=0)
    (tmp_path / ".restart_notify.json").write_text(json.dumps(
        {"platform": "telegram", "chat_id": "111", "chat_type": "dm"}), encoding="utf-8")
    (tmp_path / ".restart_pending.json").write_text("{}", encoding="utf-8")
    return runner, adapter, generation, tmp_path


def _sent(adapter):
    return [(str(c.kwargs.get("chat_id")), c.kwargs.get("text", "")) for c in adapter._bot.send_message.await_args_list]


async def _boot(runner):
    await runner._await_startup_boot_sends(
        planned_restart_notification_pending=gateway_run._planned_restart_notification_pending())


async def _settle(runner, adapter):
    for _ in range(5):
        pending = [t for t in (*runner._background_tasks, *adapter._background_tasks) if not t.done()]
        if not pending:
            return
        await asyncio.gather(*pending, return_exceptions=True)


@pytest.mark.asyncio
async def test_notices_refused_before_first_getupdates_are_sent_once_polling_heals(rig):
    runner, adapter, generation, home = rig
    await _boot(runner)
    adapter._bot.send_message.assert_not_called()  # refused locally: send_path_degraded
    assert (home / ".restart_notify.json").exists() and (home / ".restart_pending.json").exists()

    assert adapter._record_polling_progress(generation)  # the same adapter heals in place
    await _settle(runner, adapter)

    sent = _sent(adapter)
    assert [chat for chat, text in sent if RESTARTED in text] == ["111"]
    assert [chat for chat, text in sent if ONLINE in text] == ["999"]
    assert not (home / ".restart_notify.json").exists() and not (home / ".restart_pending.json").exists()

    # A later flap (degraded -> healthy again) sends nothing twice.
    generation, _ = adapter._begin_polling_generation()
    assert adapter._record_polling_progress(generation)
    await _settle(runner, adapter)
    assert len(_sent(adapter)) == len(sent)


@pytest.mark.asyncio
async def test_a_permanent_refusal_or_an_expired_marker_is_still_consumed(rig):
    runner, adapter, generation, home = rig
    assert adapter._record_polling_progress(generation)
    await _settle(runner, adapter)
    class BadRequest(Exception):  # matched by class name, as PTB's own (no SDK in the CI test env)
        pass
    adapter._bot.send_message = AsyncMock(side_effect=BadRequest("Chat not found"))
    await runner._send_restart_notification()
    assert not (home / ".restart_notify.json").exists()  # non-retryable: never retried forever

    stale = home / ".restart_notify.json"  # kept since a restart hours ago: the requester is not answered now
    stale.write_text(json.dumps({"platform": "telegram", "chat_id": "111"}), encoding="utf-8")
    os.utime(stale, (time.time() - 6 * 3600, time.time() - 6 * 3600))
    adapter._bot.send_message = AsyncMock(return_value=MagicMock(message_id=8))
    await runner._send_restart_notification()
    adapter._bot.send_message.assert_not_called()
    assert not (home / ".restart_notify.json").exists()


@pytest.mark.asyncio
async def test_a_heal_before_the_boot_pass_or_for_the_next_restart_sends_nothing(rig):
    """Before the boot pass the markers are its (every served profile must be loaded first); after it,
    a marker the boot pass never read belongs to a /restart this process has not performed yet."""
    runner, adapter, generation, home = rig
    assert adapter._record_polling_progress(generation)  # healed while secondary profiles still connect
    await _settle(runner, adapter)
    adapter._bot.send_message.assert_not_called()
    assert (home / ".restart_notify.json").exists() and (home / ".restart_pending.json").exists()

    await _boot(runner)  # the boot pass sends both
    assert len(_sent(adapter)) == 2

    # A new /restart wrote its marker; this process drains, and Telegram flaps meanwhile.
    (home / ".restart_notify.json").write_text(json.dumps(
        {"platform": "telegram", "chat_id": "222", "chat_type": "dm"}), encoding="utf-8")
    for draining in (False, True):
        runner._draining = draining
        generation, _ = adapter._begin_polling_generation()
        assert adapter._record_polling_progress(generation)
        await _settle(runner, adapter)
    assert len(_sent(adapter)) == 2
    assert (home / ".restart_notify.json").exists(), "the next boot's notice must survive"
