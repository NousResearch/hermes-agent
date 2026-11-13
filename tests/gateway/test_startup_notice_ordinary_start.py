"""An ordinary start announces itself on the home channels (#27870).

The planned-restart marker only exists when the previous process went down *for a restart*, so a
cold boot, a supervisor/crash recovery and a CLI ``stop`` + ``start`` (all of Windows) reached the
boot pass with nothing owed and left the home channel silent — on a messaging-only install that is
indistinguishable from a dead gateway. These tests pin the branch in ``_await_startup_boot_sends``,
the notice itself and the throttle that keeps a crash loop from paging (#60438).
"""

import json
import logging
import os
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

import gateway.run as gateway_run
from agent.i18n import t
from gateway.config import GatewayConfig, HomeChannel, Platform, PlatformConfig
from gateway.platforms.base import SendResult
from gateway.run import GatewayRunner

ONLINE_NOTICE = t("gateway.startup.online")


def _adapter():
    return SimpleNamespace(send=AsyncMock(return_value=SendResult(success=True, message_id="m1")))


def _make_runner(tmp_path, monkeypatch, *, adapters):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    # gateway.run snapshots the home at import; the marker writer reads it live.
    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
    runner = object.__new__(GatewayRunner)
    config = GatewayConfig(
        platforms={
            Platform.TELEGRAM: PlatformConfig(
                enabled=True,
                gateway_restart_notification=True,
                home_channel=HomeChannel(platform=Platform.TELEGRAM, chat_id="home-1", name="home-1"),
            )
        }
    )
    config.sessions_dir = tmp_path / "sessions"
    runner.config = config
    runner.adapters = adapters
    runner._profile_configs = {}
    runner._profile_adapters = {}
    runner._planned_restart_notice_lock = None
    for name in (
        "_home_channel_transports",
        "_served_home_channel_transports",
        "_send_home_channel_message",
        "_send_home_channel_startup_notifications",
        "_send_ordinary_start_notice",
        "_await_startup_boot_sends",
    ):
        method = getattr(GatewayRunner, name, None)
        if method is not None:  # absent on the pre-fix base, where the red run must still set up
            setattr(runner, name, method.__get__(runner, GatewayRunner))
    # Not under test here: the thread-target resolver reads live session state.
    runner._thread_metadata_for_target = lambda *a, **k: None
    return runner


@pytest.mark.asyncio
async def test_ordinary_start_notifies_the_home_channel(tmp_path, monkeypatch):
    adapter = _adapter()
    runner = _make_runner(tmp_path, monkeypatch, adapters={Platform.TELEGRAM: adapter})

    await runner._send_ordinary_start_notice()

    adapter.send.assert_awaited_once()
    assert adapter.send.await_args.args[:2] == ("home-1", ONLINE_NOTICE)
    assert not (tmp_path / ".restart_pending.json").exists(), "the debt must be discharged too"
    stamped = json.loads((tmp_path / ".startup_notice.json").read_text(encoding="utf-8"))
    assert stamped["sent_at"] > 0


@pytest.mark.asyncio
async def test_second_start_inside_the_window_is_throttled_and_owes_nothing(tmp_path, monkeypatch):
    """A crash loop must not page the home channel every attempt — nor leave a notice owed."""
    adapter = _adapter()
    runner = _make_runner(tmp_path, monkeypatch, adapters={Platform.TELEGRAM: adapter})

    await runner._send_ordinary_start_notice()
    await runner._send_ordinary_start_notice()

    adapter.send.assert_awaited_once()
    assert not (tmp_path / ".restart_pending.json").exists(), (
        "a throttled start must owe nothing, or the reconnect replay would page anyway"
    )


@pytest.mark.asyncio
async def test_an_unreachable_home_stays_owed_and_burns_no_throttle(tmp_path, monkeypatch):
    runner = _make_runner(tmp_path, monkeypatch, adapters={})
    # `adapters={}` alone is not enough: the delivery transport falls back to a process-wide
    # router that other suites populate, so pin "nothing is live" explicitly.
    monkeypatch.setattr(runner, "_served_home_channel_transports", lambda: iter(()))

    await runner._send_ordinary_start_notice()

    assert (tmp_path / ".restart_pending.json").exists(), (
        "a start that reached nobody must keep the notice owed for the reconnect replay"
    )
    assert not (tmp_path / ".startup_notice.json").exists(), (
        "a start that reached nobody must not burn the throttle window"
    )


@pytest.mark.asyncio
async def test_a_platform_that_connects_late_still_gets_the_notice(tmp_path, monkeypatch):
    """The point of owing the marker: offline at boot, announced when it comes back (#112109)."""
    adapter = _adapter()
    runner = _make_runner(tmp_path, monkeypatch, adapters={})
    live = runner._served_home_channel_transports
    monkeypatch.setattr(runner, "_served_home_channel_transports", lambda: iter(()))

    await runner._send_ordinary_start_notice()
    adapter.send.assert_not_awaited()

    runner.adapters = {Platform.TELEGRAM: adapter}  # the adapter reconnects
    monkeypatch.setattr(runner, "_served_home_channel_transports", live)
    await runner._replay_pending_planned_restart_notification()

    adapter.send.assert_awaited_once()
    assert adapter.send.await_args.args[:2] == ("home-1", ONLINE_NOTICE)
    assert not (tmp_path / ".restart_pending.json").exists()


@pytest.mark.asyncio
async def test_boot_pass_picks_the_notice_for_the_start_it_actually_had(tmp_path, monkeypatch):
    """Marker present -> planned-restart replay; marker absent -> ordinary notice. Never both."""
    monkeypatch.setattr(gateway_run, "_startup_restore_drain_timeout_secs", lambda: 0)
    adapter = _adapter()
    runner = _make_runner(tmp_path, monkeypatch, adapters={Platform.TELEGRAM: adapter})
    runner._claim_pending_obligations = AsyncMock(return_value=set())
    runner._send_restart_notification = AsyncMock(return_value=None)
    runner._replay_pending_planned_restart_notification = AsyncMock(return_value=None)
    runner._redeliver_claimed_obligations = AsyncMock(return_value=None)
    runner._send_ordinary_start_notice = AsyncMock(return_value=set())

    await runner._await_startup_boot_sends(planned_restart_notification_pending=True)
    runner._replay_pending_planned_restart_notification.assert_awaited_once()
    runner._send_ordinary_start_notice.assert_not_awaited()

    runner._replay_pending_planned_restart_notification.reset_mock()
    await runner._await_startup_boot_sends(planned_restart_notification_pending=False)
    runner._send_ordinary_start_notice.assert_awaited_once()
    runner._replay_pending_planned_restart_notification.assert_not_awaited()


@pytest.mark.asyncio
async def test_ordinary_start_reaches_the_home_channel_through_the_boot_pass(tmp_path, monkeypatch):
    """End to end through the real boot pass: no marker, the home channel still hears about it.

    Fails on the pre-fix base on an assertion (the channel got nothing), not on a missing symbol,
    because the only stub is ``_send_restart_notification``.
    """
    monkeypatch.setattr(gateway_run, "_startup_restore_drain_timeout_secs", lambda: 0)
    adapter = _adapter()
    runner = _make_runner(tmp_path, monkeypatch, adapters={Platform.TELEGRAM: adapter})
    runner._claim_pending_obligations = AsyncMock(return_value=set())
    runner._send_restart_notification = AsyncMock(return_value=None)
    runner._redeliver_claimed_obligations = AsyncMock(return_value=None)

    await runner._await_startup_boot_sends(planned_restart_notification_pending=False)

    adapter.send.assert_awaited_once()
    assert adapter.send.await_args.args[:2] == ("home-1", ONLINE_NOTICE)


@pytest.mark.asyncio
async def test_a_process_that_owns_this_home_announces_the_start(tmp_path, monkeypatch):
    """The pid record naming THIS process is the proof of a new start, not of an attach."""
    monkeypatch.setattr("gateway.status.live_gateway_pid_for_home", lambda home: os.getpid())
    adapter = _adapter()
    runner = _make_runner(tmp_path, monkeypatch, adapters={Platform.TELEGRAM: adapter})

    await runner._send_ordinary_start_notice()

    adapter.send.assert_awaited_once()
    assert adapter.send.await_args.args[:2] == ("home-1", ONLINE_NOTICE)


@pytest.mark.asyncio
async def test_attaching_to_a_live_gateway_announces_nothing_and_burns_no_window(tmp_path, monkeypatch):
    """A second process pointed at a home another gateway owns must not re-announce the start.

    Nothing may be owed and nothing stamped: burning the throttle here would silence the owning
    process's next real start for five minutes.
    """
    monkeypatch.setattr("gateway.status.live_gateway_pid_for_home", lambda home: os.getpid() + 1)
    adapter = _adapter()
    runner = _make_runner(tmp_path, monkeypatch, adapters={Platform.TELEGRAM: adapter})

    await runner._send_ordinary_start_notice()

    adapter.send.assert_not_awaited()
    assert not (tmp_path / ".restart_pending.json").exists(), "an attach owes no notice"
    assert not (tmp_path / ".startup_notice.json").exists(), "an attach must not burn the window"


@pytest.mark.asyncio
async def test_a_failing_owner_probe_silently_skips_the_notice(tmp_path, monkeypatch):
    """A broken probe must not page the home channel and must not break the boot pass."""

    def _boom(home):
        raise RuntimeError("pid probe exploded")

    monkeypatch.setattr("gateway.status.live_gateway_pid_for_home", _boom)
    monkeypatch.setattr(gateway_run, "_startup_restore_drain_timeout_secs", lambda: 0)
    adapter = _adapter()
    runner = _make_runner(tmp_path, monkeypatch, adapters={Platform.TELEGRAM: adapter})
    runner._claim_pending_obligations = AsyncMock(return_value=set())
    runner._send_restart_notification = AsyncMock(return_value=None)
    runner._redeliver_claimed_obligations = AsyncMock(return_value=None)

    await runner._await_startup_boot_sends(planned_restart_notification_pending=False)

    adapter.send.assert_not_awaited()
    assert not (tmp_path / ".restart_pending.json").exists()
    assert not (tmp_path / ".startup_notice.json").exists()


# --- Failure paths (D4): the notice is best-effort and must never reach the boot caller ---------


def _boot_runner(tmp_path, monkeypatch, *, send_effect=None):
    """Runner wired for the real boot pass, with the notification stubs it must not exercise."""
    monkeypatch.setattr(gateway_run, "_startup_restore_drain_timeout_secs", lambda: 0)
    adapter = SimpleNamespace(send=AsyncMock(side_effect=send_effect, return_value=SendResult(success=True, message_id="m1")))
    runner = _make_runner(tmp_path, monkeypatch, adapters={Platform.TELEGRAM: adapter})
    runner._claim_pending_obligations = AsyncMock(return_value=set())
    runner._send_restart_notification = AsyncMock(return_value=None)
    runner._redeliver_claimed_obligations = AsyncMock(return_value=None)
    return runner, adapter


@pytest.mark.asyncio
async def test_no_home_channel_configured_sends_nothing_and_boots_clean(tmp_path, monkeypatch, caplog):
    """No home is a configuration state, not an error: no request, no warning, nothing left owed."""
    runner, adapter = _boot_runner(tmp_path, monkeypatch)
    runner.config.platforms[Platform.TELEGRAM].home_channel = None

    with caplog.at_level(logging.WARNING):
        await runner._await_startup_boot_sends(planned_restart_notification_pending=False)

    adapter.send.assert_not_awaited()
    assert not (tmp_path / ".restart_pending.json").exists(), "nothing to owe, so nothing may stay owed"
    assert "Home-channel startup notification failed" not in caplog.text


@pytest.mark.asyncio
async def test_a_raising_delivery_is_swallowed_and_the_boot_pass_returns(tmp_path, monkeypatch, caplog):
    """An adapter that raises (chat not found, flood control, network) must not abort startup."""
    runner, _adapter_unused = _boot_runner(tmp_path, monkeypatch, send_effect=RuntimeError("chat not found"))

    with caplog.at_level(logging.WARNING):
        await runner._await_startup_boot_sends(planned_restart_notification_pending=False)

    runner.adapters[Platform.TELEGRAM].send.assert_awaited_once()
    assert "Home-channel startup notification failed for telegram:home-1" in caplog.text
    assert "chat not found" in caplog.text
    assert (tmp_path / ".restart_pending.json").exists(), "a failed target stays owed for the reconnect replay"


@pytest.mark.asyncio
async def test_a_failed_send_burns_no_throttle_so_the_next_start_retries(tmp_path, monkeypatch):
    """One failure must not swallow the next window: the retry still attempts the send."""
    from gateway.startup_notice import should_send_startup_notice

    runner, _failing = _boot_runner(tmp_path, monkeypatch, send_effect=RuntimeError("flood control"))
    await runner._send_ordinary_start_notice()

    assert should_send_startup_notice() is True, "a start that reached nobody must not burn the throttle"

    retry = _adapter()
    runner.adapters = {Platform.TELEGRAM: retry}
    await runner._send_ordinary_start_notice()

    retry.send.assert_awaited_once()
    assert retry.send.await_args.args[:2] == ("home-1", ONLINE_NOTICE)


# --- Throttle window (D1): one notice per window, injectable clock, fail-open -------------------


class _Clock:
    """Injectable clock: the throttle reads ``startup_notice.time.time``."""

    def __init__(self, now: float) -> None:
        self.now = now

    def time(self) -> float:
        return self.now


@pytest.mark.asyncio
async def test_the_notice_returns_after_the_window(tmp_path, monkeypatch):
    """A start past the window announces again — otherwise a recovered gateway stays silent."""
    from gateway import startup_notice

    monkeypatch.setattr(startup_notice, "time", _Clock(1_000.0))
    adapter = _adapter()
    runner = _make_runner(tmp_path, monkeypatch, adapters={Platform.TELEGRAM: adapter})

    await runner._send_ordinary_start_notice()
    adapter.send.assert_awaited_once()

    startup_notice.time.now += startup_notice.MIN_INTERVAL_SECONDS
    await runner._send_ordinary_start_notice()

    assert adapter.send.await_count == 2, "a start outside the window must announce itself again"


def test_the_window_boundary_is_inclusive(tmp_path):
    """One second short stays throttled, the window itself sends: ``now - last >= min_interval``."""
    from gateway import startup_notice as sn

    path = tmp_path / "notice.json"
    sn.stamp_startup_notice(now=1_000.0, path=path)

    assert sn.should_send_startup_notice(now=1_000.0 + sn.MIN_INTERVAL_SECONDS - 1, path=path) is False
    assert sn.should_send_startup_notice(now=1_000.0 + sn.MIN_INTERVAL_SECONDS, path=path) is True


@pytest.mark.parametrize("content", [None, "{not json", '{"sent_at": null}', '{"sent_at": "yesterday"}'])
@pytest.mark.asyncio
async def test_an_unreadable_throttle_record_fails_open(tmp_path, monkeypatch, content):
    """Missing or broken state must never suppress — or raise out of — the notice."""
    if content is not None:
        (tmp_path / ".startup_notice.json").write_text(content, encoding="utf-8")
    adapter = _adapter()
    runner = _make_runner(tmp_path, monkeypatch, adapters={Platform.TELEGRAM: adapter})

    await runner._send_ordinary_start_notice()

    adapter.send.assert_awaited_once()


@pytest.mark.asyncio
async def test_each_process_home_keeps_its_own_window(tmp_path, monkeypatch):
    """The window belongs to a process home, so a second install/profile starts with its own.

    A single boot fans the notice out to every SERVED profile's home at once, so profiles served by
    one process share one window by design (D1); a home run as its own process does not.
    """
    first, second = tmp_path / "home-a", tmp_path / "home-b"
    first.mkdir()
    second.mkdir()
    adapter_a = _adapter()
    runner_a = _make_runner(first, monkeypatch, adapters={Platform.TELEGRAM: adapter_a})
    await runner_a._send_ordinary_start_notice()

    adapter_b = _adapter()
    runner_b = _make_runner(second, monkeypatch, adapters={Platform.TELEGRAM: adapter_b})
    await runner_b._send_ordinary_start_notice()

    adapter_a.send.assert_awaited_once()
    # another home must not inherit this home's window
    adapter_b.send.assert_awaited_once()


# --- Dedupe (D6): a chat that ran /restart is not told "online" on top of "restarted" -----------

RESTARTED_NOTICE = t("gateway.startup.restarted")


def _bind_real_restart_notification(runner):
    """Bind the real restart-notice path: the key it returns is what the boot pass seeds (D6)."""
    for name in (
        "_send_restart_notification",
        "_adapters_for_profile",
        "_pending_marker_metadata",
    ):
        setattr(runner, name, getattr(GatewayRunner, name).__get__(runner, GatewayRunner))
    # ``_marker_profile`` is a staticmethod: no ``self`` to bind.
    return runner


@pytest.mark.asyncio
async def test_a_chat_restart_is_not_announced_twice(tmp_path, monkeypatch):
    """`/restart` sent from the home chat: it hears "restarted", never also "online"."""
    monkeypatch.setattr(gateway_run, "_startup_restore_drain_timeout_secs", lambda: 0)
    adapter = _adapter()
    runner = _bind_real_restart_notification(
        _make_runner(tmp_path, monkeypatch, adapters={Platform.TELEGRAM: adapter})
    )
    (tmp_path / ".restart_notify.json").write_text(
        json.dumps({"platform": "telegram", "chat_id": "home-1"}), encoding="utf-8"
    )
    runner._claim_pending_obligations = AsyncMock(return_value=set())
    runner._redeliver_claimed_obligations = AsyncMock(return_value=None)

    await runner._await_startup_boot_sends(planned_restart_notification_pending=False)

    adapter.send.assert_awaited_once()
    assert adapter.send.await_args.args[:2] == ("home-1", RESTARTED_NOTICE)
    assert not (tmp_path / ".restart_pending.json").exists(), "the requester's chat is not owed again"


@pytest.mark.asyncio
async def test_a_chat_restart_still_announces_the_other_homes(tmp_path, monkeypatch):
    """Skipping the requester must not silence the home channels that heard nothing."""
    monkeypatch.setattr(gateway_run, "_startup_restore_drain_timeout_secs", lambda: 0)
    adapter = _adapter()
    runner = _bind_real_restart_notification(
        _make_runner(tmp_path, monkeypatch, adapters={Platform.TELEGRAM: adapter})
    )
    (tmp_path / ".restart_notify.json").write_text(
        json.dumps({"platform": "telegram", "chat_id": "dm-7"}), encoding="utf-8"
    )
    runner._claim_pending_obligations = AsyncMock(return_value=set())
    runner._redeliver_claimed_obligations = AsyncMock(return_value=None)

    await runner._await_startup_boot_sends(planned_restart_notification_pending=False)

    assert {call.args[0]: call.args[1] for call in adapter.send.await_args_list} == {
        "dm-7": RESTARTED_NOTICE,
        "home-1": ONLINE_NOTICE,
    }
