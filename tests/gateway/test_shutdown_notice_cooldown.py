"""Durable cross-process cooldown for the home-channel shutdown broadcast.

Regression coverage for the notification flood observed when something outside
Hermes repeatedly cycles the gateway process: the in-process ``notified`` set
in ``_notify_active_sessions_of_shutdown`` cannot dedupe across process
boundaries, so every fresh start re-broadcast to the same home channel (240
sends on a WSL host under Windows Modern Standby).

These tests pin two properties:
  * a second gateway PROCESS is silenced inside the cooldown window (the fix),
  * everything the guard must NOT touch stays loud — active-session pings, a
    genuine restart after the window, and the disabled/misconfigured paths.
"""
import asyncio
import multiprocessing
import time
from pathlib import Path

import pytest

from gateway import shutdown_notice
from gateway.config import GatewayConfig, HomeChannel, Platform, PlatformConfig
from tests.gateway.restart_test_helpers import make_restart_runner


@pytest.fixture(autouse=True)
def _state_home(tmp_path, monkeypatch):
    """Point the durable notice state at a temp HERMES_HOME."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(
        shutdown_notice, "get_hermes_home", lambda: Path(tmp_path)
    )
    return tmp_path


def _runner_with_home(cooldown: int = 300, chat_id: str = "8737458794"):
    runner, adapter = make_restart_runner()
    runner.config = GatewayConfig(
        platforms={
            Platform.TELEGRAM: PlatformConfig(
                enabled=True,
                token="***",
                home_channel=HomeChannel(
                    platform=Platform.TELEGRAM, chat_id=chat_id, name="home"
                ),
            )
        },
        shutdown_notification_cooldown_seconds=cooldown,
    )
    return runner, adapter


def _concurrent_admission_worker(home, key, ready, start, results):
    """Compete for one admission from a separate OS process."""
    ready.put(True)
    start.wait(10)
    with shutdown_notice.home_notice_admission(
        key,
        cooldown_seconds=300,
        now=100.0,
        home=Path(home),
    ) as admission:
        results.put(admission.allowed)
        if admission.allowed:
            # Keep the lock held across the simulated awaited send.
            time.sleep(0.15)
            admission.record_success(now=100.0)


# ── The bug: repeated gateway processes re-broadcast ────────────────────


@pytest.mark.asyncio
async def test_second_process_suppressed_within_cooldown():
    """A fresh gateway process must not re-broadcast inside the window.

    Two runners with independent ``notified`` sets model two OS processes.
    Before the fix both send; after it, only the first does.
    """
    runner_a, adapter_a = _runner_with_home(cooldown=300)
    await runner_a._notify_active_sessions_of_shutdown()
    assert len(adapter_a.sent) == 1, "first process must announce"

    runner_b, adapter_b = _runner_with_home(cooldown=300)
    await runner_b._notify_active_sessions_of_shutdown()

    assert adapter_b.sent == [], (
        "a second gateway process inside the cooldown window re-broadcast "
        "the shutdown notice — the durable guard did not hold"
    )


def test_concurrent_processes_have_single_admission(tmp_path):
    """Concurrent gateway processes cannot both pass the cooldown check."""
    ctx = multiprocessing.get_context("spawn")
    ready = ctx.Queue()
    start = ctx.Event()
    results = ctx.Queue()
    key = shutdown_notice.destination_key("telegram", "chat:a:b", "thread:c")
    processes = [
        ctx.Process(
            target=_concurrent_admission_worker,
            args=(str(tmp_path), key, ready, start, results),
        )
        for _ in range(2)
    ]
    for process in processes:
        process.start()
    try:
        for _ in processes:
            assert ready.get(timeout=15) is True
        start.set()
        decisions = sorted(results.get(timeout=15) for _ in processes)
    finally:
        start.set()
        for process in processes:
            process.join(timeout=15)
            if process.is_alive():
                process.terminate()
                process.join()
    assert all(process.exitcode == 0 for process in processes)
    assert decisions == [False, True]


def test_destination_key_is_collision_free():
    """Structured keys keep colon-containing IDs distinct."""
    assert shutdown_notice.destination_key("telegram", "a:b", "c") != (
        shutdown_notice.destination_key("telegram", "a", "b:c")
    )


@pytest.mark.asyncio
async def test_flood_of_processes_collapses_to_one_message():
    """20 rapid start/stop cycles produce exactly one home-channel message."""
    sent_total = 0
    for _ in range(20):
        runner, adapter = _runner_with_home(cooldown=300)
        await runner._notify_active_sessions_of_shutdown()
        sent_total += len(adapter.sent)

    assert sent_total == 1, f"expected 1 notice across 20 cycles, got {sent_total}"


# ── What must stay loud ────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_notice_sent_again_after_cooldown_expires():
    """A genuine restart after the window is still announced."""
    runner_a, adapter_a = _runner_with_home(cooldown=300)
    await runner_a._notify_active_sessions_of_shutdown()
    assert len(adapter_a.sent) == 1

    # Age the recorded timestamp past the window.
    key = shutdown_notice.destination_key("telegram", "8737458794", None)
    import time

    shutdown_notice.record_home_notice(key, now=time.time() - 301)

    runner_b, adapter_b = _runner_with_home(cooldown=300)
    await runner_b._notify_active_sessions_of_shutdown()
    assert len(adapter_b.sent) == 1, "notice must resume once the window lapses"


@pytest.mark.asyncio
async def test_zero_cooldown_restores_always_send():
    """The documented opt-out (0) preserves the original behaviour."""
    for _ in range(3):
        runner, adapter = _runner_with_home(cooldown=0)
        await runner._notify_active_sessions_of_shutdown()
        assert len(adapter.sent) == 1


@pytest.mark.asyncio
async def test_active_session_notice_never_suppressed():
    """Per-session interrupt pings are NOT gated by the cooldown.

    They carry the 'your task was cut off' hint, so a user mid-task must hear
    about it even if the home channel was told seconds ago.
    """
    from unittest.mock import MagicMock

    # Burn the cooldown with a home-channel broadcast first.
    runner_a, adapter_a = _runner_with_home(cooldown=300)
    await runner_a._notify_active_sessions_of_shutdown()
    assert len(adapter_a.sent) == 1

    runner_b, adapter_b = _runner_with_home(cooldown=300)
    runner_b._running_agents["agent:main:telegram:dm:8737458794"] = MagicMock()
    await runner_b._notify_active_sessions_of_shutdown()

    assert len(adapter_b.sent) == 1, "active-session ping must not be suppressed"
    assert "interrupted" in adapter_b.sent[0]


@pytest.mark.asyncio
async def test_distinct_home_channels_have_independent_cooldowns():
    """Silencing chat A must not silence chat B."""
    runner_a, adapter_a = _runner_with_home(cooldown=300)
    await runner_a._notify_active_sessions_of_shutdown()
    assert len(adapter_a.sent) == 1

    runner_b, adapter_b = _runner_with_home(cooldown=300, chat_id="999999")
    await runner_b._notify_active_sessions_of_shutdown()
    assert len(adapter_b.sent) == 1, "a different chat must still be notified"


# ── Fail-open properties of the state file ─────────────────────────────


@pytest.mark.asyncio
async def test_corrupt_state_file_does_not_suppress(_state_home):
    """A malformed state file must fail toward sending, never toward silence."""
    shutdown_notice.notice_state_path().write_text("{not json", encoding="utf-8")

    runner, adapter = _runner_with_home(cooldown=300)
    await runner._notify_active_sessions_of_shutdown()
    assert len(adapter.sent) == 1


def test_future_timestamp_does_not_silence_indefinitely():
    """A backwards clock step must not lock the channel silent.

    Host suspend/resume — the very scenario this guard exists for — can move
    the wall clock. A recorded time in the future would otherwise read as a
    cooldown that never expires.
    """
    import time

    key = shutdown_notice.destination_key("telegram", "123", None)
    shutdown_notice.record_home_notice(key, now=time.time() + 86_400)

    assert shutdown_notice.should_send_home_notice(key, cooldown_seconds=300) is True


def test_record_survives_unwritable_home(monkeypatch):
    """A failed persist must not raise into the shutdown path."""
    def _boom(*_a, **_kw):
        raise OSError("read-only filesystem")

    monkeypatch.setattr(shutdown_notice, "atomic_json_write", _boom)
    # Must not raise.
    shutdown_notice.record_home_notice(
        shutdown_notice.destination_key("telegram", "123", None)
    )


def test_unknown_destination_sends():
    """No record at all means send."""
    key = shutdown_notice.destination_key("telegram", "never-seen", None)
    assert shutdown_notice.should_send_home_notice(key, cooldown_seconds=300) is True


# ── Config plumbing ────────────────────────────────────────────────────


def test_cooldown_default_and_roundtrip():
    cfg = GatewayConfig()
    assert cfg.shutdown_notification_cooldown_seconds == 300
    assert (
        GatewayConfig.from_dict(cfg.to_dict()).shutdown_notification_cooldown_seconds
        == 300
    )


def test_cooldown_read_from_flat_and_nested_config():
    assert (
        GatewayConfig.from_dict(
            {"shutdown_notification_cooldown_seconds": 60}
        ).shutdown_notification_cooldown_seconds
        == 60
    )
    assert (
        GatewayConfig.from_dict(
            {"gateway": {"shutdown_notification_cooldown_seconds": 45}}
        ).shutdown_notification_cooldown_seconds
        == 45
    )


def test_cooldown_zero_is_preserved_not_treated_as_unset():
    """0 disables the guard, so it must survive parsing rather than fall back."""
    cfg = GatewayConfig.from_dict({"shutdown_notification_cooldown_seconds": 0})
    assert cfg.shutdown_notification_cooldown_seconds == 0


@pytest.mark.parametrize("bad", ["abc", -5, True, 1.5, [], {}])
def test_malformed_cooldown_falls_back_to_default(bad):
    """A typo must never prevent the gateway from starting."""
    cfg = GatewayConfig.from_dict({"shutdown_notification_cooldown_seconds": bad})
    assert cfg.shutdown_notification_cooldown_seconds == 300


def test_cooldown_default_is_in_shared_config_defaults():
    from hermes_cli.config_defaults import DEFAULT_CONFIG

    assert DEFAULT_CONFIG["gateway"]["shutdown_notification_cooldown_seconds"] == 300


def test_cooldown_read_from_real_yaml_loader(tmp_path, monkeypatch):
    """The user-facing config.yaml path must reach GatewayConfig.from_dict."""
    from gateway.config import load_gateway_config

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "gateway:\n  shutdown_notification_cooldown_seconds: 45\n",
        encoding="utf-8",
    )
    assert load_gateway_config().shutdown_notification_cooldown_seconds == 45

    config_path.write_text(
        "gateway:\n  shutdown_notification_cooldown_seconds: 45\n"
        "shutdown_notification_cooldown_seconds: 60\n",
        encoding="utf-8",
    )
    assert load_gateway_config().shutdown_notification_cooldown_seconds == 60
