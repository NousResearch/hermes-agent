"""#71180: persisted shutdown-notification dedup across process restarts."""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest


@pytest.fixture
def hermes_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    # Refresh module path binding after HERMES_HOME change.
    import gateway.run as gr
    monkeypatch.setattr(gr, "_hermes_home", home)
    return home


def _write_config(home: Path, cooldown):
    (home / "config.yaml").write_text(
        f"gateway:\n  shutdown_notify_cooldown: {cooldown}\n",
        encoding="utf-8",
    )


def test_cooldown_reads_config_yaml(hermes_home, monkeypatch):
    import gateway.run as gr

    monkeypatch.delenv("HERMES_GATEWAY_SHUTDOWN_NOTIFY_COOLDOWN", raising=False)
    _write_config(hermes_home, 120)
    assert gr._shutdown_notify_cooldown_seconds() == 120


def test_zero_cooldown_disables_dedup(hermes_home, monkeypatch):
    import gateway.run as gr

    monkeypatch.delenv("HERMES_GATEWAY_SHUTDOWN_NOTIFY_COOLDOWN", raising=False)
    _write_config(hermes_home, 0)
    key = gr._shutdown_notify_dest_key("telegram", "1", None)
    gr._record_shutdown_notify_sent(key)
    assert gr._should_suppress_shutdown_notify(key, 0) is False


def test_cross_process_suppression_within_cooldown(hermes_home, monkeypatch):
    """A second process (fresh imports, same HERMES_HOME) must suppress."""
    import gateway.run as gr

    monkeypatch.delenv("HERMES_GATEWAY_SHUTDOWN_NOTIFY_COOLDOWN", raising=False)
    _write_config(hermes_home, 60)
    key = gr._shutdown_notify_dest_key("telegram", "home", None)
    gr._record_shutdown_notify_sent(key)
    assert gr._should_suppress_shutdown_notify(key, 60) is True

    # Simulate second process: re-bind path, do not call record again.
    path = hermes_home / ".shutdown_notify_sent.json"
    assert path.exists()
    data = json.loads(path.read_text(encoding="utf-8"))
    assert key in data
    assert gr._should_suppress_shutdown_notify(key, 60) is True


def test_expiry_allows_resend(hermes_home, monkeypatch):
    import gateway.run as gr

    key = gr._shutdown_notify_dest_key("discord", "chan", None)
    path = hermes_home / ".shutdown_notify_sent.json"
    path.write_text(json.dumps({key: time.time() - 120}), encoding="utf-8")
    assert gr._should_suppress_shutdown_notify(key, 60) is False


def test_corrupt_state_fail_open(hermes_home, monkeypatch):
    import gateway.run as gr

    path = hermes_home / ".shutdown_notify_sent.json"
    path.write_text("{not-json", encoding="utf-8")
    key = gr._shutdown_notify_dest_key("telegram", "1", None)
    assert gr._should_suppress_shutdown_notify(key, 60) is False


def test_future_timestamp_rewritten_not_infinite(hermes_home, monkeypatch):
    import gateway.run as gr

    key = gr._shutdown_notify_dest_key("telegram", "1", None)
    path = hermes_home / ".shutdown_notify_sent.json"
    future = time.time() + 10_000
    path.write_text(json.dumps({key: future}), encoding="utf-8")

    # First check clamps + rewrites. Immediate second check still suppresses
    # (cooldown from clamp time), but disk value is no longer far-future.
    assert gr._should_suppress_shutdown_notify(key, 60) is True
    data = json.loads(path.read_text(encoding="utf-8"))
    assert float(data[key]) <= time.time() + 1

    # After cooldown elapses from rewritten stamp, send is allowed.
    path.write_text(json.dumps({key: time.time() - 61}), encoding="utf-8")
    assert gr._should_suppress_shutdown_notify(key, 60) is False


def test_missing_state_file_fail_open(hermes_home):
    import gateway.run as gr

    key = gr._shutdown_notify_dest_key("telegram", "1", None)
    assert gr._should_suppress_shutdown_notify(key, 60) is False


# ── Behavioral: the actual home-channel send path ──────────────────────────
# teknium1's review asked for a test that drives _notify_active_sessions_of_shutdown
# through a real runner (not just the dedup helpers), performs a successful
# home-channel send, then constructs a second runner/process-equivalent and
# verifies the persisted state suppresses the duplicate.


@pytest.mark.asyncio
async def test_home_channel_send_persists_and_suppresses_second_process(tmp_path, monkeypatch):
    """A second gateway process must not re-broadcast the home-channel shutdown.

    First runner sends to the home channel and persists the timestamp. A second
    runner (fresh object, same HERMES_HOME) must see the persisted state and
    skip the duplicate broadcast.
    """
    from unittest.mock import AsyncMock

    import gateway.run as gateway_run
    from gateway.config import HomeChannel, Platform
    from gateway.platforms.base import SendResult
    from tests.gateway.restart_test_helpers import make_restart_runner

    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
    monkeypatch.delenv("HERMES_GATEWAY_SHUTDOWN_NOTIFY_COOLDOWN", raising=False)
    _write_config(tmp_path, 60)

    def _configure(runner, adapter):
        runner.config.platforms[Platform.TELEGRAM].home_channel = HomeChannel(
            platform=Platform.TELEGRAM,
            chat_id="home-42",
            name="Ops Home",
        )
        adapter.send = AsyncMock(return_value=SendResult(success=True, message_id="home"))

    # First process: sends and persists.
    runner1, adapter1 = make_restart_runner()
    _configure(runner1, adapter1)
    await runner1._notify_active_sessions_of_shutdown()
    assert adapter1.send.await_count == 1

    # Second process-equivalent: fresh runner, same HERMES_HOME. Must suppress.
    runner2, adapter2 = make_restart_runner()
    _configure(runner2, adapter2)
    await runner2._notify_active_sessions_of_shutdown()
    assert adapter2.send.await_count == 0


@pytest.mark.asyncio
async def test_home_channel_send_resends_after_cooldown_expiry(tmp_path, monkeypatch):
    """After the cooldown window elapses, a fresh process sends again."""
    from unittest.mock import AsyncMock

    import gateway.run as gateway_run
    from gateway.config import HomeChannel, Platform
    from gateway.platforms.base import SendResult
    from tests.gateway.restart_test_helpers import make_restart_runner

    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
    monkeypatch.delenv("HERMES_GATEWAY_SHUTDOWN_NOTIFY_COOLDOWN", raising=False)
    _write_config(tmp_path, 60)

    # Pre-seed an expired timestamp for the home channel.
    key = gateway_run._shutdown_notify_dest_key("telegram", "home-42", None)
    (tmp_path / ".shutdown_notify_sent.json").write_text(
        json.dumps({key: time.time() - 120}), encoding="utf-8"
    )

    runner, adapter = make_restart_runner()
    runner.config.platforms[Platform.TELEGRAM].home_channel = HomeChannel(
        platform=Platform.TELEGRAM,
        chat_id="home-42",
        name="Ops Home",
    )
    adapter.send = AsyncMock(return_value=SendResult(success=True, message_id="home"))

    await runner._notify_active_sessions_of_shutdown()

    assert adapter.send.await_count == 1
