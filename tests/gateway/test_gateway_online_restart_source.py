"""The "Gateway online" home-channel notice after a signal-initiated restart.

A bare external SIGTERM (``_signal_initiated_shutdown``, no in-process restart request)
exits 1 so a supervisor with a restart policy revives the gateway.
``_stop_persist_exit_state`` only wrote ``.restart_pending.json`` for a source-less
``_restart_requested`` restart, so a supervised signal-initiated restart warned "shutting
down" and home channels never heard "back online". These tests cover the marker and the
"restarting" wording for that case, the replay of the "online" notice on the next boot,
and the paths that must not change: a signal with no restarting supervisor keeps "shutting
down" and writes no marker, a chat ``/restart`` keeps its private acknowledgement and
writes no marker, and a planned stop or takeover (neither flag set) writes no marker either.
"""

import asyncio
import json
import time
from unittest.mock import AsyncMock

import pytest

import gateway.run as gateway_run
import gateway.run_shutdown as gateway_run_shutdown
import gateway.status as gateway_status
from gateway.config import HomeChannel, Platform
from gateway.platforms.base import SendResult
from gateway.run_shutdown import GatewayShutdownMixin
from gateway.session import build_session_key
from tests.gateway.restart_test_helpers import make_restart_runner, make_restart_source


@pytest.fixture
def stop_env(tmp_path, monkeypatch):
    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
    monkeypatch.setattr(gateway_status, "remove_pid_file", lambda: None)
    monkeypatch.setattr(gateway_status, "release_gateway_runtime_lock", lambda: None)
    monkeypatch.setattr(gateway_run, "_shutdown_gateway_health_export", lambda _runner: None)
    return tmp_path


@pytest.fixture(autouse=True)
def supervised(monkeypatch):
    """Run under a restarting supervisor unless a test says otherwise (``set(False)``)."""
    def _set(value):
        monkeypatch.setattr(
            gateway_run_shutdown, "is_gateway_supervisor_process", lambda *a, **k: value, raising=False
        )
    _set(True)
    return _set


def _persist_exit_state(*, restart_requested, signal_initiated, command_source=None):
    runner, _adapter = make_restart_runner()
    runner._stop_persist_exit_state = GatewayShutdownMixin._stop_persist_exit_state.__get__(
        runner, type(runner)
    )
    runner._increment_restart_failure_counts = GatewayShutdownMixin._increment_restart_failure_counts.__get__(
        runner, type(runner)
    )
    runner._restart_requested = restart_requested
    runner._signal_initiated_shutdown = signal_initiated
    runner._restart_command_source = command_source
    ctx = GatewayShutdownMixin._StopContext(deferred_count=lambda: 0)
    ctx.started_at = time.monotonic()
    asyncio.run(runner._stop_persist_exit_state(ctx))


def test_marker_written_for_signal_initiated_shutdown(stop_env):
    """A bare SIGTERM under a restarting supervisor exits 1 and the supervisor revives the
    gateway: the next boot owes the home channels an "online" notice."""
    _persist_exit_state(restart_requested=False, signal_initiated=True)

    marker = stop_env / ".restart_pending.json"
    assert marker.exists(), "a signal-initiated shutdown is revived by the service manager"
    assert "requested_at" in json.loads(marker.read_text())


def test_marker_still_written_for_source_less_restart(stop_env):
    """Existing behaviour: a source-less restart request writes the marker."""
    _persist_exit_state(restart_requested=True, signal_initiated=False)

    assert (stop_env / ".restart_pending.json").exists()


@pytest.mark.parametrize("signal_initiated", [False, True])
def test_marker_not_written_for_chat_restart(stop_env, signal_initiated):
    """A chat /restart is acknowledged privately through .restart_notify.json, not through the
    home-channel broadcast, even if a signal also arrives during the restart."""
    _persist_exit_state(
        restart_requested=True, signal_initiated=signal_initiated, command_source=make_restart_source()
    )

    assert not (stop_env / ".restart_pending.json").exists()


def test_marker_not_written_for_unsupervised_signal(stop_env, supervised):
    """With no supervisor that restarts it (docker stop, shell launch, OS shutdown) a signal is a
    real stop: no marker, so no false "online" notice later."""
    supervised(False)
    _persist_exit_state(restart_requested=False, signal_initiated=True)

    assert not (stop_env / ".restart_pending.json").exists()


def test_marker_not_written_for_planned_stop(stop_env):
    """A planned stop or takeover sets neither flag and is not expected to come back on its own."""
    _persist_exit_state(restart_requested=False, signal_initiated=False)

    assert not (stop_env / ".restart_pending.json").exists()


@pytest.mark.asyncio
async def test_signal_initiated_shutdown_uses_restarting_wording():
    """The warning promises a comeback whenever the marker above will follow through."""
    runner, adapter = make_restart_runner()
    source = make_restart_source(chat_id="active-42", chat_type="group", thread_id="topic-7")
    session_key = build_session_key(source)

    runner._restart_requested = False
    runner._signal_initiated_shutdown = True
    runner._running_agents[session_key] = object()
    runner.session_store._entries[session_key] = type("Entry", (), {"origin": source})()
    adapter.send = AsyncMock(return_value=SendResult(success=True, message_id="1"))

    await runner._notify_active_sessions_of_shutdown()

    sent_message = adapter.send.await_args.args[1]
    assert sent_message.startswith("⚠️ Hermes is restarting"), sent_message


@pytest.mark.asyncio
async def test_home_channel_hears_online_after_signal_restart(stop_env):
    """The marker left by a signal-initiated shutdown is replayed on the next boot: the home
    channel gets the "online" notice and the marker is cleared."""
    await asyncio.to_thread(_persist_exit_state, restart_requested=False, signal_initiated=True)
    marker = stop_env / ".restart_pending.json"
    assert marker.exists()

    runner, adapter = make_restart_runner()
    runner.config.platforms[Platform.TELEGRAM].home_channel = HomeChannel(
        platform=Platform.TELEGRAM, chat_id="home-42", name="Ops Home",
    )
    adapter.send = AsyncMock(return_value=SendResult(success=True, message_id="home"))

    await runner._replay_pending_planned_restart_notification()

    adapter.send.assert_awaited_once()
    assert adapter.send.await_args.args[:2] == ("home-42", "♻️ Gateway online — Hermes is back and ready.")
    assert not marker.exists(), "every owed home channel was reached, so the marker discharges"


async def _shutdown_warning(*, restart_requested, signal_initiated):
    runner, adapter = make_restart_runner()
    source = make_restart_source(chat_id="active-42", chat_type="group", thread_id="topic-7")
    session_key = build_session_key(source)

    runner._restart_requested = restart_requested
    runner._signal_initiated_shutdown = signal_initiated
    runner._running_agents[session_key] = object()
    runner.session_store._entries[session_key] = type("Entry", (), {"origin": source})()
    adapter.send = AsyncMock(return_value=SendResult(success=True, message_id="1"))

    await runner._notify_active_sessions_of_shutdown()
    return adapter.send.await_args.args[1]


@pytest.mark.asyncio
async def test_unsupervised_signal_keeps_shutting_down_wording(supervised):
    """Nothing will bring the gateway back, so the warning must not promise a restart."""
    supervised(False)

    sent_message = await _shutdown_warning(restart_requested=False, signal_initiated=True)

    assert sent_message.startswith("⚠️ Hermes is shutting down"), sent_message


@pytest.mark.asyncio
async def test_planned_stop_keeps_shutting_down_wording():
    """A planned stop sets neither flag and keeps the "shutting down" wording, supervised or not."""
    sent_message = await _shutdown_warning(restart_requested=False, signal_initiated=False)

    assert sent_message.startswith("⚠️ Hermes is shutting down"), sent_message
