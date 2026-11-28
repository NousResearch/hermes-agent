"""The "shutting down" chat notice names an unplanned stop's cause, and only an unplanned stop's.

Each case drives the real SIGTERM/SIGINT handler and the real ``stop()``, so the planned-stop and
``--replace`` takeover markers are classified exactly as in production before the notice is sent.
"""

import asyncio
import signal
from unittest.mock import MagicMock, patch

import pytest

from agent.i18n import t
from tests.gateway.restart_test_helpers import make_restart_runner

CHAT = "chat-42"


@pytest.fixture
def runner(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.delenv("INVOCATION_ID", raising=False)
    monkeypatch.setattr("gateway.drain_control.drain_notification_suppressed", lambda: False)
    # The detached ps/pstree/dmesg collector is irrelevant here; keep tests from spawning it.
    monkeypatch.setattr("gateway.shutdown_forensics.spawn_async_diagnostic", lambda *a, **k: None)
    monkeypatch.setattr("gateway.status.consume_takeover_marker_for_self", lambda: False)
    monkeypatch.setattr("gateway.status.consume_planned_stop_marker_for_self", lambda: False)
    runner, adapter = make_restart_runner()
    runner._home_shutdown_notice = None
    runner._planned_takeover_shutdown = False
    runner._signal_initiated_shutdown = False
    agent = MagicMock()
    agent.interrupt.side_effect = lambda *a, **k: runner._running_agents.clear()
    runner._running_agents = {f"agent:main:telegram:dm:{CHAT}": agent}
    return runner, adapter


async def _signal_stop(runner, received_signal) -> None:
    """Deliver ``received_signal`` to the real handler and wait for the ``stop()`` it schedules."""
    from gateway import run as gateway_run

    with patch("gateway.status.remove_pid_file"), patch("gateway.status.publish_runtime_status"):
        before = set(asyncio.all_tasks())
        gateway_run._start_gateway_make_shutdown_signal_handler(runner, [False])(received_signal)
        await asyncio.wait_for(asyncio.gather(*(set(asyncio.all_tasks()) - before)), timeout=30)


def _chat_notices(adapter) -> list:
    return [text for chat_id, text, _meta in adapter.sent_calls if chat_id == CHAT]


@pytest.mark.asyncio
async def test_external_sigterm_under_systemd_names_systemd(runner, monkeypatch):
    runner, adapter = runner
    monkeypatch.setenv("INVOCATION_ID", "unit-invocation")

    await _signal_stop(runner, signal.SIGTERM)

    reason = t("gateway.shutdown.reason_external_systemd")
    assert _chat_notices(adapter) == [t("gateway.shutdown.notice_shutdown_reason", reason=reason)]


@pytest.mark.asyncio
async def test_bare_external_sigterm_says_external_stop(runner):
    runner, adapter = runner

    await _signal_stop(runner, signal.SIGTERM)

    reason = t("gateway.shutdown.reason_external")
    assert _chat_notices(adapter) == [t("gateway.shutdown.notice_shutdown_reason", reason=reason)]


@pytest.mark.asyncio
async def test_planned_stop_marker_sigterm_sends_plain_notice(runner, monkeypatch):
    """``hermes gateway stop`` / the systemd ExecStop mark: planned, even under systemd."""
    runner, adapter = runner
    monkeypatch.setenv("INVOCATION_ID", "unit-invocation")
    monkeypatch.setattr("gateway.status.consume_planned_stop_marker_for_self", lambda: True)

    await _signal_stop(runner, signal.SIGTERM)

    assert _chat_notices(adapter) == [t("gateway.shutdown.notice_shutdown")]


@pytest.mark.asyncio
async def test_planned_takeover_marker_sigterm_sends_plain_notice(runner, monkeypatch):
    """A ``--replace`` takeover still tells a chat whose task is cut, but never as an external stop."""
    runner, adapter = runner
    monkeypatch.setenv("INVOCATION_ID", "unit-invocation")
    monkeypatch.setattr("gateway.status.consume_takeover_marker_for_self", lambda: True)

    await _signal_stop(runner, signal.SIGTERM)

    assert _chat_notices(adapter) == [t("gateway.shutdown.notice_shutdown")]


@pytest.mark.asyncio
async def test_sigint_sends_plain_notice(runner):
    runner, adapter = runner

    await _signal_stop(runner, signal.SIGINT)

    assert _chat_notices(adapter) == [t("gateway.shutdown.notice_shutdown")]


@pytest.mark.asyncio
async def test_restart_notice_is_unchanged_by_shutdown_context(runner):
    runner, adapter = runner
    runner._restart_requested = True
    runner._signal_initiated_shutdown = True
    runner._shutdown_ctx = {"signal": "SIGTERM", "under_systemd": True}

    await runner._notify_active_sessions_of_shutdown()

    assert _chat_notices(adapter) == [t("gateway.shutdown.notice_restart")]


@pytest.mark.asyncio
async def test_missing_forensics_snapshot_sends_plain_notice(runner):
    runner, adapter = runner
    runner._signal_initiated_shutdown = True
    runner._shutdown_ctx = None

    await runner._notify_active_sessions_of_shutdown()

    assert _chat_notices(adapter) == [t("gateway.shutdown.notice_shutdown")]
