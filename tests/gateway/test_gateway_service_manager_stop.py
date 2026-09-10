"""A SIGTERM from our own service manager must exit cleanly (gateway/run.py).

The planned-stop marker is written only by the Hermes CLI, so a ``systemctl
stop``/``restart`` — including the needrestart sweep that follows
``apt-daily-upgrade`` — used to look like a bare external kill: the gateway
exited non-zero, systemd marked the unit ``failed``, and the next boot pruned
live sessions as "left by a crashed gateway".

The container / OOM / bare-kill cases the non-zero exit exists for must keep it.
"""

import asyncio
import signal

import pytest

from gateway.run import (
    _shutdown_signal_is_service_manager_stop,
    _start_gateway_make_shutdown_signal_handler,
)

_SYSTEMD_CTX = {
    "signal": "SIGTERM",
    "ppid": 1,
    "systemd_invocation_id": "8f2c1d1b9b0a4d5e",
    "parent": {"pid": 1, "name": "systemd", "cmdline": "/usr/lib/systemd/systemd --system"},
}


def _ctx(**overrides):
    """A systemd system-unit shutdown context, with per-test overrides."""
    ctx = {k: (v.copy() if isinstance(v, dict) else v) for k, v in _SYSTEMD_CTX.items()}
    ctx.update(overrides)
    return ctx


def test_systemd_unit_stop_is_a_service_manager_stop():
    assert _shutdown_signal_is_service_manager_stop(signal.SIGTERM, _ctx()) is True


def test_user_unit_stop_is_a_service_manager_stop():
    """A user unit's parent is ``systemd --user`` — PID 1 is not the marker of a managed stop."""
    ctx = _ctx(ppid=920, parent={"pid": 920, "name": "systemd", "cmdline": "/usr/lib/systemd/systemd --user"})
    assert _shutdown_signal_is_service_manager_stop(signal.SIGTERM, ctx) is True


def test_container_pid1_without_invocation_id_is_not():
    """Docker's PID 1 has no INVOCATION_ID; that SIGTERM must keep the non-zero exit."""
    ctx = _ctx(parent={"pid": 0})
    ctx.pop("systemd_invocation_id")
    assert _shutdown_signal_is_service_manager_stop(signal.SIGTERM, ctx) is False


def test_bare_kill_from_a_shell_is_not():
    assert _shutdown_signal_is_service_manager_stop(
        signal.SIGTERM, _ctx(parent={"pid": 4242, "name": "bash"})) is False


def test_sigint_is_not_a_service_manager_stop():
    """Ctrl+C is already a planned stop upstream; this probe stays SIGTERM-only."""
    assert _shutdown_signal_is_service_manager_stop(signal.SIGINT, _ctx()) is False


def test_missing_context_is_not():
    """The snapshot is best-effort and may be None; absent evidence is not a managed stop."""
    assert _shutdown_signal_is_service_manager_stop(signal.SIGTERM, None) is False


class _FakeRunner:
    """Stand-in for GatewayRunner — the handler only sets a flag and awaits stop()."""

    def __init__(self):
        self._signal_initiated_shutdown = False
        self.stopped = False

    async def stop(self):
        self.stopped = True


@pytest.fixture
def quiet_forensics(monkeypatch):
    """Keep the handler off /proc and out of subprocesses; each test supplies its own context."""
    import gateway.shutdown_forensics as forensics
    import gateway.status as status

    monkeypatch.setattr(status, "consume_takeover_marker_for_self", lambda: False)
    monkeypatch.setattr(status, "consume_planned_stop_marker_for_self", lambda: False)
    monkeypatch.setattr(forensics, "format_context_for_log", lambda ctx: "ctx")
    monkeypatch.setattr(forensics, "spawn_async_diagnostic", lambda *a, **k: None)
    return forensics


async def _run_handler(ctx, forensics, monkeypatch):
    monkeypatch.setattr(forensics, "snapshot_shutdown_context", lambda received_signal=None: ctx)
    runner, flag = _FakeRunner(), [False]
    _start_gateway_make_shutdown_signal_handler(runner, flag)(signal.SIGTERM)
    await asyncio.sleep(0)  # let the task the handler scheduled for runner.stop() run
    return runner, flag


@pytest.mark.asyncio
async def test_handler_leaves_systemd_stop_unflagged(quiet_forensics, monkeypatch):
    """Unflagged means exit 0: the unit stops without being marked failed."""
    runner, flag = await _run_handler(_ctx(), quiet_forensics, monkeypatch)
    assert flag[0] is False
    assert runner._signal_initiated_shutdown is False
    assert runner.stopped is True


@pytest.mark.asyncio
async def test_handler_still_flags_a_bare_kill(quiet_forensics, monkeypatch):
    ctx = _ctx(parent={"pid": 4242, "name": "bash"})
    runner, flag = await _run_handler(ctx, quiet_forensics, monkeypatch)
    assert flag[0] is True
    assert runner._signal_initiated_shutdown is True
