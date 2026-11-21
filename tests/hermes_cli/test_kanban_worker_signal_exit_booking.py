"""A signal-stopped Kanban worker must exit 128+signal, never rc=0 (#136651).

The one-shot signal handler's Kanban fast path hard-exits the worker so a stuck
non-daemon thread cannot keep the PID alive (#28181). It used ``os._exit(0)``,
so a worker whose systemd scope is OOM-killed (``MemoryMax``) and SIGTERMs that
path was reaped with a REAL rc=0 — booked by ``_classify_dead_worker_exit`` as a
clean-exit ``protocol_violation`` (a violation-only budget that never charges
``consecutive_failures``), sending operators hunting an agent-behavior bug while
the actual cause is memory. The shell convention (128+signal) keeps the
immediate hard exit AND lets the dispatcher book an ordinary crash that trips
the breaker.
"""

from __future__ import annotations

import os
import signal
import sys
import time
from types import SimpleNamespace

import pytest

import cli
from hermes_cli import kanban_db_dispatch as kbd


class _ExitCalled(Exception):
    """Raised in place of ``os._exit`` so the handler's exit code is observable."""

    def __init__(self, code: int):
        super().__init__(f"os._exit({code})")
        self.code = code


@pytest.fixture(autouse=True)
def _restore_signal_state():
    yield
    signal.alarm(0)
    signal.signal(signal.SIGINT, signal.default_int_handler)
    signal.signal(signal.SIGTERM, signal.SIG_DFL)
    signal.signal(signal.SIGHUP, signal.SIG_DFL)


def test_reaped_143_books_as_crash_not_protocol_violation(monkeypatch):
    """Dispatcher side of the contract: a 128+SIGTERM reaped exit is an ordinary
    crash, while rc=0 stays the clean-exit protocol-violation booking."""
    monkeypatch.setattr(
        kbd, "_recent_worker_exits", {4242: (((143 & 0xFF) << 8), time.time())}
    )
    assert kbd._classify_worker_exit(4242) == ("nonzero_exit", 143)
    dead = kbd._classify_dead_worker_exit(4242, "host#w")
    assert dead.protocol_violation is False
    assert dead.run_outcome == "crashed"
    assert "143" in dead.error_text
    # rc=0 (the old signal-path exit code) is still booked as the clean-exit
    # protocol violation — that path is what mis-booked the OOM kills.
    monkeypatch.setattr(
        kbd, "_recent_worker_exits", {4243: ((0 & 0xFF) << 8, time.time())}
    )
    clean = kbd._classify_dead_worker_exit(4243, "host#w")
    assert clean.protocol_violation is True
    assert clean.event_payload["exit_code"] == 0


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX signal semantics only")
@pytest.mark.parametrize(
    ("sigattr", "expected"),
    [("SIGTERM", 143), ("SIGINT", 130), ("SIGHUP", 129)],
)
def test_kanban_signal_handler_hard_exits_128_plus_signal(
    monkeypatch, sigattr, expected
):
    """The handler's Kanban fast path must hard-exit with 128+signal so the
    reaped worker never reads as a clean exit to the dispatcher."""
    monkeypatch.setenv("HERMES_KANBAN_TASK", "t_136651")
    monkeypatch.setattr(cli, "_arm_exit_watchdog_on_shutdown_signal", lambda: None)
    monkeypatch.setattr(cli, "_interrupt_agent_for_signal", lambda agent, signum: None)
    monkeypatch.setattr(cli, "_flush_one_shot_session_store", lambda c: None)
    monkeypatch.setattr(cli, "_flush_logging_and_stdio", lambda: None)
    from tools.environments import base as environments_base

    monkeypatch.setattr(
        environments_base, "kill_live_foreground_processes", lambda now=True: None
    )
    monkeypatch.setattr(
        os, "_exit", lambda code: (_ for _ in ()).throw(_ExitCalled(code))
    )

    from hermes_cli.cli_single_query import _install_single_query_signal_handlers

    _install_single_query_signal_handlers(SimpleNamespace(agent=None))
    with pytest.raises(_ExitCalled) as excinfo:
        # raise_signal runs the handler synchronously in the main thread; the
        # patched os._exit aborts it before the real process exit.
        signal.raise_signal(getattr(signal, sigattr))
    assert excinfo.value.code == expected


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX signal semantics only")
def test_non_kanban_signal_still_unwinds_via_keyboard_interrupt(monkeypatch):
    """Without HERMES_KANBAN_TASK the handler must keep raising KeyboardInterrupt
    (the interactive ``chat -q`` unwind), not hard-exit the process."""
    monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)
    monkeypatch.setattr(cli, "_arm_exit_watchdog_on_shutdown_signal", lambda: None)
    monkeypatch.setattr(cli, "_interrupt_agent_for_signal", lambda agent, signum: None)
    exits: list[int] = []
    monkeypatch.setattr(os, "_exit", lambda code: exits.append(code))

    from hermes_cli.cli_single_query import _install_single_query_signal_handlers

    _install_single_query_signal_handlers(SimpleNamespace(agent=None))
    with pytest.raises(KeyboardInterrupt):
        signal.raise_signal(signal.SIGTERM)
    assert exits == []
