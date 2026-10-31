"""Regression test for #140890 — a repeat Ctrl+C must not unwind the TUI shutdown.

In interactive ``hermes`` the SIGINT handler is the one-shot one
(``_install_single_query_signal_handlers`` -> ``_signal_handler_q``), and it raised
``KeyboardInterrupt`` unconditionally. A second Ctrl+C while ``Shutting down…`` was on
screen therefore landed inside ``_tui_shutdown``: ``on_session_finalize``, the memory
provider shutdown and ``_release_active_session`` were skipped and the shell got a raw
traceback whose frame moved with timing (YAML load in ``relay_shared_metrics``,
``_release_active_session``, ``_pet_stop_anim``).

The fix marks the teardown on the CLI (``_tui_shutting_down``, set at the top of
``_tui_shutdown``) and has the handler return silently while it — or ``_cleanup_done``,
for the paths that only go through ``_run_cleanup`` — is set. The exit watchdog armed at
the top of the handler still covers a teardown that wedges.

These tests drive the real handler captured through a patched ``signal.signal``; the flag
ordering test drives the real ``_tui_shutdown``.
"""

from __future__ import annotations

import os
import signal
from collections.abc import Callable

import pytest

import cli
from hermes_cli.cli_single_query import _install_single_query_signal_handlers


class _StubCLI:
    """The slice of ``HermesCLI`` the signal handler touches."""

    def __init__(self, *, shutting_down: bool = False, with_agent: bool = False):
        self.agent = object() if with_agent else None
        self._tui_shutting_down = shutting_down


def _capture_handler(monkeypatch, stub) -> Callable[[int, object], None]:
    """Install the real handlers over a fake ``signal.signal`` and return ``_signal_handler_q``."""
    captured: dict[int, Callable[[int, object], None]] = {}

    monkeypatch.setattr(cli, "_arm_exit_watchdog_on_shutdown_signal", lambda: None)
    monkeypatch.setattr(signal, "signal", lambda signum, handler: captured.__setitem__(signum, handler))
    _install_single_query_signal_handlers(stub)

    assert signal.SIGINT in captured, "the handler was never installed"
    return captured[signal.SIGINT]


def _call_handler(handler, signum) -> None:
    """Invoke the handler; a ``KeyboardInterrupt`` escaping is the regression, not an error."""
    try:
        handler(signum, None)
    except KeyboardInterrupt:
        pytest.fail("KeyboardInterrupt escaped the signal handler and unwound the teardown")


@pytest.mark.parametrize("signum", [signal.SIGINT, signal.SIGTERM])
def test_before_teardown_the_signal_still_unwinds(monkeypatch, signum):
    """The first signal keeps raising: that is how Ctrl+C exits the prompt at all."""
    monkeypatch.setattr(cli, "_cleanup_done", False)
    handler = _capture_handler(monkeypatch, _StubCLI())

    with pytest.raises(KeyboardInterrupt):
        handler(signum, None)


@pytest.mark.parametrize("signum", [signal.SIGINT, signal.SIGTERM])
def test_repeat_signal_is_ignored_once_the_tui_teardown_started(monkeypatch, signum):
    """#140890: this raised into ``_tui_shutdown`` and aborted finalize/release."""
    monkeypatch.setattr(cli, "_cleanup_done", False)
    handler = _capture_handler(monkeypatch, _StubCLI(shutting_down=True))

    _call_handler(handler, signum)


@pytest.mark.parametrize("signum", [signal.SIGINT, signal.SIGTERM])
def test_repeat_signal_is_ignored_once_cleanup_started(monkeypatch, signum):
    """The classic CLI and the atexit path only flip ``_cleanup_done``."""
    monkeypatch.setattr(cli, "_cleanup_done", True)
    handler = _capture_handler(monkeypatch, _StubCLI())

    _call_handler(handler, signum)


@pytest.mark.parametrize("signum", [signal.SIGINT, signal.SIGTERM])
def test_teardown_signal_returns_before_the_agent_grace(monkeypatch, signum):
    """The guard belongs above the interrupt: that helper sleeps 1.5 s for any agent, idle or not."""
    monkeypatch.setattr(cli, "_cleanup_done", False)
    interrupts: list[int] = []
    monkeypatch.setattr(cli, "_interrupt_agent_for_signal",
                        lambda agent, sig: interrupts.append(sig))
    handler = _capture_handler(monkeypatch, _StubCLI(shutting_down=True, with_agent=True))

    _call_handler(handler, signum)

    assert interrupts == [], "a repeat signal paid the SIGTERM grace before returning"


@pytest.mark.parametrize("signum", [signal.SIGINT, signal.SIGTERM])
def test_first_signal_still_interrupts_the_agent(monkeypatch, signum):
    """The reorder must not drop the hard interrupt that kills the tool's subprocess group."""
    monkeypatch.setattr(cli, "_cleanup_done", False)
    interrupts: list[int] = []
    monkeypatch.setattr(cli, "_interrupt_agent_for_signal",
                        lambda agent, sig: interrupts.append(sig))
    handler = _capture_handler(monkeypatch, _StubCLI(with_agent=True))

    with pytest.raises(KeyboardInterrupt):
        handler(signum, None)

    assert interrupts == [signum]


def test_kanban_worker_keeps_its_hard_exit_during_teardown(monkeypatch):
    """A kanban worker never runs _tui_shutdown's unwind, so the early return must not skip it."""
    monkeypatch.setattr(cli, "_cleanup_done", False)
    monkeypatch.setattr(cli, "_interrupt_agent_for_signal", lambda agent, sig: None)
    monkeypatch.setattr(cli, "_flush_one_shot_session_store", lambda _cli: None)
    monkeypatch.setattr(cli, "_flush_logging_and_stdio", lambda: None)
    monkeypatch.setenv("HERMES_KANBAN_TASK", "t_140890")
    monkeypatch.setattr(signal, "alarm", lambda _seconds: None)

    class _HardExit(Exception):
        pass

    monkeypatch.setattr(os, "_exit", lambda code: (_ for _ in ()).throw(_HardExit(code)))

    handler = _capture_handler(monkeypatch, _StubCLI(shutting_down=True, with_agent=True))
    with pytest.raises(_HardExit):
        handler(signal.SIGTERM, None)


class _TeardownReached(Exception):
    """Raised by the probe at the first teardown step, carrying the flag it saw."""

class _TeardownProbe:
    """Stops ``_tui_shutdown`` at its first step so the flag can be read there."""

    _should_exit = False

    def _pet_stop_anim(self):
        raise _TeardownReached(getattr(self, "_tui_shutting_down", "<unset>"))


def test_tui_shutdown_marks_the_teardown_before_its_first_step():
    """The flag has to be set before the work, not after it: the interrupt lands in between."""
    from cli import HermesCLI

    with pytest.raises(_TeardownReached) as excinfo:
        HermesCLI._tui_shutdown(_TeardownProbe())

    assert excinfo.value.args == (True,)
