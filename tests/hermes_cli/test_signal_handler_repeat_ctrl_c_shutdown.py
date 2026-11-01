"""A repeat Ctrl+C during TUI shutdown must not raise KeyboardInterrupt.

The handler from ``_install_single_query_signal_handlers`` is also installed in
interactive mode; raising during ``_tui_shutdown`` / ``_run_cleanup`` aborted
session finalize mid-write and printed a traceback.
"""
from __future__ import annotations

import signal
from types import SimpleNamespace

import pytest

import cli as cli_mod
from hermes_cli import cli_single_query


@pytest.fixture
def handler(monkeypatch):
    installed = {}
    monkeypatch.setattr(signal, "signal", lambda sig, fn: installed.setdefault(sig, fn))
    monkeypatch.setattr(cli_mod, "_arm_exit_watchdog_on_shutdown_signal", lambda: None)
    monkeypatch.setattr(cli_mod, "_interrupt_agent_for_signal", lambda *a: None)
    monkeypatch.setattr(cli_mod, "_cleanup_done", False)
    monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)
    fake_cli = SimpleNamespace(agent=None)
    cli_single_query._install_single_query_signal_handlers(fake_cli)
    return fake_cli, installed[signal.SIGINT]


def _assert_ignored(fn):
    # Catch here: an escaped KeyboardInterrupt would abort the pytest session instead of failing.
    try:
        fn(signal.SIGINT, None)
    except KeyboardInterrupt:
        pytest.fail("repeat SIGINT during shutdown raised KeyboardInterrupt")


def test_sigint_before_shutdown_still_raises(handler):
    _, fn = handler
    with pytest.raises(KeyboardInterrupt):
        fn(signal.SIGINT, None)


def test_sigint_during_tui_shutdown_is_ignored(handler):
    fake_cli, fn = handler
    fake_cli._tui_shutting_down = True
    _assert_ignored(fn)


def test_sigint_after_cleanup_started_is_ignored(handler, monkeypatch):
    _, fn = handler
    monkeypatch.setattr(cli_mod, "_cleanup_done", True)
    _assert_ignored(fn)
