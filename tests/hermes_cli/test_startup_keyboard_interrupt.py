"""Verify that Ctrl-C during startup exits cleanly (exit code 130).

Regression test for #53704: pressing Ctrl+C while ``_prepare_agent_startup``
runs ``discover_plugins()`` used to propagate an uncaught ``KeyboardInterrupt``
out of ``main()``, dumping a full traceback.

``KeyboardInterrupt`` is a ``BaseException`` (not ``Exception``), so the
``except Exception`` guards *inside* ``_prepare_agent_startup`` do not catch
it.  The fix wraps the call site in ``main()`` with its own handler.
"""

from __future__ import annotations

import sys
from unittest.mock import patch

import pytest


def test_keyboard_interrupt_during_startup_exits_130(monkeypatch):
    """Ctrl-C during _prepare_agent_startup must exit 130, not traceback."""
    from hermes_cli import main as _main

    # Patch _prepare_agent_startup to raise KeyboardInterrupt immediately,
    # simulating Ctrl-C during plugin discovery.
    def _raise_keyboard_interrupt(args):
        raise KeyboardInterrupt

    monkeypatch.setattr(_main, "_prepare_agent_startup", _raise_keyboard_interrupt)

    # Use valid argv: "hermes chat -q hi" triggers _prepare_agent_startup.
    monkeypatch.setattr(sys, "argv", ["hermes", "chat", "-q", "hi"])

    # The key invariant: main() must catch KeyboardInterrupt and call
    # sys.exit(130), NOT let it propagate as an unhandled exception.
    with pytest.raises(SystemExit) as exc_info:
        _main.main()

    assert exc_info.value.code == 130, (
        f"Expected exit code 130 (SIGINT), got {exc_info.value.code}"
    )
