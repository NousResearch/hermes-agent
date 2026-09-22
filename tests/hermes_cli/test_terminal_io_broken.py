"""CLI freezes UI paints after stdout/PTY EIO (#81521).

A stream-stall interrupt that corrupts the PTY used to leave the classic
CLI invalidating hundreds of times per second (escape-sequence flood).
Once EIO is observed, paints must stop.
"""

from __future__ import annotations

import errno
from unittest.mock import MagicMock

import pytest


def _make_cli_stub():
    from cli import HermesCLI

    cli = object.__new__(HermesCLI)
    cli._terminal_io_broken = False
    cli._resize_recovery_pending = False
    cli._last_invalidate = 0.0
    cli._pet_anim_running = False
    cli._app = MagicMock()
    return cli


class TestTerminalIoBrokenFreeze:
    def test_mark_terminal_io_broken_is_idempotent(self):
        cli = _make_cli_stub()
        cli._mark_terminal_io_broken("first")
        cli._mark_terminal_io_broken("second")
        assert cli._terminal_io_broken is True

    def test_invalidate_stops_after_eio(self):
        cli = _make_cli_stub()
        cli._app.invalidate.side_effect = OSError(errno.EIO, "Input/output error")

        cli._invalidate(min_interval=0.0)

        assert cli._terminal_io_broken is True
        assert cli._app.invalidate.call_count == 1

        cli._invalidate(min_interval=0.0)
        # Frozen — no further paints.
        assert cli._app.invalidate.call_count == 1

    def test_force_full_redraw_skipped_when_broken(self):
        cli = _make_cli_stub()
        cli._terminal_io_broken = True
        cli._force_full_redraw()
        cli._app.invalidate.assert_not_called()

    def test_recover_terminal_after_interrupt_skips_when_broken(self):
        cli = _make_cli_stub()
        cli._terminal_io_broken = True
        cli._force_full_redraw = MagicMock()
        cli._recover_terminal_after_interrupt()
        cli._force_full_redraw.assert_not_called()
