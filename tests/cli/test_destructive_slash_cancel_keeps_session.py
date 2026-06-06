"""Regression tests: cancelling a destructive-confirm keeps the REPL alive.

``process_command`` is contracted to return ``bool`` — ``True`` to continue,
``False`` to exit (see cli.py). The REPL loop exits the whole session when the
call is falsy (``if not self.process_command(...): app.exit()``).

The destructive session commands (/clear, /new, /reset, /undo) gate on
``_confirm_destructive_slash``, which returns ``None`` when the user picks
"Cancel — keep current conversation". The cancel branches must return ``True``
so cancelling preserves the session instead of tearing it down.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from tests.cli.test_cli_init import _make_cli


@pytest.fixture
def cancelled_cli():
    """A CLI whose destructive-confirm prompt always cancels (returns None)."""
    cli = _make_cli()
    cli._confirm_destructive_slash = lambda *_a, **_kw: None
    # Guard rails: these must NOT run when the user cancels.
    cli.new_session = MagicMock()
    cli.undo_last = MagicMock()
    return cli


@pytest.mark.parametrize("command", ["/clear", "/new", "/reset", "/undo", "/undo 3"])
def test_cancel_returns_true_and_preserves_session(cancelled_cli, command):
    result = cancelled_cli.process_command(command)

    # A falsy return tears down the REPL — the opposite of the user's intent.
    assert result is True
    cancelled_cli.new_session.assert_not_called()
    cancelled_cli.undo_last.assert_not_called()


def test_undo_invalid_count_returns_true(cancelled_cli):
    """A bad /undo count reports the error but must not exit the session."""
    result = cancelled_cli.process_command("/undo notanumber")

    assert result is True
    cancelled_cli.undo_last.assert_not_called()
