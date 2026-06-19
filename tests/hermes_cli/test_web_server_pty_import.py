"""Test the platform-branched PTY bridge import in hermes_cli.web_server.

The /api/pty WebSocket handler in web_server.py picks its bridge at import
time via ``sys.platform.startswith("win")`` — Windows gets the ConPTY
backend, POSIX gets the fcntl/termios one.  Both branches must:

  1. Expose ``PtyBridge`` as the bridge class (or None) and
     ``PtyUnavailableError`` as an exception class.
  2. Set ``_PTY_BRIDGE_AVAILABLE`` correctly.
  3. Never raise at import time when the platform-native dependency is
     missing — the dashboard's non-chat tabs must keep loading.

This test asserts the live state on whichever platform CI runs on, plus a
source-text check confirming the branch shape is preserved so a future
refactor can't accidentally collapse it back to a POSIX-only import.
"""

from __future__ import annotations

import sys

import pytest

from hermes_cli import web_server


def test_web_server_exposes_pty_bridge_symbols():
    """The two symbols /api/pty consumes must always exist."""
    assert hasattr(web_server, "PtyBridge")
    assert hasattr(web_server, "PtyUnavailableError")
    assert hasattr(web_server, "_PTY_BRIDGE_AVAILABLE")
    # PtyUnavailableError is always an exception class — either the real
    # one from the platform bridge, or the local fallback class.
    assert isinstance(web_server.PtyUnavailableError, type)
    assert issubclass(web_server.PtyUnavailableError, BaseException)


@pytest.mark.skipif(not sys.platform.startswith("win"), reason="Windows-only")
def test_web_server_uses_win_pty_bridge_on_windows():
    """On native Windows, web_server.PtyBridge must be the ConPTY backend."""
    from hermes_cli.win_pty_bridge import WinPtyBridge

    assert web_server.PtyBridge is WinPtyBridge
    assert web_server._PTY_BRIDGE_AVAILABLE is True
    # And the error class must be the one from the same module so isinstance
    # checks in /api/pty's spawn fallback path actually work.
    from hermes_cli.win_pty_bridge import PtyUnavailableError as WinErr

    assert web_server.PtyUnavailableError is WinErr


@pytest.mark.skipif(sys.platform.startswith("win"), reason="POSIX-only")
def test_web_server_uses_posix_pty_bridge_on_posix():
    """On POSIX, the bridge must be the fcntl/termios PtyBridge."""
    from hermes_cli.pty_bridge import PtyBridge as PosixBridge
    from hermes_cli.pty_bridge import PtyUnavailableError as PosixErr

    assert web_server.PtyBridge is PosixBridge
    assert web_server._PTY_BRIDGE_AVAILABLE is True
    assert web_server.PtyUnavailableError is PosixErr


def test_pty_bridge_import_block_is_platform_branched():
    """Source-level guard: a future refactor must not collapse the branch
    back to a single POSIX import.  Reads web_server.py directly so this
    fails the same way on every OS — the runtime symbol checks above can
    pass even when the branch shape is wrong on the current platform."""
    src = pytest.importorskip("inspect").getsource(web_server)
    # The shape we expect (from PR #39913):
    #
    #   if sys.platform.startswith("win"):
    #       try:
    #           from hermes_cli.win_pty_bridge import WinPtyBridge as PtyBridge, ...
    #       except ImportError:
    #           PtyBridge = None
    #           ...
    #   else:
    #       try:
    #           from hermes_cli.pty_bridge import PtyBridge, PtyUnavailableError
    #       ...
    assert 'sys.platform.startswith("win")' in src or "sys.platform.startswith('win')" in src
    assert "from hermes_cli.win_pty_bridge import" in src
    assert "from hermes_cli.pty_bridge import" in src
