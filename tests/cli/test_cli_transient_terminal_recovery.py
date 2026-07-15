"""Regression tests for the transient-terminal-death resurrection loop in
``HermesCLI.run()``.

On hosted web terminals (Lightning.ai cloudspaces, Codespaces, ttyd/gotty
proxies, flaky SSH PTYs) the pseudo-terminal backing fd 0/1 briefly drops when
a browser tab reconnects or a proxy recycles a websocket. A stdout write during
that window raises ``OSError(EIO)`` (or ``EAGAIN``), which prompt_toolkit turns
into a fatal renderer teardown. Historically ``app.run()`` then returned and the
CLI EXITED — dumping the user to a shell and forcing a full ``hermes`` restart
(losing the live in-memory session) roughly every time the terminal blipped.

The fix classifies such errors as transient (via
``_cli_is_transient_terminal_error``) and re-enters ``app.run()`` with the same
Application instead of exiting — but ONLY while the terminal is actually alive
again, so a genuinely dead fd can't cause a hot busy-loop.

These tests pin the classifier's decision matrix. The classifier is the single
source of truth for the loop's resurrect-vs-exit branch, so covering it fully
protects the behavior without having to construct a real prompt_toolkit app.
"""

from __future__ import annotations

import errno

import pytest


class _FakeStdin:
    def __init__(self, is_tty: bool):
        self._is_tty = is_tty

    def isatty(self) -> bool:
        return self._is_tty


def _alive_fstat(fd):  # pretend fds 0/1 are healthy
    return object()


def _dead_fstat(fd):  # pretend the fd is gone
    raise OSError(errno.EBADF, "Bad file descriptor")


def _classify(exc, *, tty=True, fstat=_alive_fstat):
    import cli as cli_mod

    return cli_mod._cli_is_transient_terminal_error(
        exc, _stdin=_FakeStdin(tty), _fstat=fstat
    )


# ── transient errors, terminal recovered → resurrect ────────────────────────

def test_eio_with_live_tty_is_transient():
    assert _classify(OSError(errno.EIO, "Input/output error")) is True


def test_eagain_with_live_tty_is_transient():
    assert _classify(OSError(errno.EAGAIN, "Resource temporarily unavailable")) is True


def test_async_generator_runtimeerror_is_transient():
    # The cascading error prompt_toolkit raises after the EIO renderer crash.
    exc = RuntimeError("aclose(): asynchronous generator is already running")
    assert _classify(exc) is True


# ── genuine exit intents / dead fds → do NOT resurrect ──────────────────────

def test_ebadf_is_not_transient():
    assert _classify(OSError(errno.EBADF, "Bad file descriptor")) is False


def test_einval_is_not_transient():
    assert _classify(OSError(errno.EINVAL, "Invalid argument")) is False


def test_unrelated_runtimeerror_is_not_transient():
    assert _classify(RuntimeError("some other real bug")) is False


def test_stdin_selector_keyerror_is_not_transient():
    # KeyError isn't OSError/RuntimeError → never classified transient.
    assert _classify(KeyError("0 is not registered")) is False


# ── the dead-terminal gate: transient-looking error but fd is gone ──────────

def test_eio_with_dead_tty_is_not_transient():
    """EIO looks transient, but if the tty reports not-a-tty we must NOT
    resurrect — otherwise the loop hot-spins on a permanently dead terminal."""
    assert _classify(OSError(errno.EIO, "io"), tty=False) is False


def test_eio_with_unfstatable_fd_is_not_transient():
    """EIO + live-looking tty but fstat fails → treated as genuinely dead."""
    assert _classify(OSError(errno.EIO, "io"), tty=True, fstat=_dead_fstat) is False


# ── module import / attribute presence guard ────────────────────────────────

def test_classifier_is_module_level_and_importable():
    """The loop aliases this module-level symbol; if it's ever inlined again
    this test fails loudly so the test above don't silently rot."""
    import cli as cli_mod

    assert callable(getattr(cli_mod, "_cli_is_transient_terminal_error", None))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-v"]))
