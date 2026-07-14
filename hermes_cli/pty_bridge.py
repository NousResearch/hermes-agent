"""PTY bridge for `hermes dashboard` chat tab.

Wraps a child process behind a pseudo-terminal so its ANSI output can be
streamed to a browser-side terminal emulator (xterm.js) and typed
keystrokes can be fed back in.  The only caller today is the
``/api/pty`` WebSocket endpoint in ``hermes_cli.web_server``.

Design constraints:

* **POSIX-only.**  This module depends on ``fcntl``, ``termios``, and
  ``ptyprocess``, none of which exist on native Windows Python.  Native
  Windows ConPTY is a different API (Windows 10 build 17763+) and would
  need a separate Windows implementation (``pywinpty``) — that's tracked
  as a future enhancement.  On native Windows, importing this module
  raises :class:`ImportError` and the dashboard's ``/chat`` tab shows a
  WSL-recommended banner instead of crashing.  Every other feature in the
  dashboard (sessions, jobs, metrics, config editor) works natively.
* **Zero Node dependency on the server side.**  We use :mod:`ptyprocess`,
  which is a pure-Python wrapper around the OS calls.  The browser talks
  to the same ``hermes --tui`` binary it would launch from the CLI, so
  every TUI feature (slash popover, model picker, tool rows, markdown,
  skin engine, clarify/sudo/approval prompts) ships automatically.
* **Byte-safe I/O.**  Reads and writes go through the PTY master fd
  directly — we avoid :class:`ptyprocess.PtyProcessUnicode` because
  streaming ANSI is inherently byte-oriented and UTF-8 boundaries may land
  mid-read.
"""

from __future__ import annotations

import errno
import fcntl
import os
import select
import signal
import struct
import sys
import termios
import time
from typing import Optional, Sequence

try:
    import ptyprocess  # type: ignore
    _PTY_AVAILABLE = not sys.platform.startswith("win")
except ImportError:  # pragma: no cover - dev env without ptyprocess
    ptyprocess = None  # type: ignore
    _PTY_AVAILABLE = False


__all__ = ["PtyBridge", "PtyUnavailableError"]


# ``struct winsize`` packs rows/cols as unsigned short (0..65535).  We clamp
# well below that ceiling: real terminals never exceed a couple thousand
# columns, and a value above this is a broken probe (WSL2 reports
# columns=131072) rather than a genuine ultrawide.  Lower bound is 1 — a
# zero/negative dimension is the classic "no size yet" signal.
_MIN_DIMENSION = 1
_MAX_COLS = 2000
_MAX_ROWS = 1000


def _clamp_dimension(value: int, maximum: int) -> int:
    """Clamp a reported terminal dimension into ``[_MIN_DIMENSION, maximum]``.

    Non-integer / non-finite values fall back to ``_MIN_DIMENSION`` so a bad
    probe can never reach ``struct.pack`` and raise ``struct.error``.
    """
    try:
        n = int(value)
    except (TypeError, ValueError, OverflowError):
        return _MIN_DIMENSION
    if n < _MIN_DIMENSION:
        return _MIN_DIMENSION
    if n > maximum:
        return maximum
    return n


class PtyUnavailableError(RuntimeError):
    """Raised when a PTY cannot be created on this platform.

    Today this means native Windows (no ConPTY bindings) or a dev
    environment missing the ``ptyprocess`` dependency.  The dashboard
    surfaces the message to the user as a chat-tab banner.
    """


class PtyBridge:
    """Thin wrapper around ``ptyprocess.PtyProcess`` for byte streaming.

    Not thread-safe.  A single bridge is owned by the WebSocket handler
    that spawned it; the reader runs in an executor thread while writes
    happen on the event-loop thread.  Both sides are OK because the
    kernel PTY is the actual synchronization point — we never call
    :mod:`ptyprocess` methods concurrently, we only call ``os.read`` and
    ``os.write`` on the master fd, which is safe.
    """

    def __init__(self, proc: "ptyprocess.PtyProcess"):  # type: ignore[name-defined]
        self._proc = proc
        self._fd: int = proc.fd
        self._closed = False

    # -- lifecycle --------------------------------------------------------

    @classmethod
    def is_available(cls) -> bool:
        """True if a PTY can be spawned on this platform."""
        return bool(_PTY_AVAILABLE)

    @staticmethod
    def clamp_dimensions(cols: int, rows: int) -> tuple[int, int]:
        """Normalize browser dimensions for both initial spawn and resize."""
        return (
            _clamp_dimension(cols, _MAX_COLS),
            _clamp_dimension(rows, _MAX_ROWS),
        )

    @classmethod
    def spawn(
        cls,
        argv: Sequence[str],
        *,
        cwd: Optional[str] = None,
        env: Optional[dict] = None,
        cols: int = 80,
        rows: int = 24,
    ) -> "PtyBridge":
        """Spawn ``argv`` behind a new PTY and return a bridge.

        Raises :class:`PtyUnavailableError` if the platform can't host a
        PTY.  Raises :class:`FileNotFoundError` or :class:`OSError` for
        ordinary exec failures (missing binary, bad cwd, etc.).
        """
        if not _PTY_AVAILABLE:
            if sys.platform.startswith("win"):
                raise PtyUnavailableError(
                    "Pseudo-terminals are unavailable on this platform. "
                    "Hermes Agent supports Windows only via WSL."
                )
            if ptyprocess is None:
                raise PtyUnavailableError(
                    "The `ptyprocess` package is missing. "
                    "Install with: pip install ptyprocess "
                    "(or pip install -e '.[pty]')."
                )
            raise PtyUnavailableError("Pseudo-terminals are unavailable.")
        cols, rows = cls.clamp_dimensions(cols, rows)
        # PTY-hosted programs expect TERM to describe the terminal type.
        # CI often runs without TERM in the parent process, which makes
        # simple terminal probes like `tput cols` fail before winsize reads.
        # Preserve explicit caller overrides, but backfill a sensible default
        # when TERM is missing or blank.
        spawn_env = (os.environ.copy() if env is None else env.copy())
        if not spawn_env.get("TERM"):
            spawn_env["TERM"] = "xterm-256color"
        proc = ptyprocess.PtyProcess.spawn(  # type: ignore[union-attr]
            list(argv),
            cwd=cwd,
            env=spawn_env,
            dimensions=(rows, cols),
        )
        return cls(proc)

    @property
    def pid(self) -> int:
        return int(self._proc.pid)

    def is_alive(self) -> bool:
        if self._closed:
            return False
        try:
            return bool(self._proc.isalive())
        except Exception:
            return False

    # -- I/O --------------------------------------------------------------

    def read(self, timeout: float = 0.2) -> Optional[bytes]:
        """Read up to 64 KiB of raw bytes from the PTY master.

        Returns:
            * bytes — zero or more bytes of child output
            * empty bytes (``b""``) — no data available within ``timeout``
            * None — child has exited and the master fd is at EOF

        Never blocks longer than ``timeout`` seconds.  Safe to call after
        :meth:`close`; returns ``None`` in that case.
        """
        if self._closed:
            return None
        try:
            readable, _, _ = select.select([self._fd], [], [], timeout)
        except (OSError, ValueError):
            return None
        if not readable:
            return b""
        try:
            data = os.read(self._fd, 65536)
        except OSError as exc:
            # EIO on Linux = slave side closed.  EBADF = already closed.
            if exc.errno in {errno.EIO, errno.EBADF}:
                return None
            raise
        if not data:
            return None
        return data

    def write(self, data: bytes) -> None:
        """Write raw bytes to the PTY master (i.e. the child's stdin)."""
        if self._closed or not data:
            return
        # os.write can return a short write under load; loop until drained.
        view = memoryview(data)
        while view:
            try:
                n = os.write(self._fd, view)
            except OSError as exc:
                if exc.errno in {errno.EIO, errno.EBADF, errno.EPIPE}:
                    return
                raise
            if n <= 0:
                return
            view = view[n:]

    def resize(self, cols: int, rows: int) -> None:
        """Forward a terminal resize to the child via ``TIOCSWINSZ``.

        Dimensions are clamped to a sane range first.  Some hosts report
        garbage window sizes — the motivating case is WSL2, where xterm.js
        in the dashboard ``/chat`` tab can pick up ``columns=131072,
        rows=1`` from a broken winsize probe.  ``struct winsize`` packs each
        field as an unsigned short (max 65535), so an unclamped 131072 would
        raise ``struct.error`` (not ``OSError``) and break the resize path,
        leaving the TUI laid out for a one-row / absurdly-wide screen —
        which is what shows up as blank / disappearing text.
        """
        if self._closed:
            return
        cols = _clamp_dimension(cols, _MAX_COLS)
        rows = _clamp_dimension(rows, _MAX_ROWS)
        # struct winsize: rows, cols, xpixel, ypixel (all unsigned short)
        winsize = struct.pack("HHHH", rows, cols, 0, 0)
        try:
            fcntl.ioctl(self._fd, termios.TIOCSWINSZ, winsize)
        except OSError:
            pass

    # -- teardown ---------------------------------------------------------

    def close(self) -> None:
        """Terminate the child (SIGTERM → 0.5s grace → SIGKILL) and close fds.

        Idempotent.  Reaping the child is important so we don't leak
        zombies across the lifetime of the dashboard process.
        """
        if self._closed:
            return
        self._closed = True

        try:
            pgid = os.getpgid(self._proc.pid)  # windows-footgun: ok — POSIX-only module (imports fcntl/termios/ptyprocess at top)
        except Exception:
            pgid = None

        # SIGHUP is the conventional "your terminal went away" signal.
        # Send it to the whole foreground process group, not just the PTY
        # leader: the dashboard TUI starts helper children such as the Python
        # slash worker, and killing only the leader can strand those helpers.
        for sig in (signal.SIGHUP, signal.SIGTERM, signal.SIGKILL):  # windows-footgun: ok — POSIX-only module (imports fcntl/termios/ptyprocess at top)
            if not self._proc.isalive():
                break
            try:
                if pgid is not None:
                    os.killpg(pgid, sig)  # windows-footgun: ok — POSIX-only module (imports fcntl/termios/ptyprocess at top)
                else:
                    self._proc.kill(sig)
            except Exception:
                pass
            deadline = time.monotonic() + 0.5
            while self._proc.isalive() and time.monotonic() < deadline:
                time.sleep(0.02)

        try:
            self._proc.close(force=True)
        except Exception:
            pass

    # Context-manager sugar — handy in tests and ad-hoc scripts.
    def __enter__(self) -> "PtyBridge":
        return self

    def __exit__(self, *_exc) -> None:
        self.close()
