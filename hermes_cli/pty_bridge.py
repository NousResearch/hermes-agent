"""PTY bridge for `hermes dashboard` chat tab.

Wraps a child process behind a pseudo-terminal so its ANSI output can be
streamed to a browser-side terminal emulator (xterm.js) and typed
keystrokes can be fed back in.  The only caller today is the
``/api/pty`` WebSocket endpoint in ``hermes_cli.web_server``.

Design constraints:

* **POSIX-only.**  Hermes Agent supports Windows exclusively via WSL, which
  exposes a native POSIX PTY via ``openpty(3)``.  Native Windows Python
  has no PTY; :class:`PtyUnavailableError` is raised with a user-readable
  install/platform message so the dashboard can render a banner instead of
  crashing.
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
        # Let caller-supplied env fully override inheritance; if they pass
        # None we inherit the server's env (same semantics as subprocess).
        spawn_env = os.environ.copy() if env is None else env
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
            if exc.errno in (errno.EIO, errno.EBADF):
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
                if exc.errno in (errno.EIO, errno.EBADF, errno.EPIPE):
                    return
                raise
            if n <= 0:
                return
            view = view[n:]

    def resize(self, cols: int, rows: int) -> None:
        """Forward a terminal resize to the child via ``TIOCSWINSZ``."""
        if self._closed:
            return
        # struct winsize: rows, cols, xpixel, ypixel (all unsigned short)
        winsize = struct.pack("HHHH", max(1, rows), max(1, cols), 0, 0)
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

        # SIGHUP is the conventional "your terminal went away" signal.
        # We escalate if the child ignores it.
        for sig in (signal.SIGHUP, signal.SIGTERM, signal.SIGKILL):
            if not self._proc.isalive():
                break
            try:
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
