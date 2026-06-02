"""PTY bridge for `hermes dashboard` chat tab.

Wraps a child process behind a pseudo-terminal so its ANSI output can be
streamed to a browser-side terminal emulator (xterm.js) and typed
keystrokes can be fed back in. The only caller today is the ``/api/pty``
WebSocket endpoint in ``hermes_cli.web_server``.

Design constraints:

* Native terminal backend per platform. POSIX uses ``ptyprocess`` and Windows
  uses ConPTY through ``pywinpty``. Both are exposed through the same
  byte-oriented bridge so the dashboard endpoint does not need platform
  branches.
* Zero Node dependency on the server side. The browser talks to the same
  ``hermes --tui`` binary it would launch from the CLI, so every TUI feature
  ships automatically.
* Byte-safe I/O. POSIX reads and writes go through the PTY master fd directly;
  Windows ConPTY exposes text reads/writes, so we encode/decode at the bridge
  boundary while preserving the WebSocket's byte-stream contract.
"""

from __future__ import annotations

import errno
import os
import select
import sys
import time
from typing import Any, Optional, Sequence

_IS_WINDOWS = sys.platform.startswith("win")

if _IS_WINDOWS:
    try:
        from winpty import PtyProcess as _WinPtyProcess  # type: ignore

        _PTY_AVAILABLE = True
        _PTY_IMPORT_ERROR: Optional[BaseException] = None
    except ImportError as exc:  # pragma: no cover - env without pywinpty
        _WinPtyProcess = None  # type: ignore
        _PTY_AVAILABLE = False
        _PTY_IMPORT_ERROR = exc
    ptyprocess = None  # type: ignore
else:
    try:
        import ptyprocess  # type: ignore

        _PTY_AVAILABLE = True
        _PTY_IMPORT_ERROR = None
    except ImportError as exc:  # pragma: no cover - dev env without ptyprocess
        ptyprocess = None  # type: ignore
        _PTY_AVAILABLE = False
        _PTY_IMPORT_ERROR = exc
    _WinPtyProcess = None  # type: ignore


__all__ = ["PtyBridge", "PtyUnavailableError"]


class PtyUnavailableError(RuntimeError):
    """Raised when a PTY cannot be created on this platform.

    This usually means the platform-specific dependency is missing:
    ``ptyprocess`` on POSIX or ``pywinpty`` on Windows. The dashboard surfaces
    the message to the user as a chat-tab banner.
    """


class PtyBridge:
    """Thin wrapper around a platform PTY process for byte streaming.

    Not thread-safe. A single bridge is owned by the WebSocket handler that
    spawned it; the reader runs in an executor thread while writes happen on
    the event-loop thread. On POSIX, both sides are OK because the kernel PTY
    is the synchronization point. On Windows, reads wait on pywinpty's
    socket-backed fd before calling into ConPTY.
    """

    def __init__(self, proc: Any, *, backend: str):
        self._proc = proc
        self._fd: int = proc.fd
        self._backend = backend
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

        Raises :class:`PtyUnavailableError` if the platform can't host a PTY.
        Raises :class:`FileNotFoundError` or :class:`OSError` for ordinary exec
        failures (missing binary, bad cwd, etc.).
        """
        if not _PTY_AVAILABLE:
            if _IS_WINDOWS:
                raise PtyUnavailableError(
                    "The `pywinpty` package is missing. "
                    "Install with: pip install pywinpty."
                )
            if ptyprocess is None:
                raise PtyUnavailableError(
                    "The `ptyprocess` package is missing. "
                    "Install with: pip install ptyprocess "
                    "(or pip install -e '.[pty]')."
                )
            detail = f" ({_PTY_IMPORT_ERROR})" if _PTY_IMPORT_ERROR else ""
            raise PtyUnavailableError(f"Pseudo-terminals are unavailable{detail}.")

        # PTY-hosted programs expect TERM to describe the terminal type.
        # Preserve explicit caller overrides, but backfill a sensible default
        # when TERM is missing or blank.
        spawn_env = (os.environ.copy() if env is None else env.copy())
        if not spawn_env.get("TERM"):
            spawn_env["TERM"] = "xterm-256color"

        if _IS_WINDOWS:
            proc = _WinPtyProcess.spawn(  # type: ignore[union-attr]
                list(argv),
                cwd=cwd,
                env=spawn_env,
                dimensions=(rows, cols),
            )
            return cls(proc, backend="windows")

        proc = ptyprocess.PtyProcess.spawn(  # type: ignore[union-attr]
            list(argv),
            cwd=cwd,
            env=spawn_env,
            dimensions=(rows, cols),
        )
        return cls(proc, backend="posix")

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
            * bytes: zero or more bytes of child output
            * empty bytes (``b""``): no data available within ``timeout``
            * None: child has exited and the master fd is at EOF

        Never blocks longer than ``timeout`` seconds. Safe to call after
        :meth:`close`; returns ``None`` in that case.
        """
        if self._closed:
            return None
        try:
            readable, _, _ = select.select([self._fd], [], [], timeout)
        except (OSError, ValueError):
            return None
        if not readable:
            return None if not self.is_alive() else b""

        if self._backend == "windows":
            try:
                data = self._proc.read(65536)
            except (EOFError, OSError):
                return None
            if not data:
                return None
            if isinstance(data, bytes):
                return data
            return str(data).encode("utf-8", errors="replace")

        try:
            data = os.read(self._fd, 65536)
        except OSError as exc:
            # EIO on Linux = slave side closed. EBADF = already closed.
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
        if self._backend == "windows":
            try:
                self._proc.write(data.decode("utf-8", errors="replace"))
            except (EOFError, OSError):
                return
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
        """Forward a terminal resize to the child."""
        if self._closed:
            return
        if self._backend == "windows":
            try:
                self._proc.setwinsize(max(1, rows), max(1, cols))
            except Exception:
                pass
            return

        import fcntl
        import struct
        import termios

        # struct winsize: rows, cols, xpixel, ypixel (all unsigned short)
        winsize = struct.pack("HHHH", max(1, rows), max(1, cols), 0, 0)
        try:
            fcntl.ioctl(self._fd, termios.TIOCSWINSZ, winsize)
        except OSError:
            pass

    # -- teardown ---------------------------------------------------------

    def close(self) -> None:
        """Terminate the child and close fds.

        Idempotent. Reaping the child is important so we don't leak zombies
        across the lifetime of the dashboard process.
        """
        if self._closed:
            return
        self._closed = True

        if self._backend == "windows":
            try:
                self._proc.terminate(force=True)
            except Exception:
                pass
            try:
                self._proc.close(force=True)
            except Exception:
                pass
            return

        import signal

        # SIGHUP is the conventional "your terminal went away" signal. We
        # escalate if the child ignores it.
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

    def __enter__(self) -> "PtyBridge":
        return self

    def __exit__(self, *_exc) -> None:
        self.close()
