"""Windows ConPTY bridge for `hermes dashboard` chat tab.

Drop-in replacement for ``pty_bridge.PtyBridge`` on native Windows.
Uses ``pywinpty`` (the same ConPTY wrapper that powers VS Code's
integrated terminal) to spawn a child process behind a Windows
pseudo-console.

This module is imported *only* on Windows when ``pty_bridge`` fails to
load (due to missing fcntl/termios).  It exposes the same public API:
``PtyBridge``, ``PtyUnavailableError``.
"""

from __future__ import annotations

import os
import threading
import time
from typing import Optional, Sequence

try:
    from winpty import PtyProcess as WinPtyProcess  # type: ignore[import]
    _WINPTY_AVAILABLE = True
except ImportError:
    WinPtyProcess = None  # type: ignore[assignment,misc]
    _WINPTY_AVAILABLE = False

__all__ = ["PtyBridge", "PtyUnavailableError"]


class PtyUnavailableError(RuntimeError):
    """Raised when a PTY cannot be created on this platform.

    On Windows this means ``pywinpty`` is not installed.  The dashboard
    surfaces the message to the user as a chat-tab banner.
    """


class PtyBridge:
    """Windows ConPTY wrapper matching the POSIX PtyBridge interface.

    Uses a background reader thread to buffer output from the ConPTY,
    since pywinpty's ``read()`` is blocking with no timeout parameter.
    The public ``read(timeout)`` method drains the buffer non-blockingly.
    """

    def __init__(self, proc: "WinPtyProcess"):
        self._proc = proc
        self._closed = False
        self._buf = bytearray()
        self._buf_lock = threading.Lock()
        self._reader_thread = threading.Thread(target=self._reader_loop, daemon=True)
        self._reader_thread.start()

    def _reader_loop(self) -> None:
        """Background thread: continuously reads from the ConPTY."""
        while not self._closed:
            try:
                if not self._proc.isalive() and self._proc.eof():
                    break
                data = self._proc.read(65536)
                if data:
                    raw = data.encode("utf-8", errors="surrogateescape") if isinstance(data, str) else data
                    with self._buf_lock:
                        self._buf.extend(raw)
                else:
                    time.sleep(0.01)
            except EOFError:
                break
            except Exception:
                if self._closed:
                    break
                time.sleep(0.01)

    # -- lifecycle --------------------------------------------------------

    @classmethod
    def is_available(cls) -> bool:
        """True if pywinpty is installed and a ConPTY can be spawned."""
        return bool(_WINPTY_AVAILABLE)

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
        """Spawn ``argv`` behind a new ConPTY and return a bridge.

        Raises :class:`PtyUnavailableError` if ``pywinpty`` is missing.
        Raises :class:`FileNotFoundError` or :class:`OSError` for
        ordinary exec failures (missing binary, bad cwd, etc.).
        """
        if not _WINPTY_AVAILABLE:
            raise PtyUnavailableError(
                "pywinpty is not installed. Install with: pip install pywinpty"
            )

        spawn_env = (os.environ.copy() if env is None else env.copy())
        if not spawn_env.get("TERM"):
            spawn_env["TERM"] = "xterm-256color"

        proc = WinPtyProcess.spawn(
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
        """Read up to 64 KiB of raw bytes from the ConPTY.

        Returns:
            * bytes — child output
            * empty bytes (``b""``) — no data available within ``timeout``
            * None — child has exited and buffer is drained

        Never blocks longer than ``timeout`` seconds.
        """
        if self._closed:
            return None

        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            with self._buf_lock:
                if self._buf:
                    data = bytes(self._buf)
                    self._buf.clear()
                    return data
            if not self.is_alive():
                with self._buf_lock:
                    if self._buf:
                        data = bytes(self._buf)
                        self._buf.clear()
                        return data
                return None
            time.sleep(0.01)

        with self._buf_lock:
            if self._buf:
                data = bytes(self._buf)
                self._buf.clear()
                return data
        return b""

    def write(self, data: bytes) -> None:
        """Write raw bytes to the ConPTY (i.e. the child's stdin)."""
        if self._closed or not data:
            return
        try:
            text = data.decode("utf-8", errors="surrogateescape")
            self._proc.write(text)
        except (OSError, EOFError):
            return

    def resize(self, cols: int, rows: int) -> None:
        """Forward a terminal resize to the ConPTY."""
        if self._closed:
            return
        try:
            self._proc.setwinsize(max(1, rows), max(1, cols))
        except Exception:
            pass

    # -- teardown ---------------------------------------------------------

    def close(self) -> None:
        """Terminate the child and clean up.  Idempotent."""
        if self._closed:
            return
        self._closed = True

        if self._proc.isalive():
            try:
                self._proc.sendintr()
                deadline = time.monotonic() + 1.0
                while self._proc.isalive() and time.monotonic() < deadline:
                    time.sleep(0.05)
            except Exception:
                pass

        if self._proc.isalive():
            try:
                self._proc.terminate(force=True)
            except Exception:
                pass

        try:
            self._proc.close(force=True)
        except Exception:
            pass

    # Context-manager sugar
    def __enter__(self) -> "PtyBridge":
        return self

    def __exit__(self, *_exc) -> None:
        self.close()
