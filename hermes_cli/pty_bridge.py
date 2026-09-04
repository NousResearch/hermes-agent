"""PTY bridge for `hermes dashboard` chat tab.

Wraps a child process behind a pseudo-terminal so its ANSI output can be streamed to xterm.js and
keystrokes fed back in; the only caller is the ``/api/pty`` WebSocket endpoint in
``hermes_cli.web_server``. POSIX-only: depends on ``fcntl``, ``termios`` and ``ptyprocess`` (native
Windows would need a separate ConPTY/``pywinpty`` implementation).
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


# ``struct winsize`` packs rows/cols as unsigned short; we clamp well below that ceiling because a
# value above it is a broken probe (WSL2 reports columns=131072), not a genuine ultrawide. Lower
# bound is 1 — a zero/negative dimension is the classic "no size yet" signal.
_MIN_DIMENSION = 1
_MAX_COLS = 2000
_MAX_ROWS = 1000


def _clamp_dimension(value: int, maximum: int) -> int:
    """Clamp into ``[_MIN_DIMENSION, maximum]``; non-integer / non-finite values fall back to
    ``_MIN_DIMENSION`` so a bad probe can never reach ``struct.pack``.
    """
    try:
        n = int(value)
    except (TypeError, ValueError, OverflowError):
        return _MIN_DIMENSION
    return max(_MIN_DIMENSION, min(n, maximum))


class PtyUnavailableError(RuntimeError):
    """PTY cannot be created here (native Windows, or ``ptyprocess`` missing); the dashboard
    surfaces the message as a chat-tab banner.
    """


class PtyBridge:
    """Thin wrapper around ``ptyprocess.PtyProcess`` for byte streaming. Not thread-safe: owned by
    the WebSocket handler that spawned it; reads run in an executor thread, writes on the loop.
    """

    def __init__(self, proc: "ptyprocess.PtyProcess"):  # type: ignore[name-defined]
        self._proc = proc
        self._fd: int = proc.fd
        self._closed = False

    @classmethod
    def is_available(cls) -> bool:
        """True if a PTY can be spawned on this platform."""
        return bool(_PTY_AVAILABLE)

    @classmethod
    def spawn(
        cls, argv: Sequence[str], *, cwd: Optional[str] = None, env: Optional[dict] = None, cols: int = 80, rows: int = 24
    ) -> "PtyBridge":
        """Spawn ``argv`` behind a new PTY and return a bridge."""
        if not _PTY_AVAILABLE:
            if sys.platform.startswith("win"):
                raise PtyUnavailableError("Pseudo-terminals are unavailable on this platform. "
                                          "Hermes Agent supports Windows only via WSL.")
            raise PtyUnavailableError("The `ptyprocess` package is missing. "  # only other way _PTY_AVAILABLE is False
                                      "Install with: pip install ptyprocess (or pip install -e '.[pty]').")
        # env=None: callers own env policy (process_registry already sanitizes), so inherit via the
        # factory with exact preservation. Backfill TERM when missing/blank — CI often lacks it and
        # probes like `tput cols` then fail before winsize reads; explicit overrides are kept.
        from tools.environments.local import build_subprocess_env
        spawn_env = build_subprocess_env(scrub_secrets=False, inherit_profile_home=False) if env is None else env.copy()
        if not spawn_env.get("TERM"):
            spawn_env["TERM"] = "xterm-256color"
        proc = ptyprocess.PtyProcess.spawn(list(argv), cwd=cwd, env=spawn_env, dimensions=(rows, cols))  # type: ignore[union-attr]
        return cls(proc)

    @property
    def pid(self) -> int:
        return int(self._proc.pid)

    def is_alive(self) -> bool:
        try:
            return not self._closed and bool(self._proc.isalive())
        except Exception:
            return False

    def read(self, timeout: float = 0.2) -> Optional[bytes]:
        """Read up to 64 KiB from the PTY master, blocking at most ``timeout`` seconds.

        ``b""`` = nothing yet; ``None`` = EOF / closed (also after :meth:`close`).
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
        return data or None

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

        Clamped first: WSL2 via xterm.js reports garbage like ``columns=131072, rows=1`` and an
        unclamped unsigned-short pack raises ``struct.error`` (not ``OSError``), leaving the TUI
        laid out for a one-row screen — the blank/disappearing-text symptom.
        """
        if self._closed:
            return
        # struct winsize: rows, cols, xpixel, ypixel (all unsigned short)
        winsize = struct.pack("HHHH", _clamp_dimension(rows, _MAX_ROWS), _clamp_dimension(cols, _MAX_COLS), 0, 0)
        try:
            fcntl.ioctl(self._fd, termios.TIOCSWINSZ, winsize)
        except OSError:
            pass

    def close(self) -> None:
        """Terminate the child (SIGHUP → SIGTERM → SIGKILL, 0.5s grace each), reap it so the
        dashboard process never leaks zombies, and close fds. Idempotent.
        """
        if self._closed:
            return
        self._closed = True

        try:
            pgid = os.getpgid(self._proc.pid)  # windows-footgun: ok — POSIX-only module (imports fcntl/termios/ptyprocess at top)
        except Exception:
            pgid = None

        # Signal the whole process group, not just the PTY leader: the dashboard TUI starts helper
        # children (e.g. the Python slash worker) and killing only the leader strands them.
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

    def __enter__(self) -> "PtyBridge":
        return self

    def __exit__(self, *_exc) -> None:
        self.close()
