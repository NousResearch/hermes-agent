"""Windows ConPTY bridge for the `hermes dashboard` chat tab.

Drop-in counterpart to ``hermes_cli.pty_bridge.PtyBridge`` for native
Windows. Mirrors the exact public surface the ``/api/pty`` WebSocket
handler in ``hermes_cli.web_server`` consumes: ``spawn``, ``read``,
``write``, ``resize``, ``close``, ``is_available``, plus the
``PtyUnavailableError`` type.

Backed by ``pywinpty`` (already a declared win32 dependency in
pyproject.toml) instead of ``ptyprocess``/``fcntl``/``termios``, none of
which exist on native Windows. The read/write/terminate calls here match
the working winpty usage already shipping in ``tools/process_registry.py``.
"""

from __future__ import annotations

import os
import sys
import time
from typing import Optional, Sequence

try:
    from winpty import PtyProcess  # type: ignore
    _PTY_AVAILABLE = sys.platform.startswith("win")
except ImportError:  # pragma: no cover - non-Windows or pywinpty missing
    PtyProcess = None  # type: ignore
    _PTY_AVAILABLE = False


__all__ = ["WinPtyBridge", "PtyUnavailableError"]


# Same clamp ceiling as the POSIX bridge: a broken winsize probe must never
# reach the resize call. ConPTY tolerates large values better than ioctl,
# but we keep parity to avoid layout surprises.
_MIN_DIMENSION = 1
_MAX_COLS = 2000
_MAX_ROWS = 1000


def _clamp(value: int, maximum: int) -> int:
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
    """Raised when a PTY cannot be created on this platform."""


class WinPtyBridge:
    """pywinpty-backed bridge with the same interface as ``PtyBridge``.

    ``web_server`` calls :meth:`read` inside ``run_in_executor``, so a
    blocking/polling read here never stalls the event loop. ConPTY exposes
    no selectable fd, so we poll with a short sleep instead of ``select``.
    """

    def __init__(self, proc: "PtyProcess") -> None:  # type: ignore[name-defined]
        self._proc = proc
        self._closed = False

    # -- lifecycle --------------------------------------------------------

    @classmethod
    def is_available(cls) -> bool:
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
    ) -> "WinPtyBridge":
        if not _PTY_AVAILABLE:
            if PtyProcess is None:
                raise PtyUnavailableError(
                    "pywinpty is not installed. Install with: pip install pywinpty"
                )
            raise PtyUnavailableError("ConPTY is unavailable on this platform.")
        spawn_env = (os.environ.copy() if env is None else dict(env))
        if not spawn_env.get("TERM"):
            spawn_env["TERM"] = "xterm-256color"
        # pywinpty mirrors ptyprocess: dimensions=(rows, cols).
        # This call shape is the one already used in tools/process_registry.py.
        try:
            proc = PtyProcess.spawn(  # type: ignore[union-attr]
                list(argv),
                cwd=cwd,
                env=spawn_env,
                dimensions=(rows, cols),
            )
        except Exception as exc:
            # winpty-rs v0.4.1 panics on Chinese Windows when the ConPTY API
            # returns HRESULT(0x00000000) (S_OK) but the Rust crate wraps it
            # in an Err variant and calls unwrap().  The panic propagates as
            # a pyo3 PanicException (subclass of Exception).  Catch it here
            # and surface as PtyUnavailableError so the caller can fall back
            # gracefully instead of crashing the gateway.
            raise PtyUnavailableError(
                f"ConPTY spawn failed: {exc}"
            ) from exc
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
        """Up to 64 KiB of child output.

        Returns bytes, ``b""`` when nothing is available this tick, or
        ``None`` once the child has exited (EOF).
        """
        if self._closed:
            return None
        try:
            data = self._proc.read(65536)  # pywinpty returns str
        except EOFError:
            return None
        except Exception:
            return None
        if not data:
            # No fd to select on; poll politely so the executor thread
            # doesn't pin a core while the TUI is idle.
            time.sleep(min(timeout, 0.02))
            return b""
        if isinstance(data, bytes):
            return data
        # NOTE: pywinpty decodes internally, so a multibyte UTF-8 sequence
        # can in theory split across reads. xterm.js tolerates the rare
        # replacement char; this is the one fidelity tradeoff vs the POSIX
        # raw-fd path.
        return data.encode("utf-8", errors="replace")

    def write(self, data: bytes) -> None:
        if self._closed or not data:
            return
        try:
            # The dashboard sends raw keystroke bytes; pywinpty.write wants text.
            self._proc.write(data.decode("utf-8", errors="replace"))
        except Exception:
            return

    def resize(self, cols: int, rows: int) -> None:
        if self._closed:
            return
        cols = _clamp(cols, _MAX_COLS)
        rows = _clamp(rows, _MAX_ROWS)
        try:
            self._proc.setwinsize(rows, cols)  # pywinpty: (rows, cols)
        except Exception:
            pass

    # -- teardown ---------------------------------------------------------

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self._proc.terminate(force=True)
        except Exception:
            pass

    def __enter__(self) -> "WinPtyBridge":
        return self

    def __exit__(self, *_exc) -> None:
        self.close()
