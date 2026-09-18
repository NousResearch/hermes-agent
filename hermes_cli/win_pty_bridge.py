"""Windows ConPTY bridge for the `hermes dashboard` chat tab."""

from __future__ import annotations

import asyncio
import logging
import sys
import time
from typing import Optional, Sequence

try:
    from winpty import PtyProcess  # type: ignore
    _PTY_AVAILABLE = sys.platform.startswith("win")
except ImportError:  # pragma: no cover - non-Windows or pywinpty missing
    PtyProcess = None  # type: ignore
    _PTY_AVAILABLE = False

_log = logging.getLogger(__name__)


__all__ = ["WinPtyBridge", "PtyUnavailableError"]


# Same clamp ceiling as the POSIX bridge so a broken winsize probe never reaches the resize call.
_MIN_DIMENSION = 1
_MAX_COLS = 2000
_MAX_ROWS = 1000
_WRITE_SHUTDOWN_GRACE = 1.0


def _clamp(value: int, maximum: int) -> int:
    try:
        n = int(value)
    except (TypeError, ValueError, OverflowError):
        return _MIN_DIMENSION
    return max(_MIN_DIMENSION, min(n, maximum))


class PtyUnavailableError(RuntimeError):
    """Raised when a PTY cannot be created on this platform."""


class WinPtyBridge:
    """pywinpty-backed bridge with the same interface as ``PtyBridge``. ``read`` runs inside
    ``run_in_executor``; ConPTY has no selectable fd, so reads poll and writes run in a worker
    thread to keep the same non-blocking event-loop contract as the POSIX bridge."""

    def __init__(self, proc: "PtyProcess") -> None:  # type: ignore[name-defined]
        self._proc = proc
        self._closed = False

    @classmethod
    def is_available(cls) -> bool:
        return bool(_PTY_AVAILABLE)

    @classmethod
    def spawn(
        cls, argv: Sequence[str], *, cwd: Optional[str] = None, env: Optional[dict] = None,
        cols: int = 80, rows: int = 24) -> "WinPtyBridge":
        if not _PTY_AVAILABLE:
            if PtyProcess is None:
                raise PtyUnavailableError("pywinpty is not installed. Install with: pip install pywinpty")
            raise PtyUnavailableError("ConPTY is unavailable on this platform.")
        # See pty_bridge.py: exact-preservation factory for the env=None fallback.
        from tools.environments.local import build_subprocess_env
        spawn_env = (
            build_subprocess_env(scrub_secrets=False, inherit_profile_home=False)
            if env is None else dict(env))
        spawn_env["TERM"] = spawn_env.get("TERM") or "xterm-256color"
        # pywinpty mirrors ptyprocess: dimensions=(rows, cols).
        return cls(PtyProcess.spawn(list(argv), cwd=cwd, env=spawn_env, dimensions=(rows, cols)))  # type: ignore[union-attr]

    @property
    def pid(self) -> int:
        return int(self._proc.pid)

    def is_alive(self) -> bool:
        try:
            return not self._closed and bool(self._proc.isalive())
        except Exception:
            return False

    def read(self, timeout: float = 0.2) -> Optional[bytes]:
        """Up to 64 KiB of child output."""
        if self._closed:
            return None
        try:
            data = self._proc.read(65536)  # pywinpty returns str
        except Exception:
            return None
        if not data:
            # No fd to select on; sleep so the executor thread doesn't pin a core while idle.
            time.sleep(min(timeout, 0.02))
            return b""
        if isinstance(data, bytes):
            return data
        # pywinpty decodes internally, so a multibyte UTF-8 sequence can split across reads;
        # xterm.js tolerates the rare replacement char (the one fidelity tradeoff vs POSIX).
        return data.encode("utf-8", errors="replace")

    def _write_blocking(self, data: bytes) -> bool:
        if self._closed:
            return False
        if not data:
            return True
        try:
            self._proc.write(data.decode("utf-8", errors="replace"))  # pywinpty wants text
        except Exception:
            return False
        return True

    async def write(self, data: bytes, *, timeout: float = 10.0) -> bool:
        """Write off-loop and tear down ConPTY when its input pipe wedges.

        ``wait_for(to_thread(...))`` alone only cancels the asyncio wrapper;
        the worker remains blocked inside pywinpty. Keep the worker future,
        force-close the ConPTY on timeout, and wait briefly for that close to
        release the blocked write before returning.

        Cancellation (the owning socket went away mid-write) is not evidence
        of a wedged child: the PTY outlives its socket by design, so give the
        in-flight write the grace window and only terminate if it never lands.
        """
        if self._closed:
            return False
        if not data:
            return True
        loop = asyncio.get_running_loop()
        write_future = loop.run_in_executor(None, self._write_blocking, data)
        try:
            return await asyncio.wait_for(
                asyncio.shield(write_future),
                timeout=max(0.0, timeout),
            )
        except asyncio.TimeoutError:
            await self._stop_stalled_write(write_future)
            return False
        except asyncio.CancelledError:
            await asyncio.shield(self._settle_or_stop_write(write_future))
            raise

    async def _settle_or_stop_write(self, write_future: asyncio.Future) -> None:
        """Let a cancelled write finish within the grace window; terminate only if it stalls."""
        try:
            await asyncio.wait_for(asyncio.shield(write_future), timeout=_WRITE_SHUTDOWN_GRACE)
            return
        except asyncio.TimeoutError:
            pass
        except Exception:
            return
        await self._stop_stalled_write(write_future)

    async def _stop_stalled_write(self, write_future: asyncio.Future) -> None:
        """Close ConPTY and reap the worker that was blocked in ``write``."""
        await asyncio.to_thread(self.close)
        try:
            await asyncio.wait_for(
                asyncio.shield(write_future),
                timeout=_WRITE_SHUTDOWN_GRACE,
            )
        except asyncio.TimeoutError:
            # The worker is still parked inside pywinpty after terminate(); it
            # now occupies a default-executor thread until the process exits.
            _log.warning(
                "ConPTY write worker did not exit within %.1fs of terminate(); thread leaked",
                _WRITE_SHUTDOWN_GRACE,
            )
        except Exception:
            pass

    def resize(self, cols: int, rows: int) -> None:
        if self._closed:
            return
        cols = _clamp(cols, _MAX_COLS)
        rows = _clamp(rows, _MAX_ROWS)
        try:
            self._proc.setwinsize(rows, cols)  # pywinpty: (rows, cols)
        except Exception:
            pass

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


# ---- BEGIN PLUGIN-COMPAT (revert-scheduled; see COMPAT_MANIFEST.md) ----
# Names external plugins imported from this module before the Sep 2026 decomposition.
# Internal code MUST NOT use these (scripts/check_compat_pointers.py fails CI if it does).
# The whole block is removed by reverting the commit that added it.
import os  # noqa: F401,E402
# ---- END PLUGIN-COMPAT ----
