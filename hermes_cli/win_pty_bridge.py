"""Windows ConPTY bridge for the `hermes dashboard` chat tab."""

from __future__ import annotations

import asyncio
import logging
import select
import sys
from typing import Optional, Sequence

try:
    from winpty import PtyProcess  # type: ignore
    _PTY_AVAILABLE = sys.platform.startswith("win")
except ImportError:  # pragma: no cover - non-Windows or pywinpty missing
    PtyProcess = None  # type: ignore
    _PTY_AVAILABLE = False

_log = logging.getLogger(__name__)


__all__ = ["PtyUnavailableError", "WinPtyBridge"]


# Same clamp ceiling as the POSIX bridge so a broken winsize probe never reaches the resize call.
_MIN_DIMENSION = 1
_MAX_COLS = 2000
_MAX_ROWS = 1000
_WRITE_TIMEOUT = 10.0
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
    ``run_in_executor`` and, like the POSIX bridge, returns within its timeout; writes run in a
    worker thread to keep the same non-blocking event-loop contract as the POSIX bridge."""

    def __init__(self, proc: PtyProcess) -> None:  # type: ignore[name-defined]
        self._proc = proc
        self._closed = False
        # Deadline guards for writes whose caller left while they waited on a fence.
        self._late_writes: set[asyncio.Task] = set()

    @classmethod
    def is_available(cls) -> bool:
        return bool(_PTY_AVAILABLE)

    @classmethod
    def spawn(
        cls, argv: Sequence[str], *, cwd: Optional[str] = None, env: Optional[dict] = None,
        cols: int = 80, rows: int = 24) -> WinPtyBridge:
        if not _PTY_AVAILABLE:
            if PtyProcess is None:
                raise PtyUnavailableError("pywinpty is not installed. Run hermes pm repair, then restart Hermes.")
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
        """Up to 64 KiB of child output, waiting at most ``timeout`` seconds for it.

        ``b""`` = nothing yet; ``None`` = EOF / closed (also after :meth:`close`).
        """
        if self._closed:
            return None
        # pywinpty's read is a blocking recv on the loopback socket its reader thread
        # fills (``fileno()``). Unbounded, an idle shell would hold this executor
        # thread until it next printed, and a few quiet terminals would take every
        # default-executor thread (min(32, cpu count + 4)) from keystrokes and cleanup.
        try:
            readable, _, _ = select.select([self._proc], [], [], timeout)
            if not readable:
                return b""
            data = self._proc.read(65536)  # pywinpty returns str
        except EOFError:
            return None
        except Exception:  # health: allow BLE001 -- normalize backend failures without leaking backend exception text
            if self._closed:
                return None
            raise OSError("PTY output capture failed during read") from None
        if not data:
            return b""  # pywinpty's placeholder for an empty ConPTY read
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

    async def write(self, data: bytes, *, timeout: Optional[float] = _WRITE_TIMEOUT, fence=None) -> bool:
        """Write off-loop and tear down ConPTY when its input pipe wedges.

        ``fence`` (see ``PtyBridge.write``) wraps the whole blocking write, and
        whatever it raises propagates. Waiting to enter the fence is not a stall:
        its profile lease can be held by other work (a cold SessionDB open, an
        attachment write) for longer than the deadline, so the deadline starts
        when the bytes reach ConPTY. ConPTY takes input whether or not the child
        reads it (conhost buffers it), so a write that misses the deadline means
        ConPTY itself is wedged: ``timeout=None``, which on POSIX waits for the
        child to read, still gets that deadline here and never holds the fence
        open-ended.

        ``wait_for(to_thread(...))`` alone only cancels the asyncio wrapper;
        the worker remains blocked inside pywinpty. Keep the worker future,
        force-close the ConPTY on timeout, and wait briefly for that close to
        release the blocked write before returning.

        Cancellation (the owning socket went away mid-write) is not evidence
        of a wedged child: the PTY outlives its socket by design, so give the
        in-flight write the grace window and only terminate if it never lands.
        A write cancelled while it still waits for its fence keeps waiting off
        the loop and gets the deadline once it reaches ConPTY.
        """
        if self._closed:
            return False
        if not data:
            return True
        loop = asyncio.get_running_loop()
        entered = asyncio.Event()

        def enter_conpty(chunk: bytes) -> bool:
            loop.call_soon_threadsafe(entered.set)
            return self._write_blocking(chunk)

        if fence is None:
            entered.set()
            write = self._write_blocking
        else:
            def write(chunk: bytes):
                return fence(lambda: enter_conpty(chunk))
        write_future = loop.run_in_executor(None, write, data)
        deadline = _WRITE_TIMEOUT if timeout is None else max(0.0, timeout)
        try:
            if not await self._entered_conpty(entered, write_future):
                return await write_future  # the fence refused: its result, or what it raised
        except asyncio.CancelledError:
            if entered.is_set():
                await asyncio.shield(self._settle_or_stop_write(write_future))
            else:
                self._bound_late_write(entered, write_future, deadline)
            raise
        try:
            return await asyncio.wait_for(asyncio.shield(write_future), timeout=deadline)
        except TimeoutError:
            await self._stop_stalled_write(write_future)
            return False
        except asyncio.CancelledError:
            await asyncio.shield(self._settle_or_stop_write(write_future))
            raise

    @staticmethod
    async def _entered_conpty(entered: asyncio.Event, write_future: asyncio.Future) -> bool:
        """Wait, with no deadline, until the write reaches ConPTY or the fence returns without it."""
        if entered.is_set():
            return True
        entering = asyncio.ensure_future(entered.wait())
        try:
            await asyncio.wait({entering, write_future}, return_when=asyncio.FIRST_COMPLETED)
        finally:
            entering.cancel()
        return entered.is_set()

    def _bound_late_write(self, entered: asyncio.Event, write_future: asyncio.Future, deadline: float) -> None:
        """Hold a write cancelled before its fence admitted it to the deadline once it reaches ConPTY."""
        async def bound() -> None:
            if await self._entered_conpty(entered, write_future):
                done, _ = await asyncio.wait({write_future}, timeout=deadline)
                if not done:
                    await self._stop_stalled_write(write_future)
                    return
            # Nobody awaits this write any more, so report what it raised here.
            if not write_future.cancelled() and (error := write_future.exception()) is not None:
                _log.debug("Host terminal write failed after its caller left", exc_info=error)

        task = asyncio.get_running_loop().create_task(bound())
        self._late_writes.add(task)
        task.add_done_callback(self._late_writes.discard)

    async def _settle_or_stop_write(self, write_future: asyncio.Future) -> None:
        """Let a cancelled write finish within the grace window; terminate only if it stalls."""
        try:
            await asyncio.wait_for(asyncio.shield(write_future), timeout=_WRITE_SHUTDOWN_GRACE)
            return
        except TimeoutError:
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
        except TimeoutError:
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

    def __enter__(self) -> WinPtyBridge:
        return self

    def __exit__(self, *_exc) -> None:
        self.close()
