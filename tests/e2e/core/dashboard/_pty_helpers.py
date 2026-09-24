"""Client side of the dashboard's embedded-TUI WebSocket (``/api/pty``), as the browser xterm speaks it.

Frames are raw PTY bytes both ways; the only control frame the client sends is the resize escape
``\\x1b[RESIZE:<cols>;<rows>]`` (consumed by the server, never written to the child). The server
may send one JSON text frame (``{"type": "resume", ...}``) before the byte stream; it is recorded,
never rendered. Output is rendered through the terminal suite's VT emulator so assertions read what
the user sees, independent of how Ink split its repaints across frames.

Input follows the terminal suite's rule for Ink: the typed text and the submitting ``\\r`` are
separate writes, and the CR is only sent once the text is echoed in the composer.

Process bookkeeping (``ProcessLedger``) records every descendant of the dashboard by (pid, start
time) plus the session ids they lead: ``ptyprocess`` makes each TUI a session leader outside the
dashboard's process group, so a group kill does not reach it and only the dashboard's own shutdown
path can.
"""

from __future__ import annotations

import json
import os
import re
import signal
import threading
import time
from pathlib import Path
from typing import Any, Callable

from websockets.exceptions import ConnectionClosed
from websockets.sync.client import connect

from tests.e2e.core.terminal._vt import Screen

_WS = re.compile(r"\s+")
_DIGITS = re.compile(r"\d")
_SCROLLBAR = "│┃║▐▕█░▒▓"


def canon(text: str) -> str:
    """Whitespace-free form: independent of where the terminal (or Ink) wrapped a line."""
    return _WS.sub("", text)


def poll(fn: Callable[[], Any], timeout: float, what: str, interval: float = 0.05) -> Any:
    deadline = time.monotonic() + timeout
    while True:
        value = fn()
        if value:
            return value
        if time.monotonic() >= deadline:
            raise AssertionError(f"timed out after {timeout:.0f}s waiting for {what}")
        time.sleep(interval)


class WsTerm:
    """One browser-side xterm attached to ``/api/pty``."""

    def __init__(self, url: str, *, rows: int = 40, cols: int = 120) -> None:
        self.url = url
        self.screen = Screen(rows, cols)
        self.raw = bytearray()
        self.control: list[dict[str, Any]] = []
        self.first_frames: list[bytes] = []  # the first few binary frames, verbatim (replay proof)
        self.close_code: int | None = None
        self._lock = threading.Lock()
        self._closed = threading.Event()
        self.ws = connect(url, open_timeout=30, max_size=None, close_timeout=5)
        self.ws.send(f"\x1b[RESIZE:{cols};{rows}]")
        self._reader = threading.Thread(target=self._read_loop, name="ws-pty-reader", daemon=True)
        self._reader.start()

    # -- io -----------------------------------------------------------------------------------------

    def _read_loop(self) -> None:
        try:
            while True:
                frame = self.ws.recv()
                if isinstance(frame, str):
                    try:
                        payload = json.loads(frame)
                    except ValueError:
                        payload = None
                    if isinstance(payload, dict) and "type" in payload:
                        self.control.append(payload)
                        continue
                    frame = frame.encode("utf-8")
                elif len(self.first_frames) < 4:
                    self.first_frames.append(bytes(frame))
                with self._lock:
                    self.raw.extend(frame)
                    self.screen.feed(frame)
        except ConnectionClosed as exc:
            self.close_code = exc.rcvd.code if exc.rcvd else None
        except Exception:  # noqa: BLE001 - socket torn down under us by close()
            pass
        finally:
            self._closed.set()

    def write(self, data: str) -> None:
        self.ws.send(data.encode("utf-8"))

    def close(self) -> None:
        """Client-side close (a browser tab refresh / transient drop)."""
        try:
            self.ws.close()
        except Exception:  # noqa: BLE001
            pass
        self._closed.wait(10)

    @property
    def closed(self) -> bool:
        return self._closed.is_set()

    # -- screen views -------------------------------------------------------------------------------

    def lines(self) -> list[str]:
        with self._lock:
            rows = self.screen.display() if self.screen.alt_active else self.screen.transcript()
            cols = self.screen.cols
        return [ln[: cols - 1] if len(ln) >= cols and ln[cols - 1] in _SCROLLBAR else ln for ln in rows]

    def text(self) -> str:
        return canon("".join(self.lines()))

    def raw_text(self) -> str:
        with self._lock:
            return bytes(self.raw).decode("utf-8", "replace")

    def dump(self) -> str:
        return "\n".join(self.lines()[-60:])

    def wait_for(self, needle: str, timeout: float = 60.0) -> None:
        try:
            poll(lambda: canon(needle) in self.text() or self.closed, timeout, f"{needle[:60]!r} on the xterm")
        except AssertionError as exc:
            raise AssertionError(f"{exc}\n--- screen ---\n{self.dump()}") from None
        assert canon(needle) in self.text(), (
            f"/api/pty closed (code={self.close_code}) before {needle[:60]!r} rendered\n--- screen ---\n{self.dump()}")

    def wait_quiet(self, idle: float = 1.0, timeout: float = 60.0) -> None:
        """A settled frame: nothing but digits (status-bar clocks) changed for ``idle`` seconds."""
        state = {"frame": "", "since": time.monotonic()}

        def settled() -> bool:
            cur = _DIGITS.sub("#", "\n".join(self.lines()))
            now = time.monotonic()
            if cur != state["frame"]:
                state["frame"], state["since"] = cur, now
            return bool(cur.strip()) and now - state["since"] >= idle
        try:
            poll(settled, timeout, "a settled TUI frame", interval=0.1)
        except AssertionError as exc:
            raise AssertionError(f"{exc}\n--- screen ---\n{self.dump()}\n--- tail ---\n{bytes(self.raw[-600:])!r}") from None

    def submit(self, text: str, timeout: float = 30.0) -> None:
        """Type ``text``; once the composer echoes it, press Enter in a separate write."""
        before = self.text().count(canon(text))
        self.write(text)
        try:
            poll(lambda: self.text().count(canon(text)) > before or self.closed, timeout, f"echo of {text!r}")
        except AssertionError as exc:
            raise AssertionError(f"{exc}\n--- screen ---\n{self.dump()}") from None
        assert not self.closed, f"/api/pty closed (code={self.close_code}) while typing\n{self.dump()}"
        self.write("\r")


# -- process bookkeeping ---------------------------------------------------------------------------


def _stat(pid: int) -> list[str] | None:
    """Fields after ``comm`` of a live, non-zombie process (state is index 0)."""
    try:
        fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
    except (OSError, IndexError):
        return None
    return None if fields[0] in ("Z", "X") else fields


def _start_time(pid: int) -> int | None:
    fields = _stat(pid)
    return int(fields[19]) if fields else None


def cmdline(pid: int) -> str:
    try:
        return Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace").strip()
    except OSError:
        return "?"


def _all_stats() -> dict[int, list[str]]:
    out = {}
    for entry in os.listdir("/proc"):
        if entry.isdigit() and (fields := _stat(int(entry))) is not None:
            out[int(entry)] = fields
    return out


class ProcessLedger:
    """Every process the dashboard (transitively) spawned, keyed by (pid, start time)."""

    def __init__(self, root_pid: int) -> None:
        self.root_pid = root_pid
        self.seen: dict[tuple[int, int], str] = {}
        self.sids: set[int] = set()

    def snapshot(self) -> list[int]:
        stats = _all_stats()
        children: dict[int, list[int]] = {}
        for pid, fields in stats.items():
            children.setdefault(int(fields[1]), []).append(pid)
        found, todo = set(), [self.root_pid]
        while todo:
            for child in children.get(todo.pop(), ()):
                if child not in found:
                    found.add(child)
                    todo.append(child)
        # A PTY child is a session leader: anything in its session is ours even after reparenting.
        self.sids |= {pid for pid in found if int(stats[pid][3]) == pid}
        found |= {pid for pid, fields in stats.items() if int(fields[3]) in self.sids}
        for pid in found:
            self.seen.setdefault((pid, int(stats[pid][19])), cmdline(pid))
        return sorted(found)

    def survivors(self) -> list[str]:
        live = {pid for pid, started in self.seen if _start_time(pid) == started}
        live |= {pid for pid, fields in _all_stats().items() if int(fields[3]) in self.sids}
        return [f"{pid}: {cmdline(pid)[:160]}" for pid in sorted(live)]

    def kill_survivors(self) -> list[str]:
        """Hard cleanup of exactly the processes recorded here (identity-checked, never by pattern).
        Uses the real ``kill(2)``: a survivor reparented to init is outside pytest's subtree, which
        the suite-wide live-system guard would otherwise refuse, yet it is provably ours (same pid
        AND kernel start time as recorded while it descended from the dashboard)."""
        import posix
        left = self.survivors()
        for (pid, started) in list(self.seen):
            if _start_time(pid) == started:
                try:
                    posix.kill(pid, signal.SIGKILL)
                except (ProcessLookupError, PermissionError):
                    pass
        return left
