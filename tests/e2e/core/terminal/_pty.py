"""Drive a real ``hermes`` CLI / Ink TUI process under a real PTY, rendered through :mod:`._vt`.

The child runs with an isolated HOME/HERMES_HOME (only the fake LLM provider configured, no
inherited credentials) as the session leader of its own PTY, so SIGWINCH from a resize reaches the
foreground process group exactly as in a terminal emulator, and every process it spawns can be
found afterwards by session id (orphans keep the sid even after being reparented to init).

Input follows prompt_toolkit's rules: text and the submitting ``\\r`` never share one write (that
is a paste and inserts a newline); we wait until the typed text is echoed on screen, then send CR.
"""

from __future__ import annotations

import fcntl
import os
import re
import signal
import sqlite3
import struct
import subprocess
import sys
import termios
import threading
import time
from pathlib import Path
from typing import Callable

from tests.e2e.core.terminal._vt import Screen
from tests.fakes.fake_llm_provider import FakeLLMServer, write_hermes_home

REPO_ROOT = Path(__file__).resolve().parents[4]

# Right-edge scrollbar glyphs the Ink TUI paints in the last column of its transcript viewport.
_SCROLLBAR = "│┃║▐▕█░▒▓"
_WS = re.compile(r"\s+")
_DIGITS = re.compile(r"\d")

_SESSION_LEADER = (
    "import fcntl, os, sys, termios\n"
    "fcntl.ioctl(0, termios.TIOCSCTTY, 0)\n"
    "os.execv(sys.argv[1], sys.argv[1:])\n"
)


def canon(text: str) -> str:
    """Whitespace-free form: independent of where the terminal (or Ink) wrapped a line."""
    return _WS.sub("", text)


def poll(fn: Callable[[], object], *, timeout: float, what: str, interval: float = 0.05):
    deadline = time.monotonic() + timeout
    while True:
        value = fn()
        if value:
            return value
        if time.monotonic() >= deadline:
            raise AssertionError(f"timed out after {timeout:.0f}s waiting for {what}")
        time.sleep(interval)


def _operator_home() -> Path:
    import pwd
    return Path(pwd.getpwuid(os.getuid()).pw_dir)


def _sid_of(pid: int) -> int | None:
    try:
        stat = Path(f"/proc/{pid}/stat").read_text()
    except OSError:
        return None
    # comm may contain spaces/parens: fields after the last ')' are fixed.
    fields = stat.rsplit(")", 1)[1].split()
    if fields[0] == "Z":
        return None  # zombie: already dead, only waiting to be reaped
    return int(fields[3])


def _start_time(pid: int) -> int | None:
    """Kernel start time of a live (non-zombie) process, the stable half of its identity."""
    try:
        fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
    except OSError:
        return None
    return None if fields[0] == "Z" else int(fields[19])


def session_members(sid: int) -> list[int]:
    out = []
    for entry in os.listdir("/proc"):
        if entry.isdigit() and _sid_of(int(entry)) == sid:
            out.append(int(entry))
    return out


def cmdline(pid: int) -> str:
    try:
        return Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace")
    except OSError:
        return "?"


class PtyHermes:
    """One interactive ``hermes`` process on a PTY with an emulated screen."""

    def __init__(self, root: Path, argv_tail: list[str], llm: FakeLLMServer, *, rows: int, cols: int,
                 extra_config: str = "") -> None:
        self.root = root
        self.home = root / "home"
        self.hermes_home = self.home / ".hermes"
        operator = (_operator_home() / ".hermes").resolve()
        assert operator not in (self.hermes_home.resolve(), *self.hermes_home.resolve().parents), (
            f"sandbox {self.hermes_home} sits inside the operator's Hermes home")
        write_hermes_home(self.hermes_home, llm.base_url, extra_config=extra_config)
        self.llm = llm
        self.screen = Screen(rows, cols)
        self.raw = bytearray()
        self._lock = threading.Lock()
        (root / "tmp").mkdir(parents=True, exist_ok=True)
        (root / "work").mkdir(parents=True, exist_ok=True)
        env = {k: os.environ[k] for k in ("PATH", "LANG", "LC_ALL") if k in os.environ}
        env.update(
            HOME=str(self.home), HERMES_HOME=str(self.hermes_home), PYTHONPATH=str(REPO_ROOT),
            TMPDIR=str(root / "tmp"), TERM="xterm-256color", COLORTERM="truecolor",
            PYTHONUNBUFFERED="1", HERMES_STATE_DB_GUARD_BYPASS="1",
        )
        master, slave = os.openpty()
        self._set_winsize(master, rows, cols)
        self.master = master
        self.pts = os.ttyname(slave)
        self.proc = subprocess.Popen(
            [sys.executable, "-c", _SESSION_LEADER, sys.executable, "-m", "hermes_cli.main", *argv_tail],
            stdin=slave, stdout=slave, stderr=slave, cwd=str(root / "work"), env=env,
            start_new_session=True, close_fds=True)
        os.close(slave)
        self.sid = self.proc.pid  # start_new_session: the child is its own session leader
        self.seen_members: set[tuple[int, int]] = set()
        self._reader = threading.Thread(target=self._read_loop, name="pty-reader", daemon=True)
        self._reader.start()

    # -- io -------------------------------------------------------------------------------------

    @staticmethod
    def _set_winsize(fd: int, rows: int, cols: int) -> None:
        fcntl.ioctl(fd, termios.TIOCSWINSZ, struct.pack("HHHH", rows, cols, 0, 0))

    def _read_loop(self) -> None:
        while True:
            try:
                data = os.read(self.master, 65536)
            except OSError:
                return
            if not data:
                return
            with self._lock:
                self.raw.extend(data)
                self.screen.feed(data)

    def write(self, data: str) -> None:
        os.write(self.master, data.encode("utf-8"))

    def resize(self, rows: int, cols: int) -> None:
        with self._lock:
            self._set_winsize(self.master, rows, cols)  # kernel delivers SIGWINCH to the fg pgrp
            self.screen.resize(rows, cols)

    # -- screen views ---------------------------------------------------------------------------

    def lines(self) -> list[str]:
        """What the user can see or scroll back to: the alt screen when a fullscreen UI owns it,
        otherwise scrollback + the main screen. Scrollbar glyphs in the last column are dropped."""
        with self._lock:
            rows = self.screen.display() if self.screen.alt_active else self.screen.transcript()
            cols = self.screen.cols
        out = []
        for line in rows:
            if len(line) >= cols and line[cols - 1] in _SCROLLBAR:
                line = line[: cols - 1]
            out.append(line)
        return out

    def text(self) -> str:
        return canon("".join(self.lines()))

    def count(self, needle: str) -> int:
        return self.text().count(canon(needle))

    def wait_for_text(self, needle: str, timeout: float = 60.0) -> None:
        try:
            poll(lambda: canon(needle) in self.text() or self._dead(), timeout=timeout,
                 what=f"{needle[:40]!r} on screen")
        except AssertionError as exc:
            raise AssertionError(f"{exc}\n--- screen ---\n{self.dump()}") from None
        assert self.proc.poll() is None, f"hermes exited ({self.proc.returncode}) waiting for {needle[:40]!r}\n{self.dump()}"

    def wait_quiet(self, idle: float = 1.0, timeout: float = 30.0) -> None:
        """Wait for a settled frame: the screen has not changed for ``idle`` seconds, ignoring
        digits (status-bar clocks such as elapsed/since-last-turn tick every second while idle;
        a busy spinner or a still-streaming reply changes non-digit cells and keeps us waiting)."""
        def frame() -> str:
            return _DIGITS.sub("#", "\n".join(self.lines()))
        state = {"frame": frame(), "since": time.monotonic()}

        def settled() -> bool:
            self.track_members()
            cur = frame()
            now = time.monotonic()
            if cur != state["frame"]:
                state["frame"], state["since"] = cur, now
            return now - state["since"] >= idle
        try:
            poll(settled, timeout=timeout, what="a settled frame", interval=0.1)
        except AssertionError as exc:
            with self._lock:
                tail = bytes(self.raw[-600:])
            raise AssertionError(f"{exc}\n--- screen ---\n{self.dump()}\n--- last bytes ---\n{tail!r}") from None

    def _raw_mode(self) -> bool:
        """The app put the slave side in full raw mode (cfmakeraw / libuv RAW: no canonical
        editing, no echo, no CR->NL). Startup terminal probes use a partial raw mode that keeps
        ICRNL, and a CR typed then would reach the app as a newline, not Enter."""
        fd = os.open(self.pts, os.O_RDWR | os.O_NOCTTY)
        try:
            iflag, _o, _c, lflag = termios.tcgetattr(fd)[:4]
        finally:
            os.close(fd)
        return not (lflag & (termios.ICANON | termios.ECHO) or iflag & termios.ICRNL)

    def wait_ready(self, timeout: float = 120.0) -> None:
        """The UI owns the terminal (raw mode) and its first frame has settled."""
        deadline = time.monotonic() + timeout
        while True:
            remaining = max(1.0, deadline - time.monotonic())
            poll(lambda: self._dead() or self._raw_mode(), timeout=remaining,
                 what="the UI to take the terminal (raw mode)")
            assert self.proc.poll() is None, f"hermes exited ({self.proc.returncode}) during startup\n{self.dump()}"
            self.wait_quiet(1.0, timeout=max(1.0, deadline - time.monotonic()))
            if self._raw_mode():
                return
            if time.monotonic() >= deadline:
                raise AssertionError(f"UI never settled in raw mode\n{self.dump()}")

    def _dead(self) -> bool:
        return self.proc.poll() is not None

    def dump(self) -> str:
        return "\n".join(self.lines()[-80:])

    # -- interaction ----------------------------------------------------------------------------

    def submit(self, text: str, timeout: float = 30.0) -> None:
        """Type ``text``, wait for its echo in the composer, then press Enter in a separate write."""
        before = self.count(text)
        self.write(text)
        poll(lambda: self.count(text) > before or self._dead(), timeout=timeout, what=f"echo of {text!r}")
        # Not synchronization but input pacing: the classic CLI treats an Enter arriving within
        # 50 ms of the last buffer change as a pasted newline (_RAPID_INPUT_ENTER_WINDOW_S), so a
        # human-speed gap (10x that window) must separate the text from the submitting CR.
        time.sleep(0.5)
        self.write("\r")

    def track_members(self) -> None:
        """Remember every process in the PTY session and every descendant of the child (a
        descendant that called setsid() has left the session but is still ours)."""
        found = {(pid, _start_time(pid)) for pid in session_members(self.sid)}
        try:
            import psutil
            found |= {(c.pid, _start_time(c.pid)) for c in psutil.Process(self.proc.pid).children(recursive=True)}
        except Exception:  # noqa: BLE001 - the child may exit between the scan and the walk
            pass
        self.seen_members.update(item for item in found if item[1] is not None)

    def exit(self, command: str = "/exit", timeout: float = 60.0) -> int:
        self.track_members()
        self.submit(command)
        deadline = time.monotonic() + timeout
        while self.proc.poll() is None and time.monotonic() < deadline:
            self.track_members()
            time.sleep(0.05)
        if self.proc.poll() is None:
            raise AssertionError(f"hermes did not exit within {timeout:.0f}s of {command}\n{self.dump()}")
        return self.proc.returncode

    def leftover_processes(self, timeout: float = 15.0) -> list[str]:
        """Processes still alive in the child's session, or seen as its descendants, after exit.

        The profile's ``gateway run`` is not a leftover: ``hermes chat`` attaches to it and it
        keeps serving (cron, messaging, the next client) after the client exits; ``close()`` still
        tears it down with the rest of this harness's tree."""
        from gateway.status import looks_like_gateway_command_line

        def alive() -> list[int]:
            pids = set(session_members(self.sid))
            # Same pid AND same start time: a recycled pid is not our leftover.
            pids |= {pid for pid, started in self.seen_members if _start_time(pid) == started}
            return sorted(pid for pid in pids if not looks_like_gateway_command_line(cmdline(pid)))
        try:
            poll(lambda: not alive(), timeout=timeout, what="the PTY session to empty")
        except AssertionError:
            return [f"{p}: {cmdline(p)}" for p in alive()]
        return []

    def close(self) -> None:
        """Hard cleanup of exactly the tree this harness spawned (by session id, never by pattern)."""
        try:
            os.killpg(self.sid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass
        leftovers = set(session_members(self.sid))  # members that moved to their own process group
        leftovers |= {pid for pid, started in self.seen_members if _start_time(pid) == started}
        for pid in leftovers:
            try:
                os.kill(pid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass
        try:
            self.proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            pass
        try:
            os.close(self.master)
        except OSError:
            pass
        self._reader.join(timeout=5)
        (self.root / "pty.raw").write_bytes(bytes(self.raw))

    # -- persisted transcript -------------------------------------------------------------------

    def wait_turns_persisted(self, n_replies: int, timeout: float = 90.0) -> None:
        """Block until state.db holds ``n_replies`` non-empty assistant rows: the agent finished the
        turn (a completion signal independent of what the screen shows, so a garbled render is
        reported by the rendering assertions instead of as a timeout)."""
        def done() -> bool:
            if self._dead():
                return True
            try:
                rows = self.persisted_messages()
            except sqlite3.Error:
                return False
            return sum(1 for _s, r, c in rows if r == "assistant" and c.strip()) >= n_replies
        poll(done, timeout=timeout, what=f"{n_replies} assistant replies persisted", interval=0.1)
        assert self.proc.poll() is None, f"hermes exited ({self.proc.returncode}) mid-turn\n{self.dump()}"

    def persisted_messages(self) -> list[tuple[str, str, str]]:
        """(session_id, role, content) for every user/assistant row in the sandbox state.db."""
        db = self.hermes_home / "state.db"
        if not db.exists():
            return []
        conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
        try:
            rows = conn.execute(
                "SELECT session_id, role, COALESCE(content, '') FROM messages "
                "WHERE role IN ('user', 'assistant') ORDER BY id").fetchall()
        finally:
            conn.close()
        return [(str(s), str(r), str(c)) for s, r, c in rows]
