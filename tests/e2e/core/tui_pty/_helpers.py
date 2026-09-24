"""Drive a real ``hermes --tui`` inside a private tmux server and read what the user would see.

tmux is the terminal emulator here: it owns the grid, reflows it on resize and keeps the
scrollback, exactly as it does for a user running the TUI inside tmux. We read it back with
``capture-pane`` (``-S -`` for scrollback, ``-J`` to join rows tmux re-wrapped) and ask tmux for
the cursor position, the alternate-screen flag and the scrollback size.

Every transcript word the fake provider streams is a unique token (``<tag>w<NNN>``), so a
*word ledger* over the captured text tells lost, duplicated and reordered words apart.

Isolation: the tmux server has its own socket and a config written into the test's tmp dir;
the TUI gets a sandbox HOME/HERMES_HOME wired only to the loopback fake provider; cleanup kills
the tmux server and every process of the pane's session by session id, never by pattern.
"""

from __future__ import annotations

import os
import re
import shutil
import signal
import sqlite3
import subprocess
import sys
import termios
import time
import unicodedata
import uuid
from collections import Counter
from pathlib import Path
from typing import Callable, Iterable

import pytest

from tests.e2e.core.terminal._pty import cmdline, poll, session_members
from tests.fakes.fake_llm_provider import write_hermes_home

REPO_ROOT = Path(__file__).resolve().parents[4]
TITLE = "Scripted session title"

# Right-edge scrollbar glyphs the Ink TUI paints in the last column of its transcript viewport.
_SCROLLBAR = "│┃║▐▕█░▒▓"
_DIGITS = re.compile(r"\d")

# A quiet sandbox: no update probe, no title call eating scripted turns, no memory/skills noise.
BASE_CONFIG = (
    "updates:\n  check: false\n"
    "auxiliary:\n  title_generation:\n    enabled: false\n"
    "memory:\n  memory_enabled: false\n  user_profile_enabled: false\n"
)


def require_tui() -> None:
    """Skip (or fail in CI, where the bundle is prebuilt) when the TUI cannot run."""
    missing = [what for what, ok in (
        ("tmux", shutil.which("tmux") is not None),
        ("node", shutil.which("node") is not None),
        ("ui-tui/dist/entry.js", (REPO_ROOT / "ui-tui" / "dist" / "entry.js").is_file()),
    ) if not ok]
    if not missing:
        return
    if os.environ.get("HERMES_E2E_REQUIRE_TUI") == "1" and missing != ["tmux"]:
        pytest.fail(f"{missing} missing but HERMES_E2E_REQUIRE_TUI=1")
    pytest.skip(f"needs {missing}")


def words(tag: str, n: int, start: int = 0) -> list[str]:
    return [f"{tag}w{i:03d}" for i in range(start, start + n)]


def paragraph(tag: str, n: int) -> str:
    """One long paragraph (no hard newlines): its wrap width is the renderer's content width."""
    return " ".join(words(tag, n))


def display_width(text: str) -> int:
    return sum(2 if unicodedata.east_asian_width(ch) in "WF" else 1 for ch in text)


def ledger(text: str, expected: Iterable[str], pattern: str) -> dict[str, list[str]]:
    """Classify every expected token as lost / duplicated, and report order violations."""
    expected = list(expected)
    seen = re.findall(pattern, text)
    counts = Counter(seen)
    lost = [w for w in expected if counts[w] == 0]
    dup = [w for w in expected if counts[w] > 1]
    firsts = [w for w in dict.fromkeys(seen) if w in set(expected)]
    order = [w for w in expected if counts[w]]
    return {"lost": lost, "dup": dup, "misordered": [] if firsts == order else firsts[:5]}


def ledger_problems(text: str, expected: Iterable[str], pattern: str) -> str:
    """'' when every expected token is present exactly once and in order, else a short report."""
    report = ledger(text, expected, pattern)
    return "; ".join(f"{k}={v[:8]}{'…' if len(v) > 8 else ''} ({len(v)})" for k, v in report.items() if v)


class Cells:
    """Named verdicts of one scenario run. Each becomes its own test id so a known bug can be
    marked ``xfail(strict=True)`` on exactly the cell it breaks while the rest stay enforced."""

    def __init__(self) -> None:
        self.results: dict[str, tuple[bool, str]] = {}
        self.error: str | None = None

    def add(self, name: str, problem: str, screen: str = "") -> None:
        self.results[name] = (not problem, f"{problem}\n--- screen ---\n{screen}" if problem else "")

    def check(self, name: str) -> None:
        if self.error is not None:
            raise RuntimeError(f"scenario failed before its cells could be evaluated:\n{self.error}")
        if name not in self.results:
            raise RuntimeError(f"cell {name!r} was never evaluated (scenario ended early)")
        ok, detail = self.results[name]
        assert ok, f"[{name}] {detail}"


def cell_params(names: Iterable[str], known: dict[str, str]) -> list:
    """pytest params for ``names``; a KNOWN cell is a strict xfail that turns red once fixed."""
    return [pytest.param(n, id=n, marks=[pytest.mark.xfail(strict=True, raises=AssertionError,
                                                           reason=known[n])] if n in known else [])
            for n in names]


def run_cells(body: Callable[[Cells], None]) -> Cells:
    """Run one scenario; a harness failure (timeout, crash) is kept and re-raised by every cell."""
    cells = Cells()
    try:
        body(cells)
    except Exception as exc:  # noqa: BLE001 - surfaced verbatim by Cells.check
        cells.error = f"{type(exc).__name__}: {exc}"
    return cells


def reply_rows_width(rows: list[str], token_re: str) -> int:
    """Widest rendered row (in columns) holding a token of the given paragraph."""
    return max((display_width(r) for r in rows if re.search(token_re, r)), default=0)


def width_problem(rows: list[str], token_re: str, cols: int, slack: int = 12) -> str:
    """Content width follows the terminal: the paragraph wraps within ``slack`` of the edge and
    never past it (#35804)."""
    widest = reply_rows_width(rows, token_re)
    if widest > cols:
        return f"reply rows are {widest} cols wide on a {cols}-col terminal"
    if widest < cols - slack:
        return f"reply wraps at {widest} cols on a {cols}-col terminal (content width did not follow)"
    return ""


class TmuxTui:
    """One ``hermes --tui`` in a private tmux server."""

    def __init__(self, root: Path, base_url: str, *, cols: int = 120, rows: int = 50,
                 extra_config: str = "", args: Iterable[str] = ("--yolo",), inline: bool = False,
                 env_extra: dict[str, str] | None = None, write_home: bool = True) -> None:
        self.root = root
        self.home = root / "home"
        self.hermes_home = self.home / ".hermes"
        self.sock = f"hermes-tui-e2e-{uuid.uuid4().hex[:10]}"
        if write_home:
            write_hermes_home(self.hermes_home, base_url, extra_config=BASE_CONFIG + extra_config)
        for sub in ("tmp", "work"):
            (root / sub).mkdir(parents=True, exist_ok=True)
        conf = root / "tmux.conf"
        conf.write_text(
            # window-size manual must be set after the session exists (tmux 3.3 dies on it here).
            "set -g history-limit 100000\nset -g status off\nset -g remain-on-exit on\n"
            "set -g default-terminal tmux-256color\nset -g escape-time 10\n", encoding="utf-8")
        env = {k: v for k, v in os.environ.items()
               if not k.startswith(("HERMES_", "TMUX", "OPENAI_", "OPENROUTER_", "ANTHROPIC_"))}
        env.update(HOME=str(self.home), HERMES_HOME=str(self.hermes_home), PYTHONPATH=str(REPO_ROOT),
                   TMPDIR=str(root / "tmp"), LANG="C.UTF-8", LC_ALL="C.UTF-8", PYTHONUNBUFFERED="1",
                   HERMES_STATE_DB_GUARD_BYPASS="1", HERMES_TUI_INLINE="1" if inline else "0")
        env.update(env_extra or {})
        argv = [sys.executable, "-m", "hermes_cli.main", "--tui", *args]
        subprocess.run(["tmux", "-L", self.sock, "-f", str(conf), "new-session", "-d", "-s", "p",
                        "-x", str(cols), "-y", str(rows), "-c", str(root / "work"), *argv],
                       env=env, check=True, timeout=30)
        self.tmux("set", "-g", "window-size", "manual")
        self.pane_pid = int(self.tmux("display", "-p", "-t", "p", "#{pane_pid}").strip())
        self.pane_tty = self.tmux("display", "-p", "-t", "p", "#{pane_tty}").strip()
        self.seen: set[int] = set()

    # -- tmux ------------------------------------------------------------------------------------

    def tmux(self, *args: str) -> str:
        return subprocess.run(["tmux", "-L", self.sock, *args], capture_output=True, text=True,
                              timeout=30).stdout

    def fmt(self, spec: str) -> str:
        return self.tmux("display", "-p", "-t", "p", spec).strip()

    def alive(self) -> bool:
        return self.fmt("#{pane_dead}") == "0" and bool(self.fmt("#{pane_pid}"))

    def size(self) -> tuple[int, int]:
        cols, rows = self.fmt("#{pane_width} #{pane_height}").split()
        return int(cols), int(rows)

    def resize(self, cols: int, rows: int | None = None) -> None:
        rows = rows if rows is not None else self.size()[1]
        self.tmux("resize-window", "-t", "p", "-x", str(cols), "-y", str(rows))

    def rows(self, *, history: bool = False) -> list[str]:
        """Visible rows (or scrollback + visible) with the Ink scrollbar column stripped."""
        span = ("-S", "-", "-E", "-") if history else ()
        cols = self.size()[0]
        out = []
        for line in self.tmux("capture-pane", "-p", "-t", "p", *span).split("\n"):
            if display_width(line) >= cols and line and line[-1] in _SCROLLBAR:
                line = line[:-1]
            out.append(line.rstrip())
        return out

    def joined(self, *, history: bool = False) -> str:
        """Rows tmux re-wrapped on a resize joined back into logical lines (``-J``)."""
        span = ("-S", "-", "-E", "-") if history else ()
        return self.tmux("capture-pane", "-p", "-J", "-t", "p", *span)

    def text(self) -> str:
        return "\n".join(self.rows(history=True))

    def dump(self, n: int = 70) -> str:
        return "\n".join(self.rows()[-n:])

    # -- input -----------------------------------------------------------------------------------

    def type(self, text: str) -> None:
        self.tmux("send-keys", "-t", "p", "-l", text)

    def key(self, *keys: str) -> None:
        self.tmux("send-keys", "-t", "p", *keys)

    def submit(self, text: str, timeout: float = 30.0) -> None:
        """Type ``text``, wait for its echo in the composer, then press Enter in its own write."""
        probe = text[-12:]
        before = self.text().count(probe)
        self.type(text)
        poll(lambda: self.text().count(probe) > before or not self.alive(), timeout=timeout,
             what=f"echo of {text!r}")
        # Input pacing, not synchronization: text and Enter in one burst is a paste.
        time.sleep(0.4)
        self.key("Enter")

    # -- waiting ---------------------------------------------------------------------------------

    def wait_for(self, needle: str | Callable[[str], bool], timeout: float = 60.0,
                 *, history: bool = True) -> None:
        test = needle if callable(needle) else (lambda t: needle in t)
        what = getattr(needle, "__name__", None) if callable(needle) else repr(needle[:50])
        try:
            poll(lambda: (not self.alive()) or test("\n".join(self.rows(history=history))),
                 timeout=timeout, what=f"{what} on screen")
        except AssertionError as exc:
            raise AssertionError(f"{exc}\n--- screen ---\n{self.dump()}") from None
        assert self.alive(), f"hermes exited while waiting for {what}\n{self.dump()}"

    def wait_quiet(self, idle: float = 1.0, timeout: float = 45.0) -> None:
        """A settled frame: unchanged (ignoring ticking digits) for ``idle`` seconds."""
        def frame() -> str:
            self.track()
            return _DIGITS.sub("#", "\n".join(self.rows()))
        state = {"frame": frame(), "since": time.monotonic()}

        def settled() -> bool:
            cur, now = frame(), time.monotonic()
            if cur != state["frame"]:
                state["frame"], state["since"] = cur, now
            return now - state["since"] >= idle
        try:
            poll(settled, timeout=timeout, what="a settled frame", interval=0.1)
        except AssertionError as exc:
            raise AssertionError(f"{exc}\n--- screen ---\n{self.dump()}") from None

    def _raw_mode(self) -> bool:
        fd = os.open(self.pane_tty, os.O_RDWR | os.O_NOCTTY)
        try:
            iflag, _o, _c, lflag = termios.tcgetattr(fd)[:4]
        finally:
            os.close(fd)
        return not (lflag & (termios.ICANON | termios.ECHO) or iflag & termios.ICRNL)

    def wait_ready(self, timeout: float = 120.0) -> None:
        """The Ink UI owns the terminal (raw mode) and its composer frame has settled."""
        try:
            poll(lambda: not self.alive() or self._raw_mode(), timeout=timeout,
                 what="the TUI to take the terminal (raw mode)")
        except AssertionError as exc:
            raise AssertionError(f"{exc}\n{self.dump()}") from None
        assert self.alive(), f"hermes --tui exited during startup\n{self.dump()}"
        self.wait_quiet(1.5, timeout=timeout)

    # -- processes -------------------------------------------------------------------------------

    def track(self) -> None:
        self.seen.update(session_members(self.pane_pid))

    def exit_problem(self, timeout: float = 60.0) -> str:
        """``/exit``: the TUI exits 0 within ``timeout`` and leaves no process of its session."""
        self.track()
        self.submit("/exit")
        try:
            poll(lambda: self.fmt("#{pane_dead}") == "1", timeout=timeout, what="the TUI to exit")
        except AssertionError:
            return f"/exit did not exit within {timeout:.0f}s\n{self.dump()}"
        status = self.fmt("#{pane_dead_status}")
        try:
            poll(lambda: not [p for p in self.seen | set(session_members(self.pane_pid)) if _alive(p)],
                 timeout=15, what="the pane session to empty")
        except AssertionError:
            left = [f"{p}: {cmdline(p)}" for p in self.seen | set(session_members(self.pane_pid)) if _alive(p)]
            return f"processes left after /exit: {left}"
        return "" if status == "0" else f"/exit status {status}"

    def close(self) -> list[str]:
        """Kill the tmux server and every process of the pane's session; return the survivors
        that needed a SIGKILL (pid: cmdline) for diagnostics."""
        self.track()
        self.tmux("kill-server")
        survivors = []
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            left = [p for p in (set(session_members(self.pane_pid)) | self.seen) if _alive(p)]
            if not left:
                break
            time.sleep(0.1)
        for pid in set(session_members(self.pane_pid)) | self.seen:
            if _alive(pid):
                survivors.append(f"{pid}: {cmdline(pid)}")
                try:
                    os.kill(pid, signal.SIGKILL)
                except (ProcessLookupError, PermissionError):
                    pass
        return survivors

    # -- persisted state -------------------------------------------------------------------------

    def db_rows(self, sql: str, args: tuple = ()) -> list[tuple]:
        db = self.hermes_home / "state.db"
        if not db.exists():
            return []
        conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True, timeout=10)
        try:
            return conn.execute(sql, args).fetchall()
        except sqlite3.OperationalError:
            return []
        finally:
            conn.close()

    def messages(self) -> list[tuple[str, str, str]]:
        """(session_id, role, content) for active user/assistant rows, in insertion order."""
        return [(str(s), str(r), str(c or "")) for s, r, c in self.db_rows(
            "SELECT session_id, role, content FROM messages "
            "WHERE role IN ('user','assistant') AND active = 1 ORDER BY id")]

    def wait_replies(self, n: int, timeout: float = 90.0) -> None:
        """Block until state.db holds ``n`` non-empty assistant rows (turn finished)."""
        def done() -> bool:
            return (not self.alive()) or sum(
                1 for _s, r, c in self.messages() if r == "assistant" and c.strip()) >= n
        try:
            poll(done, timeout=timeout, what=f"{n} assistant replies persisted", interval=0.1)
        except AssertionError as exc:
            raise AssertionError(f"{exc}\n{self.dump()}") from None
        assert self.alive(), f"hermes exited mid-turn\n{self.dump()}"


def _alive(pid: int) -> bool:
    try:
        state = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[0]
    except OSError:
        return False
    return state != "Z"
