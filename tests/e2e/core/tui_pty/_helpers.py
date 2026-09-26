"""Drive a real ``hermes --tui`` inside a private tmux server and read what the user would see.

tmux is the terminal emulator here: it owns the grid, reflows it on resize and keeps the
scrollback, exactly as it does for a user running the TUI inside tmux. We read it back with
``capture-pane`` (``-S -`` for scrollback) and ask tmux for
the cursor position, the alternate-screen flag and the scrollback size.

Every transcript word the fake provider streams is a unique token (``<tag>w<NNN>``), so a
*word ledger* over the captured text tells lost, duplicated and reordered words apart.

Isolation: the tmux server has its own socket and a config written into the test's tmp dir;
the TUI gets a sandbox HOME/HERMES_HOME wired only to the loopback fake provider; cleanup kills
the tmux server and every process of the pane's session by session id, never by pattern.
"""

from __future__ import annotations

import contextlib
import os
import re
import shlex
import shutil
import signal
import sqlite3
import subprocess
import sys
import tempfile
import termios  # windows-footgun: ok — tmux/pty suite, skipped off Linux
import time
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Callable, Iterable

import pytest

from tests.e2e.core._pending_fixes import known_failure
from tests.e2e.core.terminal._pty import cmdline, poll, session_members
from tests.e2e.core.terminal._vt import Screen
from tests.fakes.fake_llm_provider import write_hermes_home

REPO_ROOT = Path(__file__).resolve().parents[4]
TITLE = "Scripted session title"

# Right-edge scrollbar glyphs the Ink TUI paints in the last column of its transcript viewport.
_SCROLLBAR = "│┃║▐▕█░▒▓"
_DIGITS = re.compile(r"\d")
_SPINNER = re.compile(r"[\u2800-\u28ff]")
_STATUS_BAR = re.compile(r"│ fake model")
# The status bar's first cell once the startup session is live: "─ ready │ fake model │ …". Before
# that it reads "summoning hermes…" / "forging session…" / "starting agent…" / "resuming…".
_READY = re.compile(r"─ ready │")
_STATUS_ROW = re.compile(r"^ ─ (.+?) │")
# AF_UNIX sun_path is 108 bytes; stay well inside it.
_SOCK_MAX = 100

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


def private_tui_dir(root: Path) -> Path:
    """A private copy of the prebuilt bundle for ``HERMES_TUI_DIR``.

    A checkout launch of ``hermes --tui`` re-runs esbuild on ``ui-tui/dist/entry.js`` every
    time, non-atomically, so a TUI started by another worker (or another e2e suite) while it
    rebuilds dies with a Node ``SyntaxError`` on a half-written bundle. The prebuilt-bundle path
    runs the same file without rebuilding; a copy that ``node --check`` accepts is immune to
    concurrent rebuilds of the shared one.
    """
    src = REPO_ROOT / "ui-tui" / "dist" / "entry.js"
    dest = root / "tui" / "dist" / "entry.js"
    dest.parent.mkdir(parents=True, exist_ok=True)
    for _ in range(20):
        shutil.copyfile(src, dest)
        if subprocess.run(["node", "--check", str(dest)], capture_output=True, timeout=60).returncode == 0:
            return dest.parent.parent
        time.sleep(0.5)  # caught a rebuild mid-write; take another copy
    raise AssertionError(f"{src} never parsed (a rebuild kept rewriting it?)")


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


class CellFailed(AssertionError):
    """A cell's verdict is a failure. Raised ONLY by ``Cells.check`` for an evaluated cell, so a
    KNOWN cell (``known_failure(..., raises=CellFailed)``) never swallows a harness failure
    (timeout, crash, precondition), which ``Cells.check`` raises as ``RuntimeError``."""


class Cells:
    """Named verdicts of one scenario run. Each becomes its own test id so a known bug can be
    pinned (``known_failure``) on exactly the cell it breaks while the rest stay enforced."""

    def __init__(self) -> None:
        self.results: dict[str, tuple[bool, str]] = {}
        self.error: str | None = None
        self.errors: dict[str, str] = {}
        self.phase = "setup"

    def add(self, name: str, problem: str, screen: str = "") -> None:
        detail = f"{problem}\n--- screen ---\n{screen}" if screen else problem
        self.results[name] = (not problem, detail if problem else "")

    def harness_error(self, name: str, detail: str) -> None:
        """The cell could not be judged (its precondition never held): a real failure, never
        swallowed by a KNOWN pin, and it leaves the scenario's other cells alone."""
        self.errors[name] = f"PHASE={self.phase}: {detail}"

    def check(self, name: str, known: dict[str, tuple[str, str]] | None = None) -> None:
        """Enforce one cell. ``known`` maps a cell to ``(pattern, "#<issue> …")``: while the cell
        fails with a message matching ``pattern`` it XFAILs; once the fix lands it passes; any
        other failure (including every harness error) stays a failure."""
        if self.error is not None:
            raise RuntimeError(f"scenario failed before its cells could be evaluated:\n{self.error}")
        if name in self.errors:
            raise RuntimeError(f"cell {name!r} could not be evaluated:\n{self.errors[name]}")
        if name not in self.results:
            raise RuntimeError(f"cell {name!r} was never evaluated (scenario ended early)")
        ok, detail = self.results[name]
        entry = (known or {}).get(name)
        guard = known_failure(*entry, raises=CellFailed) if entry else contextlib.nullcontext()
        with guard:
            if not ok:
                raise CellFailed(f"[{name}] {detail}")


def cell_params(names: Iterable[str]) -> list:
    return [pytest.param(n, id=n) for n in names]


def run_cells(body: Callable[[Cells], None]) -> Cells:
    """Run one scenario; a harness failure (timeout, crash) is kept, tagged with the phase it hit
    (``cells.phase``), and re-raised by every cell."""
    cells = Cells()
    try:
        body(cells)
    except Exception as exc:  # noqa: BLE001 - surfaced verbatim by Cells.check
        cells.error = f"PHASE={cells.phase}: {type(exc).__name__}: {exc}"
    return cells


def layout_problem(rows: list[str], cols: int, tag: str, slack: int | None = 12) -> str:
    """The paragraph of ``<tag>wNNN`` words on the visible frame is laid out for the current width
    (#35804). tmux never shows a row wider than the pane, so "fits" alone proves nothing; instead:

    * every row holding its words is only those words: a blank/``┊`` gutter, then whole words
      separated by single spaces. A frame painted for another width leaves fragments, box rules
      or words of other rows spliced into a row;
    * the rows read on from one another: each row starts with the word after the previous row's
      last one (no word missing, repeated or moved between rows);
    * the widest row reaches within ``slack`` columns of the edge (a narrower stale layout);
      ``slack=None`` skips this (and tolerates the paragraph being scrolled off) for a paragraph
      too short to fill a row.
    """
    word = rf"{tag}w\d{{3}}"
    clean = re.compile(rf"^[ ┊]*({word}(?: {word})*)$")
    para = [r for r in rows if re.search(rf"{tag}w\d", r)]
    if not para:
        return f"no row of the {tag} paragraph on screen" if slack is not None else ""
    seq: list[int] = []
    for r in para:
        m = clean.match(r)
        if not m:
            return f"a {tag} paragraph row is not whole words at this width: {r!r}"
        idx = [int(w[len(tag) + 1:]) for w in m.group(1).split(" ")]
        if (seq and idx[0] != seq[-1] + 1) or idx != list(range(idx[0], idx[0] + len(idx))):
            return f"{tag} paragraph rows do not read on (after {tag}w{seq[-1] if seq else -1:03d}): {r!r}"
        seq += idx
    widest = max(display_width(r) for r in para)
    if slack is not None and widest < cols - slack:
        return f"reply wraps at {widest} cols on a {cols}-col terminal (content width did not follow)"
    return ""


class TmuxTui:
    """One ``hermes --tui`` in a private tmux server."""

    def __init__(self, root: Path, base_url: str, *, cols: int = 120, rows: int = 50,
                 extra_config: str = "", args: Iterable[str] = ("--yolo",), inline: bool = False,
                 env_extra: dict[str, str] | None = None, write_home: bool = True,
                 tui_dir: Path | None = None) -> None:
        self.root = root
        self.home = root / "home"
        self.hermes_home = self.home / ".hermes"
        # A socket path of our own (never the shared default dir tmux-<uid> under the system temp
        # dir, where a crashed run would leave it behind); close() removes it.
        self._sock_dir: str | None = None
        sock = root / "tmux.sock"
        if len(str(sock)) > _SOCK_MAX:
            self._sock_dir = tempfile.mkdtemp(prefix="htui-")
            sock = Path(self._sock_dir) / "s"
        self.sock = str(sock)
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
                   HERMES_STATE_DB_GUARD_BYPASS="1", HERMES_TUI_INLINE="1" if inline else "0",
                   HERMES_TUI_DIR=str(tui_dir or private_tui_dir(root)))
        env.update(env_extra or {})
        argv = [sys.executable, "-m", "hermes_cli.main", "--tui", *args]
        subprocess.run(["tmux", "-S", self.sock, "-f", str(conf), "new-session", "-d", "-s", "p",
                        "-x", str(cols), "-y", str(rows), "-c", str(root / "work"), *argv],
                       env=env, check=True, timeout=30)
        self.tmux("set", "-g", "window-size", "manual")
        # Everything the pane prints, for failure reports (the screen is blank after exit).
        self._cols, self._rows = cols, rows
        self.transcript_path = root / "pty.log"
        self.tmux("pipe-pane", "-t", "p", f"cat >> {shlex.quote(str(self.transcript_path))}")
        self.pane_pid = int(self.tmux("display", "-p", "-t", "p", "#{pane_pid}").strip())
        self.pane_tty = self.tmux("display", "-p", "-t", "p", "#{pane_tty}").strip()
        self.server_pid = int(self.tmux("display", "-p", "#{pid}").strip() or 0)
        self._zombie_since = self._reap_nudged = 0.0
        self.reap_nudges = 0
        self.seen: set[int] = set()

    # -- tmux ------------------------------------------------------------------------------------

    def tmux_run(self, *args: str) -> subprocess.CompletedProcess:
        argv = ["tmux", "-S", self.sock, *args]
        return subprocess.run(argv, capture_output=True, text=True, encoding="utf-8", errors="replace",
                              timeout=30)

    def tmux(self, *args: str) -> str:
        return self.tmux_run(*args).stdout

    def fmt(self, spec: str) -> str:
        return self.tmux("display", "-p", "-t", "p", spec).strip()

    def pane_state(self) -> str:
        """``"<dead> <pid> <status>"``; a tmux client that loses the race to a loaded host
        answers empty, so retry before concluding the server is gone."""
        for _ in range(5):
            out = self.fmt("#{pane_dead} #{pane_pid} #{pane_dead_status}")
            if out:
                return out
            time.sleep(0.2)
        return ""

    def alive(self) -> bool:
        return self.pane_state().startswith("0 ")

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
        """A settled frame: unchanged for ``idle`` seconds, ignoring what ticks on its own while a
        card waits for the human (digits of clocks, braille spinners, the status bar's rotating
        verb)."""
        def frame() -> str:
            self.track()
            rows = [r for r in self.rows() if not _STATUS_BAR.search(r)]
            return _SPINNER.sub("#", _DIGITS.sub("#", "\n".join(rows)))
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

    def status(self) -> str:
        """The status bar's first cell (``ready``, ``summoning hermes…``, ``running…``), or ''."""
        for row in reversed(self.rows()):
            if m := _STATUS_ROW.match(row):
                return m.group(1)
        return ""

    def wait_raw(self, timeout: float = 120.0) -> None:
        """The Ink UI owns the terminal (raw mode): keys typed from now on reach the composer."""
        try:
            poll(lambda: not self.alive() or self._raw_mode(), timeout=timeout,
                 what="the TUI to take the terminal (raw mode)")
        except AssertionError as exc:
            raise AssertionError(f"{exc}\n{self.dump()}") from None
        state = self.pane_state()
        assert state.startswith("0 "), (
            f"hermes --tui exited during startup (pane_dead pid status={state!r}, "
            f"tmux server {'up' if state else 'gone'})\n{self.dump()}")

    def wait_ready(self, timeout: float = 120.0) -> None:
        """Raw mode, the startup session is live (status bar ``ready`` -- the composer accepts
        input long before: a slash command sent while the gateway still starts races the startup
        session and loses), and the frame has settled."""
        self.wait_raw(timeout)
        try:
            poll(lambda: not self.alive() or _READY.search("\n".join(self.rows())), timeout=timeout,
                 what="the startup session (status bar 'ready')")
        except AssertionError as exc:
            raise AssertionError(f"{exc} (status {self.status()!r})\n{self.dump()}") from None
        assert self.alive(), f"hermes --tui exited during startup\n{self.dump()}"
        self.wait_quiet(1.5, timeout=timeout)

    # -- processes -------------------------------------------------------------------------------

    def track(self) -> None:
        self.seen.update(session_members(self.pane_pid))

    def exit_report(self) -> tuple[str, str] | None:
        """``(exit status, signal)`` once tmux has reaped the pane process, else None.

        Ordering: tmux flags a pane dead (``pane_dead``) as soon as its pty reads EOF, which the
        kernel delivers when the exiting process closes its last tty fd -- before the process is
        a zombie and tmux reaps it on SIGCHLD. Until then ``pane_dead_status`` and
        ``pane_dead_signal`` are both empty; on a loaded host that gap is long enough to read.
        """
        out = self.fmt("#{pane_dead}|#{pane_dead_status}|#{pane_dead_signal}")
        dead, _, rest = out.partition("|")
        status, _, sig = rest.partition("|")
        return (status, sig) if dead == "1" and (status or sig) else None

    def reap_report(self) -> tuple[str, str] | None:
        """:meth:`exit_report`, re-signalling tmux while the pane process sits exited but unreaped.

        tmux 3.4 (the CI runner's) intermittently misses the SIGCHLD of an exited pane: the pane
        process stays a single-threaded zombie of the tmux server -- parent alive, SIGCHLD caught,
        neither blocked nor pending -- and ``pane_dead_status`` never fills. A spurious SIGCHLD only
        makes tmux run its ``waitpid`` loop, which reaps the zombie with its real exit status; a
        process that is still running (a real /exit hang) is untouched and still times out."""
        report = self.exit_report()
        if report is not None or _proc_letter(self.pane_pid) != "Z":
            return report
        now = time.monotonic()
        self._zombie_since = self._zombie_since or now
        # tmux's own SIGCHLD normally lands within milliseconds; nudge only a zombie it left behind.
        if (now - self._zombie_since >= 2.0 and now - self._reap_nudged >= 1.0 and self.server_pid
                and "tmux" in cmdline(self.server_pid)):
            self._reap_nudged, self.reap_nudges = now, self.reap_nudges + 1
            with contextlib.suppress(ProcessLookupError):
                os.kill(self.server_pid, signal.SIGCHLD)  # windows-footgun: ok — Linux-only suite
        return None

    def exit_problem(self, timeout: float = 60.0) -> str:
        """``/exit``: the TUI exits 0 within ``timeout`` and leaves no process of its session."""
        self.track()
        before = "\n".join([r for r in self.rows() if r.strip()][-30:])
        t0 = time.monotonic()
        self.submit("/exit")
        try:
            status, sig = poll(self.reap_report, timeout=timeout, what="the TUI to exit")
        except AssertionError:
            return self.exit_diagnostics(f"/exit did not exit within {timeout:.0f}s", t0, before)
        def members() -> list[int]:
            return [p for p in self.seen | set(session_members(self.pane_pid)) if _alive(p)]
        try:
            poll(lambda: not members(), timeout=15, what="the pane session to empty")
        except AssertionError:
            left = [f"{p}: {cmdline(p)}" for p in members()]
            return self.exit_diagnostics(f"processes left after /exit: {left}", t0, before)
        if status == "0" and not sig:
            return ""
        return self.exit_diagnostics(f"/exit ended with exit status {status or '-'} signal {sig or '-'}",
                                     t0, before)

    def exit_diagnostics(self, problem: str, t0: float, before: str, tail: int = 40) -> str:
        """``problem`` plus what explains it: the raw tmux answer (rc/stderr), whether the pane
        process is still alive, the elapsed time, the frame before ``/exit`` and the last lines of
        the PTY transcript (the live screen is blank once the TUI left the alternate screen)."""
        q = self.tmux_run("display", "-p", "-t", "p",
                          "dead=#{pane_dead} status=#{pane_dead_status} signal=#{pane_dead_signal}")
        proc = _proc_state(self.pane_pid)
        return (f"{problem}\n"
                f"elapsed since /exit: {time.monotonic() - t0:.1f}s\n"
                f"tmux: rc={q.returncode} out={q.stdout.strip()!r} err={q.stderr.strip()!r}\n"
                f"pane pid {self.pane_pid}: {proc}\n"
                f"tmux server {self.server_pid}: {_status_fields(self.server_pid)} "
                f"(SIGCHLD nudges sent: {self.reap_nudges})\n"
                f"--- frame before /exit ---\n{before}\n"
                f"--- last {tail} lines of the PTY transcript ---\n{self.transcript_tail(tail)}")

    def transcript_tail(self, n: int = 40) -> str:
        """Last ``n`` rows of everything the pane printed (tmux ``pipe-pane`` log, replayed
        through a VT emulator): scrollback + main screen, i.e. what a user sees after exit."""
        try:
            raw = self.transcript_path.read_bytes()
        except OSError as exc:
            return f"(no transcript: {exc})"
        screen = Screen(self._rows, self._cols)
        screen.feed(raw)
        lines = screen.transcript()
        while lines and not lines[-1]:
            lines.pop()
        return "\n".join(lines[-n:]) or f"(transcript: {len(raw)} bytes, nothing printable on the main screen)"

    def close(self) -> list[str]:
        """Kill the tmux server and every process of the pane's session; return the survivors
        that needed a SIGKILL (pid: cmdline) for diagnostics."""
        self.track()
        self.tmux("kill-server")
        Path(self.sock).unlink(missing_ok=True)
        if self._sock_dir:
            shutil.rmtree(self._sock_dir, ignore_errors=True)
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
                    os.kill(pid, signal.SIGKILL)  # windows-footgun: ok — Linux-only suite
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


def _proc_state(pid: int) -> str:
    """'gone (reaped)', or the /proc state letter (Z = zombie not yet reaped) and cmdline."""
    try:
        state = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8").rsplit(")", 1)[1].split()[0]
    except OSError:
        return "gone (reaped)"
    return (f"state {state} ({'zombie, not reaped yet' if state == 'Z' else 'still running'}): {cmdline(pid)}\n"
            f"  {_status_fields(pid)}\n  threads: {_threads(pid)}")


def _status_fields(pid: int, keys: tuple[str, ...] = ("State", "PPid", "Threads", "SigPnd", "ShdPnd", "SigBlk",
                                                      "SigIgn", "SigCgt")) -> str:
    """Selected ``/proc/<pid>/status`` lines: why a process is not reaped (a zombie leader whose
    other threads are still exiting, or a parent with SIGCHLD blocked/pending)."""
    try:
        lines = Path(f"/proc/{pid}/status").read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        return f"(no status: {exc})"
    return " ".join(line.replace("\t", "") for line in lines if line.split(":", 1)[0] in keys)


def _threads(pid: int) -> str:
    """``tid:comm:state:wchan`` for every thread of ``pid``."""
    out = []
    try:
        tids = sorted(os.listdir(f"/proc/{pid}/task"), key=int)
    except OSError as exc:
        return f"(no tasks: {exc})"
    for tid in tids:
        base = f"/proc/{pid}/task/{tid}"
        try:
            stat = Path(f"{base}/stat").read_text(encoding="utf-8")
            wchan = Path(f"{base}/wchan").read_text(encoding="utf-8") or "-"
        except OSError:
            continue
        out.append(f"{tid}:{stat[stat.index('(') + 1:stat.rindex(')')]}:{stat.rsplit(')', 1)[1].split()[0]}:{wchan}")
    return " ".join(out)


def _proc_letter(pid: int) -> str:
    """The ``/proc/<pid>/stat`` state letter, '' once the process is reaped."""
    try:
        return Path(f"/proc/{pid}/stat").read_text(encoding="utf-8").rsplit(")", 1)[1].split()[0]
    except OSError:
        return ""


def _alive(pid: int) -> bool:
    return _proc_letter(pid) not in ("", "Z")
