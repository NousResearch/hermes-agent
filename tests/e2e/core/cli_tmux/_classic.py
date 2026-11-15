"""Shared harness for the classic-CLI tmux suites (``display.interface: cli``).

A real ``python -m hermes_cli.main`` runs in a private tmux server (own ``-L`` socket) with a
hermetic fake HOME / HERMES_HOME and a config that routes the model to a loopback
``FakeLLMServer``. The CLI is a thin client of the profile's gateway daemon, which it spawns on
first connect; cleanup stops that daemon and then kills anything left whose environ carries this
run's exact ``HERMES_HOME`` (never a pattern kill).

Driver rule (the CLI reads an Enter processed within 50 ms of the last buffer change as a pasted
newline): type with ``send-keys -l``, wait until the composer shows the text, pause, then send
``Enter`` separately. Each step waits for its needle, then until the screen (digits masked) has
been static for ``quiet`` seconds.

Stale output never counts: every ``submit``/``keys``/``launch`` snapshots the scrollback first, and
``wait`` accepts a needle only when it occurs more often than in that snapshot. An unchanged
transcript (a no-op command) therefore fails the step instead of matching old output.

Linux only: process discovery and cleanup read ``/proc/<pid>/environ``.

Run environment: the CLI catches SIGTERM/SIGHUP and exits cleanly ("Shutting down…", "Goodbye!",
status 0), so a signal from outside the test looks like the CLI quitting on its own. A container
run with the host user's uid shares that uid with every host process, and a host-side same-uid
process sweep can reach the container's CLIs (observed: SIGTERM with ``si_pid=0``, i.e. from
outside the container's pid namespace). Run containers under a uid of their own.
"""

from __future__ import annotations

import json
import os
import re
import shlex
import shutil
import signal
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import pytest

from tests.fakes.fake_llm_provider import MODEL_ID, OFFER_VERSION


def require_tmux() -> None:
    """Skip without tmux, except in the CI E2E lane, where a skip is green over zero coverage.

    The release/dispatch ``e2e`` job (``.github/workflows/tests.yml``) sets
    ``HERMES_E2E_REQUIRE_TUI=1`` and GitHub sets ``CI``; under both, a missing tmux fails.
    """
    if shutil.which("tmux") is not None:
        return
    if os.environ.get("CI") and os.environ.get("HERMES_E2E_REQUIRE_TUI") == "1":
        pytest.fail("tmux missing in the CI e2e lane (HERMES_E2E_REQUIRE_TUI=1): install it in the e2e job")
    pytest.skip("needs tmux")

REPO_ROOT = Path(__file__).resolve().parents[4]
COLS, ROWS = 160, 50
PROMPT = "\u276f"
STATUS_BAR = f"\u2624 {MODEL_ID}"
_DIGITS = re.compile(r"\d")
_TRACEBACK = "Traceback (most recent call last)"


def new_needle(needles: list[str], before: str, after: str) -> str | None:
    """The first needle that occurs more often in *after* than in *before* (output produced since)."""
    for needle in needles:
        if after.count(needle) > before.count(needle):
            return needle
    return None


def home_pids(hermes_home: Path) -> set[int]:
    """PIDs (never this one) whose environ carries exactly ``HERMES_HOME=<hermes_home>``."""
    needle = f"HERMES_HOME={hermes_home}".encode()
    found: set[int] = set()
    for entry in os.listdir("/proc"):
        if not entry.isdigit() or int(entry) == os.getpid():
            continue
        try:
            with open(f"/proc/{entry}/environ", "rb") as fh:
                if needle in fh.read().split(b"\0"):
                    found.add(int(entry))
        except OSError:
            continue
    return found


def kill_home_processes(hermes_home: Path, stop: Callable[[], object] | None = None) -> None:
    """Graceful stop, then SIGTERM/SIGKILL whatever still runs against *hermes_home*.

    The signal sweep runs in ``finally``: a ``stop`` that times out or raises still leaves no
    CLI or gateway behind.
    """
    try:
        if stop is not None:
            stop()
    finally:
        for sig, grace in ((signal.SIGTERM, 8.0), (signal.SIGKILL, 2.0)):
            for pid in home_pids(hermes_home):
                try:
                    os.kill(pid, sig)
                except OSError:
                    pass
            end = time.monotonic() + grace
            while home_pids(hermes_home) and time.monotonic() < end:
                time.sleep(0.2)

BASE_CONFIG = f"""model:
  provider: custom
  base_url: {{base_url}}
  default: {MODEL_ID}
  context_length: 128000
agent:
  api_max_retries: 1
telemetry:
  shared_metrics:
    enabled: false
    send: false
    offer_version: {OFFER_VERSION}
display:
  interface: cli
approvals:
  mode: manual
  timeout: 60
skills:
  creation_nudge_interval: 0
updates:
  check: false
models_dev:
  url: http://127.0.0.1:9/api.json
auxiliary:
  title_generation:
    enabled: false
"""


@dataclass
class ClassicHome:
    root: Path
    extra_env: dict[str, str] = field(default_factory=dict)
    # (pid, argv) of every hermes process the harness started: tmux pane CLIs and piped runs.
    spawned: list[tuple[int, str]] = field(default_factory=list)

    @property
    def home(self) -> Path:
        return self.root / "home"

    @property
    def hermes_home(self) -> Path:
        return self.home / ".hermes"

    @property
    def work(self) -> Path:
        return self.root / "work"

    @property
    def stderr_log(self) -> Path:
        return self.root / "stderr.log"

    def write(self, base_url: str, extra_config: str = "") -> ClassicHome:
        self.hermes_home.mkdir(parents=True, exist_ok=True)
        self.work.mkdir(parents=True, exist_ok=True)
        (self.hermes_home / "config.yaml").write_text(BASE_CONFIG.format(base_url=base_url) + extra_config,
                                                      encoding="utf-8")
        (self.hermes_home / ".env").write_text("OPENAI_API_KEY=sk-fake-e2e\n", encoding="utf-8")
        return self

    def env(self) -> dict[str, str]:
        """Allowlisted child env: no inherited HERMES_*, provider keys, display or TMUX variables."""
        keep = ("PATH", "LANG", "USER", "LOGNAME", "SHELL", "TMPDIR")
        env = {k: os.environ[k] for k in keep if k in os.environ}
        env.update({
            "LC_ALL": "C.UTF-8", "LANG": "C.UTF-8", "TERM": "xterm-256color", "TZ": "UTC",
            "HOME": str(self.home), "HERMES_HOME": str(self.hermes_home), "PYTHONPATH": str(REPO_ROOT),
            # The child's state.db IS this tmp home's db; the live-DB guard would refuse it under pytest.
            "HERMES_STATE_DB_GUARD_BYPASS": "1", "PYTHONFAULTHANDLER": "1", "NO_GCE_CHECK": "True",
        })
        env.update(self.extra_env)
        return env

    def argv(self, *args: str) -> list[str]:
        return [sys.executable, "-m", "hermes_cli.main", *args]

    def run(self, *args: str, timeout: float = 180.0) -> subprocess.CompletedProcess:
        """A non-TTY run (stdout piped): ``chat -q`` one-shots, ``gateway stop``."""
        argv = self.argv(*args)
        with subprocess.Popen(argv, cwd=self.work, env=self.env(), stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                              stdin=subprocess.DEVNULL, text=True) as proc:
            self.spawned.append((proc.pid, " ".join(args)))
            try:
                out, err = proc.communicate(timeout=timeout)
            except subprocess.TimeoutExpired:
                proc.kill()
                out, err = proc.communicate()
                raise subprocess.TimeoutExpired(argv, timeout, out, err) from None
        return subprocess.CompletedProcess(argv, proc.returncode, out, err)

    def pids(self) -> set[int]:
        return home_pids(self.hermes_home)

    def diagnostics(self, limit: int = 3000) -> str:
        """The CLI's stderr and the gateway log tails: why a pane died, for a failure message."""
        parts = []
        for label, path in (("cli stderr", self.stderr_log), ("gateway log", self.hermes_home / "logs" / "gateway.log"),
                            ("agent log", self.hermes_home / "logs" / "agent.log")):
            try:
                text = path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            if text.strip():
                parts.append(f"--- {label} ({path.name}) ---\n{text[-limit:]}")
        return "\n".join(parts)

    def reap(self) -> None:
        """Operator stop for the profile's gateway, then SIGTERM/SIGKILL whatever this home still runs
        (even when the stop times out)."""
        kill_home_processes(self.hermes_home, lambda: self.run("gateway", "stop", timeout=60))


class Pane:
    """One private tmux server holding the CLI pane ``p``."""

    def __init__(self, ch: ClassicHome) -> None:
        self.ch = ch
        self.sock = f"hermes-e2e-classic-{uuid.uuid4().hex[:8]}"
        # Scrollback snapshot taken just before the latest input: what `wait` must see grow.
        self.baseline = ""

    def tmux(self, *args: str) -> str:
        env = {k: v for k, v in os.environ.items() if not k.startswith("TMUX")}
        return subprocess.run(["tmux", "-L", self.sock, *args], capture_output=True, text=True, encoding="utf-8",
                              errors="replace", timeout=30, check=False, env=env).stdout

    def launch(self, *args: str) -> None:
        self.tmux("kill-server")
        env = self.ch.env()
        envs = " ".join(f"{k}={shlex.quote(v)}" for k, v in env.items())
        cmd = (f"exec env -i {envs} {' '.join(shlex.quote(a) for a in self.ch.argv(*args))} "
               f"2>>{shlex.quote(str(self.ch.stderr_log))}")
        subprocess.run(["tmux", "-L", self.sock, "-f", os.devnull, "new-session", "-d", "-s", "p", "-x", str(COLS),
                        "-y", str(ROWS), "-c", str(self.ch.work), "sh", "-c", cmd], check=True, timeout=30,
                       env={k: v for k, v in os.environ.items() if not k.startswith("TMUX")})
        self.tmux("set", "-g", "remain-on-exit", "on")
        self.tmux("set", "-g", "history-limit", "50000")
        # `sh -c "exec env -i ... python"` keeps the pane's pid: this IS the CLI process.
        pid = self.tmux("display-message", "-p", "-t", "p", "#{pane_pid}").strip()
        if pid.isdigit():
            self.ch.spawned.append((int(pid), " ".join(("chat", *args)) if args else "chat"))
        self.baseline = ""  # a fresh tmux server: everything on it is new

    def screen(self) -> str:
        return self.tmux("capture-pane", "-p", "-t", "p")

    def scrollback(self) -> str:
        return self.tmux("capture-pane", "-p", "-J", "-t", "p", "-S", "-", "-E", "-")

    def dead(self) -> bool:
        return self.tmux("display-message", "-p", "-t", "p", "#{pane_dead}").strip() == "1"

    def keys(self, *k: str) -> None:
        self.baseline = self.scrollback()
        self.tmux("send-keys", "-t", "p", *k)

    def wait(self, needles: list[str], timeout: float = 60.0, *, since: str | None = None) -> str | None:
        """The first needle that appeared after the latest input (or after *since*), None on timeout/exit."""
        before = self.baseline if since is None else since
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            hit = new_needle(needles, before, self.scrollback())
            if hit is not None:
                return hit
            if self.dead():
                return None
            time.sleep(0.25)
        return None

    def settle(self, quiet: float = 2.0, timeout: float = 20.0) -> None:
        end = time.monotonic() + timeout
        last, since = None, time.monotonic()
        while time.monotonic() < end:
            text = _DIGITS.sub("#", self.screen())
            if text != last:
                last, since = text, time.monotonic()
            elif time.monotonic() - since >= quiet:
                return
            time.sleep(0.2)

    def paste_gap(self) -> None:
        """Keep the next key out of the input batch the app is still reading: prompt_toolkit takes an
        Enter that lands within 50 ms of the last buffer change for a pasted newline."""
        time.sleep(1.2)  # health: allow HX013 -- the gap IS the behaviour: the CLI's paste detection is time-based

    def submit(self, text: str) -> None:
        self.baseline = self.scrollback()
        self.tmux("send-keys", "-t", "p", "-l", text)
        self.wait([f"{PROMPT} {text}"[-40:]], 10, since=self.baseline)
        self.paste_gap()
        self.tmux("send-keys", "-t", "p", "Enter")

    def wait_dead(self, timeout: float = 30.0) -> bool:
        end = time.monotonic() + timeout
        while time.monotonic() < end and not self.dead():
            time.sleep(0.3)
        return self.dead()

    def ready(self, timeout: float = 120.0) -> str | None:
        """Wait for the running app's status bar (painted in raw mode), then let startup settle."""
        hit = self.wait([STATUS_BAR], timeout)
        self.settle(quiet=3.0, timeout=30)
        return hit

    def kill(self) -> None:
        self.tmux("kill-server")


def assert_no_traceback(text: str, where: str) -> None:
    assert _TRACEBACK not in text, f"{where}: a traceback reached the terminal:\n{text[-4000:]}"


def text_of(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(str(p.get("text", "")) if isinstance(p, dict) else str(p) for p in content)
    return json.dumps(content)


def last_user(body: dict) -> str:
    for message in reversed(body.get("messages") or []):
        if message.get("role") == "user":
            return text_of(message.get("content"))
    return ""
