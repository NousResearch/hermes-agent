"""Share this backend over tailcat: a supervised ``tailcat serve`` for a dedicated listener.

``tailcat serve`` hands every remote connection to a loopback port, so the peer
looks local. The main listener cannot be that port: a headless serve returns the
process session token to any loopback ``GET /``. The share therefore gets its own
loopback listener on the same app (``web_server_share``), where only paired
device tokens authenticate. This module owns the tailcat side: the binary, the
node key that makes the address stable, and a child process restarted within a
bounded budget.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

from hermes_cli import tailcat_share_store as store

log = logging.getLogger(__name__)

# Restart budget: a crash-looping tailcat (bad key, no network) must not spin.
RESTART_BACKOFF_S = (1.0, 2.0, 5.0, 10.0, 30.0)
RESTART_WINDOW_S = 300.0
RESTART_MAX_IN_WINDOW = 6
_ADDRESS_WAIT_S = 30.0


class TailcatUnavailable(RuntimeError):
    """No tailcat binary could be found or installed."""


def find_tailcat(*, install: bool = False) -> Optional[Path]:
    """A user-installed tailcat on PATH wins; else PM's pinned copy (installed on request)."""
    from hermes_platform.resolver import locate_command

    found = locate_command("tailcat")
    if found.command:
        return Path(found.command[0])
    import pm

    selected = pm.installed_package("tailcat")
    if selected is not None:
        return selected.binary
    if install:
        pm.ensure("tailcat", explicit=True)
        selected = pm.installed_package("tailcat")
        return selected.binary if selected is not None else None
    return None


def _hidden_kwargs() -> dict:
    if os.name != "nt":
        return {}
    from hermes_cli._subprocess_compat import windows_hide_flags

    return {"creationflags": windows_hide_flags()}


def _exit_with_parent(parent_pid: int) -> Callable[[], None]:
    """Linux ``preexec_fn``: SIGTERM tailcat when the thread that spawned it dies.

    PR_SET_PDEATHSIG is bound to the forking THREAD; the supervisor thread forks
    every tailcat and outlives each one, so it is the right owner. libc is loaded
    here, before the fork, so the child only makes the syscall.
    """
    import ctypes
    import signal

    libc = ctypes.CDLL(None, use_errno=True)

    def _arm() -> None:
        libc.prctl(1, signal.SIGTERM, 0, 0, 0)  # PR_SET_PDEATHSIG
        if os.getppid() != parent_pid:  # the parent died before the prctl
            os._exit(1)

    return _arm


def _spawn_contained(argv: list[str], **kwargs) -> tuple[subprocess.Popen, Optional[Any]]:
    """Start tailcat so it cannot outlive this process.

    A graceful shutdown stops it, but a SIGKILL, OOM kill, or the Desktop
    parent-death watchdog's hard exit skips that path; an orphaned tailcat keeps
    publishing the share and a relaunch starts a second one on the same key.
    Windows gets a KILL_ON_JOB_CLOSE job (closed with this process) and Linux a
    parent-death signal. Returns the process and the job (None off Windows).
    """
    if sys.platform == "win32":
        from hermes_cli.local_runtime.processes import spawn_server

        return spawn_server(argv, **kwargs)
    if sys.platform.startswith("linux"):
        kwargs["preexec_fn"] = _exit_with_parent(os.getpid())
    return subprocess.Popen(argv, **kwargs), None


def ensure_server_key(binary: Path, home: Optional[Path] = None) -> Path:
    """The node key is the address: generated once, kept until ``hermes share reset``.

    ``--fixed-region`` bakes the nearest relay region into the key. Without it
    the key says "pick at startup", the region is part of the address, and a
    restart that measures a different nearest region publishes a new address
    that every paired Desktop has never heard of.
    """
    key = store.server_key_path(home)
    if key.is_file():
        return key
    key.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [str(binary), "genkey", f"--key={key}", "--fixed-region"],
        check=True, capture_output=True, timeout=30, **_hidden_kwargs(),
    )
    if os.name != "nt":
        key.chmod(0o600)
    return key


@dataclass
class ShareStatus:
    state: str = "stopped"  # stopped | starting | ready | failed
    address: str = ""
    port: int = 0
    error: str = ""
    restarts: int = 0

    def as_dict(self) -> dict:
        return {"state": self.state, "address": self.address, "port": self.port,
                "error": self.error, "restarts": self.restarts}


@dataclass
class TailcatSupervisor:
    """Runs ``tailcat serve --json`` for ``port`` and restarts it within budget.

    ``on_address`` fires every time tailcat reports its listen address (the same
    one each time, because the key is saved).
    """

    binary: Path
    key: Path
    port: int
    on_address: Callable[[str], None] = lambda _addr: None
    status: ShareStatus = field(default_factory=ShareStatus)
    _stop: threading.Event = field(default_factory=threading.Event)
    _proc: Optional[subprocess.Popen] = None
    _thread: Optional[threading.Thread] = None
    _ready: threading.Event = field(default_factory=threading.Event)

    def start(self) -> None:
        self.status = ShareStatus(state="starting", port=self.port)
        self._thread = threading.Thread(target=self._run, name="tailcat-share", daemon=True)
        self._thread.start()

    def wait_ready(self, timeout: float = _ADDRESS_WAIT_S) -> bool:
        return self._ready.wait(timeout)

    def stop(self) -> None:
        self._stop.set()
        proc = self._proc
        if proc is not None and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
        if self._thread is not None:
            self._thread.join(timeout=6)
        self.status.state = "stopped"

    def _argv(self) -> list[str]:
        return [str(self.binary), "serve", "--json", f"--key={self.key}", str(self.port)]

    def _run(self) -> None:
        failures: list[float] = []
        while not self._stop.is_set():
            started = time.monotonic()
            self._run_once()
            if self._stop.is_set():
                return
            now = time.monotonic()
            failures = [t for t in failures if now - t < RESTART_WINDOW_S] + [now]
            if len(failures) > RESTART_MAX_IN_WINDOW:
                self.status.state = "failed"
                self.status.error = self.status.error or "tailcat kept exiting; sharing stopped"
                log.error("tailcat share: %s", self.status.error)
                return
            delay = RESTART_BACKOFF_S[min(len(failures) - 1, len(RESTART_BACKOFF_S) - 1)]
            if now - started > RESTART_WINDOW_S:
                delay = RESTART_BACKOFF_S[0]
            self.status.restarts += 1
            self.status.state = "starting"
            log.warning("tailcat share: tailcat exited; restarting in %.0fs", delay)
            self._stop.wait(delay)

    def _run_once(self) -> None:
        try:
            self._proc, job = _spawn_contained(
                self._argv(), stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                stderr=subprocess.PIPE, text=True, encoding="utf-8", errors="replace",
                **_hidden_kwargs(),
            )
        except OSError as exc:
            self.status.error = f"could not start tailcat: {exc}"
            return
        try:
            self._watch(self._proc)
        finally:
            if job is not None:
                job.close()

    def _watch(self, proc: subprocess.Popen) -> None:
        stderr_tail: list[str] = []
        drain = threading.Thread(target=self._drain_stderr, args=(proc, stderr_tail),
                                 name="tailcat-share-stderr", daemon=True)
        drain.start()
        assert proc.stdout is not None
        for line in proc.stdout:
            address = _listen_address(line)
            if address:
                self.status.address = address
                self.status.state = "ready"
                self.status.error = ""
                self.on_address(address)
                self._ready.set()
        code = proc.wait()
        drain.join(timeout=2)
        if not self._stop.is_set():
            self.status.error = (stderr_tail[-1] if stderr_tail else f"tailcat exited with {code}")

    @staticmethod
    def _drain_stderr(proc: subprocess.Popen, tail: list[str]) -> None:
        assert proc.stderr is not None
        for line in proc.stderr:
            text = line.strip()
            if text:
                log.debug("tailcat: %s", text)
                tail.append(text)
                del tail[:-5]


def _listen_address(line: str) -> str:
    try:
        data = json.loads(line)
    except ValueError:
        return ""
    address = data.get("listenAddr") if isinstance(data, dict) else None
    return address if isinstance(address, str) and address.startswith("tc") else ""
