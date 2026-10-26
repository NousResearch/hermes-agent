"""Host side of the shared-PID-namespace sandbox (see ``_ns_agent.py``).

``NamespaceHost(root)`` starts ONE ``bwrap`` sandbox (``_helpers.sandbox_argv``: own PID namespace,
tmpfs over ``/run/user/<uid>``, real ``~/.hermes`` read-only, only ``root`` writable) running the
agent, then runs and spawns every process of a cell inside it. Closing it kills the sandbox's init,
which takes every process of the namespace with it, detached or not, so nothing a cell starts can
outlive it; the dashboard lane's ``_reaper`` is armed around it as the leak check.
"""

from __future__ import annotations

import itertools
import json
import os
import signal
import subprocess
import sys
import threading
import time
from concurrent.futures import Future
from pathlib import Path
from typing import Sequence

from tests.e2e.core.dashboard import _reaper
from tests.e2e.core.upgrade import _helpers as H

_AGENT = Path(__file__).with_name("_ns_agent.py")


class NamespaceHost:
    def __init__(self, root: Path, env: dict[str, str]):
        self.root = root
        self.env = env
        self.transcript: list[str] = []
        self._ids = itertools.count(1)
        self._pending: dict[int, Future] = {}
        self._lock = threading.Lock()
        self._wlock = threading.Lock()  # one request line at a time: probes run from helper threads too
        _reaper.adopt_orphans()
        self._proc = subprocess.Popen(
            H.sandbox_argv([sys.executable, "-u", str(_AGENT)], writable=[root]),
            env=env, cwd=str(root), text=True, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=open(root / "ns-agent.stderr", "w"),
            start_new_session=True,
        )
        self._reader = threading.Thread(target=self._read, daemon=True)
        self._reader.start()
        self.call({"op": "ping"}, timeout=60)

    def _read(self) -> None:
        assert self._proc.stdout is not None
        for line in self._proc.stdout:
            try:
                msg = json.loads(line)
            except ValueError:
                continue
            with self._lock:
                fut = self._pending.pop(msg.get("id"), None)
            if fut is not None:
                fut.set_result(msg)
        with self._lock:
            pending, self._pending = self._pending, {}
        for fut in pending.values():
            fut.set_exception(AssertionError("sandbox agent exited:\n" + self.agent_stderr()))

    def agent_stderr(self) -> str:
        p = self.root / "ns-agent.stderr"
        return p.read_text(errors="replace")[-3000:] if p.exists() else ""

    def call(self, req: dict, *, timeout: float) -> dict:
        rid = next(self._ids)
        fut: Future = Future()
        with self._lock:
            self._pending[rid] = fut
        assert self._proc.stdin is not None
        with self._wlock:
            self._proc.stdin.write(json.dumps({**req, "id": rid}) + "\n")
            self._proc.stdin.flush()
        msg = fut.result(timeout=timeout)
        if "error" in msg:
            raise AssertionError(f"sandbox agent refused {req.get('op')}: {msg['error']}")
        return msg

    # -- commands ------------------------------------------------------------------------------

    def run(self, argv: Sequence[str], *, timeout: float = 600, env: dict[str, str] | None = None,
            cwd: Path | None = None, input: str | None = None, quiet: bool = False) -> subprocess.CompletedProcess:
        """Run to completion inside the namespace. ``quiet`` keeps a polling probe out of the transcript."""
        started = time.strftime("%H:%M:%S")
        msg = self.call({"op": "run", "argv": list(argv), "env": env or self.env, "cwd": str(cwd or self.root),
                         "timeout": timeout, "input": input}, timeout=timeout + 60)
        cp = subprocess.CompletedProcess(list(argv), msg["rc"], msg["stdout"], msg["stderr"])
        if not quiet:
            self.transcript.append(f"[{started}] rc={cp.returncode} ({msg['elapsed']:.1f}s) {' '.join(argv)[:300]}")
        if msg["timed_out"]:
            raise AssertionError(f"{list(argv)} timed out after {timeout}s inside the sandbox\n" + H.describe(cp))
        return cp

    def spawn(self, argv: Sequence[str], *, log: Path, env: dict[str, str] | None = None,
              cwd: Path | None = None) -> int:
        msg = self.call({"op": "spawn", "argv": list(argv), "env": env or self.env, "cwd": str(cwd or self.root),
                         "log": str(log)}, timeout=60)
        self.transcript.append(f"[{time.strftime('%H:%M:%S')}] spawned pid {msg['pid']}: {' '.join(argv)[:300]}")
        return int(msg["pid"])

    def procs(self) -> list[dict]:
        """Every process in the namespace (pids as the sandbox sees them)."""
        return self.call({"op": "procs"}, timeout=60)["procs"]

    def kill(self, pid: int, sig: int = signal.SIGTERM) -> bool:
        return bool(self.call({"op": "kill", "pid": pid, "sig": int(sig)}, timeout=60)["ok"])

    def alive(self, pid: int) -> bool:
        return any(p["pid"] == pid and p["state"] not in ("Z", "X") for p in self.procs())

    def ps_text(self) -> str:
        rows = [f"{p['pid']:>6} {p['ppid']:>6} {p['state']} {' '.join(p['cmdline'])[:220]}" for p in self.procs()]
        return "   PID   PPID S CMD\n" + "\n".join(sorted(rows, key=lambda r: int(r.split()[0])))

    def close(self) -> list[str]:
        """Kill the sandbox (its init takes every namespace process with it); returns leaked host pids."""
        try:
            os.killpg(self._proc.pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass
        try:
            self._proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            pass
        try:
            return _reaper.reap(_reaper.with_home(Path(self.env["HOME"])), timeout=15)
        finally:
            _reaper.release_orphans()
