"""Stdio JSON-RPC driver for a REAL ``python -m tui_gateway.entry`` subprocess.

This is the process the Ink TUI and Desktop talk to: newline-delimited JSON-RPC 2.0 on
stdin/stdout, responses carry the request ``id``, server-pushed events are
``{"method": "event", "params": {"type", "session_id", "payload"}}``. The driver keeps
one reader thread that routes responses to waiters and appends every event to a log,
so tests can poll with deadlines instead of sleeping.
"""

from __future__ import annotations

import itertools
import json
import os
import subprocess
import threading
import time
from pathlib import Path
from typing import Any, Callable

from tests.e2e.core.chaos._helpers import REPO_ROOT, python_exe


class RpcError(AssertionError):
    pass


class TuiGatewayProcess:
    def __init__(self, env: dict[str, str], cwd: Path, stderr_path: Path) -> None:
        self._stderr_fh = open(stderr_path, "wb")  # noqa: SIM115 - closed in close()
        self.stderr_path = stderr_path
        self.proc = subprocess.Popen(
            [python_exe(), "-X", "faulthandler", "-m", "tui_gateway.entry"],
            cwd=str(cwd),
            env=env,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=self._stderr_fh,
            bufsize=0,
        )
        self._ids = itertools.count(1)
        self._lock = threading.Condition()
        self._responses: dict[Any, dict[str, Any]] = {}
        self.events: list[dict[str, Any]] = []
        self.malformed: list[bytes] = []
        self._write_lock = threading.Lock()
        self._reader = threading.Thread(target=self._read_loop, name="tui-rpc-reader", daemon=True)
        self._reader.start()

    # ── wire ───────────────────────────────────────────────────────────────
    def _read_loop(self) -> None:
        assert self.proc.stdout is not None
        for raw in iter(self.proc.stdout.readline, b""):
            try:
                frame = json.loads(raw)
            except json.JSONDecodeError:
                self.malformed.append(raw[:400])
                continue
            with self._lock:
                if frame.get("method") == "event":
                    params = dict(frame.get("params") or {})
                    params["_t"] = time.monotonic()
                    params["_i"] = len(self.events)
                    self.events.append(params)
                elif "id" in frame and ("result" in frame or "error" in frame):
                    self._responses[frame["id"]] = frame
                self._lock.notify_all()
        with self._lock:
            self._lock.notify_all()

    def send(self, method: str, params: dict[str, Any] | None = None) -> int:
        rid = next(self._ids)
        line = json.dumps({"jsonrpc": "2.0", "id": rid, "method": method, "params": params or {}})
        assert self.proc.stdin is not None
        with self._write_lock:
            self.proc.stdin.write(line.encode() + b"\n")
            self.proc.stdin.flush()
        return rid

    def wait_response(self, rid: int, timeout: float) -> dict[str, Any]:
        deadline = time.monotonic() + timeout
        with self._lock:
            while rid not in self._responses:
                left = deadline - time.monotonic()
                if left <= 0:
                    raise RpcError(f"no response to rpc id={rid} within {timeout}s{self.tail()}")
                if self.proc.poll() is not None and not self._reader.is_alive():
                    raise RpcError(f"gateway exited rc={self.proc.returncode} before rpc id={rid}{self.tail()}")
                self._lock.wait(min(left, 0.5))
            return self._responses.pop(rid)

    def call(self, method: str, params: dict[str, Any] | None = None, *, timeout: float = 60.0) -> dict[str, Any]:
        frame = self.wait_response(self.send(method, params), timeout)
        if "error" in frame:
            raise RpcError(f"{method} -> {frame['error']}")
        return frame["result"]

    def timed_call(self, method: str, params: dict[str, Any] | None = None, *, timeout: float) -> float:
        t0 = time.monotonic()
        frame = self.wait_response(self.send(method, params), timeout)
        if "error" in frame:
            raise RpcError(f"{method} -> {frame['error']}")
        return time.monotonic() - t0

    # ── events ─────────────────────────────────────────────────────────────
    def event_count(self) -> int:
        with self._lock:
            return len(self.events)

    def wait_event(
        self, pred: Callable[[dict[str, Any]], bool], *, timeout: float, start: int = 0,
    ) -> dict[str, Any] | None:
        """First event at index >= ``start`` matching ``pred``; None at the deadline."""
        deadline = time.monotonic() + timeout
        seen = start
        with self._lock:
            while True:
                while seen < len(self.events):
                    ev = self.events[seen]
                    seen += 1
                    if pred(ev):
                        return ev
                left = deadline - time.monotonic()
                if left <= 0 or (self.proc.poll() is not None and not self._reader.is_alive()):
                    return None
                self._lock.wait(min(left, 0.5))

    def events_since(self, start: int, sid: str | None = None) -> list[dict[str, Any]]:
        with self._lock:
            evs = list(self.events[start:])
        return [e for e in evs if sid is None or e.get("session_id") == sid]

    # ── lifecycle ──────────────────────────────────────────────────────────
    def close_stdin_and_wait(self, timeout: float) -> int | None:
        """EOF on stdin is how the TUI/Desktop parent says goodbye; the gateway must exit
        on its own. Returns the exit code, or None if it is still alive at the deadline."""
        try:
            assert self.proc.stdin is not None
            self.proc.stdin.close()
        except OSError:
            pass
        try:
            return self.proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            return None

    def kill(self) -> None:
        if self.proc.poll() is None:
            self.proc.kill()
            try:
                self.proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                pass
        self._stderr_fh.close()

    def tail(self, n: int = 4000) -> str:
        try:
            if not self._stderr_fh.closed:
                self._stderr_fh.flush()
            data = self.stderr_path.read_bytes()[-n:].decode(errors="replace")
        except OSError:
            data = ""
        return f"\n--- gateway stderr tail ---\n{data}" if data else ""


class Heartbeat:
    """Issue cheap existing RPCs (round-robin) every ``interval`` s on their own thread and
    record each round-trip latency — the dispatcher must keep answering while a turn is wedged."""

    def __init__(self, gw: TuiGatewayProcess, calls: list[tuple[str, dict[str, Any]]], *,
                 interval: float = 0.15, per_call_timeout: float = 30.0) -> None:
        self.gw, self.calls = gw, calls
        self.interval, self.per_call_timeout = interval, per_call_timeout
        self.latencies: list[float] = []
        self.failures: list[str] = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="tui-heartbeat", daemon=True)

    def _run(self) -> None:
        n = 0
        while not self._stop.is_set():
            method, params = self.calls[n % len(self.calls)]
            n += 1
            self._inflight_since = time.monotonic()
            try:
                self.latencies.append(self.gw.timed_call(method, params, timeout=self.per_call_timeout))
            except Exception as exc:  # recorded and asserted by the test
                self.failures.append(repr(exc)[:300])
                if len(self.failures) > 3:
                    return
            finally:
                self._inflight_since = None
            self._stop.wait(self.interval)

    def __enter__(self) -> "Heartbeat":
        self._inflight_since: float | None = None
        self._thread.start()
        return self

    def __exit__(self, *_exc: object) -> None:
        self._stop.set()
        self._thread.join(5.0)
        # A call still unanswered when the window closes counts with its age so far: a
        # dispatcher wedged for the whole turn must not look healthy by answering nothing.
        since = self._inflight_since
        if self._thread.is_alive() and since is not None:
            self.latencies.append(time.monotonic() - since)


def gateway_cwd(root: Path) -> Path:
    work = root / "work"
    work.mkdir(parents=True, exist_ok=True)
    return work


# System dirs only: user-level CLIs (``gh``, ``node``…) that the gateway probes in the
# background would otherwise make process hygiene depend on the developer's machine.
_SYSTEM_PATH = ("/usr/local/bin", "/usr/bin", "/bin")


def env_for_gateway(base_env: dict[str, str], work: Path) -> dict[str, str]:
    env = dict(base_env)
    env["TERMINAL_ENV"] = "local"
    env["HERMES_YOLO_MODE"] = "1"  # the terminal tool must never wait on an approval prompt
    # The child's HOME *is* the tmp root, so the live-DB guard (pytest ancestry) would
    # take our tmp state.db for "production"; the documented child opt-out is safe here.
    env["HERMES_STATE_DB_GUARD_BYPASS"] = "1"
    env["PWD"] = str(work)
    tmp = work.parent / "tmp"
    tmp.mkdir(exist_ok=True)
    env["TMPDIR"] = str(tmp)  # tool snapshots/spill files stay inside the scenario root
    env["TERMINAL_CWD"] = str(work)
    env["PATH"] = os.pathsep.join((str(Path(python_exe()).parent), *_SYSTEM_PATH))
    for key in ("DISPLAY", "WAYLAND_DISPLAY", "DBUS_SESSION_BUS_ADDRESS", "SSH_AUTH_SOCK"):
        env.pop(key, None)
    return env
