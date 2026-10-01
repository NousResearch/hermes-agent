"""Real ``hermes serve`` backend + WebSocket JSON-RPC clients for the terminal lane E2E suites.

One backend per test module: a real ``python -m hermes_cli.main serve --port 0`` subprocess (the
exact argv Desktop spawns, see ``apps/desktop/electron/backend-command.ts``) with an isolated
HOME/HERMES_HOME, no inherited provider credentials, and only the recording fake LLM provider
configured. Clients speak the Desktop wire protocol over ``/api/ws`` (newline JSON-RPC,
``gateway.ready`` on accept, ``client.capabilities`` to answer server→client requests), so every
event, lease, reaper and approval path is the production one.
"""

from __future__ import annotations

import json
import os
import re
import secrets
import signal
import socket
import sqlite3
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Callable, Iterable

from websockets.sync.client import connect as ws_connect

from tests.fakes.fake_llm_provider import FakeLLMServer, write_hermes_home

REPO_ROOT = Path(__file__).resolve().parents[4]
READY_RE = re.compile(r"HERMES_BACKEND_READY port=(\d+)")


class RpcError(AssertionError):
    def __init__(self, method: str, error: dict) -> None:
        self.method = method
        self.code = error.get("code")
        self.message = str(error.get("message") or "")
        self.data = error.get("data")
        super().__init__(f"{method} -> {self.code}: {self.message}")


def poll_until(fn: Callable[[], Any], *, timeout: float, interval: float = 0.05, what: str = "condition") -> Any:
    """Return the first truthy ``fn()`` before ``timeout``; AssertionError naming ``what`` otherwise."""
    deadline = time.monotonic() + timeout
    last_exc: Exception | None = None
    while True:
        try:
            value = fn()
            if value:
                return value
        except AssertionError:
            raise
        except Exception as exc:  # noqa: BLE001 - surfaced in the timeout message
            last_exc = exc
        if time.monotonic() >= deadline:
            raise AssertionError(f"timed out after {timeout:.0f}s waiting for {what}"
                                 + (f" (last error: {last_exc!r})" if last_exc else ""))
        time.sleep(interval)


def _operator_home() -> str:
    """The account's real home (passwd), immune to a test-time HOME override."""
    try:
        import pwd
        return pwd.getpwuid(os.getuid()).pw_dir
    except (ImportError, KeyError):  # pragma: no cover - non-POSIX
        return os.path.expanduser("~")


def _minimal_env(home: Path, hermes_home: Path, token: str, extra: dict[str, str] | None) -> dict[str, str]:
    """A clean child env: nothing Hermes-, provider- or credential-shaped leaks in from the test runner."""
    env = {k: os.environ[k] for k in ("PATH", "LANG", "LC_ALL", "TERM", "SYSTEMROOT") if k in os.environ}
    tmpdir = home.parent / "tmp"
    tmpdir.mkdir(parents=True, exist_ok=True)
    env.update(
        HOME=str(home), HERMES_HOME=str(hermes_home), PYTHONPATH=str(REPO_ROOT),
        TMPDIR=str(tmpdir), HERMES_DASHBOARD_SESSION_TOKEN=token, PYTHONUNBUFFERED="1",
        NO_COLOR="1",
        # The child's HOME *is* the sandbox, so its "real" root (expanduser('~')/.hermes) is the tmp
        # home and the pytest-ancestry live-DB guard would refuse every open. The guard exists to keep
        # tests off the operator's state.db; Backend.start() asserts the sandbox is outside it.
        HERMES_STATE_DB_GUARD_BYPASS="1",
    )
    env.update(extra or {})
    return env


class Backend:
    """One real ``hermes serve`` process wired to a :class:`FakeLLMServer`."""

    def __init__(self, root: Path, responder, *, extra_config: str = "", env: dict[str, str] | None = None,
                 aux=None) -> None:
        self.root = root
        self.home = root / "home"
        self.hermes_home = self.home / ".hermes"
        self.llm = FakeLLMServer(responder, aux=aux)
        self.extra_config = extra_config
        self.extra_env = env or {}
        self.token = secrets.token_hex(16)
        self.proc: subprocess.Popen | None = None
        self.port: int | None = None
        self.clients: list[WSClient] = []

    def start(self, *, timeout: float = 90.0) -> "Backend":
        operator_root = (Path(_operator_home()) / ".hermes").resolve()
        assert operator_root not in (self.hermes_home.resolve(), *self.hermes_home.resolve().parents), (
            f"sandbox {self.hermes_home} sits inside the operator's Hermes home {operator_root}")
        self.llm.start()
        write_hermes_home(self.hermes_home, self.llm.base_url, extra_config=self.extra_config)
        self._stdout = open(self.root / "serve.stdout.log", "wb")
        self._stderr = open(self.root / "serve.stderr.log", "wb")
        # Own process group: stop() signals exactly the tree this fixture spawned, never a pattern match.
        self.proc = subprocess.Popen(
            [sys.executable, "-m", "hermes_cli.main", "serve", "--host", "127.0.0.1", "--port", "0"],
            cwd=str(self.root), env=_minimal_env(self.home, self.hermes_home, self.token, self.extra_env),
            stdin=subprocess.DEVNULL, stdout=self._stdout, stderr=self._stderr, start_new_session=True)

        proc = self.proc

        def ready() -> int | None:
            if proc.poll() is not None:
                raise AssertionError(f"hermes serve exited {proc.returncode} before ready:\n{self.logs()}")
            match = READY_RE.search((self.root / "serve.stdout.log").read_text(errors="replace"))
            return int(match.group(1)) if match else None

        try:
            self.port = poll_until(ready, timeout=timeout, interval=0.1, what="HERMES_BACKEND_READY")
        except BaseException:
            self.stop()
            raise
        return self

    @property
    def ws_url(self) -> str:
        return f"ws://127.0.0.1:{self.port}/api/ws?token={self.token}"

    def connect(self, name: str) -> "WSClient":
        client = WSClient(self.ws_url, name)
        self.clients.append(client)
        return client

    def logs(self, tail: int = 60) -> str:
        parts = []
        for path in (self.root / "serve.stdout.log", self.root / "serve.stderr.log",
                     self.hermes_home / "logs" / "errors.log"):
            name = path.name
            if path.exists():
                lines = path.read_text(errors="replace").splitlines()[-tail:]
                parts.append(f"--- {name} ---\n" + "\n".join(lines))
        return "\n".join(parts)

    def stop(self) -> None:
        for client in self.clients:
            client.close()
        proc, self.proc = self.proc, None
        if proc is not None and proc.poll() is None:
            try:
                os.killpg(proc.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                proc.wait(timeout=10)
        self.llm.stop()
        for fh in (getattr(self, "_stdout", None), getattr(self, "_stderr", None)):
            if fh is not None:
                fh.close()

    # state.db, read-only (never a second writer next to the backend's)
    def db_rows(self, sql: str, args: Iterable[Any] = ()) -> list[tuple]:
        path = self.hermes_home / "state.db"
        if not path.exists():
            return []
        conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=30)
        try:
            conn.execute("PRAGMA busy_timeout=30000")
            return conn.execute(sql, tuple(args)).fetchall()
        finally:
            conn.close()


class WSClient:
    """One Desktop-shaped WebSocket connection. A reader thread records every inbound frame."""

    def __init__(self, url: str, name: str, *, answers_requests: bool = True) -> None:
        self.name = name
        self.frames: list[dict] = []
        self._cond = threading.Condition()
        self._reading = threading.Event()
        self._reading.set()
        self._next_id = 0
        self.closed = False
        self.dropped = False
        self._ws = ws_connect(url, max_size=None, ping_interval=None, open_timeout=30, close_timeout=2)
        self._thread = threading.Thread(target=self._read_loop, name=f"ws-{name}", daemon=True)
        self._thread.start()
        self.wait_for(lambda f: etype(f) == "gateway.ready", timeout=60, what="gateway.ready")
        if answers_requests:
            self.call("client.capabilities", server_requests=True)

    def _read_loop(self) -> None:
        while True:
            self._reading.wait()
            try:
                raw = self._ws.recv()
            except Exception:  # noqa: BLE001 - closed/dropped socket ends the reader
                break
            for line in str(raw).splitlines():
                if line.strip():
                    frame = json.loads(line)
                    frame["_t"] = time.monotonic()
                    with self._cond:
                        self.frames.append(frame)
                        self._cond.notify_all()
        with self._cond:
            self.closed = True
            self._cond.notify_all()

    # -- absence: stop consuming; the websockets receive queue fills and TCP backpressures the backend
    def pause(self) -> None:
        self._reading.clear()

    def resume(self) -> None:
        self._reading.set()

    def snapshot(self) -> list[dict]:
        with self._cond:
            return list(self.frames)

    def wait_for(self, pred: Callable[[dict], bool], *, timeout: float, start: int = 0, what: str = "frame") -> dict:
        deadline = time.monotonic() + timeout
        with self._cond:
            idx = start
            while True:
                while idx < len(self.frames):
                    frame = self.frames[idx]
                    idx += 1
                    if pred(frame):
                        return frame
                if self.closed:
                    raise AssertionError(f"{self.name}: socket closed while waiting for {what}")
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise AssertionError(f"{self.name}: timed out after {timeout:.0f}s waiting for {what}")
                self._cond.wait(min(remaining, 0.5))

    def send(self, frame: dict) -> None:
        self._ws.send(json.dumps(frame))

    def request(self, method: str, params: dict, *, timeout: float = 60.0) -> dict:
        """Send one RPC; return the raw response frame (``result`` or ``error``)."""
        self._next_id += 1
        rid = self._next_id
        start = len(self.snapshot())
        self.send({"jsonrpc": "2.0", "id": rid, "method": method, "params": params})
        return self.wait_for(lambda f: f.get("id") == rid and "method" not in f, timeout=timeout,
                             start=start, what=f"{method} reply")

    def call(self, method: str, *, timeout: float = 60.0, **params: Any) -> dict:
        frame = self.request(method, params, timeout=timeout)
        if "error" in frame:
            raise RpcError(method, frame["error"])
        return frame.get("result") or {}

    def respond(self, request_id: str, result: dict) -> None:
        """Answer a server→client request (clarify/approval/...) with a response frame."""
        self.send({"jsonrpc": "2.0", "id": request_id, "result": result})

    def events(self, sid: str | None = None, etype: str | None = None) -> list[dict]:
        return [f for f in self.snapshot() if f.get("method") == "event"
                and (sid is None or f["params"].get("session_id") == sid)
                and (etype is None or f["params"].get("type") == etype)]

    def server_requests(self, method: str | None = None) -> list[dict]:
        return [f for f in self.snapshot() if isinstance(f.get("id"), str) and f.get("method") not in (None, "event")
                and (method is None or f.get("method") == method)]

    def drop(self) -> None:
        """Abrupt network loss: no close frame, the backend learns from the dead TCP stream."""
        self.dropped = True
        self._reading.set()
        sock = getattr(self._ws, "socket", None)
        if sock is not None:
            try:
                sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            sock.close()
        self._thread.join(timeout=10)

    def close(self) -> None:
        self._reading.set()
        try:
            self._ws.close()
        except Exception:  # noqa: BLE001 - already dropped
            pass


def etype(frame: dict) -> str:
    """The event ``type`` of a notification frame ('' for replies and server→client requests)."""
    return str((frame.get("params") or {}).get("type") or "") if frame.get("method") == "event" else ""
