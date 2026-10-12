"""Entrypoint drivers for the parity matrix: the JSON-RPC surfaces.

* ``drive_tui_gateway`` — ``python -m tui_gateway.entry`` over stdio, exactly how
  the Ink TUI spawns it (launch dir = the user's shell cwd); stops on stdin EOF.
* ``drive_serve`` — ``hermes serve --host 127.0.0.1 --port 0`` spawned the way the
  Desktop spawns it (``TERMINAL_CWD`` + ``HERMES_DASHBOARD_SESSION_TOKEN`` +
  ``HERMES_DESKTOP=1`` env, stdin closed), port read from the READY sentinel on
  stdout, one turn over the ``/api/ws`` JSON-RPC WebSocket; stops on SIGTERM.

Both speak the same contract (``tui_gateway/contracts``): ``session.create`` →
``prompt.submit`` → ``message.complete`` event. Follows the driver contract in
``_drive_cli``.
"""

from __future__ import annotations

import json
import queue
import subprocess
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable

from tests.e2e.core.parity._helpers import (
    TURN_TIMEOUT,
    DriveResult,
    ParityHome,
    hermes_argv,
    terminate,
)
from tests.fakes.fake_llm_provider import FakeLLMServer

import sys

READY_TIMEOUT = 120.0


class RpcClient:
    """Minimal JSON-RPC 2.0 client over a line/frame transport (``send`` + ``recv`` callables)."""

    def __init__(self, send: Callable[[str], None], frames: "queue.Queue[str | None]") -> None:
        self._send = send
        self._frames = frames
        self._next_id = 0
        self.events: list[dict[str, Any]] = []
        self._responses: dict[Any, dict[str, Any]] = {}

    def _pump(self, timeout: float) -> bool:
        try:
            raw = self._frames.get(timeout=timeout)
        except queue.Empty:
            return False
        if raw is None:
            raise AssertionError("JSON-RPC transport closed")
        msg = json.loads(raw)
        if msg.get("method") == "event":
            self.events.append(msg.get("params") or {})
        elif "id" in msg and "method" in msg:
            # Server→client request (approval/clarify/...): this scripted turn never needs one.
            self._send(json.dumps({"jsonrpc": "2.0", "id": msg["id"],
                                   "error": {"code": -32601, "message": "parity driver: unsupported"}}))
        elif "id" in msg:
            self._responses[msg["id"]] = msg
        return True

    def call(self, method: str, params: dict[str, Any] | None = None, timeout: float = 60.0) -> Any:
        self._next_id += 1
        rid = f"parity-{self._next_id}"
        self._send(json.dumps({"jsonrpc": "2.0", "id": rid, "method": method, "params": params or {}}))
        deadline = time.monotonic() + timeout
        while rid not in self._responses:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise AssertionError(f"{method}: no response within {timeout}s")
            self._pump(min(remaining, 0.5))
        resp = self._responses.pop(rid)
        if "error" in resp:
            raise AssertionError(f"{method} failed: {resp['error']}")
        return resp.get("result")

    def wait_event(self, etype: str, pred: Callable[[dict], bool] = lambda _e: True,
                   timeout: float = TURN_TIMEOUT) -> dict[str, Any]:
        deadline = time.monotonic() + timeout
        seen = 0
        while True:
            for ev in self.events[seen:]:
                if ev.get("type") == etype and pred(ev):
                    return ev
            seen = len(self.events)
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise AssertionError(
                    f"no {etype!r} event within {timeout}s; saw {[e.get('type') for e in self.events][-30:]}")
            self._pump(min(remaining, 0.5))


def run_turn(rpc: RpcClient, prompt: str, create_params: dict[str, Any]) -> str:
    created = rpc.call("session.create", create_params, timeout=TURN_TIMEOUT)
    sid = created["session_id"]
    rpc.call("prompt.submit", {"session_id": sid, "text": prompt}, timeout=TURN_TIMEOUT)
    done = rpc.wait_event("message.complete", lambda e: e.get("session_id") == sid)
    payload = done.get("payload") or {}
    text = payload.get("text")
    return text if isinstance(text, str) else json.dumps(text)


@dataclass
class StreamCapture:
    """Every stdout line (in order) plus the stderr text of a child, pumped on threads."""

    stdout_lines: "queue.Queue[str | None]" = field(default_factory=queue.Queue)
    stdout_seen: list[str] = field(default_factory=list)
    stderr_chunks: list[str] = field(default_factory=list)

    def start(self, proc: subprocess.Popen) -> "StreamCapture":
        def out() -> None:
            assert proc.stdout is not None
            for line in proc.stdout:
                self.stdout_seen.append(line)
                self.stdout_lines.put(line)
            self.stdout_lines.put(None)

        def err() -> None:
            assert proc.stderr is not None
            for line in proc.stderr:
                self.stderr_chunks.append(line)

        threading.Thread(target=out, daemon=True, name="parity-stdout").start()
        threading.Thread(target=err, daemon=True, name="parity-stderr").start()
        return self

    @property
    def stderr(self) -> str:
        return "".join(self.stderr_chunks)


# tui_gateway (stdio) ---------------------------------------------------------------


def spawn_tui_gateway(ph: ParityHome) -> tuple[subprocess.Popen, StreamCapture]:
    proc = subprocess.Popen(
        [sys.executable, "-m", "tui_gateway.entry"], cwd=ph.project, env=ph.env(),
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1,
    )
    return proc, StreamCapture().start(proc)


def drive_tui_gateway(ph: ParityHome, srv: FakeLLMServer, prompt: str) -> DriveResult:
    proc, cap = spawn_tui_gateway(ph)

    def send(line: str) -> None:
        assert proc.stdin is not None
        proc.stdin.write(line + "\n")
        proc.stdin.flush()

    graceful = False
    try:
        rpc = RpcClient(send, cap.stdout_lines)
        rpc.wait_event("gateway.ready", timeout=READY_TIMEOUT)
        text = run_turn(rpc, prompt, {})
    finally:
        # Normal stop for a stdio gateway: the client closes the pipe.
        if proc.stdin is not None and not proc.stdin.closed:
            proc.stdin.close()
        try:
            proc.wait(timeout=60)
            graceful = True
        except subprocess.TimeoutExpired:
            terminate(proc)
    return DriveResult(final_text=text, toolset="hermes-cli", graceful_exit=graceful,
                       extra={"exit_code": proc.returncode, "stderr_tail": cap.stderr[-2000:]})


# hermes serve (Desktop backend) ------------------------------------------------------


@dataclass
class ServeProcess:
    proc: subprocess.Popen
    cap: StreamCapture
    token: str


def spawn_serve(ph: ParityHome) -> ServeProcess:
    """Spawn ``hermes serve`` with the Desktop's argv/env/stdio shape (electron/main.ts)."""
    token = uuid.uuid4().hex
    env = ph.env({
        "TERMINAL_CWD": str(ph.project),
        "HERMES_DASHBOARD_SESSION_TOKEN": token,
        "HERMES_DESKTOP": "1",
    })
    proc = subprocess.Popen(
        hermes_argv("serve", "--host", "127.0.0.1", "--port", "0"), cwd=ph.home, env=env,
        stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1,
    )
    return ServeProcess(proc=proc, cap=StreamCapture().start(proc), token=token)


def first_stdout_line(sp: ServeProcess, timeout: float = READY_TIMEOUT) -> str:
    """The first stdout line, failing fast if the sentinel shows up on stderr instead."""
    deadline = time.monotonic() + timeout
    while True:
        try:
            line = sp.cap.stdout_lines.get(timeout=0.2)
        except queue.Empty:
            line = ""
        if line is None:
            raise AssertionError(f"serve exited {sp.proc.poll()} before any stdout line\n{sp.cap.stderr[-3000:]}")
        if line:
            return line
        if "HERMES_BACKEND_READY" in sp.cap.stderr:
            raise AssertionError(
                "READY sentinel was written to STDERR; the Desktop only watches stdout "
                f"(backend-ready.ts):\n{sp.cap.stderr[-1500:]}")
        if time.monotonic() >= deadline:
            raise AssertionError(f"no stdout line from serve within {timeout}s\n{sp.cap.stderr[-3000:]}")


def desktop_port(sp: ServeProcess, timeout: float = READY_TIMEOUT) -> int:
    """The port the Desktop would learn from this backend's stdout (junk-tolerant, like
    backend-ready.ts). The strict "sentinel is the first line" rule is test_boot_contract's."""
    from tests.e2e.core.parity._boot_contract import desktop_parse

    deadline = time.monotonic() + timeout
    seen = ""
    while True:
        remaining = deadline - time.monotonic()
        line = first_stdout_line(sp, timeout=max(remaining, 0.1))
        seen += line
        verdict = desktop_parse(seen)
        if "port" in verdict:
            return int(verdict["port"])


def ws_rpc(port: int, token: str) -> tuple[RpcClient, Callable[[], None]]:
    from websockets.sync.client import connect

    ws = connect(f"ws://127.0.0.1:{port}/api/ws?token={token}", open_timeout=30, max_size=None)
    frames: queue.Queue[str | None] = queue.Queue()

    def pump() -> None:
        try:
            for msg in ws:
                frames.put(msg if isinstance(msg, str) else msg.decode())
        except Exception:
            pass
        frames.put(None)

    threading.Thread(target=pump, daemon=True, name="parity-ws").start()
    return RpcClient(ws.send, frames), ws.close


def drive_serve(ph: ParityHome, srv: FakeLLMServer, prompt: str) -> DriveResult:
    sp = spawn_serve(ph)
    graceful = False
    try:
        port = desktop_port(sp)
        rpc, close = ws_rpc(port, sp.token)
        try:
            rpc.wait_event("gateway.ready", timeout=READY_TIMEOUT)
            text = run_turn(rpc, prompt, {"cwd": str(ph.project)})
        finally:
            close()
    finally:
        # Desktop quit path: SIGTERM, bounded wait.
        code = terminate(sp.proc, timeout=60)
        graceful = code is not None and code in (0, -15, 143)
    return DriveResult(final_text=text, toolset="hermes-cli", cwd_channel="TERMINAL_CWD + session.create cwd",
                       graceful_exit=graceful,
                       extra={"exit_code": sp.proc.returncode, "stderr_tail": sp.cap.stderr[-2000:]})
