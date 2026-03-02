"""Minimal stdio JSON-RPC client for `codex app-server`."""

from __future__ import annotations

import json
import logging
import os
import queue
import subprocess
import threading
import time
from datetime import datetime
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class CodexAppServerError(RuntimeError):
    """Raised when codex app-server request/response fails."""


@dataclass
class RpcResponse:
    result: Optional[Dict[str, Any]] = None
    error: Optional[Dict[str, Any]] = None


class CodexAppServerClient:
    """JSON-RPC client for codex app-server over stdio."""

    def __init__(self, *, binary: str = "codex", cwd: Optional[str] = None) -> None:
        self.binary = binary
        self.cwd = cwd
        self._proc: Optional[subprocess.Popen] = None
        self._reader_thread: Optional[threading.Thread] = None
        self._stderr_thread: Optional[threading.Thread] = None
        self._write_lock = threading.Lock()
        self._pending_lock = threading.Lock()
        self._next_id = 1
        self._pending: Dict[int, queue.Queue] = {}
        self._notifications: "queue.Queue[Dict[str, Any]]" = queue.Queue()
        self._closed = False
        logs_dir = Path.home() / ".hermes" / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)
        self._stderr_log_path = logs_dir / "codex-app-server.log"
        self._rpc_log_enabled = str(os.getenv("HERMES_CODEX_APP_SERVER_RPC_LOG", "")).strip().lower() in {"1", "true", "yes", "on"}
        self._rpc_log_path = logs_dir / "codex-app-server-rpc.log"

    def start(self) -> None:
        if self._proc is not None:
            return
        self._proc = subprocess.Popen(
            [self.binary, "app-server"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=self.cwd,
            text=True,
            bufsize=1,
        )
        self._reader_thread = threading.Thread(target=self._reader_loop, daemon=True)
        self._reader_thread.start()
        self._stderr_thread = threading.Thread(target=self._stderr_loop, daemon=True)
        self._stderr_thread.start()

    def close(self) -> None:
        self._closed = True
        proc = self._proc
        if proc is None:
            return
        try:
            proc.terminate()
            proc.wait(timeout=2)
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass
        self._proc = None

    def initialize(
        self,
        *,
        client_name: str,
        client_title: str,
        client_version: str,
        experimental_api: bool = False,
    ) -> None:
        caps: Dict[str, Any] = {}
        if experimental_api:
            caps["experimentalApi"] = True
        params: Dict[str, Any] = {
            "clientInfo": {
                "name": client_name,
                "title": client_title,
                "version": client_version,
            },
        }
        if caps:
            params["capabilities"] = caps
        self.call("initialize", params=params, timeout=15.0)
        self.notify("initialized", params={})

    def call(self, method: str, *, params: Optional[Dict[str, Any]] = None, timeout: float = 30.0) -> Dict[str, Any]:
        req_id = self._allocate_id()
        response_queue: "queue.Queue[RpcResponse]" = queue.Queue(maxsize=1)
        with self._pending_lock:
            self._pending[req_id] = response_queue
        self._send({"method": method, "id": req_id, "params": params or {}})
        try:
            response = response_queue.get(timeout=max(0.1, timeout))
        except queue.Empty as exc:
            with self._pending_lock:
                self._pending.pop(req_id, None)
            raise CodexAppServerError(f"Timed out waiting for response to {method}") from exc
        if response.error:
            code = response.error.get("code")
            message = response.error.get("message", "Unknown error")
            raise CodexAppServerError(f"{method} failed ({code}): {message}")
        return response.result or {}

    def notify(self, method: str, *, params: Optional[Dict[str, Any]] = None) -> None:
        self._send({"method": method, "params": params or {}})

    def next_notification(self, timeout: float = 0.1) -> Optional[Dict[str, Any]]:
        try:
            return self._notifications.get(timeout=max(0.01, timeout))
        except queue.Empty:
            return None

    def _allocate_id(self) -> int:
        with self._pending_lock:
            req_id = self._next_id
            self._next_id += 1
            return req_id

    def _send(self, payload: Dict[str, Any]) -> None:
        proc = self._proc
        if proc is None or proc.stdin is None:
            raise CodexAppServerError("codex app-server process is not running")
        line = json.dumps(payload, ensure_ascii=False)
        self._log_rpc(">>", payload)
        with self._write_lock:
            try:
                proc.stdin.write(line + "\n")
                proc.stdin.flush()
            except Exception as exc:
                raise CodexAppServerError("Failed writing to codex app-server stdin") from exc

    def _reader_loop(self) -> None:
        proc = self._proc
        if proc is None or proc.stdout is None:
            return
        try:
            for raw_line in proc.stdout:
                line = raw_line.strip()
                if not line:
                    continue
                try:
                    msg = json.loads(line)
                except Exception:
                    logger.debug("Invalid app-server JSON line: %s", line)
                    continue
                self._dispatch_message(msg)
        finally:
            if not self._closed:
                self._fail_all_pending("codex app-server stream closed")

    def _stderr_loop(self) -> None:
        proc = self._proc
        if proc is None or proc.stderr is None:
            return
        for raw_line in proc.stderr:
            line = raw_line.rstrip()
            if not line:
                continue
            logger.debug("codex app-server stderr: %s", line)
            try:
                ts = datetime.now().isoformat(timespec="seconds")
                with self._stderr_log_path.open("a", encoding="utf-8") as f:
                    f.write(f"{ts} {line}\n")
            except Exception:
                # Logging must never break the stderr reader loop.
                pass

    def _dispatch_message(self, msg: Dict[str, Any]) -> None:
        if not isinstance(msg, dict):
            return
        self._log_rpc("<<", msg)
        if "id" in msg and ("result" in msg or "error" in msg):
            req_id = msg.get("id")
            if not isinstance(req_id, int):
                return
            with self._pending_lock:
                response_queue = self._pending.pop(req_id, None)
            if response_queue is None:
                return
            response_queue.put(
                RpcResponse(
                    result=msg.get("result"),
                    error=msg.get("error"),
                )
            )
            return
        if "method" in msg:
            self._notifications.put(msg)

    def _fail_all_pending(self, reason: str) -> None:
        with self._pending_lock:
            pending = list(self._pending.items())
            self._pending.clear()
        for _, response_queue in pending:
            response_queue.put(RpcResponse(error={"code": -1, "message": reason}))

    def _log_rpc(self, direction: str, payload: Dict[str, Any]) -> None:
        if not self._rpc_log_enabled:
            return
        try:
            ts = datetime.now().isoformat(timespec="seconds")
            serialized = json.dumps(payload, ensure_ascii=False)
            with self._rpc_log_path.open("a", encoding="utf-8") as f:
                f.write(f"{ts} {direction} {serialized}\n")
        except Exception:
            pass


def wait_for_notification(
    client: CodexAppServerClient,
    *,
    method: str,
    timeout: float,
    login_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Wait for a specific notification method."""
    deadline = time.time() + max(1.0, timeout)
    while time.time() < deadline:
        msg = client.next_notification(timeout=0.2)
        if not msg:
            continue
        if msg.get("method") != method:
            continue
        params = msg.get("params", {})
        if login_id is not None and params.get("loginId") != login_id:
            continue
        return params if isinstance(params, dict) else {}
    raise TimeoutError(f"Timed out waiting for notification: {method}")
