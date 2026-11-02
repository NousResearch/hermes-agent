"""Shared bits for the dashboard lane's open-issue cells (#120937).

* ``GatewayApiServer``: a real ``python -m gateway.run`` with the API-server platform enabled via
  the profile ``.env`` exactly as a user configures it (``API_SERVER_ENABLED``/``_KEY``/``_HOST``/
  ``_PORT``), retrying the pick-then-bind port race like the parity lane's driver.
* ``KnownIssue`` assertion subclasses: a KNOWN cell gates its final check with
  ``known_gate(KNOWN, name, raises=<its subclass>)`` (tests/e2e/core/_pending_fixes.py), so only the
  bug's own signature is excused; a boot failure or any other assertion stays red, and the cell
  simply passes once the fix lands.
"""

from __future__ import annotations

import contextlib
import json
import secrets
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from tests.e2e.core.dashboard._helpers import Sandbox, kill_group


class KnownIssue(AssertionError):
    """Base for assertions that encode an OPEN issue's user-visible signature."""


class Issue120937(KnownIssue):
    """#120937: /api/sessions/{id}/chat silently truncates a long message."""


# Gateway API server ------------------------------------------------------------------------------


PORT_ATTEMPTS = 3


def _free_port() -> int:
    import socket
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def http_json(method: str, url: str, *, key: str | None = None, body: Any = None,
              timeout: float = 10.0) -> tuple[int, Any]:
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, method=method)
    if data is not None:
        req.add_header("Content-Type", "application/json")
    if key:
        req.add_header("Authorization", f"Bearer {key}")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8", "replace")
            status = resp.status
    except urllib.error.HTTPError as exc:
        raw, status = exc.read().decode("utf-8", "replace"), exc.code
    try:
        return status, json.loads(raw)
    except ValueError:
        return status, {"raw": raw}


class GatewayApiServer:
    """``python -m gateway.run`` serving the API-server platform on 127.0.0.1:<port>."""

    def __init__(self, sb: Sandbox, hermes_home: Path, log_path: Path, ready_timeout: float = 120.0) -> None:
        self.sb, self.hermes_home, self.log_path = sb, hermes_home, log_path
        self.key = secrets.token_hex(32)
        env_path = hermes_home / ".env"
        env_before = env_path.read_text(encoding="utf-8") if env_path.exists() else ""
        for _ in range(PORT_ATTEMPTS):
            port = _free_port()
            env_path.write_text(env_before + "".join(f"{k}={v}\n" for k, v in {
                "API_SERVER_ENABLED": "true", "API_SERVER_KEY": self.key,
                "API_SERVER_HOST": "127.0.0.1", "API_SERVER_PORT": str(port)}.items()), encoding="utf-8")
            self._log = open(log_path, "a", encoding="utf-8")
            self.proc = subprocess.Popen(
                [sys.executable, "-m", "gateway.run"], cwd=str(sb.home), env=sb.env({"HERMES_HOME": str(hermes_home)}),
                stdin=subprocess.DEVNULL, stdout=self._log, stderr=subprocess.STDOUT, start_new_session=True,
            )
            self.base = f"http://127.0.0.1:{port}"
            try:
                ready = self._await_ready(ready_timeout)
            except BaseException:
                self.stop()
                raise
            if ready:
                return
            self.stop()
        raise AssertionError(f"api server lost the port race {PORT_ATTEMPTS} times\n{self.log_tail()}")

    def log_tail(self, n: int = 4000) -> str:
        try:
            return self.log_path.read_text(encoding="utf-8", errors="replace")[-n:]
        except OSError:
            return ""

    def _await_ready(self, timeout: float) -> bool:
        deadline = time.monotonic() + timeout
        while True:
            if "already in use" in self.log_tail(20000):
                return False
            if self.proc.poll() is not None:
                raise AssertionError(f"gateway exited {self.proc.returncode} before ready\n{self.log_tail()}")
            if time.monotonic() >= deadline:
                raise AssertionError(f"api server never became ready on {self.base}\n{self.log_tail()}")
            with contextlib.suppress(OSError, urllib.error.URLError, ValueError):
                status, body = http_json("GET", f"{self.base}/health/detailed", key=self.key, timeout=2.0)
                if status == 200 and isinstance(body, dict) and body.get("pid") == self.proc.pid:
                    return True
            time.sleep(0.2)

    def call(self, method: str, path: str, body: Any = None, timeout: float = 60.0) -> tuple[int, Any]:
        return http_json(method, f"{self.base}{path}", key=self.key, body=body, timeout=timeout)

    def stop(self) -> None:
        if self.proc.poll() is None:
            kill_group(self.proc, signal.SIGTERM)
            try:
                self.proc.wait(timeout=20)
            except subprocess.TimeoutExpired:
                pass
        kill_group(self.proc)
        with contextlib.suppress(Exception):
            self.proc.wait(timeout=10)
        with contextlib.suppress(Exception):
            self._log.close()
