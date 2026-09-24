"""Shared bits for the dashboard lane's open-issue cells (#120527, #120937).

* ``PtyDashboard``: the lane's ``Dashboard`` harness, but spawned with a pseudo-terminal slave as
  stdin — the shape of ``hermes serve`` / ``hermes dashboard`` launched from an interactive
  terminal. The master end is held open and never written to (nobody watches that console).
  Teardown SIGKILLs the whole process group: a server wedged in a worker thread never finishes
  a graceful shutdown (the default executor is joined on exit).
* ``GatewayApiServer``: a real ``python -m gateway.run`` with the API-server platform enabled via
  the profile ``.env`` exactly as a user configures it (``API_SERVER_ENABLED``/``_KEY``/``_HOST``/
  ``_PORT``), retrying the pick-then-bind port race like the parity lane's driver.
* ``KnownIssue`` assertion subclasses: a strict xfail cell declares ``raises=<its subclass>`` so
  only the bug's own signature is excused; a boot failure or any other assertion stays red.
"""

from __future__ import annotations

import contextlib
import json
import os
import secrets
import signal
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import httpx

from tests.e2e.core.dashboard._helpers import (
    _READY_RE, _TOKEN_RE, Dashboard, Sandbox, group_pids, kill_group, poll,
)


class KnownIssue(AssertionError):
    """Base for assertions that encode an OPEN issue's user-visible signature."""


class Issue120527(KnownIssue):
    """#120527: a dashboard MCP catalog install wedges the serve process (interactive prompt)."""


class Issue120937(KnownIssue):
    """#120937: /api/sessions/{id}/chat silently truncates a long message."""


def xfail_known(known: dict[str, tuple[str, type[KnownIssue]]], name: str):
    """``pytest.mark.xfail(strict=True)`` for a KNOWN entry, excusing ONLY its issue's exception."""
    import pytest

    reason, exc = known[name]
    return pytest.mark.xfail(strict=True, raises=exc, reason=reason)


# Dashboard with a TTY stdin -----------------------------------------------------------------------


class PtyDashboard(Dashboard):
    """``hermes dashboard`` whose stdin is a pty slave (interactive-terminal launch); ``tty=False``
    gives the same process with stdin=/dev/null (service / desktop-spawned launch) as a control."""

    def __init__(self, sb: Sandbox, log_path: Path, extra_env: dict[str, str] | None = None,
                 tty: bool = True) -> None:
        self.sb = sb
        self.log_path = log_path
        self._log = open(log_path, "a", encoding="utf-8")  # noqa: SIM115 - closed in close()
        if tty:
            self.pty_master, slave = os.openpty()
        else:
            self.pty_master, slave = -1, os.open(os.devnull, os.O_RDONLY)
        env = sb.env({"HERMES_WEB_DIST": str(sb.root / "web_dist"), "TERM": "xterm-256color",
                      **(extra_env or {})})
        try:
            self.proc = subprocess.Popen(
                [sys.executable, "-m", "hermes_cli.main", "dashboard", "--no-open", "--skip-build",
                 "--host", "127.0.0.1", "--port", "0"],
                cwd=str(sb.home), env=env, stdin=slave, stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT, text=True, bufsize=1, start_new_session=True,
            )
        finally:
            os.close(slave)
        port_box: list[int] = []

        def pump() -> None:
            assert self.proc.stdout is not None
            for line in self.proc.stdout:
                self._log.write(line)
                self._log.flush()
                m = _READY_RE.search(line)
                if m and not port_box:
                    port_box.append(int(m.group(1)))
        threading.Thread(target=pump, daemon=True, name="dash-pty-stdout").start()
        try:
            self.port = poll(lambda: port_box[0] if port_box else (self.proc.poll() is not None and -1), 120,
                             "hermes dashboard (tty stdin) to report its port")
            assert self.port > 0, f"hermes dashboard exited rc={self.proc.returncode}:\n{self.log_tail()}"
            self.base = f"http://127.0.0.1:{self.port}"
            self.http = httpx.Client(base_url=self.base, timeout=30.0, trust_env=False)
            index = self.http.get("/")
            m = _TOKEN_RE.search(index.text)
            assert index.status_code == 200 and m, f"index.html carried no session token: {index.status_code}"
            self.token = m.group(1)
        except BaseException:
            self.close()
            raise

    def stdin_target(self) -> str:
        """What the child's fd 0 really is (read from the child itself): /dev/pts/N or /dev/null."""
        try:
            return os.readlink(f"/proc/{self.proc.pid}/fd/0")
        except OSError as exc:
            return f"unreadable: {exc}"

    def close(self) -> None:
        with contextlib.suppress(Exception):
            self.http.close()
        pids = group_pids(self.proc.pid)
        kill_group(self.proc)  # SIGKILL: a wedged worker thread blocks any graceful exit
        with contextlib.suppress(Exception):
            self.proc.wait(timeout=15)
        for pid in pids:  # anything that left the group (setsid'd MCP children) but was ours
            with contextlib.suppress(ProcessLookupError, PermissionError):
                os.kill(pid, signal.SIGKILL)
        if self.pty_master >= 0:
            with contextlib.suppress(OSError):
                os.close(self.pty_master)
        with contextlib.suppress(Exception):
            self._log.close()


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
            self._log = open(log_path, "a", encoding="utf-8")  # noqa: SIM115 - closed in stop()
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
