"""Test-side driver for a real ``python -m gateway.run`` process with the chaos fake platform.

The gateway child runs the production entry point (plugin discovery, platform registry,
adapter connect, message handler install, SIGTERM drain) in a hermetic home; this module
owns the loopback socket the fake adapter dials, records every adapter event, and gives
the scenarios deadline-bounded waits. It never sleeps as synchronization: every wait is a
condition-variable poll against a deadline.
"""

from __future__ import annotations

import json
import os
import signal
import socket
import subprocess
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

from tests.e2e.core.chaos import _gateway_fake_platform as fake_platform
from tests.e2e.core.chaos._helpers import hermetic_env, kill_tagged, python_exe, write_chaos_home

BOOT_DEADLINE_S = 180.0
SHUTDOWN_DEADLINE_S = 60.0

PLUGIN_MANIFEST = (
    f"name: {fake_platform.PLUGIN_DIR_NAME}\n"
    "label: Chaos Fake\n"
    "kind: platform\n"
    "version: 0.0.1\n"
    "description: in-memory platform for the gateway chaos suite\n"
)
PLUGIN_INIT = "from tests.e2e.core.chaos._gateway_fake_platform import register  # noqa: F401\n"


def gateway_extra_config() -> str:
    """config.yaml additions: enable the user plugin + its platform, terminal-only toolset."""
    name = fake_platform.PLATFORM_NAME
    return (
        "plugins:\n"
        f"  enabled: [{fake_platform.PLUGIN_DIR_NAME}]\n"
        "platforms:\n"
        f"  {name}:\n"
        "    enabled: true\n"
        "platform_toolsets:\n"
        f"  {name}: [terminal]\n"
        "terminal:\n"
        "  backend: local\n"
        # /new is a scenario step (stale-breaker recovery), not an interactive confirmation.
        "approvals:\n"
        "  destructive_slash_confirm: false\n"
    )


@dataclass
class Event:
    ev: str
    data: dict[str, Any]
    mono: float
    idx: int


@dataclass
class GatewayProc:
    root: Path
    base_url: str
    tag: str
    cfg: dict[str, Any] = field(default_factory=dict)
    extra_config: str = ""

    def __post_init__(self) -> None:
        self.home, self.hermes_home = write_chaos_home(
            self.root, self.base_url, extra=gateway_extra_config() + self.extra_config, **self.cfg)
        plugin_dir = self.hermes_home / "plugins" / fake_platform.PLUGIN_DIR_NAME
        plugin_dir.mkdir(parents=True, exist_ok=True)
        (plugin_dir / "plugin.yaml").write_text(PLUGIN_MANIFEST, encoding="utf-8")
        (plugin_dir / "__init__.py").write_text(PLUGIN_INIT, encoding="utf-8")
        self.state_db = self.hermes_home / "state.db"
        self.log_path = self.root / "gateway.log"
        self.events: list[Event] = []
        self._cond = threading.Condition()
        self._conn: Optional[socket.socket] = None
        self._listener: Optional[socket.socket] = None
        self.proc: Optional[subprocess.Popen] = None
        self.exit_code: Optional[int] = None
        self.shutdown_s: Optional[float] = None
        self._msg_seq = 0

    # -- lifecycle -------------------------------------------------------------

    def start(self) -> None:
        listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        listener.bind(("127.0.0.1", 0))
        listener.listen(1)
        listener.settimeout(1.0)
        self._listener = listener
        env = hermetic_env(self.home, self.hermes_home, self.tag)
        for key in ("NOTIFY_SOCKET", "INVOCATION_ID", "WATCHDOG_USEC", "WATCHDOG_PID", "XDG_STATE_HOME",
                    "XDG_CONFIG_HOME", "XDG_DATA_HOME", "XDG_CACHE_HOME"):
            env.pop(key, None)
        env.update({
            fake_platform.PORT_ENV: str(listener.getsockname()[1]),
            fake_platform.ALLOW_ALL_ENV: "true",
            # Host rendezvous (one-gateway-per-user lock + record) must never see the live gateway.
            "HERMES_GATEWAY_LOCK_DIR": str(self.root / "locks"),
            "HERMES_GATEWAY_MAX_STARTS": "0",
            # The child's HOME is the tmp root, so the live-DB guard (pytest ancestry) would take
            # its tmp state.db for "production"; the documented child opt-out is safe here.
            "HERMES_STATE_DB_GUARD_BYPASS": "1",
        })
        log = open(self.log_path, "wb")
        # Same process group as pytest (no start_new_session): when the runner kills a timed-out
        # file's group, the gateway goes with it instead of outliving the run.
        self.proc = subprocess.Popen(
            [python_exe(), "-m", "gateway.run"], cwd=str(self.home), env=env,
            stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
        log.close()
        deadline = time.monotonic() + BOOT_DEADLINE_S
        while True:
            try:
                conn, _ = listener.accept()
                break
            except socket.timeout:
                if self.proc.poll() is not None:
                    raise AssertionError(f"gateway exited during boot rc={self.proc.returncode}\n{self.log_tail()}")
                if time.monotonic() >= deadline:
                    raise AssertionError(f"gateway did not connect the fake platform in {BOOT_DEADLINE_S}s\n"
                                         f"{self.log_tail()}")
        conn.settimeout(None)
        self._conn = conn
        threading.Thread(target=self._pump, name=f"chaos-gw-{self.tag}", daemon=True).start()
        self.wait_for(lambda e: e.ev == "ready", BOOT_DEADLINE_S, "adapter ready")

    def _pump(self) -> None:
        assert self._conn is not None
        buf = b""
        while True:
            try:
                chunk = self._conn.recv(65536)
            except OSError:
                chunk = b""
            if not chunk:
                with self._cond:
                    self.events.append(Event("eof", {}, time.monotonic(), len(self.events)))
                    self._cond.notify_all()
                return
            buf += chunk
            while b"\n" in buf:
                line, buf = buf.split(b"\n", 1)
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    continue
                with self._cond:
                    self.events.append(Event(str(data.get("ev")), data, time.monotonic(), len(self.events)))
                    self._cond.notify_all()

    def stop(self) -> None:
        """Real SIGTERM shutdown; records wall time and exit code. Escalates to SIGKILL only
        after SHUTDOWN_DEADLINE_S (the test then fails on ``shutdown_s``/``exit_code``)."""
        if self.proc is None or self.exit_code is not None:
            return
        t0 = time.monotonic()
        if self.proc.poll() is None:
            self.proc.send_signal(signal.SIGTERM)
        try:
            self.exit_code = self.proc.wait(timeout=SHUTDOWN_DEADLINE_S)
            self.shutdown_s = time.monotonic() - t0
        except subprocess.TimeoutExpired:
            self.shutdown_s = None
            kill_tagged(self.tag)
            with _suppress_oserror():
                self.proc.kill()
            self.exit_code = self.proc.wait(timeout=30)
        finally:
            for sock in (self._conn, self._listener):
                if sock is not None:
                    with _suppress_oserror():
                        sock.close()

    # -- driving -----------------------------------------------------------------

    def send_user(self, text: str, chat: str = "chaos-chat") -> str:
        assert self._conn is not None
        self._msg_seq += 1
        msg_id = f"in-{self._msg_seq}"
        self._conn.sendall((json.dumps({"op": "msg", "text": text, "chat": chat, "id": msg_id}) + "\n").encode())
        return msg_id

    def mark(self) -> int:
        with self._cond:
            return len(self.events)

    def wait_for(self, pred: Callable[[Event], bool], timeout: float, what: str,
                 since: int = 0) -> Optional[Event]:
        """First event at index >= ``since`` matching ``pred``; None at the deadline."""
        deadline = time.monotonic() + timeout
        with self._cond:
            idx = since
            while True:
                while idx < len(self.events):
                    ev = self.events[idx]
                    idx += 1
                    if pred(ev):
                        return ev
                left = deadline - time.monotonic()
                if left <= 0:
                    return None
                self._cond.wait(min(left, 0.5))

    def events_between(self, start: int, end: int | None = None) -> list[Event]:
        with self._cond:
            return list(self.events[start:end])

    def sends_since(self, since: int) -> list[str]:
        with self._cond:
            return [e.data.get("text") or "" for e in self.events[since:] if e.ev == "send"]

    def max_heartbeat_gap(self) -> float:
        with self._cond:
            return max((float(e.data.get("max_gap") or 0) for e in self.events if e.ev == "hb"), default=0.0)

    def heartbeat_count(self) -> int:
        with self._cond:
            return sum(1 for e in self.events if e.ev == "hb")

    def log_tail(self, n: int = 80) -> str:
        try:
            lines = self.log_path.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError:
            return "<no gateway log>"
        return "\n".join(lines[-n:])


class _suppress_oserror:
    def __enter__(self) -> None:
        return None

    def __exit__(self, exc_type, *_: object) -> bool:
        return exc_type is not None and issubclass(exc_type, (OSError, ProcessLookupError))
