"""Harness for the messaging-adapter contract suite.

``GatewayUnderTest`` runs the REAL ``hermes gateway run`` (``python -m hermes_cli.main gateway run``)
in a child process on a throwaway HOME/HERMES_HOME, with the real platform adapter plugin loaded and
its SDK pointed at a local stand-in platform server (``tests/fakes/platforms``). The model is
``tests/fakes/fake_llm_provider.FakeLLMServer`` driven by a ``Director`` that answers per inbound
token. Nothing on our side of the platform boundary is mocked.
"""

from __future__ import annotations

import json
import os
import re
import signal
import sqlite3
import subprocess
import sys
import threading
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import hermes_yaml as yaml

from tests.fakes.fake_llm_provider import Text, write_hermes_home
from tests.fakes.platforms._standin import wait_until

REPO_ROOT = Path(__file__).resolve().parents[4]

_STRIP_SUFFIXES = ("_API_KEY", "_TOKEN", "_SECRET", "_PASSWORD", "_KEY")
_STRIP_PREFIXES = ("HERMES_", "TELEGRAM_", "DISCORD_", "SLACK_", "OPENAI_", "ANTHROPIC_", "OPENROUTER_",
                   "GATEWAY_", "NOUS_")
_TOKEN_RE = re.compile(r"\[in:([A-Za-z0-9_.-]+)\]")


def real_user_home() -> Path:
    import pwd

    return Path(pwd.getpwuid(os.getuid()).pw_dir).resolve()


def hermetic_env(home: Path, extra: Optional[Dict[str, str]] = None) -> Dict[str, str]:
    home = home.resolve()
    assert home != real_user_home(), home
    env = {k: v for k, v in os.environ.items()
           if not (k.endswith(_STRIP_SUFFIXES) or k.startswith(_STRIP_PREFIXES))}
    for var in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy",
                "XDG_STATE_HOME", "XDG_CONFIG_HOME", "XDG_DATA_HOME", "XDG_CACHE_HOME",
                "DBUS_SESSION_BUS_ADDRESS", "XDG_RUNTIME_DIR", "NOTIFY_SOCKET", "INVOCATION_ID",
                "PYTEST_CURRENT_TEST"):
        env.pop(var, None)
    extra = dict(extra or {})
    shim = extra.pop("PYTHONPATH_PREPEND", "")
    env.update(
        HOME=str(home), HERMES_HOME=str(home / ".hermes"), XDG_STATE_HOME=str(home / ".local" / "state"),
        # a stand-in's sitecustomize shim (if any) first, then the checkout under test
        PYTHONPATH=os.pathsep.join(p for p in (shim, str(REPO_ROOT)) if p),
        NO_COLOR="1", TERM="dumb", NO_PROXY="127.0.0.1,localhost", no_proxy="127.0.0.1,localhost",
        HERMES_STATE_DB_GUARD_BYPASS="1", HERMES_DISABLE_LAZY_INSTALLS="1", TZ="UTC", PYTHONUNBUFFERED="1",
        TIRITH_ENABLED="false", AWS_EC2_METADATA_DISABLED="true",
    )
    env.update(extra or {})
    return env


class Director:
    """Scripts the fake model per inbound ``[in:<token>]`` and counts model turns per token."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.scripts: Dict[str, List[Any]] = {}
        self.turns: Dict[str, int] = {}
        self.requests: Dict[str, List[dict]] = {}

    def script(self, token: str, *responses: Any) -> None:
        with self._lock:
            self.scripts[token] = list(responses)

    @staticmethod
    def _text(m: dict) -> str:
        c = m.get("content")
        if isinstance(c, list):
            return " ".join(p.get("text", "") for p in c if isinstance(p, dict))
        return c or ""

    def __call__(self, record: dict) -> Any:
        body = record["body"]
        users = [m for m in body.get("messages", []) if m.get("role") == "user"]
        last = self._text(users[-1]) if users else ""
        tokens = _TOKEN_RE.findall(last)
        continuation = body["messages"][-1].get("role") == "tool"
        token = tokens[-1] if tokens else "none"
        with self._lock:
            self.requests.setdefault(token, []).append(body)
            if not continuation:
                self.turns[token] = self.turns.get(token, 0) + 1
            queue = self.scripts.get(token)
            if queue:
                return queue.pop(0)
        return Text(f"reply to {token}")


class GatewayUnderTest:
    """A real ``hermes gateway run`` child on its own fake HOME; SIGTERM stop + restart on the same state."""

    def __init__(self, root: Path, *, llm_base_url: str, config: Dict[str, Any], env: Dict[str, str],
                 ready: Callable[[], bool]) -> None:
        self.root = root
        self.home = root / "home"
        self.hermes_home = self.home / ".hermes"
        self.log_path = root / "gateway.log"
        self._env = dict(env)
        self._ready = ready
        # approvals.mode manual: a dangerous command waits for a human click (``smart`` would ask the
        # fake model to judge it).
        base = {"updates": {"check": False},
                "approvals": {"mode": "manual", "destructive_slash_confirm": False}}
        cfg_path = write_hermes_home(self.hermes_home, llm_base_url) / "config.yaml"
        merged = _deep_merge(_deep_merge(yaml.safe_load(cfg_path.read_text()), base), config)
        cfg_path.write_text(yaml.safe_dump(merged, sort_keys=False), encoding="utf-8")
        self.proc: Optional[subprocess.Popen] = None
        self.pids: List[int] = []

    @property
    def db_path(self) -> Path:
        return self.hermes_home / "state.db"

    def start(self, timeout: float = 60.0) -> "GatewayUnderTest":
        assert self.proc is None or self.proc.poll() is not None
        log = open(self.log_path, "a", encoding="utf-8")  # noqa: SIM115 - handed to the child
        self.proc = subprocess.Popen(
            [sys.executable, "-m", "hermes_cli.main", "gateway", "run"], cwd=str(self.home),
            env=hermetic_env(self.home, dict(self._env)), stdin=subprocess.DEVNULL, stdout=log,
            stderr=subprocess.STDOUT, start_new_session=True)
        log.close()
        self.pids.append(self.proc.pid)
        wait_until(lambda: self._ready() or self.proc.poll() is not None, "adapter connected to the stand-in",
                   timeout=timeout, on_timeout=self.tail)
        assert self.proc.poll() is None, f"gateway exited rc={self.proc.returncode}\n{self.tail()}"
        return self

    def stop(self, timeout: float = 60.0) -> Optional[int]:
        if self.proc is None:
            return None
        proc = self.proc
        if proc.poll() is None:
            try:
                os.killpg(proc.pid, signal.SIGTERM)
                proc.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait(timeout=15)
            except ProcessLookupError:
                pass
        try:
            os.killpg(proc.pid, signal.SIGKILL)  # stragglers in the session
        except (ProcessLookupError, PermissionError):
            pass
        return proc.returncode

    def run_cli(self, *argv: str, timeout: float = 120.0) -> subprocess.CompletedProcess:
        return subprocess.run([sys.executable, "-m", "hermes_cli.main", *argv], cwd=str(self.home),
                              env=hermetic_env(self.home, dict(self._env)), stdin=subprocess.DEVNULL,
                              capture_output=True, text=True, timeout=timeout)

    def alive(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def tail(self, n: int = 6000) -> str:
        out = []
        # the child's stdout/stderr, then its INFO-level log (where turn/delivery progress lands)
        for label, path in (("gateway stdout", self.log_path),
                            ("gateway.log (INFO)", self.hermes_home / "logs" / "gateway.log")):
            try:
                out.append(f"--- {label} tail\n" + path.read_text(errors="replace")[-n:])
            except OSError:
                out.append(f"--- (no {label})")
        return "\n".join(out)

    def active_agents(self) -> Optional[int]:
        """In-flight turns as the gateway persists them to ``gateway_state.json`` at every turn boundary."""
        try:
            return int(json.loads((self.hermes_home / "gateway_state.json").read_text()).get("active_agents"))
        except (OSError, ValueError, TypeError):
            return None

    def wait_idle(self, timeout: float = 30.0) -> None:
        """No agent run is in flight (``active_agents == 0`` in gateway_state.json).

        NOT delivery or adapter quiescence: the count drops when the agent run returns, before the
        adapter sends the normal final reply and closes the turn, and a same-chat inbound in that
        window can be dropped (#121393). Ordering after a turn's delivery comes from a same-chat
        barrier (``_contract.barrier``); this only keeps a new scenario from racing the last one."""
        wait_until(lambda: self.active_agents() == 0 or not self.alive(), "gateway idle (active_agents == 0)",
                   timeout=timeout, on_timeout=self.tail)

    def grep(self, pattern: str, limit: int = 30) -> str:
        """Gateway log lines matching ``pattern`` (case-insensitive), for assertion context."""
        try:
            lines = self.log_path.read_text(errors="replace").splitlines()
        except OSError:
            return ""
        rx = re.compile(pattern, re.I)
        return "\n".join([ln for ln in lines if rx.search(ln)][-limit:])

    def user_rows(self, needle: str) -> List[str]:
        if not self.db_path.exists():
            return []
        conn = sqlite3.connect(f"file:{self.db_path}?mode=ro", uri=True, timeout=30)
        try:
            return [r[0] for r in conn.execute(
                "SELECT content FROM messages WHERE role='user' AND content LIKE ?", (f"%{needle}%",))]
        finally:
            conn.close()


def _deep_merge(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(a)
    for k, v in b.items():
        out[k] = _deep_merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out
