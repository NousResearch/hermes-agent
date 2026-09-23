"""Lane-private harness for the C11 routing truth table.

A ``Fleet`` is K real loopback provider hosts (``FakeLLMServer``), one per provider
identity, each accepting only its own randomly minted key. ``check_routing`` is the one
invariant every scenario shares: a request on a host the scenario did not select, or a
selected host receiving any bearer other than its own, is a credential leak.

Hermes itself runs for real in child processes (``hermes chat -q`` / ``hermes -z`` /
``python -m tui_gateway.entry``) with HOME and HERMES_HOME inside the test's tmp dir,
every credential/endpoint env var stripped, and only these fake hosts configured.
"""

from __future__ import annotations

import json
import os
import pwd
import queue
import secrets
import signal
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable, Iterable

import yaml

from tests.fakes.fake_llm_provider import Error, FakeLLMServer, Text

REPO_ROOT = Path(__file__).resolve().parents[4]

# Env that could route or authenticate a child Hermes outside the fake fleet.
_STRIP_SUFFIXES = ("_API_KEY", "_TOKEN", "_BASE_URL", "_SECRET", "_ACCESS_KEY", "_KEY_ID")
_STRIP_PREFIXES = ("HERMES_", "OPENAI", "ANTHROPIC", "OPENROUTER", "AWS_", "AZURE_", "GOOGLE_",
                   "GEMINI", "PYTEST_", "NOUS_", "XAI_", "LLM_", "CUSTOM_")


def hermetic_env(home: Path, hermes_home: Path | None = None, proxy: str | None = None) -> dict[str, str]:
    real_home = Path(pwd.getpwuid(os.getuid()).pw_dir).resolve()
    fake = home.resolve()
    # The state-db guard is bypassed below, so prove first that this child can never reach the
    # real install: its HOME (and therefore every profile root) lives outside the real home.
    assert fake != real_home and real_home / ".hermes" not in (fake, *fake.parents), fake
    env = {k: v for k, v in os.environ.items()
           if not (k.endswith(_STRIP_SUFFIXES) or k.startswith(_STRIP_PREFIXES))}
    env.update(
        HOME=str(home),
        HERMES_HOME=str(hermes_home or home / ".hermes"),
        PYTHONPATH=str(REPO_ROOT),
        NO_COLOR="1",
        TERM="dumb",
        # A pytest-descendant child treats $HOME/.hermes/state.db as production; that HOME is ours.
        HERMES_STATE_DB_GUARD_BYPASS="1",
    )
    for var in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "NO_PROXY", "http_proxy", "https_proxy", "all_proxy", "no_proxy"):
        env.pop(var, None)
    if proxy:
        env.update(HTTP_PROXY=proxy, HTTPS_PROXY=proxy, ALL_PROXY=proxy, http_proxy=proxy, https_proxy=proxy,
                   all_proxy=proxy, NO_PROXY="127.0.0.1,localhost", no_proxy="127.0.0.1,localhost")
    return env


# Fleet --------------------------------------------------------------------------


def _bearer(record: dict[str, Any]) -> str:
    auth = record.get("auth") or ""
    if auth.lower().startswith("bearer "):
        return auth[7:]
    return (record.get("headers") or {}).get("x-api-key", "") or auth


@dataclass
class Fleet:
    """One FakeLLMServer per provider identity; every identity owns distinct secrets."""

    roles: Iterable[str]
    key_counts: dict[str, int] = field(default_factory=dict)
    servers: dict[str, FakeLLMServer] = field(init=False, default_factory=dict)
    keys: dict[str, tuple[str, ...]] = field(init=False, default_factory=dict)

    def __post_init__(self) -> None:
        for role in self.roles:
            n = self.key_counts.get(role, 1)
            self.keys[role] = tuple(f"sk-c11-{role}-{i}-{secrets.token_hex(6)}" for i in range(n))
            srv = FakeLLMServer(self._dispatch(role), api_key=self.keys[role],
                                aux=self._aux(role), record_get=True)
            self.servers[role] = srv
        self._scripts: dict[str, Callable[[dict[str, Any]], Any]] = {}

    # scripting: a callable(record) -> Response per role; default answers name the host
    def script(self, role: str, responder: Callable[[dict[str, Any]], Any]) -> None:
        self._scripts[role] = responder

    def _dispatch(self, role: str) -> Callable[[dict[str, Any]], Any]:
        def respond(record: dict[str, Any]) -> Any:
            fn = self._scripts.get(role)
            resp = fn(record) if fn else None
            return resp if resp is not None else Text(f"answer-from-{role}")
        return respond

    def _aux(self, role: str) -> Callable[[dict[str, Any]], Any]:
        def respond(record: dict[str, Any]) -> Any:
            fn = self._scripts.get(f"{role}:aux")
            resp = fn(record) if fn else None
            return resp if resp is not None else Text(f"aux-from-{role}")
        return respond

    def key(self, role: str, i: int = 0) -> str:
        return self.keys[role][i]

    def url(self, role: str) -> str:
        return self.servers[role].base_url

    def start(self) -> "Fleet":
        for srv in self.servers.values():
            srv.start()
        return self

    def stop(self) -> None:
        for srv in self.servers.values():
            srv.stop()

    def marks(self) -> dict[str, int]:
        return {role: len(srv.requests) for role, srv in self.servers.items()}

    def since(self, marks: dict[str, int] | None = None) -> dict[str, list[dict[str, Any]]]:
        marks = marks or {}
        return {role: list(srv.requests[marks.get(role, 0):]) for role, srv in self.servers.items()}

    def owner_of(self, secret: str) -> str:
        for role, keys in self.keys.items():
            if secret in keys:
                return role
        return "<no identity>"


def check_routing(
    fleet: Fleet,
    log: dict[str, list[dict[str, Any]]],
    expect: dict[str, Iterable[str]],
    must_hit: dict[str, int] | None = None,
) -> list[str]:
    """Return every routing violation in ``log`` (empty list == correct routing).

    ``expect`` maps each host the scenario may reach to the secrets it may carry there;
    ``must_hit`` maps hosts to the minimum number of chat POSTs they must have served.
    """
    allowed = {role: set(keys) for role, keys in expect.items()}
    problems: list[str] = []
    for role, records in log.items():
        for r in records:
            secret = _bearer(r)
            where = f"{r.get('kind')} {r.get('path')}"
            if r.get("kind") == "get":
                # Model-list/metadata probes carry no prompt: they may reach any configured host,
                # but never with another identity's key.
                owner = fleet.owner_of(secret)
                if secret and owner not in {role, "<no identity>"}:
                    problems.append(f"LEAK: host '{role}' got {where} carrying the key of '{owner}' ({secret!r})")
                continue
            if role not in allowed:
                problems.append(f"LEAK: unselected host '{role}' got {where} carrying the key of "
                                f"'{fleet.owner_of(secret)}' ({secret!r})")
            elif secret not in allowed[role]:
                problems.append(f"LEAK/WRONG KEY: host '{role}' got {where} carrying the key of "
                                f"'{fleet.owner_of(secret)}' ({secret!r})")
    for role, n in (must_hit or {}).items():
        got = sum(1 for r in log.get(role, []) if r.get("kind") in {"main", "aux"})
        if got < n:
            problems.append(f"MISSING: host '{role}' served {got} chat request(s), expected >= {n}")
    return problems


def describe(log: dict[str, list[dict[str, Any]]]) -> str:
    rows = [f"  {role}: " + ", ".join(f"{r.get('kind')}[{_bearer(r)[:22]}]" for r in recs)
            for role, recs in log.items() if recs]
    return "\n".join(rows) or "  (no requests anywhere)"


# Egress trap ------------------------------------------------------------------------


class EgressTrap:
    """HTTP(S)_PROXY for the child: records every non-loopback destination and refuses it.

    The fleet is reached directly (NO_PROXY=loopback), so anything arriving here is traffic to a
    host the scenario never configured -- e.g. a fall-through to the real OpenRouter/OpenAI API.
    """

    def __init__(self) -> None:
        self.attempts: list[tuple[str, str]] = []
        trap = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *_a: object) -> None:
                pass

            def _refuse(self) -> None:
                target = self.path if self.command != "CONNECT" else f"https://{self.path}"
                trap.attempts.append((self.command, target))
                self.send_response(403)
                self.send_header("Content-Length", "0")
                self.send_header("Connection", "close")
                self.end_headers()
                self.close_connection = True

            do_CONNECT = do_GET = do_POST = do_PUT = do_HEAD = _refuse  # noqa: N815

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self._server.daemon_threads = True
        threading.Thread(target=self._server.serve_forever, daemon=True, name="c11-egress").start()

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self._server.server_address[1]}"

    def stop(self) -> None:
        self._server.shutdown()
        self._server.server_close()


def inference_hosts() -> frozenset[str]:
    """Hostnames of every inference API Hermes itself knows (its provider registry + OpenRouter).

    Read from the product, never copied: a CONNECT to one of these from a scenario that
    configured only loopback hosts is a prompt leaving for a real provider.
    """
    from urllib.parse import urlparse

    from hermes_cli.auth import PROVIDER_REGISTRY
    from hermes_constants import OPENROUTER_BASE_URL

    urls = [getattr(p, "inference_base_url", "") or "" for p in PROVIDER_REGISTRY.values()] + [OPENROUTER_BASE_URL]
    return frozenset(h for h in (urlparse(u).hostname for u in urls) if h and h not in {"127.0.0.1", "localhost"})


# HERMES_HOME writing ------------------------------------------------------------


def write_home(hermes_home: Path, config: dict[str, Any], env: dict[str, str],
               auth: dict[str, Any] | None = None) -> Path:
    hermes_home.mkdir(parents=True, exist_ok=True)
    (hermes_home / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    (hermes_home / ".env").write_text("".join(f"{k}={v}\n" for k, v in env.items()), encoding="utf-8")
    if auth is not None:
        (hermes_home / "auth.json").write_text(json.dumps(auth), encoding="utf-8")
    return hermes_home


def pool_auth(provider: str, keys: Iterable[str]) -> dict[str, Any]:
    return {"version": 1, "providers": {}, "credential_pool": {provider: [
        {"id": f"k{i}", "label": f"key-{i}", "auth_type": "api_key", "priority": i,
         "source": "manual", "access_token": key}
        for i, key in enumerate(keys)
    ]}}


# Child Hermes processes ---------------------------------------------------------


@dataclass
class RunResult:
    argv: list[str]
    rc: int | None
    stdout: str
    stderr: str
    seconds: float


def run_hermes(argv: list[str], home: Path, *, hermes_home: Path | None = None, proxy: str | None = None,
               timeout: float = 240.0) -> RunResult:
    """Run the real ``hermes`` entry point in its own process group; kill only that group."""
    start = time.monotonic()
    proc = subprocess.Popen(
        [sys.executable, "-m", "hermes_cli.main", *argv],
        cwd=str(home), env=hermetic_env(home, hermes_home, proxy), stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, start_new_session=True,
    )
    try:
        out, err = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        _kill_group(proc)
        out, err = proc.communicate()
        err += f"\n[c11 harness] killed after {timeout}s"
    _kill_group(proc)  # reap any grandchild left in our own session
    return RunResult(argv, proc.returncode, out, err, time.monotonic() - start)


def _kill_group(proc: subprocess.Popen) -> None:
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        pass


class TuiGateway:
    """The real stdio JSON-RPC backend (``python -m tui_gateway.entry``) the TUI/Desktop drive."""

    def __init__(self, home: Path, proxy: str | None = None) -> None:
        self.home = home
        self._stderr = open(home / "tui_gateway.stderr.log", "w", encoding="utf-8")  # noqa: SIM115
        self.proc = subprocess.Popen(
            [sys.executable, "-m", "tui_gateway.entry"], cwd=str(home), env=hermetic_env(home, proxy=proxy),
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self._stderr, text=True, bufsize=1,
            start_new_session=True,
        )
        self._q: queue.Queue[dict[str, Any]] = queue.Queue()
        self._rid = 0
        self.seen: list[dict[str, Any]] = []
        threading.Thread(target=self._read, daemon=True, name="c11-tui-reader").start()

    def _read(self) -> None:
        assert self.proc.stdout is not None
        for line in self.proc.stdout:
            try:
                self._q.put(json.loads(line))
            except json.JSONDecodeError:
                continue

    def wait(self, pred: Callable[[dict[str, Any]], bool], timeout: float = 120.0) -> dict[str, Any]:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self.proc.poll() is not None and self._q.empty():
                break
            try:
                msg = self._q.get(timeout=0.25)
            except queue.Empty:
                continue
            self.seen.append(msg)
            if pred(msg):
                return msg
        tail = [(m.get("method"), (m.get("params") or {}).get("type"), m.get("id")) for m in self.seen[-25:]]
        raise AssertionError(f"tui_gateway: condition not met within {timeout}s (rc={self.proc.poll()}); "
                             f"last frames {tail}")

    def seen_or_wait(self, pred: Callable[[dict[str, Any]], bool], timeout: float = 120.0) -> dict[str, Any]:
        """Like ``wait`` but also satisfied by a frame already consumed (events may race a turn's end)."""
        return next((m for m in self.seen if pred(m)), None) or self.wait(pred, timeout)

    @staticmethod
    def event(kind: str, sid: str | None = None) -> Callable[[dict[str, Any]], bool]:
        def pred(m: dict[str, Any]) -> bool:
            p = m.get("params") or {}
            return m.get("method") == "event" and p.get("type") == kind and (sid is None or p.get("session_id") == sid)
        return pred

    def call(self, method: str, params: dict[str, Any], timeout: float = 120.0) -> dict[str, Any]:
        self._rid += 1
        rid = self._rid
        assert self.proc.stdin is not None
        self.proc.stdin.write(json.dumps({"jsonrpc": "2.0", "id": rid, "method": method, "params": params}) + "\n")
        self.proc.stdin.flush()
        return self.wait(lambda m: m.get("id") == rid and "method" not in m, timeout)

    def turn(self, sid: str, text: str, timeout: float = 120.0) -> dict[str, Any]:
        """Submit one prompt and wait for the turn's terminal event (complete or error)."""
        reply = self.call("prompt.submit", {"session_id": sid, "text": text})
        assert "result" in reply, f"prompt.submit refused: {reply}"
        return self.wait(lambda m: self.event("message.complete", sid)(m) or self.event("error", sid)(m), timeout)

    def close(self) -> None:
        try:
            if self.proc.stdin:
                self.proc.stdin.close()
            self.proc.wait(timeout=30)
        except (subprocess.TimeoutExpired, OSError, ValueError):
            pass
        _kill_group(self.proc)
        try:
            self.proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            pass
        self._stderr.close()


def always(resp_factory: Callable[[], Any]) -> Callable[[dict[str, Any]], Any]:
    return lambda _record: resp_factory()


def reject_key(bad_key: str, status: int) -> Callable[[dict[str, Any]], Any]:
    """Answer ``status`` for requests carrying ``bad_key`` (a pooled key that is rate limited)."""
    return lambda record: Error(status, "scripted pool-key failure") if _bearer(record) == bad_key else None
