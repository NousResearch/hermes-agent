"""Lane-private harness for the C7 two-tenant canary suites.

Every tenant (profile) owns a set of random canaries: its provider key (same env var NAME in every
profile, different value), its API-server key, a non-secret ``.env`` marker, its model id, memory
text, SOUL text, terminal cwd and a cron prompt. Each tenant also owns its own loopback provider
(``FakeLLMServer``) that accepts only its own key, so every request the provider records is proof of
WHO sent it. The invariant shared by every scenario is ``leaks(...) == []``: no tenant's canary may
appear in another tenant's provider request, tool-subprocess env snapshot, or on-disk file.

Hermes runs for real in child processes with HOME=<tmp>/home and HERMES_HOME=<tmp>/home/.hermes
(profiles resolve under $HOME, never the real install), every credential env var stripped.
"""

from __future__ import annotations

import json
import os
import secrets
import signal
import socket
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable

import yaml

from tests.fakes.fake_llm_provider import FakeLLMServer, Response, Text, ToolCall

REPO_ROOT = Path(__file__).resolve().parents[4]
ENV_PROBE = "echo SNAPSHOT-BEGIN; env | sort; echo CWD=$(pwd); echo SNAPSHOT-END"
PROVIDER_KEY_ENV = "TENANT_PROVIDER_KEY"  # same NAME in every profile's .env, distinct VALUE

_STRIP_SUFFIXES = ("_API_KEY", "_TOKEN", "_BASE_URL", "_SECRET", "_ACCESS_KEY", "_KEY_ID", "_KEY")
_STRIP_PREFIXES = ("HERMES_", "OPENAI", "ANTHROPIC", "OPENROUTER", "AWS_", "AZURE_", "GOOGLE_", "GEMINI",
                   "PYTEST_", "NOUS_", "XAI_", "LLM_", "CUSTOM_", "TERMINAL_", "TENANT_", "API_SERVER_")


def real_user_home() -> Path:
    import pwd
    return Path(pwd.getpwuid(os.getuid()).pw_dir).resolve()


def hermetic_env(home: Path, extra: dict[str, str] | None = None) -> dict[str, str]:
    """Child env: fake HOME (profile root anchor) + HERMES_HOME under it, nothing credential-shaped."""
    home = home.resolve()
    assert home != real_user_home() and home / ".hermes" != real_user_home() / ".hermes", home
    env = {k: v for k, v in os.environ.items()
           if not (k.endswith(_STRIP_SUFFIXES) or k.startswith(_STRIP_PREFIXES))}
    for var in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy", "XDG_STATE_HOME",
                "XDG_CONFIG_HOME", "XDG_DATA_HOME", "XDG_CACHE_HOME",
                # no route to the developer's systemd --user bus: a child can never see or touch the
                # live hermes-gateway unit, whatever it decides about service management.
                "DBUS_SESSION_BUS_ADDRESS", "XDG_RUNTIME_DIR"):
        env.pop(var, None)
    env.update(
        HOME=str(home),
        HERMES_HOME=str(home / ".hermes"),
        XDG_STATE_HOME=str(home / ".local" / "state"),
        PYTHONPATH=str(REPO_ROOT),
        NO_COLOR="1",
        TERM="dumb",
        NO_PROXY="127.0.0.1,localhost",
        no_proxy="127.0.0.1,localhost",
        # The live-DB guard treats $HOME/.hermes/state.db of a pytest descendant as production;
        # this HOME is the test's own tmp dir (asserted above).
        HERMES_STATE_DB_GUARD_BYPASS="1",
        HERMES_ACCEPT_HOOKS="1",
    )
    env.update(extra or {})
    return env


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def poll(pred: Callable[[], Any], timeout: float, what: str, interval: float = 0.1) -> Any:
    deadline = time.monotonic() + timeout
    while True:
        got = pred()
        if got:
            return got
        if time.monotonic() > deadline:
            raise AssertionError(f"timed out after {timeout}s waiting for {what}")
        time.sleep(interval)


# Tenants -------------------------------------------------------------------------------------------


@dataclass
class Tenant:
    name: str
    home: Path  # the profile's HERMES_HOME
    workdir: Path
    tag: str = field(default_factory=lambda: secrets.token_hex(5))
    srv: FakeLLMServer | None = None
    extra: dict[str, str] = field(default_factory=dict)  # canaries minted mid-scenario (RPC writes)

    # canaries (each embeds the random tag, so a hit can never be a coincidence)
    @property
    def provider_key(self) -> str:
        return f"sk-prov-{self.name}-{self.tag}"

    @property
    def api_server_key(self) -> str:
        return f"apisrv-{self.name}-{self.tag}"

    @property
    def env_marker(self) -> str:
        return f"envmark-{self.name}-{self.tag}"

    @property
    def model(self) -> str:
        return f"model-{self.name}-{self.tag}"

    @property
    def memory(self) -> str:
        return f"memory-canary-{self.name}-{self.tag}"

    @property
    def soul(self) -> str:
        return f"soul-canary-{self.name}-{self.tag}"

    @property
    def shell_state(self) -> str:
        """Exported into the tool shell AFTER each env dump: a shell reused by another tenant shows it."""
        return f"shellstate-{self.name}-{self.tag}"

    @property
    def cron_prompt(self) -> str:
        return f"cron-canary-{self.name}-{self.tag}: run the environment probe"

    def canaries(self) -> dict[str, str]:
        return {
            "provider_key": self.provider_key, "api_server_key": self.api_server_key,
            "env_marker": self.env_marker, "model": self.model, "memory": self.memory, "soul": self.soul,
            "workdir": str(self.workdir), "cron_prompt": self.cron_prompt.split(":")[0],
            "shell_state": self.shell_state, **self.extra,
        }

    def secrets(self) -> dict[str, str]:
        return {"provider_key": self.provider_key, "api_server_key": self.api_server_key}


def _responder(t: Tenant) -> Callable[[dict[str, Any]], Response]:
    """Every user turn runs the env probe in the terminal tool; the follow-up answers with text."""
    def respond(record: dict[str, Any]) -> Response:
        msgs = record["body"].get("messages") or []
        last = msgs[-1] if msgs else {}
        if last.get("role") == "tool":
            return Text(f"probe done for {t.name}")
        return ToolCall("terminal", {"command": f"{ENV_PROBE}; export TENANT_SHELL_STATE={t.shell_state}"})
    return respond


def make_tenants(root: Path, names: Iterable[str], launch: str = "default") -> dict[str, Tenant]:
    """Build tenant homes on disk: launch profile at HOME/.hermes, the rest under profiles/<name>."""
    hermes_home = root / "home" / ".hermes"
    tenants: dict[str, Tenant] = {}
    for name in names:
        home = hermes_home if name == launch else hermes_home / "profiles" / name
        t = Tenant(name=name, home=home, workdir=root / "work")
        t.workdir = root / f"work-{name}-{t.tag}"
        t.workdir.mkdir(parents=True)
        t.srv = FakeLLMServer(_responder(t), api_key=t.provider_key)
        tenants[name] = t
    return tenants


def write_tenant_home(t: Tenant, extra_config: dict[str, Any] | None = None,
                      extra_env: dict[str, str] | None = None) -> None:
    assert t.srv is not None
    t.home.mkdir(parents=True, exist_ok=True)
    cfg: dict[str, Any] = {
        "model": {"provider": "custom", "base_url": t.srv.base_url, "default": t.model,
                  "key_env": PROVIDER_KEY_ENV, "context_length": 128000},
        "agent": {"api_max_retries": 1},
        "terminal": {"backend": "local", "cwd": str(t.workdir)},
        "memory": {"memory_enabled": True},
        "compression": {"enabled": False},
    }
    for k, v in (extra_config or {}).items():
        cfg[k] = {**cfg.get(k, {}), **v} if isinstance(v, dict) and isinstance(cfg.get(k), dict) else v
    (t.home / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    env = {PROVIDER_KEY_ENV: t.provider_key, "API_SERVER_KEY": t.api_server_key, "TENANT_MARKER": t.env_marker,
           **(extra_env or {})}
    (t.home / ".env").write_text("".join(f"{k}={v}\n" for k, v in env.items()), encoding="utf-8")
    (t.home / "memories").mkdir(exist_ok=True)
    (t.home / "memories" / "MEMORY.md").write_text(f"{t.memory}\n", encoding="utf-8")
    (t.home / "SOUL.md").write_text(f"You are {t.soul}.\n", encoding="utf-8")


def assert_profiles_root_under(root: Path, home: Path) -> None:
    """The profile root is HOME-anchored: prove it resolves inside ``root`` before any write."""
    probe = subprocess.run(
        [sys.executable, "-c", "from hermes_cli.profiles import _get_profiles_root as r; print(r())"],
        env=hermetic_env(home), cwd=str(home), capture_output=True, text=True, timeout=120,
        stdin=subprocess.DEVNULL,
    )
    assert probe.returncode == 0, probe.stderr[-2000:]
    got = Path(probe.stdout.strip().splitlines()[-1]).resolve()
    assert str(got).startswith(str(root.resolve())), f"profiles root escaped the sandbox: {got}"


# Invariants ----------------------------------------------------------------------------------------


def _bearer(record: dict[str, Any]) -> str:
    auth = record.get("auth") or ""
    return auth[7:] if auth.lower().startswith("bearer ") else auth


def request_leaks(tenants: dict[str, Tenant]) -> list[str]:
    """Every provider request carries its own tenant's key/model and no other tenant's canary."""
    problems: list[str] = []
    for t in tenants.values():
        assert t.srv is not None
        for i, r in enumerate(list(t.srv.requests)):
            where = f"provider[{t.name}] request #{i} ({r['kind']} {r['path']})"
            if _bearer(r) != t.provider_key:
                owner = next((u.name for u in tenants.values() if u.provider_key == _bearer(r)), "<none>")
                problems.append(f"{where}: Authorization is {owner!r}'s key, not {t.name!r}'s")
            if r["kind"] == "main" and r["body"].get("model") != t.model:
                problems.append(f"{where}: model {r['body'].get('model')!r} != {t.model!r}")
            blob = json.dumps(r["body"]) + json.dumps(r.get("headers") or {})
            problems += [f"{where}: carries {u.name}'s {kind} ({value})"
                         for u in tenants.values() if u is not t
                         for kind, value in u.canaries().items() if value in blob]
    return problems


def env_snapshots(t: Tenant) -> list[tuple[str, str]]:
    """``(origin, output)`` of every ENV_PROBE terminal call as sent back to the tenant's provider;
    origin is ``cron job`` when the conversation is the tenant's cron prompt, else ``turn``."""
    assert t.srv is not None
    seen: dict[str, tuple[str, str]] = {}  # tool_call_id -> (origin, output); results are re-sent later
    cron_mark = t.cron_prompt.split(":")[0]
    for r in list(t.srv.requests):
        origin = "cron job" if cron_mark in json.dumps(r["body"].get("messages") or []) else "turn"
        for m in r["body"].get("messages") or []:
            content = m.get("content")
            text = content if isinstance(content, str) else json.dumps(content)
            if m.get("role") == "tool" and "SNAPSHOT-BEGIN" in text:
                seen.setdefault(str(m.get("tool_call_id") or text), (origin, text))
    return list(seen.values())


def snapshot_problems(tenants: dict[str, Tenant], min_per_tenant: int = 1) -> list[str]:
    """Inside a tool subprocess: cwd is the tenant's own, and no other tenant's canary is visible."""
    problems: list[str] = []
    for t in tenants.values():
        snaps = env_snapshots(t)
        if len(snaps) < min_per_tenant:
            problems.append(f"{t.name}: {len(snaps)} env snapshot(s) reached its provider, expected >= {min_per_tenant}")
        for origin, s in snaps:
            text = s.encode().decode("unicode_escape", errors="ignore") if "\\n" in s else s
            own_dirs = [str(t.workdir), *(v for k, v in t.extra.items() if k.startswith("workdir"))]
            if not any(f"CWD={d}\n" in text + "\n" for d in own_dirs):
                cwd = next((ln for ln in text.splitlines() if ln.startswith("CWD=")), "<no CWD line>")
                problems.append(f"{t.name}: terminal ({origin}) ran in {cwd.strip()!r}, not its own {own_dirs}")
            problems += [f"{t.name}: tool subprocess ({origin}) carries {u.name}'s {kind}: "
                         f"{next((ln.strip() for ln in text.splitlines() if value in ln), value)[:300]!r}"
                         for u in tenants.values() if u is not t
                         for kind, value in u.canaries().items() if value in text]
    return problems


def file_leaks(tenants: dict[str, Tenant], launch: str = "default") -> list[str]:
    """Byte-scan every file of every tenant home (state.db + WAL, logs, sessions, config, .env,
    memories, cron output) for any other tenant's canary. The launch home's ``profiles/`` subtree
    belongs to the other tenants and is scanned as theirs."""
    problems: list[str] = []
    for t in tenants.values():
        foreign = [(u.name, kind, value.encode()) for u in tenants.values() if u is not t
                   for kind, value in u.canaries().items()]
        for path in t.home.rglob("*"):
            if not path.is_file() or (t.name == launch and "profiles" in path.relative_to(t.home).parts[:1]):
                continue
            try:
                data = path.read_bytes()
            except OSError:
                continue
            problems += [f"{t.name}: {path.relative_to(t.home)} contains {owner}'s {kind}"
                         for owner, kind, value in foreign if value in data]
    return problems


def check_isolation(tenants: dict[str, Tenant], *, launch: str = "default", min_snapshots: int = 1,
                    extra: Iterable[str] = ()) -> None:
    problems = [*request_leaks(tenants), *snapshot_problems(tenants, min_snapshots), *file_leaks(tenants, launch),
                *extra]
    assert not problems, "cross-tenant leak(s):\n  " + "\n  ".join(dict.fromkeys(problems))


def text_leaks(label: str, text: str, tenants: dict[str, Tenant], owner: str | None = None) -> list[str]:
    """Canaries of any tenant other than ``owner`` in a log/stream (``owner=None``: secrets only)."""
    out: list[str] = []
    for u in tenants.values():
        if u.name == owner:
            continue
        values = u.canaries() if owner is not None else u.secrets()
        out += [f"{label}: carries {u.name}'s {kind}" for kind, value in values.items() if value in text]
    return out


_SEED_CRON = ("import sys; from cron.jobs import create_job, trigger_job, list_jobs; "
              "jobs = [j for j in list_jobs(include_disabled=True) if j.get('name') == 'tenancy-canary']; "
              "j = jobs[0] if jobs else create_job(prompt=sys.argv[1], schedule='every 1h', name='tenancy-canary', "
              "deliver='local'); trigger_job(j['id']); print(j['id'])")


def seed_due_cron_job(t: Tenant, home: Path) -> str:
    """Create (once) an hourly job in the tenant's own cron store and mark it due now (the state the
    "run on next tick" action produces), so the multiplexed ticker's first tick fires it."""
    r = subprocess.run([sys.executable, "-c", _SEED_CRON, t.cron_prompt], cwd=str(home),
                       env=hermetic_env(home, {"HERMES_HOME": str(t.home)}), capture_output=True, text=True,
                       timeout=120, stdin=subprocess.DEVNULL)
    assert r.returncode == 0, r.stderr[-2000:]
    return r.stdout.strip().splitlines()[-1]


def cron_requests(t: Tenant) -> int:
    """Provider requests driven by the tenant's cron job (its prompt canary is in the user turn)."""
    assert t.srv is not None
    return sum(t.cron_prompt.split(":")[0] in json.dumps(r["body"]) for r in list(t.srv.requests))


# Processes -----------------------------------------------------------------------------------------


def kill_group(proc: subprocess.Popen, sig: int = signal.SIGKILL) -> None:
    try:
        os.killpg(proc.pid, sig)
    except (ProcessLookupError, PermissionError):
        pass


def run_hermes(argv: list[str], home: Path, timeout: float = 120.0,
               extra_env: dict[str, str] | None = None) -> subprocess.CompletedProcess:
    proc = subprocess.Popen(
        [sys.executable, "-m", "hermes_cli.main", *argv], cwd=str(home), env=hermetic_env(home, extra_env),
        stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        start_new_session=True,
    )
    try:
        out, err = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        kill_group(proc)
        out, err = proc.communicate()
        err += f"\n[harness] killed after {timeout}s"
    kill_group(proc)
    return subprocess.CompletedProcess(argv, proc.returncode, out, err)


class TuiBackend:
    """The real stdio JSON-RPC backend the TUI/Desktop drive (``python -m tui_gateway.entry``)."""

    def __init__(self, home: Path, log_path: Path, extra_env: dict[str, str] | None = None) -> None:
        import queue
        import threading

        self._log = open(log_path, "a", encoding="utf-8")  # noqa: SIM115 - closed in close()
        self.proc = subprocess.Popen(
            [sys.executable, "-m", "tui_gateway.entry"], cwd=str(home), env=hermetic_env(home, extra_env),
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self._log, text=True, bufsize=1,
            start_new_session=True,
        )
        self._q: queue.Queue[dict[str, Any]] = queue.Queue()
        self._rid = 0
        self.seen: list[dict[str, Any]] = []
        threading.Thread(target=self._read, daemon=True, name="tenancy-tui-reader").start()

    def _read(self) -> None:
        assert self.proc.stdout is not None
        for line in self.proc.stdout:
            try:
                self._q.put(json.loads(line))
            except json.JSONDecodeError:
                continue

    def wait(self, pred: Callable[[dict[str, Any]], bool], timeout: float = 90.0) -> dict[str, Any]:
        import queue

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
        tail = [(m.get("method"), (m.get("params") or {}).get("type"), m.get("id")) for m in self.seen[-20:]]
        raise AssertionError(f"tui_gateway: condition not met within {timeout}s (rc={self.proc.poll()}); last {tail}")

    def call(self, method: str, params: dict[str, Any], timeout: float = 90.0) -> dict[str, Any]:
        self._rid += 1
        rid = self._rid
        assert self.proc.stdin is not None
        self.proc.stdin.write(json.dumps({"jsonrpc": "2.0", "id": rid, "method": method, "params": params}) + "\n")
        self.proc.stdin.flush()
        return self.wait(lambda m: m.get("id") == rid and "method" not in m, timeout)

    def ok(self, method: str, params: dict[str, Any], timeout: float = 90.0) -> dict[str, Any]:
        reply = self.call(method, params, timeout)
        assert "result" in reply, f"{method} failed: {reply.get('error')}"
        return reply["result"]

    def turn(self, sid: str, text: str, timeout: float = 90.0) -> dict[str, Any]:
        """Submit one prompt and wait for that session's terminal turn event."""
        self.ok("prompt.submit", {"session_id": sid, "text": text})

        def done(m: dict[str, Any]) -> bool:
            p = m.get("params") or {}
            return (m.get("method") == "event" and p.get("session_id") == sid
                    and p.get("type") in {"message.complete", "error"})
        ev = self.wait(done, timeout)
        assert ev["params"]["type"] == "message.complete", f"turn failed: {ev}"
        return ev

    def close(self) -> None:
        try:
            if self.proc.stdin:
                self.proc.stdin.close()
            self.proc.wait(timeout=60)
        except (subprocess.TimeoutExpired, OSError, ValueError):
            pass
        kill_group(self.proc)
        try:
            self.proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            pass
        self._log.close()


class ServeBackend(TuiBackend):
    """The real Desktop backend: ``hermes serve --port 0`` with the JSON-RPC surface on ``/api/ws``.

    ``HERMES_DESKTOP=1`` marks it app-spawned, which is also what runs the cron ticker in-process.
    """

    def __init__(self, home: Path, log_path: Path, extra_env: dict[str, str] | None = None) -> None:
        import queue
        import re
        import threading

        from websockets.sync.client import connect  # ``websockets`` is a core dependency

        self.token = secrets.token_urlsafe(24)
        self._log = open(log_path, "a", encoding="utf-8")  # noqa: SIM115 - closed in close()
        env = {"HERMES_DASHBOARD_SESSION_TOKEN": self.token, "HERMES_DESKTOP": "1", **(extra_env or {})}
        self.proc = subprocess.Popen(
            [sys.executable, "-m", "hermes_cli.main", "serve", "--host", "127.0.0.1", "--port", "0"],
            cwd=str(home), env=hermetic_env(home, env), stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, text=True, bufsize=1, start_new_session=True,
        )
        port_box: list[int] = []

        def pump() -> None:
            assert self.proc.stdout is not None
            for line in self.proc.stdout:
                self._log.write(line)
                self._log.flush()
                m = re.search(r"backend listening on [\d.]+:(\d+)", line)
                if m and not port_box:
                    port_box.append(int(m.group(1)))
        threading.Thread(target=pump, daemon=True, name="tenancy-serve-stdout").start()
        self.port = poll(lambda: port_box[0] if port_box else (self.proc.poll() is not None and -1), 120,
                         "hermes serve to report its port")
        assert self.port > 0, f"hermes serve exited rc={self.proc.returncode}"
        self.ws = connect(f"ws://127.0.0.1:{self.port}/api/ws?token={self.token}", open_timeout=90, max_size=None)
        self._q: queue.Queue[dict[str, Any]] = queue.Queue()
        self._rid = 0
        self.seen: list[dict[str, Any]] = []
        threading.Thread(target=self._read, daemon=True, name="tenancy-ws-reader").start()

    def _read(self) -> None:
        while True:
            try:
                raw = self.ws.recv()
            except Exception:
                return
            if not raw:
                return
            try:
                self._q.put(json.loads(raw))
            except json.JSONDecodeError:
                continue

    def call(self, method: str, params: dict[str, Any], timeout: float = 90.0) -> dict[str, Any]:
        self._rid += 1
        rid = self._rid
        self.ws.send(json.dumps({"jsonrpc": "2.0", "id": rid, "method": method, "params": params}))
        return self.wait(lambda m: m.get("id") == rid and "method" not in m, timeout)

    def close(self) -> None:
        try:
            self.ws.close()
        except Exception:
            pass
        kill_group(self.proc, signal.SIGTERM)
        try:
            self.proc.wait(timeout=60)
        except subprocess.TimeoutExpired:
            kill_group(self.proc)
            self.proc.wait(timeout=10)
        kill_group(self.proc)
        self._log.close()


def pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    stat = Path(f"/proc/{pid}/stat")
    return not (stat.exists() and stat.read_text().split(") ", 1)[-1].startswith("Z"))
