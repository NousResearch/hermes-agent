"""Lane-private harness for the dashboard (``hermes dashboard``) E2E suite.

Every scenario drives the REAL web server as a child process: ``python -m hermes_cli.main dashboard
--no-open --skip-build --port 0`` with HOME=<tmp>/home and HERMES_HOME=<tmp>/home/.hermes (profiles
resolve under $HOME, never the real install), every credential env var stripped, the LLM vendor
replaced by ``tests.fakes.fake_llm_provider.FakeLLMServer``. The client side speaks only what the SPA
speaks: it reads the session token out of the served ``index.html`` (the browser's bootstrap), sends
it as ``X-Hermes-Session-Token`` on REST and ``?token=`` on WebSocket upgrades, and appends
``?profile=<name>`` exactly like the profile switcher in ``web/src/lib/api.ts``.
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
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import httpx
import yaml

from tests.fakes.fake_llm_provider import FakeLLMServer

REPO_ROOT = Path(__file__).resolve().parents[4]
TOKEN_HEADER = "X-Hermes-Session-Token"
PROVIDER_KEY_ENV = "DASH_PROVIDER_KEY"  # same NAME in every profile's .env, distinct VALUE

_STRIP_SUFFIXES = ("_API_KEY", "_TOKEN", "_BASE_URL", "_SECRET", "_ACCESS_KEY", "_KEY_ID", "_KEY")
_STRIP_PREFIXES = ("HERMES_", "OPENAI", "ANTHROPIC", "OPENROUTER", "AWS_", "AZURE_", "GOOGLE_", "GEMINI",
                   "PYTEST_", "NOUS_", "XAI_", "LLM_", "CUSTOM_", "TERMINAL_", "DASH_")
_TOKEN_RE = re.compile(r'__HERMES_SESSION_TOKEN__\s*=\s*"([^"]+)"')
_READY_RE = re.compile(r"HERMES_DASHBOARD_READY port=(\d+)")


def real_user_home() -> Path:
    import pwd
    return Path(pwd.getpwuid(os.getuid()).pw_dir).resolve()


def poll(pred: Callable[[], Any], timeout: float, what: str, interval: float = 0.05) -> Any:
    deadline = time.monotonic() + timeout
    while True:
        got = pred()
        if got:
            return got
        if time.monotonic() > deadline:
            raise AssertionError(f"timed out after {timeout}s waiting for {what}")
        time.sleep(interval)


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def hermetic_env(home: Path, extra: dict[str, str] | None = None) -> dict[str, str]:
    """Child env: fake HOME (profile-root anchor) + HERMES_HOME under it, nothing credential-shaped."""
    home = home.resolve()
    assert home != real_user_home() and not str(home).startswith(str(real_user_home() / ".hermes" / "profiles")), home
    env = {k: v for k, v in os.environ.items()
           if not (k.endswith(_STRIP_SUFFIXES) or k.startswith(_STRIP_PREFIXES))}
    for var in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy",
                "XDG_STATE_HOME", "XDG_CONFIG_HOME", "XDG_DATA_HOME", "XDG_CACHE_HOME",
                "DBUS_SESSION_BUS_ADDRESS", "XDG_RUNTIME_DIR"):
        env.pop(var, None)
    env.update(
        HOME=str(home),
        HERMES_HOME=str(home / ".hermes"),
        XDG_STATE_HOME=str(home / ".local" / "state"),
        PYTHONPATH=str(REPO_ROOT),
        NO_COLOR="1",
        NO_PROXY="127.0.0.1,localhost",
        no_proxy="127.0.0.1,localhost",
        # The live-DB guard treats $HOME/.hermes/state.db of a pytest descendant as production;
        # this HOME is the test's own tmp dir (asserted above).
        HERMES_STATE_DB_GUARD_BYPASS="1",
    )
    env.update(extra or {})
    return env


# Profiles ------------------------------------------------------------------------------------------


@dataclass
class Profile:
    """One profile home with its own provider and random canaries (a hit is never a coincidence)."""

    name: str
    home: Path  # the profile's HERMES_HOME
    tag: str = field(default_factory=lambda: secrets.token_hex(5))
    srv: FakeLLMServer | None = None

    @property
    def provider_key(self) -> str:
        return f"sk-dash-{self.name}-{self.tag}"

    @property
    def model(self) -> str:
        return f"model-{self.name}-{self.tag}"

    @property
    def marker(self) -> str:
        return f"cfgmark-{self.name}-{self.tag}"

    @property
    def db(self) -> Path:
        return self.home / "state.db"

    def canaries(self) -> dict[str, str]:
        return {"provider_key": self.provider_key, "model": self.model, "marker": self.marker}

    def config(self) -> dict[str, Any]:
        return yaml.safe_load((self.home / "config.yaml").read_text(encoding="utf-8")) or {}


@dataclass
class Sandbox:
    root: Path
    profiles: dict[str, Profile]

    @property
    def home(self) -> Path:
        return self.root / "home"

    @property
    def hermes_home(self) -> Path:
        return self.home / ".hermes"

    def env(self, extra: dict[str, str] | None = None) -> dict[str, str]:
        return hermetic_env(self.home, extra)

    def stop(self) -> None:
        for p in self.profiles.values():
            if p.srv is not None:
                p.srv.stop()


def write_profile_home(p: Profile, extra_config: dict[str, Any] | None = None) -> None:
    assert p.srv is not None
    p.home.mkdir(parents=True, exist_ok=True)
    cfg: dict[str, Any] = {
        "model": {"provider": "custom", "base_url": p.srv.base_url, "default": p.model,
                  "key_env": PROVIDER_KEY_ENV, "context_length": 128000},
        "agent": {"api_max_retries": 1},
        "compression": {"enabled": False},
        "updates": {"check": False},
        "dash_canary": {"marker": p.marker},
    }
    cfg.update(extra_config or {})
    (p.home / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    (p.home / ".env").write_text(f"{PROVIDER_KEY_ENV}={p.provider_key}\n", encoding="utf-8")


def make_sandbox(root: Path, names: tuple[str, ...] = ("default",),
                 responder: Callable[[Profile], Any] | None = None) -> Sandbox:
    """Launch profile at HOME/.hermes, the rest under profiles/<name>; each owns a started provider
    that accepts only its own key, so every recorded request proves WHO sent it."""
    hermes_home = root / "home" / ".hermes"
    profiles: dict[str, Profile] = {}
    for name in names:
        home = hermes_home if name == "default" else hermes_home / "profiles" / name
        p = Profile(name=name, home=home)
        script = responder(p) if responder else None
        p.srv = FakeLLMServer(script, default_text=f"reply-from-{p.name}-{p.tag}", api_key=p.provider_key)
        p.srv.start()
        write_profile_home(p)
        profiles[name] = p
    sb = Sandbox(root=root, profiles=profiles)
    _assert_profiles_root_under(sb)
    (root / "web_dist").mkdir(exist_ok=True)
    (root / "web_dist" / "index.html").write_text(
        "<!doctype html><html><head><title>hermes</title></head><body><div id=root></div></body></html>",
        encoding="utf-8")
    return sb


def _assert_profiles_root_under(sb: Sandbox) -> None:
    """The profile root is HOME-anchored: prove it resolves inside the sandbox before any write."""
    probe = subprocess.run(
        [sys.executable, "-c", "from hermes_cli.profiles import _get_profiles_root as r; print(r())"],
        env=sb.env(), cwd=str(sb.home), capture_output=True, text=True, timeout=120, stdin=subprocess.DEVNULL,
    )
    assert probe.returncode == 0, probe.stderr[-2000:]
    got = Path(probe.stdout.strip().splitlines()[-1]).resolve()
    assert str(got).startswith(str(sb.root.resolve())), f"profiles root escaped the sandbox: {got}"


# The dashboard process -----------------------------------------------------------------------------


def kill_group(proc: subprocess.Popen, sig: int = signal.SIGKILL) -> None:
    try:
        os.killpg(proc.pid, sig)
    except (ProcessLookupError, PermissionError):
        pass


class Dashboard:
    """``hermes dashboard`` bound to 127.0.0.1:<ephemeral>, token scraped from index.html."""

    def __init__(self, sb: Sandbox, log_path: Path, extra_env: dict[str, str] | None = None,
                 argv: tuple[str, ...] = ()) -> None:
        self.sb = sb
        self.log_path = log_path
        self._log = open(log_path, "a", encoding="utf-8")  # noqa: SIM115 - closed in close()
        env = sb.env({"HERMES_WEB_DIST": str(sb.root / "web_dist"), **(extra_env or {})})
        self.proc = subprocess.Popen(
            [sys.executable, "-m", "hermes_cli.main", "dashboard", "--no-open", "--skip-build",
             "--host", "127.0.0.1", "--port", "0", *argv],
            cwd=str(sb.home), env=env, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, text=True, bufsize=1, start_new_session=True,
        )
        port_box: list[int] = []

        def pump() -> None:
            assert self.proc.stdout is not None
            for line in self.proc.stdout:
                self._log.write(line)
                self._log.flush()
                m = _READY_RE.search(line)
                if m and not port_box:
                    port_box.append(int(m.group(1)))
        threading.Thread(target=pump, daemon=True, name="dash-stdout").start()
        self.port = poll(lambda: port_box[0] if port_box else (self.proc.poll() is not None and -1), 120,
                         "hermes dashboard to report its port")
        assert self.port > 0, f"hermes dashboard exited rc={self.proc.returncode}:\n{self.log_tail()}"
        self.base = f"http://127.0.0.1:{self.port}"
        self.http = httpx.Client(base_url=self.base, timeout=60.0, trust_env=False)
        index = self.http.get("/")
        m = _TOKEN_RE.search(index.text)
        assert index.status_code == 200 and m, f"index.html carried no session token: {index.status_code}"
        self.token = m.group(1)

    def log_tail(self, n: int = 4000) -> str:
        try:
            return self.log_path.read_text(encoding="utf-8", errors="replace")[-n:]
        except OSError:
            return ""

    # REST with the SPA's credential
    def request(self, method: str, path: str, profile: str | None = None, **kw: Any) -> httpx.Response:
        params = dict(kw.pop("params", None) or {})
        if profile is not None:
            params["profile"] = profile
        headers = {TOKEN_HEADER: self.token, **(kw.pop("headers", None) or {})}
        return self.http.request(method, path, params=params, headers=headers, **kw)

    def ok(self, method: str, path: str, profile: str | None = None, **kw: Any) -> Any:
        r = self.request(method, path, profile, **kw)
        assert r.status_code == 200, f"{method} {path} profile={profile}: {r.status_code} {r.text[:600]}"
        return r.json()

    def ws_url(self, path: str, **query: str | None) -> str:
        from urllib.parse import urlencode
        q = urlencode({k: v for k, v in query.items() if v is not None})
        return f"ws://127.0.0.1:{self.port}{path}{'?' + q if q else ''}"

    def close(self) -> None:
        try:
            self.http.close()
        except Exception:
            pass
        kill_group(self.proc, signal.SIGTERM)
        try:
            self.proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            kill_group(self.proc)
            self.proc.wait(timeout=10)
        kill_group(self.proc)  # anything left in the group (tui children, gateways)
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


def group_pids(pgid: int) -> list[int]:
    """Live, non-zombie pids of a process group (the dashboard runs as its own group leader)."""
    out: list[int] = []
    for d in Path("/proc").iterdir():
        if not d.name.isdigit():
            continue
        try:
            fields = (d / "stat").read_text().rsplit(")", 1)[1].split()
        except OSError:
            continue
        if fields[0] != "Z" and int(fields[2]) == pgid:
            out.append(int(d.name))
    return out


# State readers -------------------------------------------------------------------------------------


def db_rows(db: Path, sql: str, args: tuple = ()) -> list[tuple]:
    conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True, timeout=30)
    try:
        return conn.execute(sql, args).fetchall()
    finally:
        conn.close()


def run_py(sb: Sandbox, code: str, *args: str, hermes_home: Path | None = None,
           timeout: float = 120.0) -> subprocess.CompletedProcess:
    extra = {"HERMES_HOME": str(hermes_home)} if hermes_home else None
    return subprocess.run([sys.executable, "-c", code, *args], env=sb.env(extra), cwd=str(sb.home),
                          capture_output=True, text=True, timeout=timeout, stdin=subprocess.DEVNULL)


SEED_SESSIONS = """
import sys
from hermes_state import SessionDB
prefix, n = sys.argv[1], int(sys.argv[2])
db = SessionDB()
for i in range(n):
    sid = f"{prefix}-{i:04d}"
    db.create_session(sid, source="cli", model="seed")
    db.append_message(sid, "user", f"hello {sid}")
    db.append_message(sid, "assistant", f"answer {sid}")
db.close()
print("seeded", n)
"""


def seed_sessions(sb: Sandbox, p: Profile, prefix: str, n: int) -> None:
    r = run_py(sb, SEED_SESSIONS, prefix, str(n), hermes_home=p.home)
    assert r.returncode == 0, r.stderr[-2000:]


def json_blob(obj: Any) -> str:
    return json.dumps(obj, default=str)
