"""Shared harness for the native-Windows end-to-end suite.

Every test drives REAL Hermes processes (``hermes.exe`` / ``python -m
hermes_cli.main``) on a real Windows host against the recording loopback
provider (``tests/fakes/fake_llm_provider.py``). Nothing in Hermes is mocked;
verdicts come from what reached the provider wire, what landed in ``state.db``
/ on disk, and the live process table (psutil).

Each test gets a fresh fake user profile under ``tmp_path``: ``USERPROFILE`` /
``HOME`` / ``LOCALAPPDATA`` / ``APPDATA`` all point inside it and
``HERMES_HOME`` is ``<profile>/.hermes``, so no child can read or write the
runner's real Hermes state.
"""

from __future__ import annotations

import json
import os
import sqlite3
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable

import yaml

from tests.fakes.fake_llm_provider import FakeLLMServer, write_hermes_home

REPO_ROOT = Path(__file__).resolve().parents[4]
TURN_TIMEOUT = 180.0

# Windows needs these for a child to start at all (CreateProcess, Winsock, temp dirs,
# Program Files lookups the Git Bash resolver uses). Credentials never pass.
_PASSTHROUGH_ENV = frozenset({
    "PATH", "PATHEXT", "SYSTEMROOT", "SystemRoot", "SYSTEMDRIVE", "SystemDrive", "COMSPEC",
    "WINDIR", "TEMP", "TMP", "OS", "PROCESSOR_ARCHITECTURE", "NUMBER_OF_PROCESSORS",
    "ProgramFiles", "ProgramFiles(x86)", "ProgramW6432", "ProgramData", "CommonProgramFiles",
    "CommonProgramFiles(x86)", "CommonProgramW6432", "USERNAME", "USERDOMAIN", "COMPUTERNAME",
    "PSModulePath", "LANG", "TZ",
})
_SECRET_SUFFIXES = ("_API_KEY", "_TOKEN", "_SECRET", "_ACCESS_KEY")


@dataclass
class WinHome:
    """One hermetic fake Windows user profile with a Hermes home inside it."""

    root: Path
    profile: Path
    hermes_home: Path
    project: Path
    extra_env: dict[str, str] = field(default_factory=dict)

    def env(self, extra: dict[str, str] | None = None) -> dict[str, str]:
        env = {
            k: v for k, v in os.environ.items()
            if k in _PASSTHROUGH_ENV and not k.upper().endswith(_SECRET_SUFFIXES)
        }
        local = self.profile / "AppData" / "Local"
        roaming = self.profile / "AppData" / "Roaming"
        local.mkdir(parents=True, exist_ok=True)
        roaming.mkdir(parents=True, exist_ok=True)
        env.update({
            "USERPROFILE": str(self.profile),
            "HOME": str(self.profile),
            "LOCALAPPDATA": str(local),
            "APPDATA": str(roaming),
            "HERMES_HOME": str(self.hermes_home),
            "PYTHONPATH": str(REPO_ROOT),
            "PYTHONUNBUFFERED": "1",
            "NO_COLOR": "1",
            # The child's state.db lives under tmp_path; under a pytest ancestor the live-DB
            # guard would refuse it. Documented child-process escape hatch (tests/conftest.py).
            "HERMES_STATE_DB_GUARD_BYPASS": "1",
        })
        env.update(self.extra_env)
        env.update(extra or {})
        return env

    def update_config(self, mutate: Callable[[dict], None]) -> None:
        path = self.hermes_home / "config.yaml"
        cfg = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        mutate(cfg)
        path.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True), encoding="utf-8")

    @property
    def db_path(self) -> Path:
        return self.hermes_home / "state.db"


def make_home(tmp_path: Path, base_url: str, *, extra_config: str = "") -> WinHome:
    root = tmp_path
    profile = root / "Users" / "e2e"
    hermes_home = profile / ".hermes"
    project = root / "project"
    project.mkdir(parents=True, exist_ok=True)
    write_hermes_home(hermes_home, base_url, extra_config=extra_config)
    home = WinHome(root=root, profile=profile, hermes_home=hermes_home, project=project)

    def _hermetic(cfg: dict) -> None:
        cfg["updates"] = {"check": False}  # offline: no GitHub round trip / git lazy fetch
        cfg.setdefault("display", {})["compact"] = True

    home.update_config(_hermetic)
    return home


def hermes_argv(*args: str) -> list[str]:
    return [sys.executable, "-m", "hermes_cli.main", *args]


def hermes_exe() -> Path:
    """The console-script shim users actually launch (``<venv>\\Scripts\\hermes.exe``)."""
    exe = Path(sys.executable).with_name("hermes.exe")
    assert exe.is_file(), f"hermes.exe console script missing next to {sys.executable}"
    return exe


@dataclass
class Run:
    returncode: int
    stdout: str
    stderr: str

    def tail(self, n: int = 3000) -> str:
        return f"rc={self.returncode}\n--stdout--\n{self.stdout[-n:]}\n--stderr--\n{self.stderr[-n:]}"


def run(argv: list[str], home: WinHome, *, cwd: Path | None = None, timeout: float = TURN_TIMEOUT,
        env_extra: dict[str, str] | None = None, stdin: bytes | None = None) -> Run:
    proc = subprocess.run(
        argv, cwd=cwd or home.project, env=home.env(env_extra), capture_output=True,
        input=stdin, stdin=None if stdin is not None else subprocess.DEVNULL, timeout=timeout,
    )
    return Run(proc.returncode, _decode(proc.stdout), _decode(proc.stderr))


def hermes(home: WinHome, *args: str, **kw: Any) -> Run:
    return run(hermes_argv(*args), home, **kw)


def _decode(raw: bytes) -> str:
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        return raw.decode("mbcs" if sys.platform == "win32" else "latin-1", errors="replace")


# Provider wire ---------------------------------------------------------------


def text_of(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(p.get("text", "") for p in content if isinstance(p, dict))
    return ""


def system_prompt(body: dict[str, Any]) -> str:
    return "\n".join(text_of(m.get("content")) for m in body.get("messages") or [] if m.get("role") == "system")


def last_user(body: dict[str, Any]) -> str:
    users = [text_of(m.get("content")) for m in body.get("messages") or [] if m.get("role") == "user"]
    return users[-1] if users else ""


def tool_results(srv: FakeLLMServer) -> list[str]:
    """Tool-result messages Hermes sent back to the model, in order, deduped across requests."""
    seen: dict[str, str] = {}
    for body in srv.main_requests():
        for m in body.get("messages") or []:
            if m.get("role") == "tool":
                seen.setdefault(m.get("tool_call_id") or str(len(seen)), text_of(m.get("content")))
    return list(seen.values())


def parse_tool_json(result: str) -> dict[str, Any]:
    """The terminal tool's JSON payload (subdirectory hints may be appended after it)."""
    start = result.find("{")
    assert start >= 0, f"tool result carries no JSON payload: {result[:500]!r}"
    obj, _end = json.JSONDecoder().raw_decode(result[start:])
    return obj


# state.db --------------------------------------------------------------------


def db_rows(home: WinHome, sql: str, params: Iterable[Any] = ()) -> list[sqlite3.Row]:
    assert home.db_path.exists(), f"no state.db at {home.db_path}"
    conn = sqlite3.connect(f"file:{home.db_path.as_posix()}?mode=ro", uri=True, timeout=30)
    conn.row_factory = sqlite3.Row
    try:
        return list(conn.execute(sql, tuple(params)))
    finally:
        conn.close()


def persisted_messages(home: WinHome) -> list[sqlite3.Row]:
    return db_rows(home, "SELECT session_id, role, content FROM messages ORDER BY id")


# Processes -------------------------------------------------------------------


def wait_until(pred: Callable[[], Any], timeout: float, what: str, interval: float = 0.1) -> Any:
    deadline = time.monotonic() + timeout
    while True:
        value = pred()
        if value:
            return value
        if time.monotonic() >= deadline:
            raise AssertionError(f"timed out after {timeout}s waiting for {what}")
        time.sleep(interval)


def process_tree(pid: int) -> list[Any]:
    """psutil handles for ``pid`` and every live descendant (identity-checked by psutil)."""
    import psutil

    try:
        root = psutil.Process(pid)
        return [root, *root.children(recursive=True)]
    except psutil.NoSuchProcess:
        return []


def kill_tree(procs: Iterable[Any]) -> None:
    """Test cleanup only: hard-kill whatever this test spawned that is still alive."""
    import psutil

    for p in procs:
        try:
            p.kill()
        except psutil.Error:
            pass


def _under(path: str, root: str) -> bool:
    path = os.path.normcase(os.path.normpath(path))
    return path == root or path.startswith(root + os.sep)


def _owned_by(proc: Any, root: str, home: str) -> bool:
    """One process belongs to the fake profile if its HERMES_HOME is that home, or its
    cwd or any argv element lives under the profile root. Environment and cwd are
    inherited by detached grandchildren, so they survive a broken parent link."""
    import psutil

    try:
        env = {k.upper(): v for k, v in proc.environ().items()}
        if "HERMES_HOME" in env and os.path.normcase(os.path.normpath(env["HERMES_HOME"])) == home:
            return True
        if _under(proc.cwd(), root):
            return True
        return any(_under(arg, root) or root + os.sep in os.path.normcase(arg) for arg in proc.cmdline())
    except (psutil.Error, OSError):
        return False


def owned_processes(home: WinHome, *, since: float) -> list[Any]:
    """Every live process created at/after ``since`` (epoch s) that belongs to ``home``,
    whatever its parent. Windows never re-parents: a child of a killed process keeps a
    dangling ppid, so ``Process.children()`` (and ``taskkill /T``) cannot see it. Finding
    orphans needs ownership, not ancestry."""
    import psutil

    root = os.path.normcase(os.path.normpath(str(home.root)))
    hermes_home = os.path.normcase(os.path.normpath(str(home.hermes_home)))
    me = os.getpid()
    owned = []
    for proc in psutil.process_iter():
        try:
            if proc.pid == me or proc.create_time() < since - 1.0:  # 1 s: create_time rounding
                continue
        except psutil.Error:
            continue
        if _owned_by(proc, root, hermes_home):
            owned.append(proc)
    return owned


def describe(procs: Iterable[Any]) -> list[str]:
    import psutil

    out = []
    for p in procs:
        try:
            out.append(f"pid={p.pid} ppid={p.ppid()} {' '.join(p.cmdline())[:200]}")
        except psutil.Error:
            continue
    return out


def owned_survivors(home: WinHome, *, since: float, timeout: float) -> list[str]:
    """Poll until nothing owned by ``home`` is alive; describe what is left at the deadline."""
    deadline = time.monotonic() + timeout
    while True:
        left = describe(owned_processes(home, since=since))
        if not left or time.monotonic() >= deadline:
            return left
        time.sleep(0.25)


def kill_owned(home: WinHome, *, since: float) -> None:
    """Test cleanup only: hard-kill every process this test's profile still owns."""
    kill_tree(owned_processes(home, since=since))


def taskkill_tree(pid: int) -> subprocess.CompletedProcess:
    """The Desktop's backend quit path on Windows (electron/backend-child.ts): taskkill /T /F."""
    return subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"], capture_output=True, timeout=60)


def nonce(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex[:10]}"


# Known bugs ------------------------------------------------------------------


class KnownBugSymptom(Exception):
    """Raised ONLY at the assertion that pins a tracked bug's symptom. Not an AssertionError: a
    file's ``KNOWN`` gate (``known_gate(KNOWN, key, raises=KnownBugSymptom)``) accepts just this,
    so a harness failure or any other broken invariant in the same test still fails loudly."""


def expect(ok: bool, message: str) -> None:
    if not ok:
        raise KnownBugSymptom(message)
