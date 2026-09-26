"""Shared harness for the native-dialect provider wire suites (``test_native_*.py``).

Every scenario drives the REAL ``hermes`` CLI (``python -m hermes_cli.main chat -q ... -Q``) as a
subprocess with a hermetic fake HOME / HERMES_HOME, a config.yaml that selects a native provider,
and the provider's endpoint redirected to a loopback fake from ``tests/fakes/providers/``. Only the
vendor boundary is faked; runtime resolution, the adapter, the agent loop, tools and SQLite are real.

What a test asserts: the NEXT wire request Hermes sends (captured by the fake), the persisted
``state.db`` rows, and the CLI's user-visible output — never Hermes source text.
"""

from __future__ import annotations

import json
import os
import sqlite3
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import hermes_yaml as yaml

REPO_ROOT = Path(__file__).resolve().parents[4]
TURN_TIMEOUT = 180.0

_SECRET_ENV_SUFFIXES = ("_API_KEY", "_TOKEN", "_SECRET", "_ACCESS_KEY", "_SESSION_TOKEN")
_PASSTHROUGH_ENV = frozenset({
    "PATH", "LANG", "LANGUAGE", "USER", "LOGNAME", "SHELL", "TMPDIR", "TZ",
    "SYSTEMROOT", "SystemRoot", "COMSPEC", "PATHEXT", "WINDIR", "TEMP", "TMP",
})


@dataclass
class NativeHome:
    """One hermetic fake HOME with ``HOME/.hermes`` as HERMES_HOME and a project dir as cwd."""

    root: Path

    @property
    def home(self) -> Path:
        return self.root / "home"

    @property
    def hermes_home(self) -> Path:
        return self.home / ".hermes"

    @property
    def project(self) -> Path:
        return self.root / "project"

    @property
    def db_path(self) -> Path:
        return self.hermes_home / "state.db"

    def env(self, extra: dict[str, str] | None = None) -> dict[str, str]:
        """Allowlisted env: no inherited credentials, HERMES_* or TERMINAL_* can reroute the child."""
        env = {
            k: v for k, v in os.environ.items()
            if (k in _PASSTHROUGH_ENV or k.startswith("LC_")) and not k.endswith(_SECRET_ENV_SUFFIXES)
        }
        env.update({
            "HOME": str(self.home),
            "HERMES_HOME": str(self.hermes_home),
            "PYTHONPATH": str(REPO_ROOT),
            "PYTHONUNBUFFERED": "1",
            "NO_COLOR": "1",
            "TERM": "dumb",
            # The child's ~/.hermes/state.db IS the tmp home's db; under a pytest ancestor the
            # live-DB guard would refuse it. Documented child escape hatch; path is tmp by construction.
            "HERMES_STATE_DB_GUARD_BYPASS": "1",
            # Never reach real AWS/GCP metadata endpoints or shared config from a fake home.
            "AWS_EC2_METADATA_DISABLED": "true",
            "AWS_CONFIG_FILE": str(self.home / ".aws" / "config"),
            "AWS_SHARED_CREDENTIALS_FILE": str(self.home / ".aws" / "credentials"),
            "NO_GCE_CHECK": "True",
        })
        env.update(extra or {})
        return env


def make_home(root: Path, model: dict[str, Any], *, env_file: dict[str, str] | None = None,
              extra_config: dict[str, Any] | None = None) -> NativeHome:
    """Write config.yaml (``model`` block + offline defaults + ``extra_config``) and ``.env``."""
    nh = NativeHome(root)
    for d in (nh.hermes_home, nh.project):
        d.mkdir(parents=True, exist_ok=True)
    cfg: dict[str, Any] = {
        "model": model,
        "agent": {"api_max_retries": 2},
        "updates": {"check": False},
        "auxiliary": {"title_generation": {"enabled": False}},
        "memory": {"memory_enabled": False, "user_profile_enabled": False},
    }
    for key, value in (extra_config or {}).items():
        if isinstance(value, dict) and isinstance(cfg.get(key), dict):
            cfg[key].update(value)
        else:
            cfg[key] = value
    (nh.hermes_home / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    lines = [f"{k}={v}" for k, v in (env_file or {}).items()]
    (nh.hermes_home / ".env").write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return nh


@dataclass
class ChatResult:
    returncode: int
    stdout: str
    stderr: str
    wall_s: float

    def describe(self) -> str:
        return (f"exit={self.returncode} wall={self.wall_s:.1f}s\n--- stdout ---\n{self.stdout[-2500:]}"
                f"\n--- stderr ---\n{self.stderr[-4000:]}")


def run_chat(nh: NativeHome, prompt: str, *, resume: str | None = None, env: dict[str, str] | None = None,
             args: tuple[str, ...] = (), timeout: float = TURN_TIMEOUT) -> ChatResult:
    """One real ``hermes chat -q`` turn (optionally ``--resume <id>``) against the fake provider."""
    argv = [sys.executable, "-m", "hermes_cli.main", "chat", "-q", prompt, "-Q", *args]
    if resume:
        argv += ["--resume", resume]
    started = time.monotonic()
    proc = subprocess.run(argv, cwd=nh.project, env=nh.env(env), capture_output=True, text=True,
                          timeout=timeout, stdin=subprocess.DEVNULL)
    return ChatResult(proc.returncode, proc.stdout, proc.stderr, time.monotonic() - started)


def _connect(nh: NativeHome) -> sqlite3.Connection:
    conn = sqlite3.connect(f"file:{nh.db_path}?mode=ro", uri=True, timeout=10)
    conn.row_factory = sqlite3.Row
    return conn


def session_ids(nh: NativeHome) -> list[str]:
    """All session ids, oldest first."""
    if not nh.db_path.exists():
        return []
    with _connect(nh) as conn:
        return [r["id"] for r in conn.execute("SELECT id FROM sessions ORDER BY started_at, rowid")]


def latest_session(nh: NativeHome) -> str:
    ids = session_ids(nh)
    assert ids, f"no session persisted in {nh.db_path}"
    return ids[-1]


def messages(nh: NativeHome, session_id: str | None = None, *, active_only: bool = True) -> list[dict[str, Any]]:
    """Persisted message rows (dicts) for ``session_id`` (default: every session), in insertion order."""
    if not nh.db_path.exists():
        return []
    where, params = [], []
    if session_id:
        where.append("session_id = ?")
        params.append(session_id)
    if active_only:
        where.append("active = 1")
    sql = "SELECT * FROM messages" + (f" WHERE {' AND '.join(where)}" if where else "") + " ORDER BY id"
    with _connect(nh) as conn:
        return [dict(r) for r in conn.execute(sql, params)]


def tool_calls_of(row: dict[str, Any]) -> list[dict[str, Any]]:
    raw = row.get("tool_calls")
    return json.loads(raw) if raw else []


class KnownSymptom(AssertionError):
    """Raised ONLY at a tracked bug's exact symptom.

    It is the type ``known_gate``/``known_failure`` accept (``raises=KnownSymptom``), so a harness failure
    (process death, timeout, precondition assert, fixture teardown error) fails for real instead of
    counting as the known bug.
    """


def wait_until(predicate: Callable[[], Any], timeout: float, what: str, interval: float = 0.05,
               error: type[AssertionError] = AssertionError) -> Any:
    """Poll ``predicate`` until truthy or raise ``error`` naming ``what`` (no bare sleeps as synchronization)."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        value = predicate()
        if value:
            return value
        time.sleep(interval)
    raise error(f"timed out after {timeout}s waiting for {what}")


def assert_no_duplicate_assistant_text(rows: list[dict[str, Any]], needle: str) -> None:
    """A retried/dropped stream must never persist the same assistant content twice."""
    hits = [r["id"] for r in rows if r["role"] == "assistant" and needle in (r.get("content") or "")]
    assert len(hits) <= 1, f"assistant text {needle!r} persisted {len(hits)}x (rows {hits})"
