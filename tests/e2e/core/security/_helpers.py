"""Shared harness for the security-boundary E2E suite.

Every scenario runs real Hermes processes (``hermes`` CLI, ``hermes serve``, the gateway,
``tui_gateway``) with HOME=<tmp>/home and HERMES_HOME=<tmp>/home/.hermes, every credential env var
stripped, and the model served by ``tests/fakes/fake_llm_provider.FakeLLMServer``. Assertions read
the boundary's observable outcome: files on disk, state.db rows, logs, and the next wire request.

``BoundaryBreach`` is raised (never a bare ``assert``) when the guarded boundary itself fails, so a
``KNOWN`` strict xfail (``raises=BoundaryBreach``) matches only the tracked bug; a harness failure
(boot, timeout, lost turn) still fails the test loudly.
"""

from __future__ import annotations

import os
import pwd
import secrets
import signal
import socket
import sqlite3
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable, Iterable

import pytest

REPO_ROOT = Path(__file__).resolve().parents[4]


class BoundaryBreach(AssertionError):
    """A security boundary did not hold (secret leaked, file outside a store touched, approval bypassed)."""


def known_param(name: str, known: dict[str, str]) -> Any:
    """``pytest.param`` for a scenario, marked strict-xfail on ``BoundaryBreach`` when it is KNOWN."""
    if name in known:
        return pytest.param(name, marks=pytest.mark.xfail(strict=True, raises=BoundaryBreach, reason=known[name]))
    return name


def canary(label: str) -> str:
    """A random, unmistakable value: a hit can never be a coincidence."""
    return f"{label}-{secrets.token_hex(8)}"


_STRIP_SUFFIXES = ("_API_KEY", "_TOKEN", "_BASE_URL", "_SECRET", "_ACCESS_KEY", "_KEY_ID", "_KEY", "_PASSWORD")
_STRIP_PREFIXES = ("HERMES_", "OPENAI", "ANTHROPIC", "OPENROUTER", "AWS_", "AZURE_", "GOOGLE_", "GEMINI",
                   "PYTEST_", "NOUS_", "XAI_", "LLM_", "CUSTOM_", "TERMINAL_", "API_SERVER_")
_DROP = ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy",
         "XDG_STATE_HOME", "XDG_CONFIG_HOME", "XDG_DATA_HOME", "XDG_CACHE_HOME",
         # no route to the developer's systemd --user bus
         "DBUS_SESSION_BUS_ADDRESS", "XDG_RUNTIME_DIR")


def real_user_home() -> Path:
    return Path(pwd.getpwuid(os.getuid()).pw_dir).resolve()


def hermetic_env(home: Path, extra: dict[str, str] | None = None) -> dict[str, str]:
    """Child env: fake HOME (profile-root anchor) + HERMES_HOME under it, nothing credential-shaped,
    no yolo/approval env inherited from the invoking agent."""
    home = home.resolve()
    assert home != real_user_home(), f"refusing to run a probe against the real HOME: {home}"
    env = {k: v for k, v in os.environ.items()
           if not (k.endswith(_STRIP_SUFFIXES) or k.startswith(_STRIP_PREFIXES)) and k not in _DROP}
    env.update(
        HOME=str(home),
        HERMES_HOME=str(home / ".hermes"),
        XDG_STATE_HOME=str(home / ".local" / "state"),
        PYTHONPATH=str(REPO_ROOT),
        NO_COLOR="1",
        TERM="dumb",
        NO_PROXY="127.0.0.1,localhost",
        no_proxy="127.0.0.1,localhost",
        # the live-DB guard treats $HOME/.hermes/state.db of a pytest descendant as production;
        # this HOME is the test's own tmp dir (asserted above).
        HERMES_STATE_DB_GUARD_BYPASS="1",
        HERMES_ACCEPT_HOOKS="1",
    )
    env.update(extra or {})
    return env


def write_home(hermes_home: Path, base_url: str, *, api_key: str, config: str = "",
               env: dict[str, str] | None = None) -> Path:
    """config.yaml routing the model to the fake provider + .env with the (canary) key.

    ``config`` is appended verbatim (YAML top-level sections)."""
    hermes_home.mkdir(parents=True, exist_ok=True)
    (hermes_home / "config.yaml").write_text(
        "model:\n"
        "  provider: custom\n"
        f"  base_url: {base_url}\n"
        "  default: fake-model\n"
        "  key_env: OPENAI_API_KEY\n"
        "  context_length: 128000\n"
        "agent:\n"
        "  api_max_retries: 1\n"
        "updates:\n"
        "  check: false\n"
        "compression:\n"
        "  enabled: false\n"
        + config,
        encoding="utf-8",
    )
    lines = {"OPENAI_API_KEY": api_key, **(env or {})}
    (hermes_home / ".env").write_text("".join(f"{k}={v}\n" for k, v in lines.items()), encoding="utf-8")
    return hermes_home


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


def kill_group(proc: subprocess.Popen, sig: int = signal.SIGKILL) -> None:
    try:
        os.killpg(proc.pid, sig)
    except (ProcessLookupError, PermissionError):
        pass


def run_hermes(argv: list[str], home: Path, *, timeout: float = 120.0, cwd: Path | None = None,
               extra_env: dict[str, str] | None = None, stdin: str | None = None) -> subprocess.CompletedProcess:
    """``python -m hermes_cli.main <argv>`` in its own process group; the group is always reaped."""
    proc = subprocess.Popen(
        [sys.executable, "-m", "hermes_cli.main", *argv], cwd=str(cwd or home), env=hermetic_env(home, extra_env),
        stdin=subprocess.PIPE if stdin is not None else subprocess.DEVNULL,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, start_new_session=True,
    )
    try:
        out, err = proc.communicate(input=stdin, timeout=timeout)
    except subprocess.TimeoutExpired:
        kill_group(proc)
        out, err = proc.communicate()
        err += f"\n[harness] killed after {timeout}s"
    kill_group(proc)
    return subprocess.CompletedProcess(argv, proc.returncode, out, err)


def run_python(code: str, home: Path, *args: str, timeout: float = 120.0,
               extra_env: dict[str, str] | None = None) -> subprocess.CompletedProcess:
    """A fresh interpreter importing the worktree's Hermes under the hermetic env."""
    return subprocess.run([sys.executable, "-c", code, *args], cwd=str(home), env=hermetic_env(home, extra_env),
                          capture_output=True, text=True, timeout=timeout, stdin=subprocess.DEVNULL)


def db_blob(db: Path) -> str:
    """Every text/blob cell of every table of a SQLite file (read-only), for substring scans."""
    if not db.exists():
        return ""
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    try:
        parts: list[str] = []
        tables = [r[0] for r in con.execute("select name from sqlite_master where type='table'")]
        for t in tables:
            try:
                for row in con.execute(f'select * from "{t}"'):
                    parts += [c.decode("utf-8", "replace") if isinstance(c, bytes) else str(c)
                              for c in row if c is not None]
            except sqlite3.DatabaseError:
                continue
        return "\n".join(parts)
    finally:
        con.close()


def files_containing(root: Path, needles: Iterable[str]) -> list[str]:
    """``<relpath>: <needle>`` for every file under ``root`` whose raw bytes contain a needle."""
    wanted = [(n, n.encode()) for n in needles if n]
    hits: list[str] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        try:
            data = path.read_bytes()
        except OSError:
            continue
        hits += [f"{path.relative_to(root)}: {n}" for n, raw in wanted if raw in data]
    return hits
