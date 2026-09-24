"""Shared harness for the OpenAI-dialect provider wire-conformance suite.

Every test drives REAL Hermes processes (``hermes -z`` oneshot, ``--resume``, the
``tui_gateway`` stdio server) against a loopback fake of the vendor HTTP API
(``tests/fakes/providers``) and asserts on the next wire request Hermes sends, the
user-visible answer, and persisted ``state.db`` rows. Homes are hermetic: a fake HOME
whose ``.hermes`` is the HERMES_HOME, an allowlisted env with no credentials.
"""

from __future__ import annotations

import json
import os
import sqlite3
import subprocess
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterator

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[4]
TURN_TIMEOUT = 150.0
_SECRET_ENV_SUFFIXES = ("_API_KEY", "_TOKEN", "_SECRET", "_ACCESS_KEY")
_PASSTHROUGH_ENV = frozenset({"PATH", "LANG", "LANGUAGE", "USER", "LOGNAME", "SHELL", "TMPDIR", "TZ"})


# KNOWN-bug plumbing ------------------------------------------------------------------


class HarnessError(RuntimeError):
    """The harness broke (a process died, a reply or event never came, a turn overran its
    budget). Never an ``AssertionError``, so a KNOWN strict xfail can never excuse it."""


class KnownBugError(AssertionError):
    """Raised ONLY from inside ``bug_assertions()``: the one exception a KNOWN strict xfail
    accepts. Preconditions, waits and teardown outside that block fail the test for real."""


def known_marks(known: dict[str, str], name: str) -> list:
    """Marks for a scenario: a strict xfail while ``name`` is in ``known`` (red the moment
    the bug is fixed, forcing the entry out), nothing once it is gone."""
    if name not in known:
        return []
    return [pytest.mark.xfail(strict=True, raises=KnownBugError, reason=known[name])]


@contextmanager
def bug_assertions() -> Iterator[None]:
    """Wrap ONLY the final behavioural assertions that name the bug, after every wait has
    settled; an ``AssertionError`` raised inside becomes a ``KnownBugError``."""
    try:
        yield
    except KnownBugError:
        raise
    except AssertionError as exc:
        raise KnownBugError(str(exc)) from exc


# Vendor-boundary sitecustomize shims ---------------------------------------------------

_CHAIN_ORIGINAL_SITECUSTOMIZE = '''\
def _chain_original_sitecustomize():
    """This shim dir is first on PYTHONPATH and shadows any sitecustomize the interpreter
    already had (a venv or distro one): find the next one on sys.path and run it too."""
    import importlib.machinery
    import importlib.util
    import os
    import sys

    here = os.path.dirname(os.path.abspath(__file__))
    rest = [p for p in sys.path if os.path.abspath(p or os.curdir) != here]
    spec = importlib.machinery.PathFinder.find_spec("sitecustomize", rest)
    if spec is None or spec.loader is None:
        return
    module = importlib.util.module_from_spec(spec)
    sys.modules["_e2e_original_sitecustomize"] = module
    spec.loader.exec_module(module)


_chain_original_sitecustomize()
'''


def write_sitecustomize_shim(shim_dir: Path, body: str) -> Path:
    """Write ``body`` as ``shim_dir/sitecustomize.py`` after a prologue that chain-imports
    the interpreter's original ``sitecustomize`` (the shim must not shadow it)."""
    shim_dir.mkdir(parents=True, exist_ok=True)
    (shim_dir / "sitecustomize.py").write_text(_CHAIN_ORIGINAL_SITECUSTOMIZE + "\n\n" + body, encoding="utf-8")
    return shim_dir


# One tool Hermes always offers on the CLI toolset and that has an observable,
# side-effect-free result: the model reads a file the fixture wrote.
READ_TOOL = "read_file"


@dataclass
class Home:
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
        """Allowlisted env: the runner may itself be a Hermes process whose HERMES_* or
        credential env would silently reroute the child."""
        import pwd  # the suite is Linux-gated

        real_root = Path(pwd.getpwuid(os.getuid()).pw_dir, ".hermes").resolve()  # windows-footgun: ok — every file here is skipif(not linux)
        fixture = self.hermes_home.resolve()
        assert fixture != real_root and fixture.parent != real_root / "profiles", "fixture is a live home"
        env = {k: v for k, v in os.environ.items()
               if (k in _PASSTHROUGH_ENV or k.startswith("LC_")) and not k.endswith(_SECRET_ENV_SUFFIXES)}
        env.update({
            "HOME": str(self.home), "HERMES_HOME": str(self.hermes_home), "PYTHONPATH": str(REPO_ROOT),
            "PYTHONUNBUFFERED": "1", "NO_COLOR": "1", "TERM": "dumb",
            # The child's state.db lives under tmp_path; the live-DB guard's documented child escape hatch.
            "HERMES_STATE_DB_GUARD_BYPASS": "1",
        })
        env.update(extra or {})
        return env

    def write(self, config: dict[str, Any], dotenv: dict[str, str] | None = None,
              auth: dict[str, Any] | None = None) -> "Home":
        self.hermes_home.mkdir(parents=True, exist_ok=True)
        self.project.mkdir(parents=True, exist_ok=True)
        base = {"updates": {"check": False}, "agent": {"api_max_retries": 2},
                "terminal": {"cwd": str(self.project)}, "memory": {"memory_enabled": False}}
        _deep_merge(base, config)
        (self.hermes_home / "config.yaml").write_text(yaml.safe_dump(base, sort_keys=False), encoding="utf-8")
        (self.hermes_home / ".env").write_text(
            "".join(f"{k}={v}\n" for k, v in (dotenv or {}).items()), encoding="utf-8")
        if auth is not None:
            (self.hermes_home / "auth.json").write_text(json.dumps(auth), encoding="utf-8")
        return self

    def update_config(self, mutate: Callable[[dict], None]) -> None:
        path = self.hermes_home / "config.yaml"
        cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
        mutate(cfg)
        path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")


def _deep_merge(dst: dict, src: dict) -> None:
    for k, v in src.items():
        if isinstance(v, dict) and isinstance(dst.get(k), dict):
            _deep_merge(dst[k], v)
        else:
            dst[k] = v


# Provider configs ----------------------------------------------------------------


def responses_provider_config(base_url: str, *, name: str = "resp-relay", model: str = "fake-responses-model") -> dict:
    """A config-defined provider that opts into the Responses transport (a Responses relay)."""
    return {
        "model": {"provider": name, "default": model, "context_length": 128000},
        "providers": {name: {"name": name, "base_url": base_url, "api_key": "sk-fake-relay",
                             "transport": "codex_responses", "default_model": model}},
    }


def custom_chat_config(base_url: str, *, model: str = "fake-model") -> dict:
    """Plain ``provider: custom`` chat-completions endpoint (Ollama / vLLM / LiteLLM shape)."""
    return {"model": {"provider": "custom", "base_url": base_url, "default": model, "context_length": 128000}}


# Processes -------------------------------------------------------------------------


@dataclass
class Run:
    proc: subprocess.CompletedProcess
    usage: dict[str, Any]

    @property
    def stdout(self) -> str:
        return self.proc.stdout

    @property
    def session_id(self) -> str | None:
        return self.usage.get("session_id")

    def describe(self) -> str:
        return (f"exit={self.proc.returncode}\nusage={json.dumps(self.usage)[:1500]}\n"
                f"stdout={self.proc.stdout[-1500:]!r}\nstderr={self.proc.stderr[-4000:]}")


def hermes_argv(*args: str) -> list[str]:
    return [sys.executable, "-m", "hermes_cli.main", *args]


def oneshot(h: Home, prompt: str, *args: str, resume: str | None = None, timeout: float = TURN_TIMEOUT,
            env: dict[str, str] | None = None) -> Run:
    """One real ``hermes -z`` turn; the usage file names the session for ``--resume``."""
    usage_file = h.root / f"usage-{time.monotonic_ns()}.json"
    argv = ["-z", prompt, "--usage-file", str(usage_file), *args]
    if resume:
        argv += ["--resume", resume]
    proc = subprocess.run(hermes_argv(*argv), cwd=h.project, env=h.env(env), capture_output=True,
                          text=True, encoding="utf-8", errors="replace", timeout=timeout, stdin=subprocess.DEVNULL)
    usage = json.loads(usage_file.read_text(encoding="utf-8")) if usage_file.exists() else {}
    return Run(proc, usage)


def bounded_turn(h: Home, prompt: str, budget: float, **kw: Any) -> Run:
    """One ``oneshot`` turn that must finish inside ``budget`` seconds; overrunning it is a
    ``HarnessError`` (never an ``AssertionError`` a KNOWN xfail could swallow)."""
    try:
        return oneshot(h, prompt, timeout=budget, **kw)
    except subprocess.TimeoutExpired as exc:
        raise HarnessError(f"turn {prompt!r} still running after {budget}s") from exc


def wait_until(pred: Callable[[], Any], timeout: float, what: str, interval: float = 0.05) -> Any:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        value = pred()
        if value:
            return value
        time.sleep(interval)
    raise HarnessError(f"timed out after {timeout}s waiting for {what}")


# Persisted state ---------------------------------------------------------------------


def db_messages(h: Home, session_id: str | None = None) -> list[dict[str, Any]]:
    """Active message rows (oldest first), read-only; every session when ``session_id`` is None."""
    if not h.db_path.exists():
        return []
    con = sqlite3.connect(f"file:{h.db_path}?mode=ro", uri=True, timeout=10)
    con.row_factory = sqlite3.Row
    try:
        sql = "SELECT * FROM messages WHERE active = 1"
        params: tuple = ()
        if session_id:
            sql += " AND session_id = ?"
            params = (session_id,)
        return [dict(r) for r in con.execute(sql + " ORDER BY id", params)]
    finally:
        con.close()


def db_tool_calls(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Every persisted assistant tool call (parsed), in order."""
    out: list[dict[str, Any]] = []
    for r in rows:
        if r["role"] == "assistant" and r.get("tool_calls"):
            out.extend(json.loads(r["tool_calls"]))
    return out


def tool_call_args(tc: dict[str, Any]) -> Any:
    fn = tc.get("function") or {}
    raw = fn.get("arguments", tc.get("arguments"))
    try:
        return json.loads(raw) if isinstance(raw, str) else raw
    except json.JSONDecodeError:
        return raw


# Wire inspection ---------------------------------------------------------------------


def responses_input_items(body: dict[str, Any], item_type: str) -> list[dict[str, Any]]:
    return [i for i in body.get("input") or [] if isinstance(i, dict) and i.get("type") == item_type]


def chat_messages(body: dict[str, Any], role: str | None = None) -> list[dict[str, Any]]:
    msgs = [m for m in body.get("messages") or [] if isinstance(m, dict)]
    return [m for m in msgs if role is None or m.get("role") == role]
