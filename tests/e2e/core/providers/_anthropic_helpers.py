"""Hermetic rig for the Anthropic Messages wire-conformance E2E suite.

Hermes' native ``anthropic`` provider runs with NO base URL override, i.e. the
exact production route to ``https://api.anthropic.com`` (native thinking-signature
policy, native headers). The child reaches the scripted fake through an
``HTTPS_PROXY`` that terminates TLS for ``api.anthropic.com`` with a leaf signed
by a throwaway CA trusted via ``SSL_CERT_FILE``; only the vendor HTTP boundary
is faked. Every child runs with an allowlisted env, ``HOME``/``HERMES_HOME``
under ``tmp_path`` and a unique tag so cleanup signals only its own tree.
"""

from __future__ import annotations

import copy
import ctypes
import json
import os
import signal
import sqlite3
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import hermes_yaml as yaml

from tests.fakes.providers.anthropic_messages import MODEL_ID, AnthropicMessagesServer, Response, Responder
from tests.fakes.providers.oauth_token_server import TLSInterceptProxy, make_test_ca

REPO_ROOT = Path(__file__).resolve().parents[4]
TAG_VAR = "ANTHROPIC_E2E_TAG"
VENDOR_HOST = "api.anthropic.com"
TURN_TIMEOUT = 180.0

_SECRET_ENV_SUFFIXES = ("_API_KEY", "_TOKEN", "_SECRET", "_ACCESS_KEY")
_PASSTHROUGH_ENV = frozenset({"PATH", "LANG", "LANGUAGE", "USER", "LOGNAME", "SHELL", "TMPDIR", "TZ"})


@dataclass
class Rig:
    root: Path
    home: Path
    hermes_home: Path
    project: Path
    srv: AnthropicMessagesServer
    proxy: TLSInterceptProxy
    ca_pem: Path
    tag: str = field(default_factory=lambda: uuid.uuid4().hex)

    def env(self, extra: dict[str, str] | None = None) -> dict[str, str]:
        import pwd  # POSIX-only; the suite is Linux-gated

        real_root = Path(pwd.getpwuid(os.getuid()).pw_dir, ".hermes").resolve()
        fixture = self.hermes_home.resolve()
        assert fixture != real_root and fixture.parent != real_root / "profiles", (
            f"fixture HERMES_HOME {fixture} is the real install's live home")
        env = {k: v for k, v in os.environ.items()
               if (k in _PASSTHROUGH_ENV or k.startswith("LC_")) and not k.endswith(_SECRET_ENV_SUFFIXES)}
        loopback = "127.0.0.1,localhost"
        env.update({
            "HOME": str(self.home), "HERMES_HOME": str(self.hermes_home), "PYTHONPATH": str(REPO_ROOT),
            "PYTHONUNBUFFERED": "1", "NO_COLOR": "1", "TERM": "dumb", TAG_VAR: self.tag,
            "HTTPS_PROXY": self.proxy.url, "https_proxy": self.proxy.url,
            "NO_PROXY": loopback, "no_proxy": loopback, "SSL_CERT_FILE": str(self.ca_pem),
            # The child's ~/.hermes/state.db IS the tmp home's db (see parity/_helpers.py).
            "HERMES_STATE_DB_GUARD_BYPASS": "1",
        })
        env.update(extra or {})
        return env

    def config(self) -> dict[str, Any]:
        return yaml.safe_load((self.hermes_home / "config.yaml").read_text(encoding="utf-8"))

    def update_config(self, mutate: Callable[[dict[str, Any]], None]) -> None:
        cfg = self.config()
        mutate(cfg)
        (self.hermes_home / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")

    def run(self, *args: str, timeout: float = TURN_TIMEOUT, extra_env: dict[str, str] | None = None,
            ) -> subprocess.CompletedProcess:
        return subprocess.run(hermes_argv(*args), cwd=self.project, env=self.env(extra_env), capture_output=True,
                              text=True, timeout=timeout, stdin=subprocess.DEVNULL)

    def stop(self) -> None:
        reap_adopted(kill_tagged(self.tag))
        self.proxy.stop()
        self.srv.stop()

    def log_tail(self, chars: int = 3000, pattern: str = "") -> str:
        """The child's agent.log tail (CI prints only the assertion message, so failures carry it)."""
        log = self.hermes_home / "logs" / "agent.log"
        if not log.exists():
            return "<no agent.log>"
        lines = log.read_text(encoding="utf-8", errors="replace").splitlines()
        return "\n".join(ln for ln in lines if pattern.lower() in ln.lower())[-chars:]

    # persisted state ------------------------------------------------------------
    def _db(self) -> sqlite3.Connection:
        return sqlite3.connect(f"file:{self.hermes_home / 'state.db'}?mode=ro", uri=True)

    def session_ids(self) -> list[str]:
        with self._db() as db:
            return [r[0] for r in db.execute("select id from sessions order by started_at")]

    def messages(self, session_id: str) -> list[dict[str, Any]]:
        with self._db() as db:
            db.row_factory = sqlite3.Row
            return [dict(r) for r in db.execute(
                "select * from messages where session_id = ? order by id", (session_id,))]


def hermes_argv(*args: str) -> list[str]:
    return [sys.executable, "-m", "hermes_cli.main", *args]


_PR_SET_CHILD_SUBREAPER = 36


def become_subreaper() -> None:
    """Adopt this test process's orphaned descendants (Linux ``PR_SET_CHILD_SUBREAPER``).

    A Hermes child that daemonises a helper leaves it reparented to init, outside the test's
    process tree, so teardown could neither reap it nor (under the local live-system guard)
    signal it. As subreaper the orphans stay our children: ``kill_tagged`` stays in-tree and
    ``reap_adopted`` (``Rig.stop``) collects them instead of leaving zombies."""
    try:
        ctypes.CDLL(None, use_errno=True).prctl(_PR_SET_CHILD_SUBREAPER, 1, 0, 0, 0)
    except (OSError, AttributeError):
        pass


def start_rig(root: Path, script: list[Response] | Responder, *, config: dict[str, Any] | None = None,
              aux: Responder | None = None) -> Rig:
    become_subreaper()
    srv = AnthropicMessagesServer(script, aux=aux).start()
    ca = make_test_ca(root / "ca", [VENDOR_HOST])
    proxy = TLSInterceptProxy(srv, ca, [VENDOR_HOST]).start()  # type: ignore[arg-type]
    home = root / "home"
    rig = Rig(root=root, home=home, hermes_home=home / ".hermes", project=root / "project", srv=srv,
              proxy=proxy, ca_pem=ca.ca_pem)
    rig.hermes_home.mkdir(parents=True)
    rig.project.mkdir()
    cfg: dict[str, Any] = {
        # No base_url: the production native route (the proxy intercepts api.anthropic.com).
        "model": {"provider": "anthropic", "default": MODEL_ID, "context_length": 200000},
        "agent": {"api_max_retries": 1, "reasoning_effort": "medium"},
        "auxiliary": {"title_generation": {"enabled": False}},
        "updates": {"check": False},
        "display": {"compact": True},
    }
    for key, value in (config or {}).items():
        cfg[key] = {**cfg.get(key, {}), **value} if isinstance(value, dict) else value
    (rig.hermes_home / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    (rig.hermes_home / ".env").write_text("ANTHROPIC_API_KEY=sk-ant-api03-e2e-fake-key\n", encoding="utf-8")
    return rig


def _tagged_pids(tag: str) -> list[int]:
    """Live (non-zombie) PIDs carrying ``ANTHROPIC_E2E_TAG=<tag>``, i.e. this test's tree only."""
    needle = f"{TAG_VAR}={tag}".encode()
    me, out = os.getpid(), []
    for entry in os.listdir("/proc"):
        if not entry.isdigit() or int(entry) == me:
            continue
        try:
            with open(f"/proc/{entry}/environ", "rb") as fh:
                env = fh.read()
        except OSError:
            continue
        if needle in env.split(b"\0"):
            out.append(int(entry))
    return out


def kill_tagged(tag: str) -> list[int]:
    """SIGKILL every live process of this test's tree; returns the PIDs signalled."""
    killed = []
    for pid in _tagged_pids(tag):
        try:
            os.kill(pid, signal.SIGKILL)
            killed.append(pid)
        except OSError:
            pass
    return killed


def reap_adopted(pids: list[int], timeout: float = 10.0) -> None:
    """Collect the exit status of killed children, including orphans adopted as subreaper.

    A killed orphan reparented to this process stays a zombie until waited for. Each PID that is
    our child is waited for (bounded), then any other exited child is drained. Never raises: it
    runs in fixture teardown."""
    deadline = time.monotonic() + timeout
    for pid in pids:
        while time.monotonic() < deadline:
            try:
                done, _status = os.waitpid(pid, os.WNOHANG)  # windows-footgun: ok — Linux-gated suite
            except OSError:  # ChildProcessError: not our child (a grandchild or already reaped)
                break
            if done:
                break
            time.sleep(0.02)
    while True:
        try:
            done, _status = os.waitpid(-1, os.WNOHANG)  # windows-footgun: ok — Linux-gated suite
        except OSError:
            return
        if not done:
            return


def wait_until(pred: Callable[[], Any], timeout: float, what: str, interval: float = 0.05) -> Any:
    deadline = time.monotonic() + timeout
    while True:
        value = pred()
        if value:
            return value
        if time.monotonic() >= deadline:
            raise AssertionError(f"timed out after {timeout}s waiting for {what}")
        time.sleep(interval)


# wire views -----------------------------------------------------------------------


def blocks(message: dict[str, Any]) -> list[dict[str, Any]]:
    content = message.get("content")
    if isinstance(content, str):
        return [{"type": "text", "text": content}]
    return list(content or [])


def thinking_of(message: dict[str, Any]) -> list[tuple[str, str]]:
    return [(b.get("thinking", ""), b.get("signature", "")) for b in blocks(message) if b.get("type") == "thinking"]


def assistant_messages(body: dict[str, Any]) -> list[dict[str, Any]]:
    return [m for m in body.get("messages") or [] if m.get("role") == "assistant"]


def without_cache_control(value: Any) -> Any:
    """History with the moving ``cache_control`` breakpoints removed (they legitimately shift per turn)."""
    value = copy.deepcopy(value)
    if isinstance(value, dict):
        value.pop("cache_control", None)
        return {k: without_cache_control(v) for k, v in value.items()}
    if isinstance(value, list):
        return [without_cache_control(v) for v in value]
    return value


def normalised(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Messages with string content expanded to one text block and cache markers dropped."""
    return [{"role": m["role"], "content": without_cache_control(blocks(m))} for m in messages]


def tool_pairing_problems(body: dict[str, Any]) -> list[str]:
    """Anthropic's pairing rule: every tool_use id is answered by a tool_result in the NEXT user
    message, and every tool_result answers a tool_use of the immediately preceding assistant message."""
    problems: list[str] = []
    msgs = body.get("messages") or []
    for i, m in enumerate(msgs):
        if m.get("role") == "assistant":
            uses = [b["id"] for b in blocks(m) if b.get("type") == "tool_use"]
            nxt = msgs[i + 1] if i + 1 < len(msgs) else None
            results = [b.get("tool_use_id") for b in blocks(nxt or {}) if b.get("type") == "tool_result"]
            if uses and sorted(uses) != sorted(results):
                problems.append(f"msg {i}: tool_use {uses} answered by {results}")
        else:
            prev = msgs[i - 1] if i else {}
            uses = {b.get("id") for b in blocks(prev) if b.get("type") == "tool_use"}
            orphans = [b.get("tool_use_id") for b in blocks(m)
                       if b.get("type") == "tool_result" and b.get("tool_use_id") not in uses]
            if orphans:
                problems.append(f"msg {i}: orphan tool_result {orphans}")
    return problems


def dump(body: dict[str, Any], limit: int = 3000) -> str:
    return json.dumps(without_cache_control(body.get("messages")), default=str)[-limit:]
