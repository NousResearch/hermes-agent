"""Shared harness for the MCP + plugin conformance suite.

Every test drives a REAL Hermes process (``hermes chat -q`` or the ``tui_gateway``
stdio host) against the recording fake LLM provider and one or more REAL MCP
servers built with the installed ``mcp`` SDK (``mcp_fixture_server.py``), over
stdio or streamable HTTP. Fakes sit only at boundaries we do not own (the LLM
vendor, the MCP server); assertions read what the MCP server received, what the
next provider request carried, or what the Hermes process printed/persisted.
"""

from __future__ import annotations

import contextlib
import json
import os
import re
import signal
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterator

import pytest
import hermes_yaml as yaml

from tests.e2e.core.parity._helpers import hermes_argv, kill_tagged, tagged_pids, wait_until
from tests.fakes.fake_llm_provider import FakeLLMServer, Text, ToolCall, write_hermes_home

REPO_ROOT = Path(__file__).resolve().parents[4]
FIXTURE_SERVER = Path(__file__).with_name("mcp_fixture_server.py")
FINAL = "MCPE2E-TURN-COMPLETE"
TURN_TIMEOUT = 240.0

_SECRET_ENV_SUFFIXES = ("_API_KEY", "_TOKEN", "_SECRET", "_ACCESS_KEY")
_PASSTHROUGH_ENV = frozenset({"PATH", "LANG", "LANGUAGE", "USER", "LOGNAME", "SHELL", "TMPDIR", "TZ"})

__all__ = ["FINAL", "E2EHome", "HttpMcpServer", "KnownSymptom", "build_home", "stdio_server",
           "http_server_cfg", "script", "call_tool", "run_chat_q", "inbound", "calls_received", "tool_results",
           "tool_name", "tool_names", "payload", "symptom", "kill_tagged", "tagged_pids", "wait_until"]


def tool_name(server: str, tool: str) -> str:
    return f"mcp__{server}__{tool}"


# Known open bugs ----------------------------------------------------------------------------------


class KnownSymptom(Exception):
    """Raised ONLY by :func:`symptom`, the assertion that observes a KNOWN bug's symptom. It is the
    type ``known_gate(KNOWN, request.node.name, raises=KnownSymptom)`` accepts around that assertion,
    so every other failure in a KNOWN cell (a server that never starts, a timeout, a precondition, a
    crashed host, teardown) is a plain error and stays red."""


def symptom(ok: Any, message: str) -> None:
    """Assert the property a KNOWN bug breaks; its violation raises :class:`KnownSymptom`."""
    if not ok:
        raise KnownSymptom(message)


def payload(result: str) -> dict[str, Any]:
    """The JSON object inside a tool result's untrusted-content wrapper."""
    match = re.search(r"^\{.*\}$", result, re.M | re.S)
    assert match, f"no JSON payload in tool result: {result!r}"
    return json.loads(match.group(0))


def tool_names(body: dict[str, Any]) -> set[str]:
    """Function names in a provider request's ``tools[]``."""
    return {str((t.get("function") or {}).get("name")) for t in body.get("tools") or []}


@dataclass
class E2EHome:
    root: Path
    home: Path
    hermes_home: Path
    project: Path
    tag: str
    extra_env: dict[str, str] = field(default_factory=dict)

    def env(self, extra: dict[str, str] | None = None) -> dict[str, str]:
        """Hermetic child env: fake HOME (so no ``~/.hermes`` of the real user is reachable)."""
        import pwd  # the suite is Linux-gated

        real_root = Path(pwd.getpwuid(os.getuid()).pw_dir, ".hermes").resolve()  # windows-footgun: ok — module is skipif(not linux)
        fixture = self.hermes_home.resolve()
        assert fixture != real_root and fixture.parent != real_root / "profiles", fixture
        assert fixture == (self.home / ".hermes").resolve(), fixture
        env = {k: v for k, v in os.environ.items()
               if (k in _PASSTHROUGH_ENV or k.startswith("LC_")) and not k.endswith(_SECRET_ENV_SUFFIXES)}
        env.update({
            "HOME": str(self.home), "HERMES_HOME": str(self.hermes_home), "PYTHONPATH": str(REPO_ROOT),
            "PYTHONUNBUFFERED": "1", "NO_COLOR": "1", "TERM": "dumb",
            "PARITY_TREE_TAG": self.tag,  # orphan-scan tag inherited by the whole tree
            "HERMES_STATE_DB_GUARD_BYPASS": "1",  # child HOME is tmp_path by construction
        })
        env.update(self.extra_env)
        env.update(extra or {})
        return env

    @property
    def config_path(self) -> Path:
        return self.hermes_home / "config.yaml"

    def update_config(self, mutate: Callable[[dict], None]) -> None:
        cfg = yaml.safe_load(self.config_path.read_text(encoding="utf-8"))
        mutate(cfg)
        self.config_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")


def _select_test_dependencies(eh: E2EHome) -> None:
    from tests.e2e.core._pm_dependencies import select_test_dependencies

    select_test_dependencies(eh.hermes_home, REPO_ROOT)


def build_home(root: Path, base_url: str, *, mcp_servers: dict[str, dict] | None = None,
               extra: dict[str, Any] | None = None) -> E2EHome:
    home = root / "home"
    hermes_home = home / ".hermes"
    project = root / "project"
    for d in (hermes_home, project):
        d.mkdir(parents=True, exist_ok=True)
    eh = E2EHome(root=root, home=home, hermes_home=hermes_home, project=project, tag=uuid.uuid4().hex)
    write_hermes_home(hermes_home, base_url)
    cfg = yaml.safe_load(eh.config_path.read_text(encoding="utf-8"))
    cfg["mcp_servers"] = dict(mcp_servers or {})
    # Discovery must be complete before the first agent build (interactive surfaces
    # wait only ~1.5 s by default, by design); the join returns as soon as it finishes.
    cfg["mcp_discovery_timeout"] = 120
    cfg["mcp_single_query_discovery_timeout"] = 120
    cfg.setdefault("display", {})["compact"] = True
    cfg["updates"] = {"check": False}
    cfg.update(extra or {})
    eh.config_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    return eh


def stdio_server(name: str, log: Path, tag: str, **env: str) -> dict[str, Any]:
    """``mcp_servers`` entry for a stdio fixture server logging its inbound wire to ``log``."""
    return {
        "command": sys.executable,
        "args": [str(FIXTURE_SERVER)],
        "env": {"MCPE2E_LOG": str(log), "MCPE2E_NAME": name, "PARITY_TREE_TAG": tag,
                "PYTHONPATH": str(REPO_ROOT), **env},
        "connect_timeout": 60,
        "timeout": 60,
    }


def http_server_cfg(url: str, **extra: Any) -> dict[str, Any]:
    return {"url": url, "connect_timeout": 30, "timeout": 30, **extra}


class HttpMcpServer:
    """A streamable-HTTP fixture server subprocess that can be crashed and restarted on its port."""

    def __init__(self, root: Path, tag: str, **env: str) -> None:
        self.root = root
        self.tag = tag
        self.log = root / "http_inbound.jsonl"
        self.port_file = root / "http_port"
        self.env = env
        self.port = 0
        self.proc: subprocess.Popen | None = None
        self.stderr_path = root / "http_server.stderr"

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}/mcp"

    def start(self) -> "HttpMcpServer":
        with contextlib.suppress(FileNotFoundError):
            self.port_file.unlink()
        env = {k: v for k, v in os.environ.items() if k in _PASSTHROUGH_ENV}
        env.update({"PYTHONPATH": str(REPO_ROOT), "MCPE2E_TRANSPORT": "http", "MCPE2E_LOG": str(self.log),
                    "MCPE2E_PORT_FILE": str(self.port_file), "MCPE2E_PORT": str(self.port),
                    "PARITY_TREE_TAG": self.tag, **self.env})
        with open(self.stderr_path, "ab") as err:
            self.proc = subprocess.Popen([sys.executable, str(FIXTURE_SERVER)], env=env,
                                         stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=err)
        text = wait_until(lambda: self.port_file.exists() and self.port_file.read_text(encoding="utf-8").strip(), 60,
                          f"HTTP MCP server bind ({self.stderr_path})")
        self.port = int(text.split()[0])
        return self

    def wait_exit(self, timeout: float = 30.0) -> int:
        assert self.proc is not None
        return self.proc.wait(timeout=timeout)

    def stop(self) -> None:
        if self.proc is not None and self.proc.poll() is None:
            self.proc.send_signal(signal.SIGTERM)
            try:
                self.proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait(timeout=10)


# Scripted provider -----------------------------------------------------------------------------


def _tool_msgs_this_turn(body: dict[str, Any]) -> list[dict]:
    msgs = body.get("messages") or []
    last_user = max((i for i, m in enumerate(msgs) if m.get("role") == "user"), default=-1)
    return [m for m in msgs[last_user + 1:] if m.get("role") == "tool"]


def script(*calls: tuple[str, dict[str, Any] | str]) -> Callable[[dict[str, Any]], Any]:
    """Stateless per-turn script: issue ``calls[k]`` while k tool results exist this turn, then answer.

    Stateless (keyed on the request's own history) so a retried request cannot desynchronise it,
    and it restarts on every new user message (multi-turn sessions)."""

    def respond(record: dict[str, Any]):
        body = record["body"]
        done = len(_tool_msgs_this_turn(body))
        if done >= len(calls):
            return Text(FINAL)
        return call_tool(body, *calls[done])

    return respond


def call_tool(body: dict[str, Any], name: str, args: dict[str, Any] | str) -> ToolCall:
    """Call ``name`` directly when offered, else through the Tool Search ``tool_call`` bridge."""
    offered = tool_names(body)
    if name in offered or "tool_call" not in offered:
        return ToolCall(name, args)
    return ToolCall("tool_call", {"calls": [{"name": name, "arguments": args}]})


def run_chat_q(eh: E2EHome, prompt: str, *, timeout: float = TURN_TIMEOUT,
               env: dict[str, str] | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(hermes_argv("chat", "-q", prompt, "-Q"), cwd=eh.project, env=eh.env(env),
                          capture_output=True, text=True, timeout=timeout, stdin=subprocess.DEVNULL)


# Observation ------------------------------------------------------------------------------------


def inbound(log: Path) -> list[dict[str, Any]]:
    if not log.exists():
        return []
    return [json.loads(line)["msg"] for line in log.read_text(encoding="utf-8").splitlines() if line.strip()]


def calls_received(log: Path, tool: str) -> list[dict[str, Any]]:
    """``tools/call`` request params the server received for ``tool`` (bare MCP tool name)."""
    return [m.get("params") or {} for m in inbound(log)
            if isinstance(m, dict) and m.get("method") == "tools/call"
            and (m.get("params") or {}).get("name") == tool]


def _text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(p.get("text", "") for p in content if isinstance(p, dict))
    return ""


def tool_results(srv: FakeLLMServer) -> list[str]:
    """Tool-result contents in the LAST main request (what the model saw)."""
    mains = srv.main_requests()
    if not mains:
        return []
    return [_text(m.get("content")) for m in mains[-1].get("messages") or [] if m.get("role") == "tool"]


@contextlib.contextmanager
def provider(responder: Callable[[dict[str, Any]], Any]) -> Iterator[FakeLLMServer]:
    srv = FakeLLMServer(responder)
    srv.start()
    try:
        yield srv
    finally:
        srv.stop()

