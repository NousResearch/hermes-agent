"""Plugin-side helpers for the MCP + plugin suite: portable (Agent Plugins v1) packages on disk,
and a ``tui_gateway`` stdio host (the Ink TUI's backend) driven over JSON-RPC in an ``E2EHome``."""

from __future__ import annotations

import contextlib
import json
import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

from tests.e2e.core.mcp_plugins._helpers import FIXTURE_SERVER, TURN_TIMEOUT, E2EHome, tagged_pids
from tests.e2e.core.parity._drive_rpc import READY_TIMEOUT, RpcClient, StreamCapture
from tests.e2e.core.parity._helpers import terminate

PLUGIN_SCHEMA = "https://agent-plugins.org/schemas/1.0.0/plugin.schema.json"
MCP_SCHEMA = "https://agent-plugins.org/schemas/1.0.0/mcp.schema.json"


LAUNCHER = "run-fixture"


def portable_stdio(log: Path, tag: str, **env: str) -> dict[str, Any]:
    """A portable ``mcp.json`` stdio entry for the fixture server. v1 only allows a bare executable or a
    ``./`` in-root path as ``command``, so it points at the package's launcher script."""
    return {"type": "stdio", "command": f"./{LAUNCHER}", "args": [],
            "env": {"MCPE2E_LOG": str(log), "PARITY_TREE_TAG": tag, **env}}


def write_portable_plugin(eh: E2EHome, dirname: str, servers: dict[str, dict[str, Any]], *,
                          name: str | None = None, version: str = "1.0.0") -> Path:
    """``<HERMES_HOME>/plugins/<dirname>/`` with ``plugin.json`` (manifest ``name``), ``mcp.json`` and an
    in-root launcher that execs the fixture MCP server with this interpreter."""
    root = eh.hermes_home / "plugins" / dirname
    root.mkdir(parents=True, exist_ok=True)
    (root / "plugin.json").write_text(json.dumps(
        {"$schema": PLUGIN_SCHEMA, "name": name or dirname, "version": version}), encoding="utf-8")
    (root / "mcp.json").write_text(json.dumps({"$schema": MCP_SCHEMA, "mcpServers": servers}), encoding="utf-8")
    launcher = root / LAUNCHER
    launcher.write_text(f"#!/bin/sh\nexec '{sys.executable}' '{FIXTURE_SERVER}' \"$@\"\n", encoding="utf-8")
    launcher.chmod(0o755)
    return root


@dataclass
class TuiHost:
    proc: subprocess.Popen
    cap: StreamCapture
    rpc: RpcClient

    def new_session(self) -> str:
        return self.rpc.call("session.create", {}, timeout=READY_TIMEOUT)["session_id"]

    def turn(self, sid: str, text: str, timeout: float = TURN_TIMEOUT) -> str:
        """Submit one prompt and wait for THIS turn's ``message.complete`` (not an earlier turn's)."""
        seen = len(self.rpc.events)
        self.rpc.call("prompt.submit", {"session_id": sid, "text": text}, timeout=READY_TIMEOUT)
        deadline = time.monotonic() + timeout
        while True:
            for ev in self.rpc.events[seen:]:
                if ev.get("type") == "message.complete" and ev.get("session_id") == sid:
                    out = (ev.get("payload") or {}).get("text")
                    return out if isinstance(out, str) else json.dumps(out)
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise AssertionError(f"no message.complete for {sid} within {timeout}s:\n{self.cap.stderr[-2000:]}")
            self.rpc._pump(min(remaining, 0.5))  # test-harness sibling: same pump wait_event uses


@contextlib.contextmanager
def tui_host(eh: E2EHome, env: dict[str, str] | None = None) -> Iterator[TuiHost]:
    """``python -m tui_gateway.entry`` exactly as the Ink TUI spawns it, in the fake home."""
    proc = subprocess.Popen([sys.executable, "-m", "tui_gateway.entry"], cwd=eh.project, env=eh.env(env),
                            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            text=True, bufsize=1)
    cap = StreamCapture().start(proc)

    def send(line: str) -> None:
        assert proc.stdin is not None
        proc.stdin.write(line + "\n")
        proc.stdin.flush()

    try:
        rpc = RpcClient(send, cap.stdout_lines)
        rpc.wait_event("gateway.ready", timeout=READY_TIMEOUT)
        yield TuiHost(proc, cap, rpc)
    finally:
        if proc.poll() is None:
            if proc.stdin is not None and not proc.stdin.closed:
                with contextlib.suppress(OSError):
                    proc.stdin.close()
            try:
                proc.wait(timeout=60)
            except subprocess.TimeoutExpired:
                terminate(proc)
        reap_tagged(eh)


def reap_tagged(eh: E2EHome) -> None:
    """Cleanup: SIGKILL any process still carrying this home's unique tag (reparented orphans too)."""
    for pid in tagged_pids(eh.tag):
        with contextlib.suppress(OSError):
            os.kill(pid, signal.SIGKILL)
