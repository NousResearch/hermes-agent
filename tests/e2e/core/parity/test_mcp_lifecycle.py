"""MCP server lifecycle on the long-lived host (issue class C15).

Uses the tui_gateway stdio host (the Ink TUI's backend; the Desktop's WS backend
runs the same MCP runtime) with the real stdio MCP fixture:

* fast death: the MCP server crashes WHILE its call is in flight, leaving a helper
  that inherited its stdio (npx/uvx wrapper shape) (#81995). The call must fail
  as a tool error EXACTLY once (outcome uncertain: never silently replayed onto
  the respawned server), the host must survive, the turn complete well inside the configured MCP call
  timeout (bound 300 s: >10x the ~18 s observed idle, 12x under the 3600 s timeout), and a normal host
  shutdown afterwards still reaps the pipe holder.
* host crash: the host is SIGKILLed after a turn (no shutdown code runs at all).
  The MCP server AND its grandchild must still be reaped (death supervisor +
  process-group kill), because a crashed Desktop/TUI backend otherwise leaves a
  pile of MCP processes behind on every restart.

The parity matrix covers the normal-shutdown reap and the healthy call on every
entrypoint; this file covers the two failure transitions.
"""

from __future__ import annotations

import contextlib
import os
import signal
import sys
import time
import uuid
from pathlib import Path

import pytest

from tests.e2e.core.parity._drive_rpc import READY_TIMEOUT, RpcClient, spawn_tui_gateway
from tests.e2e.core.parity._helpers import (
    FINAL_ANSWER,
    build_parity_home,
    collect_observation,
    describe_pids,
    kill_tagged,
    mcp_pids,
    start_provider,
    terminate,
    wait_no_orphans,
    wait_until,
)

# Same rationale as test_entrypoint_parity: Linux /proc scans, and the bypass is
# for teardown of reparented orphans carrying this run's unique tag only.
pytestmark = [
    pytest.mark.skipif(not sys.platform.startswith("linux"), reason="process-tree checks use /proc"),
    pytest.mark.live_system_guard_bypass,
]

MCP_DIE_TOOL = "mcp__parity__parity_die"
MCP_CALL_TIMEOUT = 3600
FAST_DEATH_BOUND = 300.0
REAP_BOUND = 30.0


@contextlib.contextmanager
def tui_host(tmp_path: Path, *, tool: str | None = None, **home_kwargs):
    nonce = uuid.uuid4().hex[:8]
    srv = start_provider(nonce, tool) if tool else start_provider(nonce)
    ph = build_parity_home(tmp_path, srv.base_url, **home_kwargs)
    proc, cap = spawn_tui_gateway(ph)

    def send(line: str) -> None:
        assert proc.stdin is not None
        proc.stdin.write(line + "\n")
        proc.stdin.flush()

    try:
        rpc = RpcClient(send, cap.stdout_lines)
        rpc.wait_event("gateway.ready", timeout=READY_TIMEOUT)
        yield ph, srv, proc, cap, rpc
    finally:
        if proc.poll() is None:
            if proc.stdin is not None and not proc.stdin.closed:
                with contextlib.suppress(OSError):
                    proc.stdin.close()
            try:
                proc.wait(timeout=60)
            except Exception:
                terminate(proc)
        kill_tagged(ph)
        srv.stop()


def _alive(pid: int) -> bool:
    try:
        with open(f"/proc/{pid}/stat", "rb") as fh:
            return fh.read().rsplit(b")", 1)[1].split()[0] != b"Z"
    except OSError:
        return False


def _turn(rpc: RpcClient, timeout: float) -> str:
    sid = rpc.call("session.create", {}, timeout=READY_TIMEOUT)["session_id"]
    rpc.call("prompt.submit", {"session_id": sid, "text": "Use the parity tool, then report."}, timeout=READY_TIMEOUT)
    done = rpc.wait_event("message.complete", lambda e: e.get("session_id") == sid, timeout=timeout)
    return str((done.get("payload") or {}).get("text") or "")


def test_mcp_server_death_mid_call_fails_fast_and_the_turn_completes(tmp_path: Path) -> None:
    with tui_host(tmp_path, tool=MCP_DIE_TOOL, grandchild=False, death_tool=True,
                  mcp_timeout=MCP_CALL_TIMEOUT) as (ph, srv, proc, cap, rpc):
        started = time.monotonic()
        text = _turn(rpc, timeout=FAST_DEATH_BOUND)
        elapsed = time.monotonic() - started

        obs = collect_observation("tui_gateway", ph, srv, text)
        assert FINAL_ANSWER in text, f"turn did not complete after the MCP server died: {text[-300:]!r}"
        assert obs.tool_results, "the dead MCP call produced no tool result for the model"
        assert not any(ph.canaries["mcp"] in r for r in obs.tool_results), obs.tool_results
        assert elapsed < FAST_DEATH_BOUND, elapsed
        # The server really died mid-call (not a schema/dispatch error before the RPC).
        dying = mcp_pids(ph).get("dying_server") or []
        # Exactly once: a call that died mid-flight may have had side effects, so it
        # must surface as outcome-uncertain, never be silently replayed.
        assert len(dying) == 1, (
            f"parity_die ran {len(dying)}x (want exactly 1): {mcp_pids(ph)}; results {obs.tool_results}")
        wait_until(lambda: not _alive(dying[0]), REAP_BOUND, "fixture MCP server exit")
        assert proc.poll() is None, f"host died with the MCP server:\n{cap.stderr[-2000:]}"

        assert mcp_pids(ph).get("pipe_holder"), "fixture pipe holder never started"
        proc.stdin.close()
        proc.wait(timeout=60)
        survivors = wait_no_orphans(ph, timeout=REAP_BOUND)
        assert not survivors, f"dead server's pipe holder survived host shutdown: {describe_pids(survivors)}"


def test_host_crash_still_reaps_the_mcp_server_and_its_grandchild(tmp_path: Path) -> None:
    with tui_host(tmp_path) as (ph, srv, proc, cap, rpc):
        text = _turn(rpc, timeout=READY_TIMEOUT)
        assert FINAL_ANSWER in text, text[-300:]
        pids = mcp_pids(ph)
        assert pids.get("server") and pids.get("grandchild"), f"MCP tree never spawned: {pids}"

        os.kill(proc.pid, signal.SIGKILL)  # no atexit, no shutdown hook, no finally
        proc.wait(timeout=30)

        survivors = wait_no_orphans(ph, timeout=REAP_BOUND)
        assert not survivors, (
            f"MCP tree survived a host crash for {REAP_BOUND}s: {describe_pids(survivors)} (pid log {pids})")
