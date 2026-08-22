"""LIVE Windows E2E for the gateway control socket named-pipe transport.

Runs ONLY on a real Windows host (the on-demand ``windows-venv-e2e.yml``
lane). Spawns a REAL child process that binds the REAL named pipe via the
proactor event loop with the DEFAULT verb handlers, then drives the real
sync client and the real fleet consumers against it — no mocks anywhere.

Proves, on windows-latest:
  1. `GatewayControlServer` binds ``\\\\.\\pipe\\hermes-gateway-<hash>`` via
     ``loop.start_serving_pipe`` and answers ``identify``/``status``.
  2. The sync client's pipe transport (open/write/read/busy-retry) works
     against a live server and returns the child's true pid + code identity.
  3. ``collect_fleet_versions()`` prefers the socket (``source: socket``).
  4. After the server process is force-killed, the client returns None
     (FileNotFoundError on the pipe — no stale-file hazard on Windows) and
     consumers fall back to the state-file/scan layer.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(
    sys.platform != "win32", reason="live Windows named-pipe E2E"
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]

_CHILD_CODE = r"""
import asyncio, os, sys
sys.path.insert(0, sys.argv[1])
os.environ["HERMES_HOME"] = sys.argv[2]
from gateway.control_socket import GatewayControlServer

async def main():
    server = GatewayControlServer()
    ok = await server.start()
    print("SERVER_STARTED" if ok else "SERVER_FAILED", flush=True)
    if not ok:
        return
    await asyncio.sleep(120)

asyncio.run(main())
"""


@pytest.fixture()
def live_server(tmp_path: Path):
    home = tmp_path / ".hermes"
    home.mkdir()
    proc = subprocess.Popen(
        [sys.executable, "-c", _CHILD_CODE, str(PROJECT_ROOT), str(home)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        cwd=str(PROJECT_ROOT),
    )
    line = proc.stdout.readline().strip()
    if line != "SERVER_STARTED":
        err = proc.stderr.read() if proc.poll() is not None else ""
        proc.kill()
        pytest.fail(f"pipe server child failed to start: {line!r} {err}")
    yield proc, home
    if proc.poll() is None:
        proc.kill()
        proc.wait()


def test_named_pipe_identify_status_and_fleet_consumer(live_server, monkeypatch):
    proc, home = live_server
    from gateway.control_socket import identify_gateway, query_gateway_control

    ident = identify_gateway(home, timeout=5.0)
    assert ident is not None, "identify returned None against a live pipe server"
    assert ident["pid"] == proc.pid
    assert ident["protocol"] == 1
    assert ident["kind"] == "hermes-gateway"
    assert ident["supervisor"] in {"systemd", "launchd", "desktop", "external", "manual"}

    status = query_gateway_control(home, "status", timeout=5.0)
    assert status is not None
    assert status["answering_pid"] == proc.pid

    # Real fleet consumer prefers the pipe
    import hermes_cli.update_receipt as ur

    monkeypatch.setattr(
        "hermes_cli.build_info.get_code_identity",
        lambda refresh=False: {"sha": ident.get("code_sha") or "X", "version": "t"},
    )
    monkeypatch.setattr("hermes_cli.profiles._get_default_hermes_home", lambda: home)
    monkeypatch.setattr(
        "hermes_cli.profiles._get_profiles_root", lambda: home / "no-profiles"
    )
    fleet = ur.collect_fleet_versions()
    assert len(fleet) == 1, fleet
    assert fleet[0]["source"] == "socket"
    assert fleet[0]["pid"] == proc.pid


def test_pipe_gone_after_kill_falls_back(live_server, monkeypatch):
    proc, home = live_server
    from gateway.control_socket import identify_gateway

    assert identify_gateway(home, timeout=5.0) is not None
    proc.kill()
    proc.wait()
    time.sleep(0.5)

    assert identify_gateway(home, timeout=2.0) is None

    # Consumer falls back to the state file (live pid = this test process)
    import hermes_cli.update_receipt as ur

    (home / "gateway_state.json").write_text(
        json.dumps(
            {"pid": os.getpid(), "code_sha": "OLD", "kind": "hermes-gateway"}
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "hermes_cli.build_info.get_code_identity",
        lambda refresh=False: {"sha": "NEW", "version": "t"},
    )
    monkeypatch.setattr("hermes_cli.profiles._get_default_hermes_home", lambda: home)
    monkeypatch.setattr(
        "hermes_cli.profiles._get_profiles_root", lambda: home / "no-profiles"
    )
    fleet = ur.collect_fleet_versions()
    assert len(fleet) == 1, fleet
    assert "source" not in fleet[0]
    assert fleet[0]["state"] == "stale"
