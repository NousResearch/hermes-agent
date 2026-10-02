"""The display websocket dials a sandbox-hosted screen under its profile's FULL runtime scope.

The Desktop's backend serves every profile from one process. A Docker-backed profile served beside a
local launch profile must relay into ITS container: with only HERMES_HOME bound, ``TERMINAL_*`` stayed
the launch profile's (local), the sandbox was looked up under a local key, and the live screen never
connected while every ``display.*`` RPC (``_profile_scoped``) saw it running.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from starlette.websockets import WebSocketDisconnect

import tui_gateway.server as server
from hermes_cli.dashboard_auth import ws_tickets
from tools import terminal_tool as tt
from tools.bot_desktop import lease, sandbox_host
from tui_gateway import launch_profile_policy as lpp

pytestmark = pytest.mark.platforms("posix")  # the stand-in docker client is a shell script

BANNER = b"RFB 003.008\n"
CONTAINER = "b-container"


@pytest.fixture
def served(tmp_path, monkeypatch):
    """Launch profile ``default`` on the local backend + profile ``b`` on Docker, one multiplexed process,
    b's screen up in its container (registered the way ``display.observe`` re-attaches it)."""
    from tools.environments.docker import DockerEnvironment

    root = tmp_path / ".hermes"
    b = root / "profiles" / "b"
    b.mkdir(parents=True)
    (root / "config.yaml").write_text("terminal:\n  backend: local\n", encoding="utf-8")
    (b / "config.yaml").write_text("terminal:\n  backend: docker\n", encoding="utf-8")
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(root))
    monkeypatch.setenv("TERMINAL_ENV", "local")  # the launch process bridged its own config at startup
    monkeypatch.setattr(server, "_hermes_home", root)
    monkeypatch.setattr(server, "_served_profile_homes", set())
    monkeypatch.setattr(lpp, "_snapshot", None)
    monkeypatch.setattr("agent.secret_scope._MULTIPLEX_ACTIVE", False)
    lpp.activate_multi_profile_hosting()

    # Stand-in docker client: `inspect` says running; `exec` records who it dialled, then IS the relay.
    calls = tmp_path / "docker-calls"
    docker = tmp_path / "docker"
    docker.write_text(
        "#!/bin/sh\n"
        'if [ "$1" = inspect ]; then echo true; exit 0; fi\n'
        f'echo "$1 $2 $3 $4 $5 ${{8##* }}" >> "{calls}"\n'
        "printf 'RFB 003.008\\n'\n"
        "exec cat\n",
        encoding="utf-8",
    )
    docker.chmod(0o755)
    env = DockerEnvironment.__new__(DockerEnvironment)
    env._container_id, env._docker_exe, env._bd_desktop_user = CONTAINER, str(docker), "pn"

    monkeypatch.setattr(tt, "_last_activity", {})
    monkeypatch.setattr(sandbox_host, "_ALIVE_CACHE", {})
    with server._session_profile_runtime_scope({"profile_home": str(b)}, hydrate_secrets=False):
        monkeypatch.setattr(tt, "_active_environments", {tt._resolve_container_task_id(None): env})
        rdir = sandbox_host._remote_dir(env, "b")
        sandbox_host._record(env, rdir, "b", {"DISPLAY": ":20"})
    lease._reset_for_tests()
    yield root, b, calls, rdir
    lease._reset_for_tests()


@pytest.fixture
def client(monkeypatch):
    from starlette.testclient import TestClient

    from hermes_cli import web_server

    monkeypatch.setattr(web_server.app.state, "auth_required", False, raising=False)
    monkeypatch.setattr(web_server.app.state, "bound_host", None, raising=False)
    ws_tickets._reset_for_tests()
    c = TestClient(web_server.app)
    try:
        yield c
    finally:
        c.close()
        ws_tickets._reset_for_tests()


def _connect(client, home: Path):
    """What the renderer does with ``display.observe``'s ticket for ``home``."""
    ticket = ws_tickets.mint_ticket(user_id="display:v", provider="bot-desktop",
                                    extra={"hermes_home": str(home), "viewer_id": "v"})
    return client.websocket_connect(f"/api/display/ws?display_ticket={ticket}")


def test_docker_profile_screen_relays_from_its_own_container_beside_a_local_launch_profile(served, client):
    root, b, calls, rdir = served
    for home in (b, root, b):
        with _connect(client, home) as conn:
            if home == b:
                assert conn.receive_bytes() == BANNER
            else:  # the launch profile has no screen of its own; b's must not leak into it
                with pytest.raises(WebSocketDisconnect) as exc:
                    conn.receive_bytes()
                assert exc.value.code == 4001
    assert calls.read_text(encoding="utf-8").splitlines() == [f"exec -i -u pn {CONTAINER} {rdir}/rfb.sock"] * 2


def test_sandbox_environment_gone_since_observe_closes_as_desktop_gone(served, client, monkeypatch):
    """The container still runs (its marker is alive) but no environment is registered for it any more:
    the renderer must get the 4001 "screen is gone" close, not a crashed handler."""
    _, b, calls, _ = served
    monkeypatch.setattr(tt, "_active_environments", {})
    with _connect(client, b) as conn:
        with pytest.raises(WebSocketDisconnect) as exc:
            conn.receive_bytes()
    assert exc.value.code == 4001
    assert not calls.exists()
