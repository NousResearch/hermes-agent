"""Saved SSH terminals use the SSH shell transport, not Webapp on remote serve (#150864)."""
from types import SimpleNamespace
from urllib.parse import urlencode
import subprocess

import pytest
from starlette.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from tests.hermes_cli.test_web_host_terminal_persistence import metadata
from tests.hermes_cli.test_webapp_socket_tickets import _SESSION, serve_webapp


@pytest.fixture
def ssh_webapp(tmp_path, monkeypatch):
    from hermes_cli import gateway_connections as store
    from hermes_cli import gateway_connections_ssh as ssh
    from hermes_cli import webapp_relay_target as reach
    from hermes_cli.gateway_connections_ssh_transport import SshEndpoint
    from hermes_cli.web_routers import webapp_relay_proxy as relay

    served = serve_webapp(tmp_path, monkeypatch)
    served.host.include_router(relay.router)
    served.host.state.ui_surface = "webapp"
    served.host.state.bound_host = "localhost"
    served.host.state.auth_required = False
    entry, _ = store.save_connection({"id": "remote-shell", "kind": "ssh", "label": "Remote shell",
                                     "host": "remote.example", "user": "operator"})
    endpoint = SshEndpoint(host=entry["host"], user=entry["user"], control_path=str(tmp_path / "control"))
    tunnel = SimpleNamespace(ssh=SimpleNamespace(endpoint=endpoint, ssh_binary="/usr/bin/ssh"),
                             fingerprint=ssh._fingerprint(entry))
    monkeypatch.setattr(ssh, "_tunnels", {entry["id"]: tunnel})
    monkeypatch.setattr(ssh, "ensure_tunnel", lambda *_args, **_kwargs: SimpleNamespace(
        base_url="http://remote.example:8000", token="owned-serve-token"))

    async def no_remote_webapp(_target, _path):
        raise reach.Unreachable("Remote shell did not issue a host-terminal ticket (404).")

    monkeypatch.setattr(reach, "socket_credential", no_remote_webapp)
    served.entry, served.tunnel = entry, tunnel
    return served


def _open(served, client, *, connection="remote-shell", profile="remote-profile", **params):
    ticket = served.client.post("/api/webapp/socket-ticket?socket=relay", headers=_SESSION)
    assert ticket.status_code == 200
    query = urlencode({"relay_ticket": ticket.json()["ticket"], "profile": profile, **params})
    return client.websocket_connect(f"ws://localhost/api/relay/{connection}/api/host-terminal?{query}",
                                    headers={"Origin": "http://localhost"})


def test_saved_ssh_terminal_starts_and_reattaches_without_remote_webapp(ssh_webapp):
    served = ssh_webapp
    with TestClient(served.host, base_url="http://localhost") as client:
        with _open(served, client) as ws:
            first = metadata(ws)
            assert first["shell"] == "ssh"
            assert not first["reconnected"]
        with _open(served, client, attach=first["terminalId"]) as ws:
            assert metadata(ws)["reconnected"]
        with _open(served, client, attach=first["terminalId"], action="close") as ws:
            assert metadata(ws)["closed"]
    assert len(served.shells) == 1


def test_saved_ssh_terminal_does_not_attach_under_another_remote_profile(ssh_webapp):
    served = ssh_webapp
    with TestClient(served.host, base_url="http://localhost") as client:
        with _open(served, client) as ws:
            token = metadata(ws)["terminalId"]
        with pytest.raises(WebSocketDisconnect) as closed:
            with _open(served, client, attach=token, profile="different-profile") as ws:
                metadata(ws)
        assert closed.value.code == 4403


def _fish_cd_target(command: str) -> str:
    """The ``cd`` argument as fish reads it. Unlike POSIX, fish decodes ``\\\\`` and ``\\'`` inside '…'."""
    escapes = {None: None, "'": "\\'", '"': '\\"$'}
    assert command.startswith("cd ")
    out, i, quote = [], 3, None
    while i < len(command) and (quote or command[i] != " "):
        ch = command[i]
        if ch in "'\"" and quote in (None, ch):
            quote = None if quote else ch
        elif ch == "\\" and (quote is None or command[i + 1] in escapes[quote]):
            i += 1
            out.append(command[i])
        else:
            out.append(ch)
        i += 1
    return "".join(out)


@pytest.mark.platforms("posix")
def test_ssh_spawn_uses_the_saved_master_and_quotes_remote_cwd(ssh_webapp, monkeypatch, tmp_path):
    from hermes_cli import web_server_chat

    seen = []
    spawn = web_server_chat.PtyBridge.spawn

    def record(argv, **kwargs):
        seen.append((argv, kwargs))
        return spawn(argv, **kwargs)

    monkeypatch.setattr(web_server_chat.PtyBridge, "spawn", record)
    remote = tmp_path / "work's \\\\dir\\; not a local command"
    remote.mkdir()
    cwd = str(remote)
    with TestClient(ssh_webapp.host) as client:
        with _open(ssh_webapp, client, cwd=cwd) as ws:
            assert metadata(ws)["cwd"] == cwd
    argv, options = seen[0]
    assert argv[:2] == ["/usr/bin/ssh", "-tt"]
    assert argv[-2] == "operator@remote.example"
    assert argv[-1].endswith(' 2>/dev/null; exec "$SHELL" -l')
    assert _fish_cd_target(argv[-1]) == cwd  # a fish login shell reaches the same directory
    shell = tmp_path / "print-pwd"
    shell.write_text("#!/bin/sh\npwd -P\n")
    shell.chmod(0o755)
    landed = subprocess.run(["sh", "-c", argv[-1]], capture_output=True, text=True, check=False,
                            env={"SHELL": str(shell), "PATH": "/usr/bin:/bin"})
    assert landed.stdout.strip() == str(remote.resolve())
    assert f"ControlPath={ssh_webapp.tunnel.ssh.endpoint.control_path}" in argv
    assert "BatchMode=yes" in argv and "StrictHostKeyChecking=accept-new" in argv
    assert options["cwd"] != cwd


@pytest.mark.parametrize("mutation", ["remove", "host", "user", "port", "keyPath", "remoteProfile"])
def test_detached_ssh_shell_is_retired_when_its_saved_target_changes(ssh_webapp, mutation):
    from hermes_cli import gateway_connections as store

    served = ssh_webapp
    with TestClient(served.host) as client:
        with _open(served, client) as ws:
            token = metadata(ws)["terminalId"]
        if mutation == "remove":
            store.remove_connection(served.entry["id"])
        else:
            store.save_connection({**served.entry, mutation: 2222 if mutation == "port" else "changed"})
        registry = served.host.state.host_terminals
        assert client.portal is not None
        client.portal.call(registry.reap_idle)
        assert token not in registry._sessions
        assert not served.shells[0].is_alive()


def test_ssh_shell_cannot_be_attached_as_a_local_shell_or_another_saved_connection(ssh_webapp):
    from hermes_cli import gateway_connections as store
    from tests.hermes_cli.test_webapp_socket_tickets import _mint

    served = ssh_webapp
    store.save_connection({"id": "other", "kind": "ssh", "label": "Other", "host": "another.example"})
    with TestClient(served.host) as client:
        with _open(served, client) as ws:
            token = metadata(ws)["terminalId"]
        local_ticket = _mint(served.client, "host-terminal")
        for context in (client.websocket_connect(f"ws://localhost/api/host-terminal?ticket={local_ticket}&attach={token}"),
                        _open(served, client, connection="other", attach=token)):
            with pytest.raises(WebSocketDisconnect) as closed:
                with context as ws:
                    metadata(ws)
            assert closed.value.code == 4403
    assert len(served.shells) == 1


def test_cold_ssh_dial_does_not_hold_up_other_new_terminals(ssh_webapp, monkeypatch):
    """A cold connect can take minutes; it must not run under the registry's spawn lock."""
    import threading

    from hermes_cli import gateway_connections_ssh as ssh
    from tests.hermes_cli.test_webapp_socket_tickets import _mint

    dialing, release = threading.Event(), threading.Event()
    calls = []

    def cold_dial(*_args, **_kwargs):
        calls.append(None)
        if len(calls) == 1:
            dialing.set()
            release.wait(30)

    monkeypatch.setattr(ssh, "ensure_tunnel", cold_dial)
    served = ssh_webapp
    timer = threading.Timer(10, release.set)  # unblocks the base, where the local open waits on the dial
    timer.start()
    try:
        with TestClient(served.host) as client, _open(served, client) as remote:
            assert dialing.wait(10)
            local_ticket = _mint(served.client, "host-terminal")
            with client.websocket_connect(f"ws://localhost/api/host-terminal?ticket={local_ticket}") as local:
                assert metadata(local)["shell"] != "ssh"
                assert not release.is_set(), "the local terminal waited for the SSH dial"
            release.set()
            assert metadata(remote)["shell"] == "ssh"
    finally:
        timer.cancel()
        release.set()
    assert len(served.shells) == 2


@pytest.mark.parametrize("surface", ["serve", "dashboard", "gateway"])
def test_ssh_terminal_does_not_enable_shells_on_other_surfaces(ssh_webapp, monkeypatch, surface):
    served = ssh_webapp
    with TestClient(served.host) as client:
        context = _open(served, client)
        monkeypatch.setattr(served.server.app.state, "ui_surface", surface)
        with pytest.raises(WebSocketDisconnect) as closed:
            with context as ws:
                metadata(ws)
        assert closed.value.code == 4403
    assert not served.shells


def test_ssh_terminal_keeps_relay_ticket_origin_and_identity_checks(ssh_webapp):
    from hermes_cli.web_routers.webapp import mint_socket_ticket
    from hermes_cli.web_session_sockets import RELAY_SOCKET
    from tests.hermes_cli.test_webapp_socket_tickets import _mint, _TOKEN

    served = ssh_webapp
    with TestClient(served.host) as client:
        with _open(served, client) as ws:
            token = metadata(ws)["terminalId"]
        other_identity = mint_socket_ticket({"user_id": "other-user", "provider": "test"}, RELAY_SOCKET,
                                            session_token=_TOKEN)
        url = f"ws://localhost/api/relay/remote-shell/api/host-terminal?profile=remote-profile&attach={token}"
        with pytest.raises(WebSocketDisconnect) as closed:
            with client.websocket_connect(f"{url}&relay_ticket={other_identity}",
                                          headers={"Origin": "http://localhost"}) as ws:
                metadata(ws)
        assert closed.value.code == 4403

        for ticket, origin in (("", "http://localhost"), (_mint(served.client, "host-terminal"), "http://localhost"),
                               (_mint(served.client, "relay"), "https://other.example"),
                               (other_identity, "http://localhost")):
            with pytest.raises(WebSocketDisconnect) as closed:
                with client.websocket_connect(f"{url}&relay_ticket={ticket}", headers={"Origin": origin}) as ws:
                    metadata(ws)
            assert closed.value.code == 4401
    assert len(served.shells) == 1
