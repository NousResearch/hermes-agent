"""Native shell continuity through the real /api/host-terminal route, isolated from user state."""
import asyncio
import json
from pathlib import Path

import pytest
from fastapi import FastAPI
from starlette.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from hermes_cli import web_host_terminal, web_server, web_server_chat
from hermes_cli.web_routers.webapp import router
from hermes_cli.web_session_sockets import HOST_TERMINAL_SOCKET
from tests.waiting import async_wait_until


@pytest.fixture
def host_app(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    (home / "config.yaml").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(web_server.app.state, "ui_surface", "webapp", raising=False)
    monkeypatch.setattr(web_server.app.state, "bound_host", "localhost", raising=False)
    monkeypatch.setattr(web_server.app.state, "auth_required", False, raising=False)
    # Do not run login scripts from the real user's shell.
    from hermes_cli import web_host_terminal
    monkeypatch.setattr(web_host_terminal, "shell_spec", lambda: (["/bin/sh", "-i"], "sh"))
    app = FastAPI()
    app.include_router(router)
    from hermes_cli.web_host_terminal_sessions import host_terminal_lifespan
    app.router.lifespan_context = host_terminal_lifespan
    yield app


def url(extra=""):
    """A host-terminal URL with a fresh terminal ticket minted with this process's session token."""
    from hermes_cli.web_routers.webapp import SESSION_TOKEN_OWNER, mint_socket_ticket
    ticket = mint_socket_ticket(SESSION_TOKEN_OWNER, HOST_TERMINAL_SOCKET, session_token=web_server._SESSION_TOKEN)
    return f"ws://localhost/api/host-terminal?ticket={ticket}{extra}"


def metadata(ws):
    text = ws.receive_text()
    assert text.startswith(web_host_terminal.META_PREFIX)
    return json.loads(text.removeprefix(web_host_terminal.META_PREFIX))


def output_until(ws, marker):
    output = b""
    while marker not in output:
        output += ws.receive_bytes()
    return output


def output_field(ws, name):
    """The digits/hex a command printed as ``NAME=value`` (markers split in source survive echo)."""
    import re
    pattern = re.compile(re.escape(name) + rb"=([0-9a-f:]+)\r?\n")
    output = b""
    while (match := pattern.search(output)) is None:
        output += ws.receive_bytes()
    return match[1]


@pytest.mark.platforms("linux")
@pytest.mark.parametrize("named_launch", [False, True])
def test_shell_scopes_passthrough_after_eager_activation(
    host_app, tmp_path, monkeypatch, named_launch,
):
    from agent import secret_scope
    from hermes_constants import get_hermes_home_override
    from tools.terminal_scope import get_terminal_scope
    from tui_gateway import launch_profile_policy

    root = tmp_path / ".hermes"
    launch = root / "profiles" / "launch" if named_launch else root
    other = root / "profiles" / "other"
    passthrough = "WEBAPP_TEST_TOKEN"
    launch_only = "WEBAPP_LAUNCH_ONLY_TOKEN"
    other_only = "WEBAPP_OTHER_ONLY_TOKEN"
    for home in (launch, other):
        home.mkdir(parents=True, exist_ok=True)
        (home / "config.yaml").write_text(
            f"terminal:\n  env_passthrough: [{passthrough}, {launch_only}, {other_only}]\n",
            encoding="utf-8",
        )
    (other / ".env").write_text(
        f"{passthrough}=other-value\n{other_only}=other-only\n", encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_HOME", str(launch))
    monkeypatch.setenv(passthrough, "launch-value")
    monkeypatch.setenv(launch_only, "launch-only")
    monkeypatch.delenv(other_only, raising=False)
    monkeypatch.delenv("GATEWAY_MULTIPLEX_PROFILES", raising=False)
    monkeypatch.setattr(secret_scope, "_MULTIPLEX_ACTIVE", False)
    monkeypatch.setattr(launch_profile_policy, "_snapshot", None)
    assert launch_profile_policy.activate_multi_profile_hosting_eagerly()
    # A later secondary's process-env residue must not replace the frozen launch value.
    monkeypatch.setenv(passthrough, "ambient-poison")

    command = (
        "printf 'SCO%s=%s:%s:%s\\n' PE "
        f'"${{{passthrough}}}" "${{{launch_only}-missing}}" "${{{other_only}-missing}}"\n'
    ).encode()
    launch_name = "launch" if named_launch else "default"
    with TestClient(host_app, base_url="http://localhost") as client:
        for profile, expected in (
            ("current", b"launch-value:launch-only:missing"),
            ("other", b"other-value:missing:other-only"),
            (launch_name, b"launch-value:launch-only:missing"),
        ):
            with client.websocket_connect(url(f"&profile={profile}")) as ws:
                assert metadata(ws)["shell"] == "sh"
                ws.send_bytes(command)
                output = output_until(ws, b"SCOPE=")
                assert b"SCOPE=" + expected in output
                assert b"ambient-poison" not in output
            assert secret_scope.current_secret_scope() is None
            assert get_hermes_home_override() is None
            assert get_terminal_scope() is None
    # The fix must bind the caller, not disable the process-wide fail-closed guard.
    with pytest.raises(secret_scope.UnscopedSecretError):
        secret_scope.get_secret(passthrough)


@pytest.mark.platforms("linux")
def test_real_shell_survives_disconnect_and_explicit_close(host_app, tmp_path):
    import shlex
    import sys
    with TestClient(host_app, base_url="http://localhost") as client:
        with client.websocket_connect(url("&cwd=" + str(tmp_path))) as ws:
            first = metadata(ws)
            token = first["terminalId"]
            bridge = host_app.state.host_terminals._sessions[token].bridge
            ws.send_bytes(b"stty -echo; continuity=retained; printf 'PID=%s\\n' $$\n")
            output_until(ws, b"PID=")
            # Echo can contain the command: use a response marker split in source.
            ws.send_bytes(b"printf 'REA%s=%s\\n' DY $$\n")
            ready = output_until(ws, b"READY=")
            pid = ready.split(b"READY=")[-1].splitlines()[0].strip()
            program = 'import os,sys; print("CHILD="+str(os.getpid()), flush=True); print("INPUT="+repr(sys.stdin.readline()), flush=True)'
            ws.send_bytes(f"{shlex.quote(sys.executable)} -u -c {shlex.quote(program)}\n".encode())
            child = output_until(ws, b"CHILD=")
            child_pid = child.split(b"CHILD=")[-1].splitlines()[0].strip()
        with client.websocket_connect(url("&attach=" + token)) as ws:
            resumed = metadata(ws)
            assert resumed["reconnected"] is True
            assert resumed["terminalId"] == token
            assert b"CHILD=" + child_pid in output_until(ws, b"CHILD=" + child_pid)
            ws.send_bytes(b"foreground survived\n")
            assert b"INPUT='foreground survived\\n'" in output_until(ws, b"INPUT='foreground survived\\n'")
            ws.send_bytes(b"printf 'STATE=%s:%s\\n' \"$continuity\" $$\n")
            assert b"STATE=retained:" + pid in output_until(ws, b"STATE=retained:" + pid)
            program = 'import os,time; print("CLOSING="+str(os.getpid()), flush=True); time.sleep(120)'
            ws.send_bytes(f"{shlex.quote(sys.executable)} -u -c {shlex.quote(program)}\n".encode())
            running = output_until(ws, b"CLOSING=")
            foreground_pid = int(running.split(b"CLOSING=")[-1].splitlines()[0].strip())
        with client.websocket_connect(url("&attach=" + token + "&action=close")) as ws:
            assert metadata(ws)["closed"] is True
            with pytest.raises(WebSocketDisconnect) as closed:
                ws.receive_bytes()
            assert closed.value.code == 1000
            assert not bridge.is_alive()
            import psutil
            try:
                child_process = psutil.Process(foreground_pid)
            except psutil.NoSuchProcess:
                pass
            else:
                assert child_process.status() == psutil.STATUS_ZOMBIE
        with client.websocket_connect(url("&attach=" + token)) as ws:
            with pytest.raises(WebSocketDisconnect) as expired:
                ws.receive_bytes()
            assert expired.value.code == 4410


@pytest.mark.platforms("posix")
def test_paste_waits_for_a_foreground_program_that_is_not_reading(host_app, tmp_path, monkeypatch):
    """A paste bigger than the PTY input queue (about 1 KB on macOS, 15 KB on Linux)
    while ``sleep`` or ``npm install`` runs waits for the program, however long it
    takes: the shell survives and the program then reads every byte in order."""
    import hashlib
    import shlex
    import sys
    # Shrink the bridge's default write deadline (the chat TUI's 10 s) so a short
    # stall outlasts it: a host terminal that writes with the default loses its shell.
    bridge_write = web_server_chat.PtyBridge.write

    async def write(self, data, *, timeout=0.5, **options):
        return await bridge_write(self, data, timeout=timeout, **options)

    monkeypatch.setattr(web_server_chat.PtyBridge, "write", write)
    paste = b"".join(b"%05d %s\n" % (n, b"x" * 57) for n in range(1024))  # 64 KiB
    stall = 2  # longer than the shrunken default deadline
    program = (
        f"import hashlib,sys,time; print('BUS'+'Y=1', flush=True); time.sleep({stall}); "
        f"data = sys.stdin.buffer.read({len(paste)}); "
        "print('GOT=%d:%s' % (len(data), hashlib.sha256(data).hexdigest()), flush=True)")
    with TestClient(host_app, base_url="http://localhost") as client:
        with client.websocket_connect(url("&cwd=" + str(tmp_path))) as ws:
            metadata(ws)
            ws.send_bytes(b"stty -echo; printf 'SHE%s=%s\\n' LL $$\n")
            shell = output_field(ws, b"SHELL")
            ws.send_bytes(f"{shlex.quote(sys.executable)} -c {shlex.quote(program)}\n".encode())
            output_field(ws, b"BUSY")
            ws.send_bytes(paste)
            got = output_field(ws, b"GOT")
            assert got == b"%d:%s" % (len(paste), hashlib.sha256(paste).hexdigest().encode())
            ws.send_bytes(b"printf 'SHE%s=%s\\n' LL $$\n")
            assert output_field(ws, b"SHELL") == shell


@pytest.mark.platforms("linux")
@pytest.mark.parametrize("spawn_fails", [False, True])
def test_route_cancelled_spawn_is_owned_through_lifespan(host_app, monkeypatch, spawn_fails):
    import threading

    original_spawn = web_server_chat.PtyBridge.spawn
    entered, release, returned = threading.Event(), threading.Event(), threading.Event()
    closing = threading.Event()
    bridges = []
    releaser = None

    def spawn(argv, **kwargs):
        entered.set()
        assert release.wait(5)
        try:
            if spawn_fails:
                raise OSError("fork failed after disconnect")
            bridge = original_spawn(["/bin/sh", "-c", "sleep 120"], **kwargs)
            bridges.append(bridge)
            return bridge
        finally:
            returned.set()

    def release_on_shutdown():
        if closing.wait(3):
            release.set()

    monkeypatch.setattr(web_server_chat.PtyBridge, "spawn", spawn)
    try:
        with TestClient(host_app) as client:
            registry = host_app.state.host_terminals
            original_close_all = registry.close_all

            async def close_all():
                closing.set()
                await original_close_all()

            monkeypatch.setattr(registry, "close_all", close_all)
            with client.websocket_connect(url()):
                assert entered.wait(2)
            # TestClient cancels the request in its real AnyIO scope. It must
            # not wait for fork/exec or own the cleanup that shutdown awaits.
            assert not returned.is_set()
            releaser = threading.Thread(target=release_on_shutdown)
            releaser.start()
        assert closing.is_set() and returned.is_set()
        assert not registry._sessions and not registry.owners
        assert len(bridges) == (0 if spawn_fails else 1)
        assert all(not bridge.is_alive() for bridge in bridges)
    finally:
        release.set()
        if releaser is not None:
            releaser.join(3)
        for bridge in bridges:
            bridge.close()


class IdleBridge:
    def __init__(self):
        import threading
        self.closed = threading.Event()
        self.wrote = threading.Event()
        self.writes = []
        self.resizes = []

    def read(self, timeout):
        self.closed.wait(timeout)
        return None if self.closed.is_set() else b""

    async def write(self, data, *, timeout=10.0, fence=None):
        def record():
            self.writes.append(data)
            self.wrote.set()
            return True
        return record() if fence is None else await asyncio.to_thread(fence, record)

    def resize(self, cols, rows):
        self.resizes.append((cols, rows))

    def close(self):
        self.closed.set()

    def is_alive(self):
        return not self.closed.is_set()


@pytest.fixture
def fake_bridges(monkeypatch):
    bridges = []

    def spawn(*args, **kwargs):
        bridge = IdleBridge()
        bridges.append(bridge)
        return bridge

    monkeypatch.setattr(web_server_chat.PtyBridge, "spawn", spawn)
    return bridges


def rejection(client, request, expected):
    # Handles pre-accept gate failures and post-accept terminal refusals.
    with pytest.raises(WebSocketDisconnect) as error:
        with client.websocket_connect(request) as ws:
            ws.receive_text()
    assert error.value.code == expected


def ticket_url(user="owner", extra=""):
    """A host-terminal URL with a fresh terminal ticket for a signed-in *user*."""
    from hermes_cli.web_routers.webapp import mint_socket_ticket
    ticket = mint_socket_ticket({"user_id": user, "provider": "test-provider"}, HOST_TERMINAL_SOCKET)
    return f"ws://localhost/api/host-terminal?ticket={ticket}{extra}"


def test_identity_profile_deletion_and_auth_gate(host_app, fake_bridges, tmp_path, monkeypatch):
    from hermes_constants import mark_named_profile_deleted
    home = tmp_path / ".hermes"
    for name in ("alpha", "beta"):
        profile = home / "profiles" / name
        profile.mkdir(parents=True)
        (profile / "config.yaml").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(web_server.app.state, "auth_required", True)
    with TestClient(host_app) as client:
        with client.websocket_connect(ticket_url(extra="&profile=alpha")) as ws:
            token = metadata(ws)["terminalId"]
        rejection(client, ticket_url("other", "&profile=alpha&attach=" + token), 4403)
        rejection(client, ticket_url(extra="&profile=beta&attach=" + token), 4403)
        rejection(client, ticket_url("other", "&profile=alpha&attach=" + token + "&action=close"), 4403)
        rejection(client, ticket_url(extra="&profile=beta&attach=" + token + "&action=close"), 4403)
        rejection(client, url("&profile=alpha&attach=" + token), 4401)
        rejection(client, url("&profile=alpha&attach=" + token + "&action=close"), 4401)
        assert len(fake_bridges) == 1 and not fake_bridges[0].closed.is_set()
        with client.websocket_connect(ticket_url(extra="&profile=alpha&attach=" + token)) as ws:
            assert metadata(ws)["reconnected"]
        assert fake_bridges[0].writes == []  # never Ctrl-L into a host shell
        mark_named_profile_deleted(home / "profiles" / "alpha")
        rejection(client, ticket_url(extra="&profile=alpha&attach=" + token), 4410)
        rejection(client, ticket_url(extra="&profile=alpha&attach=" + token + "&action=close"), 4410)
        assert len(fake_bridges) == 1
        assert fake_bridges[0].closed.wait(3)  # live reaper retires a deleted profile's shells


@pytest.mark.platforms("linux")
def test_real_profile_a_b_a_and_process_exit_never_respawns(host_app, tmp_path):
    homes = {}
    for name in ("alpha", "beta"):
        homes[name] = tmp_path / ".hermes" / "profiles" / name
        homes[name].mkdir(parents=True)
        (homes[name] / "config.yaml").write_text("{}", encoding="utf-8")
    with TestClient(host_app) as client:
        tokens = {}
        for name in ("alpha", "beta", "alpha"):
            extra = "&profile=" + name
            if name in tokens:
                extra += "&attach=" + tokens[name]
            with client.websocket_connect(url(extra)) as ws:
                tokens[name] = metadata(ws)["terminalId"]
                ws.send_bytes(b"printf 'HO%s=%s\\n' ME \"$HERMES_HOME\"\n")
                assert b"HOME=" + str(homes[name]).encode() in output_until(ws, b"HOME=" + str(homes[name]).encode())
        token = tokens["alpha"]
        with client.websocket_connect(url("&profile=alpha&attach=" + token)) as ws:
            metadata(ws)
            ws.send_bytes(b"exit\n")
            with pytest.raises(WebSocketDisconnect) as error:
                while True:
                    ws.receive_bytes()
            assert error.value.code == 4410
        rejection(client, url("&profile=alpha&attach=" + token), 4410)
        assert set(host_app.state.host_terminals._sessions) <= set(tokens.values())


@pytest.mark.platforms("linux")
def test_real_shell_rejects_input_after_profile_deletion(host_app, tmp_path):
    import os
    import shlex
    import subprocess
    import sys

    profile = tmp_path / ".hermes" / "profiles" / "alpha"
    profile.mkdir(parents=True)
    (profile / "config.yaml").write_text("{}", encoding="utf-8")
    before, after = tmp_path / "before", tmp_path / "after"
    with TestClient(host_app) as client:
        with client.websocket_connect(url("&profile=alpha")) as ws:
            metadata(ws)
            ws.send_bytes((
                f"stty -echo; printf before > {shlex.quote(str(before))}; printf 'REA%s\\n' DY\n"
            ).encode())
            output_until(ws, b"READY")
            assert before.read_text(encoding="utf-8-sig") == "before"
            # A different process cannot notify the serving process's registry.
            subprocess.run([
                sys.executable, "-c",
                "import sys; from hermes_constants import mark_named_profile_deleted; "
                "mark_named_profile_deleted(sys.argv[1])",
                str(profile),
            ], env={**os.environ, "HOME": str(tmp_path)}, check=True, timeout=10)
            ws.send_bytes((
                f"printf executed > {shlex.quote(str(after))}; printf 'STA%s\\n' LE\n"
            ).encode())
            with pytest.raises(WebSocketDisconnect) as error:
                output_until(ws, b"STALE")
            assert error.value.code == 4410
        assert not after.exists()


def test_capacity_expiry_close_idempotence_and_lifespan(host_app, fake_bridges):
    with TestClient(host_app) as client:
        registry = host_app.state.host_terminals
        registry._max = 1
        with client.websocket_connect(url()) as ws:
            token = metadata(ws)["terminalId"]
            rejection(client, url(), 1013)
            assert not fake_bridges[0].closed.is_set()
        detached = registry._sessions[token].last_detached_at
        assert detached is not None
        registry._sessions[token].last_detached_at = detached - registry._ttl - 1
        rejection(client, url("&attach=" + token), 4410)
        assert len(fake_bridges) == 1
        # A fresh request may create, an expired attach never does.
        with client.websocket_connect(url()) as ws:
            next_token = metadata(ws)["terminalId"]
            assert next_token != token
        for _ in range(2):
            with client.websocket_connect(url("&attach=" + next_token + "&action=close")) as ws:
                assert metadata(ws)["closed"]
        with client.websocket_connect(url()) as ws:
            metadata(ws)
    assert all(bridge.closed.is_set() for bridge in fake_bridges)
    assert not registry._sessions and not registry.owners
    assert not hasattr(host_app.state, "host_terminals")


def test_simultaneous_viewer_supersedes_without_respawn(host_app, fake_bridges):
    with TestClient(host_app) as client:
        with client.websocket_connect(url()) as old:
            token = metadata(old)["terminalId"]
            assert old.receive_bytes() == b""  # explicit empty replay boundary
            with client.websocket_connect(url("&attach=" + token)) as new:
                assert metadata(new)["reconnected"]
                with pytest.raises(WebSocketDisconnect) as error:
                    old.receive_bytes()
                assert error.value.code == 4409
                new.send_bytes(b"current")
                assert fake_bridges[0].wrote.wait(3)
        assert len(fake_bridges) == 1
        assert fake_bridges[0].writes == [b"current"]


def test_ownership_checks_never_block_the_event_loop(host_app, fake_bridges, tmp_path, monkeypatch):
    """Ownership reads the profile's home and tombstone; on a slow disk that stalls
    every socket the process serves. Create, reattach, input and the reaper all
    check off-loop, and a deleted profile still loses its attached shell."""
    import asyncio
    from hermes_cli.web_host_terminal_sessions import HostOwner
    from hermes_constants import mark_named_profile_deleted

    profile = tmp_path / ".hermes" / "profiles" / "alpha"
    profile.mkdir(parents=True)
    (profile / "config.yaml").write_text("{}", encoding="utf-8")
    checks = []
    original_current = HostOwner.current

    def record():
        try:
            asyncio.get_running_loop()
            checks.append("loop")
        except RuntimeError:
            checks.append("thread")

    def current(owner):
        record()
        return original_current(owner)

    monkeypatch.setattr(HostOwner, "current", current)
    with TestClient(host_app) as client:
        with client.websocket_connect(url("&profile=alpha")) as ws:
            token = metadata(ws)["terminalId"]
        with client.websocket_connect(url("&profile=alpha&attach=" + token)) as ws:
            assert metadata(ws)["reconnected"]
            ws.send_bytes(b"input")
            assert fake_bridges[0].wrote.wait(3)
            mark_named_profile_deleted(profile)
            with pytest.raises(WebSocketDisconnect) as error:
                while True:
                    ws.receive_bytes()
            assert error.value.code == 4410
        assert fake_bridges[0].closed.wait(3)
        assert fake_bridges[0].writes == [b"input"]
    assert checks and set(checks) == {"thread"}


def test_input_frames_that_arrive_together_share_ownership_checks(tmp_path, monkeypatch):
    """A paste lands as many frames. Frames that arrived together share off-loop
    fenced writes, every byte lands in order, and a deleted home receives none."""
    from types import SimpleNamespace
    from hermes_cli.web_host_terminal_sessions import HostOwner, _pump_input

    frames = [f"line {i}\n".encode() for i in range(40)]
    checks = []
    original_current = HostOwner.current

    def current(owner):
        try:
            asyncio.get_running_loop()
            checks.append("loop")
        except RuntimeError:
            checks.append("thread")
        return original_current(owner)

    monkeypatch.setattr(HostOwner, "current", current)

    class Socket:
        def __init__(self):
            # Every frame is already buffered when the pump starts reading.
            self.inbound = [{"type": "websocket.receive", "bytes": frame} for frame in frames]
            self.inbound.append({"type": "websocket.disconnect"})

        async def receive(self):
            return self.inbound.pop(0)

    async def pump(home):
        checks.clear()
        written, removed = [], []
        # Not a named profile home: it stays owned while it exists.
        owner = HostOwner(("loopback", "session-token"), home, "sh", str(tmp_path))

        async def write(_ws, data, *, fence, **_bridge_options):
            return await asyncio.to_thread(fence, lambda: written.append(data) or True)

        async def remove(token):
            removed.append(token)

        ws = Socket()
        session = SimpleNamespace(_ws=ws, alive=True, write=write, resize=lambda *args, **kwargs: None)
        await _pump_input(ws, SimpleNamespace(remove=remove), "terminal", session, owner)
        return written, removed

    written, removed = asyncio.run(pump(tmp_path))
    assert b"".join(written) == b"".join(frames) and removed == []
    assert set(checks) == {"thread"} and len(checks) < len(frames)
    written, removed = asyncio.run(pump(tmp_path / "deleted"))
    assert written == [] and removed == ["terminal"]
    assert set(checks) == {"thread"}


def test_cross_process_deletion_closes_attached_shell(host_app, fake_bridges, tmp_path):
    import os
    import subprocess
    import sys
    profile = tmp_path / ".hermes" / "profiles" / "alpha"
    profile.mkdir(parents=True)
    (profile / "config.yaml").write_text("{}", encoding="utf-8")
    with TestClient(host_app) as client:
        with client.websocket_connect(url("&profile=alpha")) as ws:
            metadata(ws)
            assert ws.receive_bytes() == b""  # empty replay precedes deletion
            # A different process cannot notify the serving process's registry.
            subprocess.run([
                sys.executable, "-c",
                "import sys; from hermes_constants import mark_named_profile_deleted; "
                "mark_named_profile_deleted(sys.argv[1])",
                str(profile),
            ], env={**os.environ, "HOME": str(tmp_path)}, check=True, timeout=5)
            with pytest.raises(WebSocketDisconnect) as error:
                ws.receive_bytes()
            assert error.value.code == 4410
        assert fake_bridges[0].closed.wait(3)


@pytest.mark.parametrize("failure", ["metadata", "replay"])
def test_route_lost_initial_delivery_detaches_and_expires(host_app, fake_bridges, monkeypatch, failure):
    from starlette.websockets import WebSocket
    from hermes_cli.pty_session import PtySession
    original_start = PtySession.start

    async def start(session):
        session.buffer.append(b"startup output")
        await original_start(session)

    async def broken(*args, **kwargs):
        raise RuntimeError("transport lost")

    monkeypatch.setattr(PtySession, "start", start)
    monkeypatch.setattr(WebSocket, "send_text" if failure == "metadata" else "send_bytes", broken)
    with TestClient(host_app) as client:
        with client.websocket_connect(url()) as ws:
            if failure == "replay":
                metadata(ws)
            with pytest.raises(WebSocketDisconnect) as error:
                ws.receive_bytes()
            assert error.value.code == 1001  # reconnect; the shell is retained
        registry = host_app.state.host_terminals
        session = next(iter(registry._sessions.values()))
        assert not session.attached and session.last_detached_at is not None
        assert client.portal is not None
        client.portal.call(registry.reap_idle, session.last_detached_at + registry._ttl + 1)
        assert not registry._sessions and fake_bridges[0].closed.is_set()


def test_host_policy_and_missing_profile_do_not_spawn(host_app, fake_bridges, monkeypatch):
    with TestClient(host_app) as client:
        rejection(client, url("&profile=missing"), 4410)
        rejection(client, url("&attach=" + "x" * 43), 4410)
        monkeypatch.setattr(web_server.app.state, "ui_surface", "dashboard")
        for extra in ("", "&attach=" + "x" * 43, "&attach=" + "x" * 43 + "&action=close"):
            rejection(client, url(extra), 4403)
        assert not fake_bridges
