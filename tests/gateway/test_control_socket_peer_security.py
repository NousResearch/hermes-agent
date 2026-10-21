"""Control-socket trust boundary: server identity, home writability, pointer mode, fallback root."""
import os
import socket
from pathlib import Path

import pytest

pytestmark = pytest.mark.platforms("linux")  # SO_PEERCRED / POSIX ACL / AF_UNIX semantics


def test_symlinked_home_still_authenticates_same_uid_peers(tmp_path):
    from gateway.control_socket import GatewayControlServer
    real = tmp_path / "real"
    real.mkdir(mode=0o700)
    (tmp_path / "link").symlink_to(real)
    ours, _theirs = socket.socketpair(socket.AF_UNIX)

    class Writer:
        def get_extra_info(self, _key):
            return ours
    assert GatewayControlServer(tmp_path / "link")._posix_peer_subject(Writer()) == f"uid:{os.getuid()}"


def test_restart_heals_a_world_readable_pointer_left_by_an_older_gateway(tmp_path):
    import asyncio
    from gateway.control_socket import GatewayControlServer
    from hermes_cli.gateway_runtime_discovery import _socket_path
    home = tmp_path / ("p" * 90) / ".hermes"
    home.mkdir(parents=True, mode=0o700)
    pointer = home / "gateway.sock.path"
    pointer.write_text("/old/control.sock")
    pointer.chmod(0o644)  # main wrote it with write_text under umask 022; a crash leaves it

    async def run():
        server = GatewayControlServer(home)
        assert await server.start()
        try:
            assert _socket_path(home).name == "control.sock"
        finally:
            await server.stop()
    asyncio.run(run())


def test_group_bits_that_are_an_acl_mask_never_count_as_a_private_group(tmp_path, monkeypatch):
    # With a POSIX ACL (setfacl u:nobody:rwx) the stat group bits are the mask: a 0770 home whose
    # group is the owner's private one may still grant another named user write.
    import grp
    import pwd
    from gateway.control_socket import GatewayControlServer
    from hermes_cli.gateway_runtime_discovery import DiscoveryError, _socket_path
    home = tmp_path / "home"
    home.mkdir()
    home.chmod(0o770)
    user = pwd.getpwuid(os.getuid())
    monkeypatch.setattr(pwd, "getpwall", lambda: [user])
    monkeypatch.setattr(grp, "getgrgid", lambda _gid: grp.struct_group((user.pw_name, "x", user.pw_gid, [])))
    monkeypatch.setattr(os, "listxattr", lambda *_a, **_k: ["system.posix_acl_access"])
    with pytest.raises(DiscoveryError, match="unsafe_control_permissions"):
        _socket_path(home)
    ours, _theirs = socket.socketpair(socket.AF_UNIX)

    class Writer:
        def get_extra_info(self, _key):
            return ours
    assert GatewayControlServer(home)._posix_peer_subject(Writer()) is None


def test_clients_refuse_a_control_listener_run_by_another_uid(tmp_path, monkeypatch):
    # lstat validates the path, but connect() follows a symlink swapped in afterwards: the
    # client must authenticate the listening process itself (SO_PEERCRED / getpeereid).
    import asyncio
    import inspect
    from gateway import control_socket, session_hosted_transport
    from hermes_cli import gateway_client
    from hermes_cli import gateway_runtime_discovery as discovery
    home = tmp_path / "home"
    home.mkdir(mode=0o700)
    real_uid = os.getuid()

    async def run():
        server = control_socket.GatewayControlServer(home, verb_handlers={"identify": lambda: {"pid": 1}})
        assert await server.start()
        try:
            path = discovery._socket_path(home)
            monkeypatch.setattr(discovery.os, "getuid", lambda: real_uid + 1)
            monkeypatch.setattr(discovery, "_socket_path", lambda _home: path)
            with pytest.raises(discovery.DiscoveryError, match="unsafe_control_peer"):
                await asyncio.to_thread(discovery.query_identify, home, timeout=2)
        finally:
            monkeypatch.undo()
            await server.stop()
    asyncio.run(run())
    import importlib.util
    spec = importlib.util.spec_from_file_location("tui_bootstrap", "ui-tui/scripts/gateway_bootstrap.py")
    tui_bootstrap = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tui_bootstrap)
    for client in (gateway_client._session_ticket, session_hosted_transport.owner_request, tui_bootstrap.bootstrap):
        source = inspect.getsource(client)
        assert "connect_private(" in source and "AF_UNIX" not in source, client.__qualname__


def test_long_home_survives_a_squatted_shared_fallback_directory(tmp_path, monkeypatch):
    # Any local user can pre-create the predictable shared-temp name; that must not stop the
    # gateway from publishing its control socket (the bootstrap treats that as fatal).
    import asyncio
    from gateway.control_socket import GatewayControlServer, resolve_server_socket_path
    from hermes_cli.gateway_runtime_discovery import _socket_path
    monkeypatch.delenv("XDG_RUNTIME_DIR", raising=False)
    home = tmp_path / ("p" * 90) / ".hermes"
    home.mkdir(parents=True, mode=0o700)
    squatted = resolve_server_socket_path(home)[0].parent
    squatted.write_text("not yours")

    async def run():
        server = GatewayControlServer(home, verb_handlers={"identify": lambda: {"pid": 1}})
        assert await server.start()
        try:
            assert _socket_path(home).parent.parent != squatted.parent
        finally:
            await server.stop()
    try:
        asyncio.run(run())
    finally:
        squatted.unlink()


def test_long_home_prefers_the_private_runtime_dir(tmp_path, monkeypatch):
    import tempfile
    from gateway.control_socket import resolve_server_socket_path
    runtime = Path(tempfile.mkdtemp(prefix="xr-"))
    try:
        monkeypatch.setenv("XDG_RUNTIME_DIR", str(runtime))
        home = tmp_path / ("p" * 90) / ".hermes"
        home.mkdir(parents=True, mode=0o700)
        assert resolve_server_socket_path(home)[0].parent.parent == runtime
        runtime.chmod(0o755)  # not a private runtime dir: never trusted
        assert resolve_server_socket_path(home)[0].parent.parent != runtime
    finally:
        runtime.rmdir()
