"""Control listeners stay private when a shared filesystem cannot chmod sockets."""

import asyncio
import errno
import os
from pathlib import Path
import socket
import stat
import tempfile

import pytest

from gateway.control_socket import (
    GatewayControlServer,
    identify_gateway,
    resolve_client_socket_path,
)
from hermes_cli.gateway_runtime_discovery import (
    DiscoveryError,
    resolve_private_socket_path,
    query_identify,
)

pytestmark = pytest.mark.platforms("posix")


@pytest.fixture
def short_root():
    # no-tmp: ok -- real AF_UNIX tests need a path below macOS's sun_path limit.
    with tempfile.TemporaryDirectory(prefix="hperm-", dir="/tmp") as root:
        yield Path(root).resolve()


@pytest.mark.parametrize("profile", ["default", "secondary"])
@pytest.mark.parametrize("chmod_result", ["einval", "ignored"])
@pytest.mark.parametrize("retained_inode", [False, True])
def test_unprivate_socket_falls_back_and_recovers_on_restart(
    short_root, monkeypatch, profile, chmod_result, retained_inode
):
    home = short_root / ("home" if profile == "default" else "profiles/secondary")
    home.mkdir(parents=True, mode=0o700)
    alias = short_root / "linked-home"
    alias.symlink_to(home, target_is_directory=True)
    direct = home / "gateway.sock"
    real_chmod, real_unlink = os.chmod, os.unlink
    real_start = asyncio.start_unix_server
    listeners = []
    refused_chmod = []

    def shared_chmod(path, mode, *args, **kwargs):
        if Path(path) == direct and mode == 0o600:
            real_chmod(path, 0o755)
            refused_chmod.append(path)
            if chmod_result == "einval":
                raise OSError(errno.EINVAL, "socket chmod unsupported")
            return
        return real_chmod(path, mode, *args, **kwargs)

    def shared_unlink(path, *args, **kwargs):
        if Path(path) == direct and retained_inode and refused_chmod:
            raise OSError(errno.EINVAL, "stale socket cannot be removed")
        return real_unlink(path, *args, **kwargs)

    async def start(*args, **kwargs):
        listener = await real_start(*args, **kwargs)
        listeners.append(listener)
        return listener

    monkeypatch.setattr(os, "chmod", shared_chmod)
    monkeypatch.setattr(os, "unlink", shared_unlink)
    monkeypatch.setattr(asyncio, "start_unix_server", start)

    async def scenario():
        for _ in range(2):
            server = GatewayControlServer(
                home, verb_handlers={"identify": lambda: {"pid": 123}}
            )
            assert await server.start()
            try:
                pointer = home / "gateway.sock.path"
                target = Path(pointer.read_text(encoding="utf-8"))
                assert server._bind_path == target and target != direct
                assert stat.S_IMODE(target.lstat().st_mode) == 0o600
                assert stat.S_IMODE(target.parent.lstat().st_mode) == 0o700
                assert stat.S_IMODE(pointer.lstat().st_mode) == 0o600
                assert all(not listener.is_serving() for listener in listeners[:-1])
                result = await asyncio.to_thread(
                    lambda: (query_identify(home, timeout=3), identify_gateway(alias))
                )
                assert result == ({"pid": 123}, {"pid": 123})
            finally:
                await server.stop()
            assert not pointer.exists() and not target.exists()
            assert all(not listener.is_serving() for listener in listeners)

    asyncio.run(scenario())
    assert refused_chmod
    assert direct.exists() is retained_inode


def test_private_socket_does_not_need_chmod_support(short_root, monkeypatch):
    home = short_root / "home"
    home.mkdir(mode=0o700)
    direct = home / "gateway.sock"
    real_chmod = os.chmod

    def refuse_chmod(path, mode, *args, **kwargs):
        if Path(path) == direct:
            raise OSError(errno.EINVAL, "socket chmod unsupported")
        return real_chmod(path, mode, *args, **kwargs)

    monkeypatch.setattr(os, "chmod", refuse_chmod)

    async def scenario():
        server = GatewayControlServer(
            home, verb_handlers={"identify": lambda: {"pid": 123}}
        )
        assert await server.start()
        try:
            assert server._bind_path == direct
            assert stat.S_IMODE(direct.lstat().st_mode) == 0o600
            assert not (home / "gateway.sock.path").exists()
            assert await asyncio.to_thread(query_identify, home, timeout=3) == {
                "pid": 123
            }
        finally:
            await server.stop()

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "untrusted", ["pointer", "directory", "socket", "profile", "missing"]
)
def test_unprivate_direct_socket_does_not_bypass_pointer_checks(short_root, untrusted):
    from gateway.control_socket import _home_hash

    home = short_root / "home"
    home.mkdir(mode=0o700)
    pointer_home = short_root / "other" if untrusted == "profile" else home
    directory = short_root / f"hermes-gw-{_home_hash(pointer_home)}"
    directory.mkdir(mode=0o700)
    target = directory / "control.sock"
    pointer = home / "gateway.sock.path"
    with (
        socket.socket(socket.AF_UNIX) as direct,
        socket.socket(socket.AF_UNIX) as fallback,
    ):
        direct.bind(str(home / "gateway.sock"))
        (home / "gateway.sock").chmod(0o755)
        fallback.bind(str(target))
        target.chmod(0o600)
        pointer.write_text(str(target), encoding="utf-8")
        pointer.chmod(0o600)
        if untrusted == "missing":
            pointer.unlink()
        elif untrusted != "profile":
            {"pointer": pointer, "directory": directory, "socket": target}[
                untrusted
            ].chmod(0o755)
        with pytest.raises(DiscoveryError, match="unsafe_control_permissions"):
            resolve_private_socket_path(home)
        assert resolve_client_socket_path(home) is None


def test_fallback_that_cannot_stay_private_fails_closed(short_root, monkeypatch):
    home = short_root / "home"
    home.mkdir(mode=0o700)
    real_chmod, real_start = os.chmod, asyncio.start_unix_server
    listeners = []

    def chmod(path, mode, *args, **kwargs):
        if Path(path).name in {"gateway.sock", "control.sock"}:
            real_chmod(path, 0o755)
            raise OSError(errno.EINVAL, "socket chmod unsupported")
        return real_chmod(path, mode, *args, **kwargs)

    async def start(*args, **kwargs):
        listener = await real_start(*args, **kwargs)
        listeners.append(listener)
        return listener

    monkeypatch.setattr(os, "chmod", chmod)
    monkeypatch.setattr(asyncio, "start_unix_server", start)

    async def scenario():
        server = GatewayControlServer(home)
        assert not await server.start()
        assert len(listeners) == 2
        assert all(not listener.is_serving() for listener in listeners)
        assert not (home / "gateway.sock").exists()
        assert not (home / "gateway.sock.path").exists()
        assert server._server is None

    asyncio.run(scenario())
