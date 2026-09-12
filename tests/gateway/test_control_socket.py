"""Tests for the gateway control socket (#92091 migration step 1)."""

import asyncio
import errno
import json
import socket
import sys
from pathlib import Path

import pytest

from gateway.control_socket import (
    CONTROL_PROTOCOL_VERSION,
    GatewayControlServer,
    _fallback_socket_path,
    client_socket_candidates,
    identify_gateway,
    query_gateway_control,
    resolve_client_socket_path,
    resolve_server_socket_path,
    windows_pipe_name,
)

pytestmark = pytest.mark.skipif(
    sys.platform == "win32",
    reason="Unix-socket transport; the named-pipe half is covered on the wine2e lane",
)


def _run(coro):
    return asyncio.run(coro)


@pytest.fixture()
def home(tmp_path: Path) -> Path:
    d = tmp_path / "home" / ".hermes"
    d.mkdir(parents=True)
    return d


def _serve(home: Path, handlers=None):
    """Context helper: start a server in a fresh loop, yield inside coro."""
    return GatewayControlServer(home, verb_handlers=handlers)


# ---------------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------------

def test_short_home_binds_in_home(tmp_path: Path):
    # A home short enough for sun_path binds in-home with no pointer.
    # tmp_path can exceed the limit on CI runners, so build one in the
    # system temp root directly.
    import tempfile

    try:
        short_root = Path(tempfile.mkdtemp(prefix="hgw-", dir="/tmp"))
    except OSError:
        pytest.skip("/tmp not writable on this host")
    try:
        short_home = short_root / ".hermes"
        short_home.mkdir()
        assert len(str(short_home / "gateway.sock").encode()) <= 100
        bind, pointer = resolve_server_socket_path(short_home)
        assert bind == short_home / "gateway.sock"
        assert pointer is None
    finally:
        import shutil

        shutil.rmtree(short_root, ignore_errors=True)


def test_long_home_uses_pointer_fallback(tmp_path: Path):
    deep = tmp_path / ("x" * 120) / ".hermes"
    deep.mkdir(parents=True)
    bind, pointer = resolve_server_socket_path(deep)
    assert bind != deep / "gateway.sock"
    assert len(str(bind).encode()) <= 100
    assert pointer == deep / "gateway.sock.path"


def test_client_resolution_prefers_pointer_then_direct(home: Path, tmp_path: Path):
    assert resolve_client_socket_path(home) is None
    assert client_socket_candidates(home) == []
    # direct file alone
    direct = home / "gateway.sock"
    direct.touch()
    assert resolve_client_socket_path(home) == direct
    # A pointer is written only when the server deliberately bound elsewhere,
    # so when it names a present target it is the authoritative answer; the
    # direct path stays as the fallback candidate.
    target = tmp_path / "elsewhere.sock"
    target.touch()
    (home / "gateway.sock.path").write_text(str(target))
    assert client_socket_candidates(home) == [target, direct]
    assert resolve_client_socket_path(home) == target
    # a pointer to a missing target is ignored
    target.unlink()
    assert client_socket_candidates(home) == [direct]


def test_client_resolution_survives_unstatable_direct_path(home: Path, tmp_path: Path, monkeypatch):
    # Docker Desktop bind mount: a socket inode from an earlier container
    # session makes Path.exists() RAISE EOPNOTSUPP. Resolution must not crash
    # and must still find the pointer the fallback server wrote.
    direct = home / "gateway.sock"
    direct.touch()
    target = tmp_path / "elsewhere.sock"
    target.touch()
    (home / "gateway.sock.path").write_text(str(target))
    real_exists = Path.exists

    def unstatable(self, *a, **k):
        if self == direct:
            raise OSError(errno.EOPNOTSUPP, "Operation not supported")
        return real_exists(self, *a, **k)

    monkeypatch.setattr(Path, "exists", unstatable)
    assert client_socket_candidates(home) == [target]
    assert resolve_client_socket_path(home) == target


def test_query_falls_through_a_dead_pointer_to_the_live_direct_socket(short_home: Path, tmp_path: Path):
    # A crashed fallback server leaves pointer + dead temp socket file behind;
    # the next server binds in-home. Clients must reach it anyway.
    home = short_home
    dead = tmp_path / "dead.sock"
    dead.touch()  # exists, but nothing listens
    (home / "gateway.sock.path").write_text(str(dead))

    async def scenario():
        server = GatewayControlServer(home, verb_handlers={"identify": lambda: {"pid": 21}})
        assert await server.start()
        try:
            assert server._bind_path == home / "gateway.sock"
            # binding in-home cleans the stale pointer up...
            assert not (home / "gateway.sock.path").exists()
            # ...and even with a stale pointer re-planted, the client falls through.
            (home / "gateway.sock.path").write_text(str(dead))
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(None, lambda: identify_gateway(home))
        finally:
            await server.stop()

    assert _run(scenario()) == {"pid": 21}


def test_windows_pipe_name_is_stable_and_home_scoped(tmp_path: Path):
    a = windows_pipe_name(tmp_path / "a")
    b = windows_pipe_name(tmp_path / "b")
    assert a.startswith(r"\\.\pipe\hermes-gateway-")
    assert a != b
    assert a == windows_pipe_name(tmp_path / "a")


# ---------------------------------------------------------------------------
# Server lifecycle + verbs (real sockets, real event loop)
# ---------------------------------------------------------------------------

def test_server_answers_identify_and_status(home: Path):
    async def scenario():
        server = GatewayControlServer(
            home,
            verb_handlers={
                "identify": lambda: {"pid": 4242, "code_sha": "abc123", "protocol": 1},
                "status": lambda: {"gateway_state": "running"},
            },
        )
        assert await server.start()
        try:
            loop = asyncio.get_running_loop()
            ident = await loop.run_in_executor(
                None, lambda: query_gateway_control(home, "identify")
            )
            status = await loop.run_in_executor(
                None, lambda: query_gateway_control(home, "status")
            )
            return ident, status
        finally:
            await server.stop()

    ident, status = _run(scenario())
    assert ident == {"pid": 4242, "code_sha": "abc123", "protocol": 1}
    assert status == {"gateway_state": "running"}


def test_unknown_verb_and_malformed_request(home: Path):
    async def scenario():
        server = GatewayControlServer(
            home, verb_handlers={"identify": lambda: {"pid": 1}}
        )
        assert await server.start()
        try:
            loop = asyncio.get_running_loop()
            unknown = await loop.run_in_executor(
                None, lambda: query_gateway_control(home, "restart")
            )

            def raw_garbage():
                path = resolve_client_socket_path(home)
                with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as s:
                    s.settimeout(2)
                    s.connect(str(path))
                    s.sendall(b"this is not json\n")
                    return s.recv(65536)

            garbage_reply = await loop.run_in_executor(None, raw_garbage)
            return unknown, garbage_reply
        finally:
            await server.stop()

    unknown, garbage_reply = _run(scenario())
    # unknown verb → ok:false → client returns None (fallback signal)
    assert unknown is None
    payload = json.loads(garbage_reply.decode())
    assert payload["ok"] is False
    assert payload["protocol"] == CONTROL_PROTOCOL_VERSION


def test_stop_removes_socket_and_pointer(home: Path):
    async def scenario():
        server = GatewayControlServer(
            home, verb_handlers={"identify": lambda: {"pid": 1}}
        )
        assert await server.start()
        bind, _ = resolve_server_socket_path(home)
        assert bind.exists()
        await server.stop()
        return bind

    bind = _run(scenario())
    assert not bind.exists()
    assert resolve_client_socket_path(home) is None
    # queries after stop cleanly return None
    assert query_gateway_control(home, "identify") is None


def test_stale_socket_file_is_replaced_on_bind(home: Path):
    # Plant the stale file at wherever the server will actually bind
    # (in-home OR the temp-dir fallback, depending on path length).
    bind, _ = resolve_server_socket_path(home)
    bind.parent.mkdir(parents=True, exist_ok=True)
    bind.touch()  # crashed predecessor's leftover

    async def scenario():
        server = GatewayControlServer(
            home, verb_handlers={"identify": lambda: {"pid": 7}}
        )
        assert await server.start()
        try:
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(None, lambda: identify_gateway(home))
        finally:
            await server.stop()

    assert _run(scenario()) == {"pid": 7}


@pytest.fixture()
def short_home():
    """A home short enough that the server binds IN-HOME (no pointer fallback).

    pytest's tmp_path can exceed sun_path on macOS/CI, in which case the
    server already uses the temp-dir fallback and an in-home bind refusal can
    never be exercised — so build the home directly under /tmp, like
    test_short_home_binds_in_home does.
    """
    import shutil
    import tempfile
    try:
        root = Path(tempfile.mkdtemp(prefix="hgw-", dir="/tmp"))
    except OSError:
        pytest.skip("/tmp not writable on this host")
    home = root / ".hermes"
    home.mkdir()
    if resolve_server_socket_path(home)[1] is not None:
        shutil.rmtree(root, ignore_errors=True)
        pytest.skip("even /tmp yields a too-long socket path here")
    try:
        yield home
    finally:
        shutil.rmtree(root, ignore_errors=True)


def _refusing_start_unix_server(monkeypatch, *, refuse_under: Path, err: int):
    """start_unix_server that refuses paths under ``refuse_under`` with ``err``
    and otherwise delegates — the behaviour of a Docker Desktop bind mount."""
    real = asyncio.start_unix_server
    attempts: list[str] = []

    async def fake(client_connected_cb, path=None, **kwargs):
        attempts.append(str(path))
        if Path(path).is_relative_to(refuse_under):
            raise OSError(err, "Operation not supported" if err == errno.EOPNOTSUPP else "refused")
        return await real(client_connected_cb, path=path, **kwargs)

    monkeypatch.setattr(asyncio, "start_unix_server", fake)
    return attempts


def test_bind_refused_in_home_falls_back_to_tempdir_with_pointer(short_home: Path, monkeypatch):
    """The documented Docker layout bind-mounts HERMES_HOME (~/.hermes:/opt/data);
    Docker Desktop answers AF_UNIX bind there with EOPNOTSUPP. The server must
    fall back to the temp-dir socket + pointer file (the long-path mechanism),
    so clients still find it and pause-for-update / identify keep working."""
    home = short_home
    attempts = _refusing_start_unix_server(monkeypatch, refuse_under=home, err=errno.EOPNOTSUPP)
    fallback = _fallback_socket_path(home)

    async def scenario():
        server = GatewayControlServer(home, verb_handlers={"identify": lambda: {"pid": 11}})
        assert await server.start()
        try:
            assert server._bind_path == fallback
            assert (home / "gateway.sock.path").read_text(encoding="utf-8").strip() == str(fallback)
            assert resolve_client_socket_path(home) == fallback
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(None, lambda: identify_gateway(home))
        finally:
            await server.stop()

    assert _run(scenario()) == {"pid": 11}
    assert attempts == [str(home / "gateway.sock"), str(fallback)]
    assert not fallback.exists()
    assert not (home / "gateway.sock.path").exists()


def test_bind_refused_for_other_reasons_still_fails(short_home: Path, monkeypatch):
    # A live sibling (EADDRINUSE) must keep failing loudly — no silent fallback
    # that would let two gateways for one home both look healthy.
    home = short_home
    attempts = _refusing_start_unix_server(monkeypatch, refuse_under=home, err=errno.EADDRINUSE)

    async def scenario():
        server = GatewayControlServer(home)
        return await server.start()

    assert _run(scenario()) is False
    assert attempts == [str(home / "gateway.sock")]
    assert not (home / "gateway.sock.path").exists()


def test_stale_socket_is_unlinked_even_when_it_cannot_be_stat_ed(short_home: Path, monkeypatch):
    # On the same bind mounts a stale socket inode cannot be stat'ed, so an
    # exists()-gated cleanup never fires and asyncio logs its own ERROR. The
    # server-side cleanup must not depend on exists(). (The fake is disarmed
    # once the server is up so the CLIENT's own exists() probe is untouched.)
    home = short_home
    bind = home / "gateway.sock"
    bind.touch()  # crashed predecessor's leftover
    real_exists = Path.exists
    armed = {"on": True}

    def unstatable(self, *a, **k):
        if armed["on"] and self == bind:
            raise OSError(errno.EOPNOTSUPP, "Operation not supported")
        return real_exists(self, *a, **k)

    monkeypatch.setattr(Path, "exists", unstatable)

    async def scenario():
        server = GatewayControlServer(home, verb_handlers={"identify": lambda: {"pid": 12}})
        assert await server.start()
        armed["on"] = False
        try:
            assert server._bind_path == bind
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(None, lambda: identify_gateway(home))
        finally:
            await server.stop()

    assert _run(scenario()) == {"pid": 12}


def test_long_home_end_to_end_via_pointer(tmp_path: Path):
    deep = tmp_path / ("p" * 120) / ".hermes"
    deep.mkdir(parents=True)

    async def scenario():
        server = GatewayControlServer(
            deep, verb_handlers={"identify": lambda: {"pid": 9}}
        )
        assert await server.start()
        try:
            assert (deep / "gateway.sock.path").is_file()
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(None, lambda: identify_gateway(deep))
        finally:
            await server.stop()

    assert _run(scenario()) == {"pid": 9}
    assert not (deep / "gateway.sock.path").exists()


def test_no_socket_returns_none_fast(home: Path):
    assert identify_gateway(home) is None
    assert query_gateway_control(home, "status") is None


def test_default_identify_payload_shape(home: Path, monkeypatch):
    """The real identify handler carries the fleet-consumer contract fields."""
    monkeypatch.setenv("HERMES_HOME", str(home))

    async def scenario():
        server = GatewayControlServer(home)  # default handlers
        assert await server.start()
        try:
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(None, lambda: identify_gateway(home))
        finally:
            await server.stop()

    ident = _run(scenario())
    assert ident is not None
    assert ident["protocol"] == CONTROL_PROTOCOL_VERSION
    assert ident["pid"] == __import__("os").getpid()
    # contract keys exist even when values are None/absent-degradable
    for key in ("hermes_home", "supervisor", "kind", "start_time"):
        assert key in ident
    assert ident["supervisor"] in {
        "systemd",
        "launchd",
        "desktop",
        "external",
        "manual",
    }


# ---------------------------------------------------------------------------
# Consumer integration: fleet matrix + inventory prefer socket, fall back
# ---------------------------------------------------------------------------

def _fake_identity(pid: int, sha: str):
    return {
        "protocol": 1,
        "pid": pid,
        "code_sha": sha,
        "code_version": "9.9.9",
        "supervisor": "systemd",
        "kind": "hermes-gateway",
    }


def test_collect_fleet_versions_prefers_socket(tmp_path: Path, monkeypatch):
    import hermes_cli.update_receipt as ur

    home = tmp_path / ".hermes"
    home.mkdir()

    monkeypatch.setattr(
        "hermes_cli.build_info.get_code_identity",
        lambda refresh=False: {"sha": "HEADSHA", "version": "1.0"},
    )
    monkeypatch.setattr(
        "hermes_cli.profiles._get_default_hermes_home", lambda: home
    )
    monkeypatch.setattr(
        "hermes_cli.profiles._get_profiles_root", lambda: tmp_path / "no-profiles"
    )
    # stale state file that would report a WRONG pid — socket must win
    (home / "gateway_state.json").write_text(
        json.dumps({"pid": 1, "code_sha": "stalefile", "kind": "hermes-gateway"})
    )
    monkeypatch.setattr(
        "gateway.control_socket.identify_gateway",
        lambda h, **kw: _fake_identity(31337, "HEADSHA"),
    )

    fleet = ur.collect_fleet_versions()
    assert len(fleet) == 1
    entry = fleet[0]
    assert entry["pid"] == 31337
    assert entry["state"] == "current"
    assert entry["source"] == "socket"


def test_collect_fleet_versions_falls_back_to_state_file(tmp_path: Path, monkeypatch):
    import os

    import hermes_cli.update_receipt as ur

    home = tmp_path / ".hermes"
    home.mkdir()

    monkeypatch.setattr(
        "hermes_cli.build_info.get_code_identity",
        lambda refresh=False: {"sha": "HEADSHA", "version": "1.0"},
    )
    monkeypatch.setattr(
        "hermes_cli.profiles._get_default_hermes_home", lambda: home
    )
    monkeypatch.setattr(
        "hermes_cli.profiles._get_profiles_root", lambda: tmp_path / "no-profiles"
    )
    monkeypatch.setattr(
        "gateway.control_socket.identify_gateway", lambda h, **kw: None
    )
    (home / "gateway_state.json").write_text(
        json.dumps(
            {
                "pid": os.getpid(),  # a live pid so _pid_exists passes
                "code_sha": "OLDSHA",
                "kind": "hermes-gateway",
            }
        )
    )

    fleet = ur.collect_fleet_versions()
    assert len(fleet) == 1
    assert fleet[0]["pid"] == os.getpid()
    assert fleet[0]["state"] == "stale"
    assert "source" not in fleet[0]


def test_runtime_inventory_dedupes_same_pid_across_homes(tmp_path: Path, monkeypatch):
    """One multiplex gateway answering identify for two profile homes must
    yield exactly ONE runtime record (reviewer point on #92447)."""
    import hermes_cli.update_inventory as ui

    home = tmp_path / ".hermes"
    home.mkdir()
    profiles_root = tmp_path / "profiles"
    (profiles_root / "coder").mkdir(parents=True)

    monkeypatch.setattr(
        "hermes_cli.profiles._get_default_hermes_home", lambda: home
    )
    monkeypatch.setattr(
        "hermes_cli.profiles._get_profiles_root", lambda: profiles_root
    )
    monkeypatch.setattr(
        "hermes_cli.gateway._get_service_pids", lambda all_profiles=False: set()
    )
    monkeypatch.setattr(
        "hermes_cli.gateway.find_profile_gateway_processes", lambda: []
    )
    monkeypatch.setattr(
        "gateway.control_socket.identify_gateway",
        lambda h, **kw: _fake_identity(777, "SHA777"),
    )

    plan = ui.collect_runtime_inventory()
    gws = [r for r in plan.runtimes if r.kind == "gateway"]
    assert len(gws) == 1, [r.__dict__ for r in gws]
    assert gws[0].pid == 777


def test_runtime_inventory_prefers_socket_supervisor(tmp_path: Path, monkeypatch):
    import hermes_cli.update_inventory as ui

    home = tmp_path / ".hermes"
    home.mkdir()

    monkeypatch.setattr(
        "hermes_cli.profiles._get_default_hermes_home", lambda: home
    )
    monkeypatch.setattr(
        "hermes_cli.profiles._get_profiles_root", lambda: tmp_path / "no-profiles"
    )
    monkeypatch.setattr(
        "hermes_cli.gateway._get_service_pids", lambda all_profiles=False: set()
    )
    monkeypatch.setattr(
        "hermes_cli.gateway.find_profile_gateway_processes", lambda: []
    )
    monkeypatch.setattr(
        "gateway.control_socket.identify_gateway",
        lambda h, **kw: _fake_identity(555, "SHA555"),
    )

    plan = ui.collect_runtime_inventory()
    gws = [r for r in plan.runtimes if r.kind == "gateway"]
    assert len(gws) == 1
    assert gws[0].pid == 555
    # supervisor comes from the gateway's own declaration, not a PID scan
    assert gws[0].supervisor == "systemd"
    assert gws[0].code_sha == "SHA555"
