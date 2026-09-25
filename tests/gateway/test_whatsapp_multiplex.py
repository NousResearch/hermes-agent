"""Profile-local WhatsApp credentials and non-destructive secondary startup."""
import socket
import os
from unittest.mock import AsyncMock

import pytest

from gateway.config import Platform, PlatformConfig
from gateway.run import _platform_has_bot_credential, _profile_runtime_scope
from plugins.platforms.whatsapp.adapter import WhatsAppAdapter


def test_credentials_follow_profile_home(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    for home in (a, b):
        (home / "platforms/whatsapp/session").mkdir(parents=True)
    (a / "platforms/whatsapp/session/creds.json").write_text("{}")
    for home, expected in ((a, True), (b, False), (a, True)):
        with _profile_runtime_scope(home, hydrate_secrets=False):
            assert _platform_has_bot_credential(Platform.WHATSAPP, PlatformConfig()) is expected
            overridden = PlatformConfig(extra={"session_path": str(a / "platforms/whatsapp/session")})
            assert _platform_has_bot_credential(Platform.WHATSAPP, overridden)
    from gateway.status import write_runtime_status
    from hermes_cli import gateway_multiplex_served as served
    from unittest.mock import patch
    write_runtime_status(platform="work:whatsapp", platform_state="disabled",
                         error_code="whatsapp_unpaired", error_message="pair it: hermes -p work whatsapp")
    with patch.object(served, "live_default_gateway_pid", return_value=os.getpid()):
        assert served.served_profile_unserved_platforms("work") == {
            "whatsapp": "pair it: hermes -p work whatsapp"}


@pytest.mark.asyncio
async def test_secondary_ports_and_foreign_listener(tmp_path, monkeypatch):
    import plugins.platforms.whatsapp.adapter as module
    adapters = []
    listener = socket.socket()
    monkeypatch.setattr(module, "_write_bridge_pidfile", lambda *a: None)
    for name in ("a", "b"):
        home = tmp_path / name
        home.mkdir()
        with _profile_runtime_scope(home, hydrate_secrets=False):
            adapter = WhatsAppAdapter(PlatformConfig())
        adapter._hermes_profile_name = name
        adapter._runtime_status_platform_key = f"{name}:whatsapp"
        monkeypatch.setattr(adapter, "_preflight", lambda: True)
        monkeypatch.setattr(adapter, "_ensure_bridge_deps", lambda path: False)
        await adapter.connect()
        adapters.append(adapter)
        if name == "a":
            listener.bind(("127.0.0.1", adapter._bridge_port))
            listener.listen()
    assert 3001 <= adapters[0]._bridge_port <= 3999
    # Reserve the first profile's actual listener before the next startup.
    with listener:
        with _profile_runtime_scope(tmp_path / "b", hydrate_secrets=False):
            explicit = WhatsAppAdapter(PlatformConfig(extra={"bridge_port": adapters[0]._bridge_port}))
        explicit._hermes_profile_name = "b"
        explicit._runtime_status_platform_key = "b:whatsapp"
        monkeypatch.setattr(explicit, "_preflight", lambda: True)
        monkeypatch.setattr(explicit, "_ensure_bridge_deps", lambda path: True)
        probe = AsyncMock(side_effect=AssertionError("must not probe foreign bridge"))
        monkeypatch.setattr(explicit, "_probe_bridge_health", probe)
        def no_kill(*args):
            raise AssertionError("must not kill foreign bridge")
        monkeypatch.setattr(module, "_kill_port_process", no_kill)
        monkeypatch.setattr(module, "_kill_stale_bridge_by_pidfile", no_kill)
        assert not await explicit.connect()
        assert explicit.has_fatal_error
        assert not explicit.fatal_error_retryable
        assert str(explicit._bridge_port) in explicit.fatal_error_message
        assert "platforms.whatsapp.extra.bridge_port" in explicit.fatal_error_message
        probe.assert_not_called()
        await explicit.disconnect()
    assert adapters[0]._bridge_port != adapters[1]._bridge_port
    for name, adapter in zip(("a", "b"), adapters):
        assert int((tmp_path / name / "platforms/whatsapp/bridge_port").read_text()) == adapter._bridge_port
        with _profile_runtime_scope(tmp_path / name, hydrate_secrets=False):
            restart = WhatsAppAdapter(PlatformConfig())
        restart._runtime_status_platform_key = f"{name}:whatsapp"
        monkeypatch.setattr(restart, "_preflight", lambda: False)
        await restart.connect()
        assert restart._bridge_port == adapter._bridge_port


def _hold_port():
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    sock.listen()
    return sock, sock.getsockname()[1]


def test_ownership_verdicts(tmp_path):
    """free: unbound port (a stale pidfile changes nothing, and nothing is signalled). ours: bound port
    and this profile's pidfile names a live process by pid + kernel start time. Anything else bound
    is a fatal, so a secondary never probes or signals another profile's bridge."""
    from gateway import status
    from plugins.platforms.whatsapp.adapter import _write_bridge_pidfile
    from plugins.platforms.whatsapp.bridge_ownership import check_secondary_ownership
    session = tmp_path / "session"
    session.mkdir()
    (session / "bridge.pid").write_text(str(os.getpid()))  # legacy, no start time
    sock, port = _hold_port()
    with sock:
        with pytest.raises(ValueError, match="does not own"):
            check_secondary_ownership(session, port)  # bound, pidfile unverifiable: fail closed
        _write_bridge_pidfile(session, os.getpid())    # pid + start time of a live process
        assert check_secondary_ownership(session, port) == "ours"
        (session / "bridge.pid").write_text(f"{os.getpid()}\n{status.get_process_start_time(os.getpid()) + 1}\n")
        with pytest.raises(ValueError, match="does not own"):
            check_secondary_ownership(session, port)  # recycled pid: fingerprint mismatch
    assert check_secondary_ownership(session, port) == "free"  # unbound now, pidfile irrelevant
    assert (session / "bridge.pid").exists()  # nothing was signalled or unlinked


@pytest.mark.asyncio
async def test_secondary_adopts_its_own_orphaned_bridge(tmp_path, monkeypatch):
    """Gateway crash, bridge child survives on the recorded port: the restart adopts it when healthy
    instead of dying with a port conflict. Only the identity-checked pidfile path may reap it."""
    import plugins.platforms.whatsapp.adapter as module
    from plugins.platforms.whatsapp.adapter import _write_bridge_pidfile
    home = tmp_path / "c"
    home.mkdir()
    with _profile_runtime_scope(home, hydrate_secrets=False):
        adapter = WhatsAppAdapter(PlatformConfig())
    adapter._runtime_status_platform_key = "c:whatsapp"
    monkeypatch.setattr(adapter, "_preflight", lambda: True)
    monkeypatch.setattr(adapter, "_ensure_bridge_deps", lambda path: True)
    sock, port = _hold_port()
    (home / "platforms/whatsapp").mkdir(parents=True)
    (home / "platforms/whatsapp/bridge_port").write_text(str(port))
    adapter._session_path.mkdir(parents=True, exist_ok=True)
    _write_bridge_pidfile(adapter._session_path, os.getpid())
    reuse = AsyncMock(return_value=True)
    monkeypatch.setattr(adapter, "_reuse_running_bridge", reuse)
    def no_port_kill(*args):
        raise AssertionError("a secondary never clears a port by scan")
    monkeypatch.setattr(module, "_kill_port_process", no_port_kill)
    with sock:
        assert await adapter.connect()
    reuse.assert_awaited_once()
    assert not adapter.has_fatal_error
    await adapter.disconnect()
