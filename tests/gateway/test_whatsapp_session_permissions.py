"""The WhatsApp session store and bridge.log are owner-only (#80090).

creds.json and the signal keys grant full access to the account, and bridge.log carries the QR
pairing payloads. Baileys writes the keys from the Node child at the process umask, so under the
common 022 umask they landed 0644 in a 0755 directory: readable by every local account. Each
test runs a real child (a Python stand-in for bridge.js) that writes into ``--session`` the way
Baileys does, then reads the modes it left behind.
"""
import os
import stat
import sys
import textwrap

import pytest

pytestmark = pytest.mark.platforms("posix")  # POSIX permission bits

_FAKE_BRIDGE = textwrap.dedent("""
    import sys, pathlib
    session = pathlib.Path(sys.argv[sys.argv.index("--session") + 1])
    (session / "creds.json").write_text('{"me": {"id": "15550001111:1@s.whatsapp.net"}}')
    (session / "session-15550001111.0.json").write_text("{}")
    print("QR: 2@pairing-payload", flush=True)
""")


def _mode(path) -> int:
    return stat.S_IMODE(os.stat(path).st_mode)


@pytest.fixture
def host(tmp_path, monkeypatch):
    """Unmanaged, non-container host with the 022 umask the issue reproduces under."""
    from hermes_agent import hermes_constants

    monkeypatch.delenv("HERMES_MANAGED", raising=False)
    monkeypatch.setattr(hermes_constants, "_container_or_chmod_skipped", lambda: False)
    previous = os.umask(0o022)
    bridge = tmp_path / "bridge" / "bridge.js"
    bridge.parent.mkdir()
    bridge.write_text(_FAKE_BRIDGE, encoding="utf-8")
    yield bridge
    os.umask(previous)


def _assert_private(session):
    assert _mode(session) == 0o700
    for name in ("creds.json", "session-15550001111.0.json"):
        assert _mode(session / name) == 0o600, name


def _connectable_adapter(host, tmp_path, monkeypatch):
    from unittest.mock import AsyncMock, MagicMock

    from hermes_agent.gateway.config import Platform
    from plugins.platforms.whatsapp import adapter as wa

    monkeypatch.setattr(wa, "find_node_executable", lambda name: sys.executable)
    monkeypatch.setattr(wa.asyncio, "to_thread", AsyncMock(return_value=True))  # preflight + reaps
    monkeypatch.setattr(wa, "_write_bridge_pidfile", lambda *args: None)
    adapter = wa.WhatsAppAdapter.__new__(wa.WhatsAppAdapter)
    adapter.platform, adapter.config = Platform.WHATSAPP, MagicMock(extra={})
    adapter._bridge_script, adapter._bridge_port = str(host), 19899
    adapter._profile_home = tmp_path
    adapter._session_path = tmp_path / "platforms" / "whatsapp" / "session"
    adapter._bridge_log_fh = adapter._bridge_log = adapter._bridge_process = None
    adapter._running = False
    adapter._foreign_bridge_session, adapter._bridge_probe_timed_out = None, False
    adapter._bridge_holder_unproven = False
    adapter._acquire_platform_lock = MagicMock(return_value=True)
    adapter._release_platform_lock = MagicMock()
    adapter._ensure_bridge_deps = MagicMock(return_value=True)
    adapter._reuse_running_bridge = AsyncMock(return_value=False)
    adapter._wait_bridge_port_free = AsyncMock(return_value=True)
    adapter._bridge_env = lambda: dict(os.environ)
    return adapter


@pytest.mark.asyncio
async def test_gateway_bridge_writes_session_and_log_owner_only(host, tmp_path, monkeypatch):
    adapter = _connectable_adapter(host, tmp_path, monkeypatch)

    async def wait_for_exit():
        adapter._bridge_process.wait(timeout=30)
        return False

    adapter._wait_for_bridge = wait_for_exit
    assert await adapter.connect() is False

    _assert_private(adapter._session_path)
    log = adapter._session_path.parent / "bridge.log"
    assert "pairing-payload" in log.read_text(encoding="utf-8")
    assert _mode(log) == 0o600


@pytest.mark.asyncio
async def test_adopting_a_running_bridge_tightens_its_existing_log(host, tmp_path, monkeypatch):
    """An adopted bridge keeps appending to the bridge.log an older version created 0644."""
    from unittest.mock import AsyncMock

    adapter = _connectable_adapter(host, tmp_path, monkeypatch)
    adapter._reuse_running_bridge = AsyncMock(return_value=True)
    log = adapter._session_path.parent / "bridge.log"
    log.parent.mkdir(parents=True)
    log.write_text("QR: 2@old-pairing-payload\n", encoding="utf-8")
    os.chmod(log, 0o644)

    assert await adapter.connect() is True

    assert _mode(log) == 0o600


def test_dashboard_pairing_tightens_an_existing_world_readable_session(host, tmp_path, monkeypatch):
    from hermes_agent import hermes_constants
    from hermes_agent.gateway.platforms import whatsapp_common
    from hermes_agent.hermes_cli.web_routers import messaging

    monkeypatch.setattr(whatsapp_common, "resolve_whatsapp_bridge_dir", lambda: host.parent)
    monkeypatch.setattr(messaging, "_ensure_whatsapp_bridge_dependencies", lambda path: None)
    monkeypatch.setattr(hermes_constants, "find_node_executable", lambda name: sys.executable)
    session = tmp_path / "platforms" / "whatsapp" / "session"
    session.mkdir(parents=True, mode=0o755)  # what an earlier version left behind

    proc = messaging._spawn_whatsapp_pairing_process(session, "bot")
    assert "pairing-payload" in proc.communicate(timeout=30)[0]

    _assert_private(session)


def test_cli_pairing_writes_the_session_owner_only(host, tmp_path, monkeypatch):
    """`hermes whatsapp` runs the pairing child itself, so its creds need the store's umask too."""
    from types import SimpleNamespace

    from hermes_agent import hermes_constants
    from hermes_agent.gateway.platforms import whatsapp_common
    from hermes_agent.hermes_cli import main_platform_setup as setup

    home = tmp_path / "hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("WHATSAPP_MODE", "bot")
    monkeypatch.setenv("WHATSAPP_ALLOWED_USERS", "15550001111")
    monkeypatch.setattr("builtins.input", lambda _prompt="": "n")
    monkeypatch.setattr("hermes_agent.hermes_cli.main._require_tty", lambda *_a, **_kw: None)
    monkeypatch.setattr(whatsapp_common, "resolve_whatsapp_bridge_dir", lambda: host.parent)
    monkeypatch.setattr(setup, "_whatsapp_install_bridge", lambda path: True)
    monkeypatch.setattr(hermes_constants, "find_node_executable", lambda name: sys.executable)

    setup.cmd_whatsapp(SimpleNamespace())

    _assert_private(home / "whatsapp" / "session")


def test_managed_install_keeps_its_group_share_modes(host, tmp_path, monkeypatch):
    """NixOS/systemd installs share HERMES_HOME with a group on purpose (#77579): the store and the
    child's files follow the service umask instead of being forced owner-only."""
    from hermes_agent import hermes_constants

    monkeypatch.setenv("HERMES_MANAGED", "nixos")
    session = tmp_path / "session"
    assert hermes_constants.ensure_private_store(session, home=tmp_path) == -1
    assert _mode(session) == 0o755
