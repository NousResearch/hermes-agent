"""Behavior tests: the old WhatsApp bridge must release its port before the new one spawns.

``connect()`` used to SIGTERM the previous bridge and sleep a blind 1s, then spawn the new
bridge regardless. A bridge that exited slowly (or ignored SIGTERM) still held
``127.0.0.1:<port>`` when the new bridge called ``listen`` — EADDRINUSE crashed it, and the
adapter retried forever. The fix: wait for the exit (bounded) and SIGKILL survivors, wait for
the port to actually come free (a real bind probe, like the bridge's own ``listen``), and fail
with a named retryable error instead of spawning while the port is busy.

Every test here uses a REAL process holding a REAL bound socket on a free port — only the
spawned bridge (``Popen``/node) and the health probe are mocked.
"""

import asyncio
import contextlib
import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gateway.config import Platform

linux = pytest.mark.platforms("linux")  # real POSIX signals + psutil /proc listening scan


# A child that binds 127.0.0.1:<argv[1]> and IGNORES SIGTERM — the slow-to-die old bridge.
_HOLDER_IGNORING_SIGTERM = (
    "import signal, socket, sys, time\n"
    "signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
    "s = socket.socket()\n"
    "s.bind(('127.0.0.1', int(sys.argv[1])))\n"
    "s.listen(5)\n"
    "print('ready', flush=True)\n"
    "time.sleep(60)\n"
)

# Same, but honoring SIGTERM (a stranger that is merely slow to be discovered, or an old
# bridge without the identity markers — it must NOT be killed at all).
_HOLDER_POLITE = (
    "import socket, sys, time\n"
    "s = socket.socket()\n"
    "s.bind(('127.0.0.1', int(sys.argv[1])))\n"
    "s.listen(5)\n"
    "print('ready', flush=True)\n"
    "time.sleep(60)\n"
)


def _wait_port_held(port: int, held: bool = True, timeout: float = 10.0) -> None:
    """Wait until a real bind probe agrees the port is (not) free — the holder is a fresh child."""
    from plugins.platforms.whatsapp.bridge_ownership import port_is_free

    deadline = time.monotonic() + timeout
    while port_is_free(port) is held and time.monotonic() < deadline:
        time.sleep(0.05)
    assert port_is_free(port) is (not held), f"port {port} never became {'held' if held else 'free'}"


def _spawn_holder(script: str, port: int, node_named_dir: Optional[Path] = None) -> subprocess.Popen:
    """Spawn a port-holding child; with *node_named_dir*, its executable is named ``node`` so the
    adapter's node-bridge identity check accepts it (a symlink keeps /proc comm = 'node')."""
    if node_named_dir is not None:
        exe = node_named_dir / "node"
        exe.symlink_to(sys.executable)
        argv = [str(exe)]
    else:
        argv = [sys.executable]
    return subprocess.Popen([*argv, "-c", script, str(port)], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)


@pytest.fixture(autouse=True)
def _pm_node(monkeypatch):
    """Stand-in for PM's Node/npm; the user's PATH copy is never picked up."""
    from plugins.platforms.whatsapp import adapter as whatsapp_adapter
    monkeypatch.setattr(whatsapp_adapter, "find_node_executable", lambda name: f"/pm/{name}")


@pytest.fixture(autouse=True)
def _fast_grace(monkeypatch):
    """Shorten the SIGTERM grace and the port-free wait so the escalation tests stay fast
    (``raising=False``: on the unfixed base these knobs do not exist and the tests must fail on
    behavior, not on the fixture)."""
    from plugins.platforms.whatsapp import adapter as whatsapp_adapter
    monkeypatch.setattr(whatsapp_adapter, "_SIGTERM_GRACE_S", 1.2, raising=False)
    monkeypatch.setattr(whatsapp_adapter, "_BRIDGE_PORT_FREE_TIMEOUT_S", 1.5, raising=False)


def _make_adapter(bridge_script: str, session_path: Path, port: int):
    """Create a WhatsAppAdapter with test attributes (bypass __init__)."""
    from plugins.platforms.whatsapp.adapter import WhatsAppAdapter

    adapter = WhatsAppAdapter.__new__(WhatsAppAdapter)
    adapter.platform = Platform.WHATSAPP
    adapter.config = MagicMock()
    adapter._bridge_port = port
    adapter._bridge_script = bridge_script
    adapter._session_path = Path(os.path.abspath(os.path.expanduser(str(session_path))))
    adapter._foreign_bridge_session = None
    adapter._bridge_probe_timed_out = False
    adapter._bridge_log_fh = None
    adapter._bridge_log = None
    adapter._bridge_process = None
    adapter._reply_prefix = None
    adapter._send_read_receipts = False
    adapter._dm_policy = adapter._group_policy = "pairing"
    adapter._allow_from = adapter._group_allow_from = set()
    adapter._running = False
    adapter._message_handler = None
    adapter._fatal_error_code = None
    adapter._fatal_error_message = None
    adapter._fatal_error_retryable = True
    adapter._fatal_error_handler = None
    adapter._active_sessions = {}
    adapter._pending_messages = {}
    adapter._background_tasks = set()
    adapter._auto_tts_disabled_chats = set()
    adapter._message_queue = asyncio.Queue()
    adapter._http_session = None
    return adapter


class _AsyncCM:
    def __init__(self, value):
        self.value = value

    async def __aenter__(self):
        return self.value

    async def __aexit__(self, *exc):
        return False


def _mock_health_refused():
    """/health probe that fails like a port holder that is not our bridge: a synchronous
    connection error — NOT a timeout (a timeout proves no ownership and connect() must stop)."""
    mock_session = MagicMock()
    mock_session.get = MagicMock(side_effect=OSError("connection refused"))
    mock_session.close = AsyncMock()
    return MagicMock(return_value=_AsyncCM(mock_session))


def _setup_bridge_dir(tmp_path: Path) -> Path:
    """Real bridge dir with bridge.js + package.json + a fresh node_modules dep stamp + creds."""
    from plugins.platforms.whatsapp.adapter import _file_content_hash

    bridge_dir = tmp_path / "whatsapp-bridge"
    bridge_dir.mkdir()
    (bridge_dir / "bridge.js").write_text("// current bridge code\n", encoding="utf-8")
    (bridge_dir / "package.json").write_text('{"name": "bridge"}\n', encoding="utf-8")
    nm = bridge_dir / "node_modules"
    nm.mkdir()
    (nm / ".hermes-pkg-hash").write_text(_file_content_hash(bridge_dir / "package.json"))
    session_path = tmp_path / "session"
    session_path.mkdir()
    (session_path / "creds.json").write_text("{}", encoding="utf-8")
    return bridge_dir


def _connect_patches(adapter, spawn_bridge):
    """Patches for a real connect() run. Only the BRIDGE spawn (``/pm/node ...``) is faked:
    ``subprocess.run`` builds on ``Popen``, so a blanket Popen patch would also blind the real
    lsof/ss listener scan that the reap depends on."""
    real_popen = subprocess.Popen

    def _popen(cmd, *args, **kwargs):
        if isinstance(cmd, (list, tuple)) and cmd and cmd[0] == "/pm/node":
            return spawn_bridge(cmd, *args, **kwargs)
        return real_popen(cmd, *args, **kwargs)

    return (
        patch("plugins.platforms.whatsapp.adapter.check_whatsapp_requirements", return_value=True),
        patch("aiohttp.ClientSession", _mock_health_refused()),
        patch("plugins.platforms.whatsapp.adapter.asyncio.sleep", new_callable=AsyncMock),
        patch("subprocess.Popen", side_effect=_popen),
        patch.object(adapter, "_acquire_platform_lock", return_value=True, create=True),
    )


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _stop(proc: subprocess.Popen) -> None:
    if proc.poll() is None:
        proc.kill()
    proc.wait()


class TestPortReleasedBeforeSpawn:
    @linux
    @pytest.mark.asyncio
    async def test_slow_exiting_bridge_is_escalated_and_port_is_free_at_spawn(self, tmp_path):
        """The full connect() flow against a real bridge that ignores SIGTERM: the reap waits out the
        grace, SIGKILLs it, and the new bridge is spawned ONCE — with the port already free."""
        from plugins.platforms.whatsapp import adapter as whatsapp_adapter
        from plugins.platforms.whatsapp.bridge_ownership import port_is_free

        bridge_dir = _setup_bridge_dir(tmp_path)
        port = _free_port()
        holder = _spawn_holder(_HOLDER_IGNORING_SIGTERM, port, node_named_dir=tmp_path)
        try:
            _wait_port_held(port, held=True)
            adapter = _make_adapter(str(bridge_dir / "bridge.js"), tmp_path / "session", port)
            spawn_state = {}

            def _record_spawn(*args, **kwargs):
                spawn_state["spawns"] = spawn_state.get("spawns", 0) + 1
                spawn_state["port_free_at_spawn"] = port_is_free(port)
                proc = MagicMock()
                proc.poll.return_value = 1  # the fake bridge exits; only the spawn moment matters here
                proc.returncode = 1
                proc.pid = 4190000  # a PID that is not alive
                return proc

            with patch.object(whatsapp_adapter, "_kill_stale_bridge_by_pidfile", wraps=whatsapp_adapter._kill_stale_bridge_by_pidfile):
                patches = _connect_patches(adapter, _record_spawn)
                with patches[0], patches[1], patches[2], patches[3], patches[4]:
                    await adapter.connect()

            assert adapter._fatal_error_code != "whatsapp_bridge_port_busy"
            assert spawn_state.get("spawns") == 1, "the new bridge was not spawned exactly once"
            assert spawn_state["port_free_at_spawn"] is True, "spawned while the port was still held"
            assert holder.poll() is not None, "the SIGTERM-ignoring holder was never reaped"
            assert holder.returncode == -signal.SIGKILL, f"expected SIGKILL escalation, got {holder.returncode}"
        finally:
            _stop(holder)

    @linux
    @pytest.mark.asyncio
    async def test_stranger_holding_port_is_left_alone_and_no_spawn(self, tmp_path):
        """A non-node holder that never answers /health is not ours: never signalled, and connect()
        fails with the named retryable port-busy error instead of spawning onto a busy port."""
        from plugins.platforms.whatsapp import adapter as whatsapp_adapter

        bridge_dir = _setup_bridge_dir(tmp_path)
        port = _free_port()
        stranger = _spawn_holder(_HOLDER_POLITE, port)
        try:
            _wait_port_held(port, held=True)
            adapter = _make_adapter(str(bridge_dir / "bridge.js"), tmp_path / "session", port)

            with patch.object(whatsapp_adapter, "_kill_stale_bridge_by_pidfile", wraps=whatsapp_adapter._kill_stale_bridge_by_pidfile):
                bridge_spawns = []
                patches = _connect_patches(adapter, lambda *a, **kw: bridge_spawns.append(a) or MagicMock())
                with patches[0], patches[1], patches[2], patches[3], patches[4]:
                    assert await adapter.connect() is False

            assert adapter._fatal_error_code == "whatsapp_bridge_port_busy"
            assert adapter._fatal_error_retryable is True
            assert str(port) in (adapter._fatal_error_message or "")
            assert bridge_spawns == [], "a bridge was spawned onto a port that never came free"
            assert stranger.poll() is None, "a non-node stranger must never be signalled"
        finally:
            _stop(stranger)


class TestListenerDiscovery:
    @linux
    @pytest.mark.asyncio
    async def test_time_wait_from_old_bridge_does_not_block_respawn(self):
        """The old bridge closed its keep-alive connections first, so its port sits in TIME_WAIT
        (up to 60s) with no listener. node's listen (SO_REUSEADDR) succeeds; the wait must too."""
        from plugins.platforms.whatsapp.adapter import WhatsAppAdapter

        srv = socket.socket()
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)  # as libuv does for the bridge
        srv.bind(("127.0.0.1", 0))
        srv.listen(1)
        port = srv.getsockname()[1]
        cli = socket.create_connection(("127.0.0.1", port))
        conn, _ = srv.accept()
        conn.close()  # server side closes first -> TIME_WAIT on 127.0.0.1:<port>
        srv.close()
        time.sleep(0.1)
        cli.close()

        adapter = WhatsAppAdapter.__new__(WhatsAppAdapter)
        adapter.platform = Platform.WHATSAPP
        adapter._bridge_port = port
        adapter._fatal_error_code = adapter._fatal_error_message = None
        adapter._fatal_error_retryable = True
        adapter._fatal_error_handler = None
        assert await adapter._wait_bridge_port_free() is True
        assert adapter._fatal_error_code is None

    @linux
    def test_psutil_fallback_when_lsof_and_ss_missing(self, monkeypatch):
        """Neither lsof nor ss on PATH (the repo image ships neither): the psutil TCP scan still
        finds the listener, so the port holder can be identified and reaped."""
        from plugins.platforms.whatsapp.adapter import _listener_pids_on_port

        srv = socket.socket()
        srv.bind(("127.0.0.1", 0))
        srv.listen(5)
        port = srv.getsockname()[1]
        try:

            def _no_tools(*args, **kwargs):
                raise FileNotFoundError("lsof/ss not installed")

            monkeypatch.setattr("subprocess.run", _no_tools)
            assert os.getpid() in _listener_pids_on_port(port)
        finally:
            srv.close()


class TestPidfileHygiene:
    @linux
    @pytest.mark.asyncio
    async def test_spawn_death_unlinks_pidfile(self, tmp_path):
        """A bridge that dies during the post-spawn health wait must not leave a dead-PID
        bridge.pid: the next connect()'s pidfile reap would find nothing there while the real
        port holder (if any) survives its port scan."""
        from plugins.platforms.whatsapp.adapter import _write_bridge_pidfile

        adapter = _make_adapter("/tmp/test-bridge.js", tmp_path / "session", 19876)
        (tmp_path / "session").mkdir(exist_ok=True)
        _write_bridge_pidfile(tmp_path / "session", 4190000)
        dead_proc = MagicMock()
        dead_proc.poll.return_value = 1
        dead_proc.returncode = 1
        adapter._bridge_process = dead_proc
        with patch("plugins.platforms.whatsapp.adapter.asyncio.sleep", new_callable=AsyncMock), \
             patch.object(adapter, "_probe_bridge_health", side_effect=AssertionError("must not probe a dead process")):
            assert await adapter._wait_for_bridge() is False

        assert not (tmp_path / "session" / "bridge.pid").exists()

    @linux
    def test_pidfile_reap_escalates_to_sigkill(self, tmp_path):
        """A pidfile-identified bridge that ignores SIGTERM is SIGKILLed after the grace window —
        identity re-checked first, so only the recorded bridge itself is escalated."""
        from gateway.status import get_process_start_time
        from plugins.platforms.whatsapp.adapter import _kill_stale_bridge_by_pidfile, _write_bridge_pidfile

        port = _free_port()
        holder = _spawn_holder(_HOLDER_IGNORING_SIGTERM, port, node_named_dir=tmp_path)
        try:
            _wait_port_held(port, held=True)
            _write_bridge_pidfile(tmp_path, holder.pid)
            start = get_process_start_time(holder.pid)
            assert start is not None

            _kill_stale_bridge_by_pidfile(tmp_path)

            # SIGKILL is asynchronous; wait (bounded) for delivery instead of an instant poll().
            with contextlib.suppress(subprocess.TimeoutExpired):
                holder.wait(timeout=5)
            assert holder.poll() is not None, "the SIGTERM-ignoring bridge was never reaped"
            assert holder.returncode == -signal.SIGKILL
            assert not (tmp_path / "bridge.pid").exists()
        finally:
            _stop(holder)


class TestDisconnectWaits:
    @linux
    @pytest.mark.asyncio
    async def test_disconnect_force_kill_waits_for_exit(self, tmp_path):
        """disconnect() escalates to force kill for a bridge that ignores SIGTERM and then WAITS
        for the exit, so an in-process restart does not race the dying bridge for the port."""
        from plugins.platforms.whatsapp.adapter import WhatsAppAdapter

        adapter = WhatsAppAdapter.__new__(WhatsAppAdapter)
        adapter.platform = Platform.WHATSAPP
        adapter._shutting_down = False
        adapter._session_path = tmp_path
        adapter._bridge_log_fh = None
        adapter._bridge_log = None
        adapter._poll_task = None
        adapter._http_session = None
        adapter._release_platform_lock = MagicMock()
        adapter._mark_disconnected = MagicMock()

        port = _free_port()
        holder = _spawn_holder(_HOLDER_IGNORING_SIGTERM, port, node_named_dir=tmp_path)
        try:
            _wait_port_held(port, held=True)
            adapter._bridge_process = holder

            with patch("plugins.platforms.whatsapp.adapter.asyncio.sleep", new_callable=AsyncMock):
                await adapter.disconnect()

            assert holder.poll() is not None, "disconnect() returned before the force-killed bridge exited"
            assert holder.returncode == -signal.SIGKILL
        finally:
            _stop(holder)
