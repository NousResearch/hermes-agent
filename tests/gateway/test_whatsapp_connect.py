"""Tests for WhatsApp connect() error handling.

Regression tests for two bugs in WhatsAppAdapter.connect():

1. Uninitialized ``data`` variable: when ``resp.json()`` raised after the
   health endpoint returned HTTP 200, ``http_ready`` was set to True but
   ``data`` was never assigned.  The subsequent ``data.get("status")``
   check raised ``NameError``.

2. Bridge log file handle leaked on error paths: the file was opened before
   the health-check loop but never closed when ``connect()`` returned False.
   Repeated connection failures accumulated open file descriptors.
"""

import asyncio
import signal
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gateway.config import Platform


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _AsyncCM:
    """Minimal async context manager returning a fixed value."""

    def __init__(self, value):
        self.value = value

    async def __aenter__(self):
        return self.value

    async def __aexit__(self, *exc):
        return False


def _make_adapter():
    """Create a WhatsAppAdapter with test attributes (bypass __init__)."""
    from plugins.platforms.whatsapp.adapter import WhatsAppAdapter

    adapter = WhatsAppAdapter.__new__(WhatsAppAdapter)
    adapter.platform = Platform.WHATSAPP
    adapter.config = MagicMock()
    adapter._bridge_port = 19876
    adapter._bridge_script = "/tmp/test-bridge.js"
    adapter._session_path = Path("/tmp/test-wa-session")
    adapter._bridge_log_fh = None
    adapter._bridge_log = None
    adapter._bridge_process = None
    adapter._reply_prefix = None
    adapter._send_read_receipts = False
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


def _mock_aiohttp(status=200, json_data=None, json_side_effect=None):
    """Build a mock ``aiohttp.ClientSession`` returning a fixed response."""
    mock_resp = MagicMock()
    mock_resp.status = status
    if json_side_effect:
        mock_resp.json = AsyncMock(side_effect=json_side_effect)
    else:
        mock_resp.json = AsyncMock(return_value=json_data or {})

    mock_session = MagicMock()
    mock_session.get = MagicMock(return_value=_AsyncCM(mock_resp))

    return MagicMock(return_value=_AsyncCM(mock_session))


def _connect_patches(mock_proc, mock_fh, mock_client_cls=None):
    """Return a dict of common patches needed to reach the health-check loop."""
    patches = {
        "plugins.platforms.whatsapp.adapter.check_whatsapp_requirements": True,
        "plugins.platforms.whatsapp.adapter.asyncio.create_task": MagicMock(),
    }
    base = [
        patch("plugins.platforms.whatsapp.adapter.check_whatsapp_requirements", return_value=True),
        patch.object(Path, "exists", return_value=True),
        patch.object(Path, "mkdir", return_value=None),
        patch("subprocess.run", return_value=MagicMock(returncode=0)),
        patch("subprocess.Popen", return_value=mock_proc),
        patch("builtins.open", return_value=mock_fh),
        patch("plugins.platforms.whatsapp.adapter.asyncio.sleep", new_callable=AsyncMock),
        patch("plugins.platforms.whatsapp.adapter.asyncio.create_task"),
        patch("plugins.platforms.whatsapp.adapter._wait_port_free", new_callable=AsyncMock, return_value=True),
    ]
    if mock_client_cls is not None:
        base.append(patch("aiohttp.ClientSession", mock_client_cls))
    return base


# ---------------------------------------------------------------------------
# _close_bridge_log() unit tests
# ---------------------------------------------------------------------------

class TestCloseBridgeLog:
    """Direct tests for the _close_bridge_log() helper method."""

    @staticmethod
    def _bare_adapter():
        from plugins.platforms.whatsapp.adapter import WhatsAppAdapter
        a = WhatsAppAdapter.__new__(WhatsAppAdapter)
        a._bridge_log_fh = None
        return a

    def test_closes_open_handle(self):
        adapter = self._bare_adapter()
        mock_fh = MagicMock()
        adapter._bridge_log_fh = mock_fh

        adapter._close_bridge_log()

        mock_fh.close.assert_called_once()
        assert adapter._bridge_log_fh is None


# ---------------------------------------------------------------------------
# data variable initialization
# ---------------------------------------------------------------------------

class TestDataInitialized:
    """Verify ``data = {}`` prevents NameError when resp.json() fails."""

    @pytest.mark.asyncio
    async def test_no_name_error_when_json_always_fails(self):
        """HTTP 200 sets http_ready but json() always raises.

        Without the fix, ``data`` was never assigned and the Phase 2 check
        ``data.get("status")`` raised NameError.  With ``data = {}``, the
        check evaluates to ``None != "connected"`` and Phase 2 runs normally.
        """
        adapter = _make_adapter()

        mock_proc = MagicMock()
        mock_proc.poll.return_value = None  # bridge stays alive

        mock_client_cls = _mock_aiohttp(
            status=200, json_side_effect=ValueError("bad json"),
        )
        mock_fh = MagicMock()

        patches = _connect_patches(mock_proc, mock_fh, mock_client_cls)

        with patches[0], patches[1], patches[2], patches[3], patches[4], \
             patches[5], patches[6], patches[7], patches[8], patches[9], \
             patch.object(type(adapter), "_poll_messages", return_value=MagicMock()):
            # Must NOT raise NameError
            result = await adapter.connect()

        # connect() returns True (warn-and-proceed path)
        assert result is True
        assert adapter._running is True


# ---------------------------------------------------------------------------
# File handle cleanup on error paths
# ---------------------------------------------------------------------------

class TestFileHandleClosedOnError:
    """Verify the bridge log file handle is closed on every failure path."""

    @pytest.mark.asyncio
    async def test_closed_when_bridge_dies_phase1(self):
        """Bridge process exits during Phase 1 health-check loop."""
        adapter = _make_adapter()

        mock_proc = MagicMock()
        mock_proc.poll.return_value = 1  # dead immediately
        mock_proc.returncode = 1

        mock_fh = MagicMock()
        patches = _connect_patches(mock_proc, mock_fh)

        with patches[0], patches[1], patches[2], patches[3], patches[4], \
             patches[5], patches[6], patches[7], patches[8]:
            result = await adapter.connect()

        assert result is False
        mock_fh.close.assert_called_once()
        assert adapter._bridge_log_fh is None


class TestConnectCleanup:
    """Verify failure paths release the scoped session lock."""

    @pytest.mark.asyncio
    async def test_releases_lock_when_npm_install_fails(self):
        adapter = _make_adapter()

        def _path_exists(path_obj):
            return not str(path_obj).endswith("node_modules")

        install_result = MagicMock(returncode=1, stderr="install failed")

        with patch("plugins.platforms.whatsapp.adapter.check_whatsapp_requirements", return_value=True), \
             patch.object(Path, "exists", autospec=True, side_effect=_path_exists), \
             patch("subprocess.run", return_value=install_result), \
             patch("gateway.status.acquire_scoped_lock", return_value=(True, None)), \
             patch("gateway.status.release_scoped_lock") as mock_release:
            result = await adapter.connect()

        assert result is False
        assert adapter.fatal_error_code == "whatsapp_npm_install_failed"
        assert adapter.fatal_error_retryable is False
        assert "npm install failed" in (adapter.fatal_error_message or "")
        mock_release.assert_called_once_with("whatsapp-session", str(adapter._session_path))
        assert adapter._platform_lock_identity is None


class TestBridgeRuntimeFailure:
    """Verify runtime bridge death is surfaced as a fatal adapter error."""

    @pytest.mark.asyncio
    async def test_send_marks_retryable_fatal_when_managed_bridge_exits(self):
        adapter = _make_adapter()
        fatal_handler = AsyncMock()
        adapter.set_fatal_error_handler(fatal_handler)
        adapter._running = True
        adapter._http_session = MagicMock()  # Persistent session active
        mock_fh = MagicMock()
        adapter._bridge_log_fh = mock_fh

        mock_proc = MagicMock()
        mock_proc.poll.return_value = 7
        adapter._bridge_process = mock_proc

        result = await adapter.send("chat-123", "hello")

        assert result.success is False
        assert "exited unexpectedly" in result.error
        assert adapter.fatal_error_code == "whatsapp_bridge_exited"
        assert adapter.fatal_error_retryable is True
        fatal_handler.assert_awaited_once()
        mock_fh.close.assert_called_once()
        assert adapter._bridge_log_fh is None

    @pytest.mark.asyncio
    async def test_send_normalizes_bare_phone_numbers_to_jid(self):
        """A bare phone target (with or without +) becomes a full JID.

        Baileys' jidDecode crashes on a bare number (#8637); the adapter
        must rewrite it to ``<digits>@s.whatsapp.net`` before the bridge
        call. Regression guard for that crash.
        """
        adapter = _make_adapter()
        adapter._running = True
        adapter._bridge_process = None  # unmanaged bridge — skip exit check

        mock_resp = MagicMock()
        mock_resp.status = 200
        mock_resp.json = AsyncMock(return_value={"messageId": "msg-1"})
        mock_session = MagicMock()
        mock_session.post = MagicMock(return_value=_AsyncCM(mock_resp))
        adapter._http_session = mock_session

        result = await adapter.send("+50766715226", "hello")

        assert result.success is True
        payload = mock_session.post.call_args.kwargs["json"]
        assert payload["chatId"] == "50766715226@s.whatsapp.net"


    @pytest.mark.asyncio
    async def test_closed_when_bridge_dies_phase2(self):
        """Bridge alive during Phase 1 but dies during Phase 2."""
        adapter = _make_adapter()

        # Phase 1 (15 iterations): alive.  Phase 2 (iteration 16): dead.
        call_count = [0]

        def poll_side_effect():
            call_count[0] += 1
            return None if call_count[0] <= 15 else 1

        mock_proc = MagicMock()
        mock_proc.poll.side_effect = poll_side_effect
        mock_proc.returncode = 1

        # Health returns 200 with status != "connected" -> triggers Phase 2
        mock_client_cls = _mock_aiohttp(
            status=200, json_data={"status": "disconnected"},
        )
        mock_fh = MagicMock()
        patches = _connect_patches(mock_proc, mock_fh, mock_client_cls)

        with patches[0], patches[1], patches[2], patches[3], patches[4], \
             patches[5], patches[6], patches[7], patches[8], patches[9]:
            result = await adapter.connect()

        assert result is False
        mock_fh.close.assert_called_once()
        assert adapter._bridge_log_fh is None


# ---------------------------------------------------------------------------
# _kill_port_process() cross-platform tests
# ---------------------------------------------------------------------------

class TestKillPortProcess:
    """Verify _kill_port_process uses platform-appropriate commands."""

    @pytest.mark.windows_only
    def test_uses_netstat_and_taskkill_on_windows(self):
        """``windows_only``: netstat/taskkill are Windows binaries. The old
        ``_IS_WINDOWS`` patch selected this branch on Linux, where neither
        exists, so the mocked argv was the only thing under test."""
        from plugins.platforms.whatsapp.adapter import _kill_port_process

        netstat_output = (
            "  Proto  Local Address          Foreign Address        State           PID\n"
            "  TCP    0.0.0.0:3000           0.0.0.0:0              LISTENING       12345\n"
            "  TCP    0.0.0.0:3001           0.0.0.0:0              LISTENING       99999\n"
        )
        mock_netstat = MagicMock(stdout=netstat_output)
        mock_taskkill = MagicMock()

        def run_side_effect(cmd, **kwargs):
            if cmd[0] == "netstat":
                return mock_netstat
            if cmd[0] == "taskkill":
                return mock_taskkill
            return MagicMock()

        with patch("plugins.platforms.whatsapp.adapter.subprocess.run", side_effect=run_side_effect) as mock_run, \
             patch("plugins.platforms.whatsapp.adapter._pid_looks_like_node_bridge",
                   return_value=True):
            _kill_port_process(3000)

        # netstat called
        assert any(
            call.args[0][0] == "netstat" for call in mock_run.call_args_list
        )
        # taskkill called with correct PID
        assert any(
            call.args[0] == ["taskkill", "/PID", "12345", "/F"]
            for call in mock_run.call_args_list
        )

    @pytest.mark.windows_only
    def test_windows_refuses_taskkill_on_non_bridge_pid(self):
        """#89614 class: the netstat-scanned PID is a bare number — if the
        live process is not a node bridge, taskkill must never fire."""
        from plugins.platforms.whatsapp.adapter import _kill_port_process

        netstat_output = (
            "  Proto  Local Address          Foreign Address        State           PID\n"
            "  TCP    0.0.0.0:3000           0.0.0.0:0              LISTENING       12345\n"
        )

        def run_side_effect(cmd, **kwargs):
            if cmd[0] == "netstat":
                return MagicMock(stdout=netstat_output)
            return MagicMock()

        with patch("plugins.platforms.whatsapp.adapter.subprocess.run", side_effect=run_side_effect) as mock_run, \
             patch("plugins.platforms.whatsapp.adapter._pid_looks_like_node_bridge",
                   return_value=False):
            _kill_port_process(3000)

        assert not any(
            call.args[0][0] == "taskkill" for call in mock_run.call_args_list
        )


    @pytest.mark.linux_only
    def test_kills_only_listeners_on_linux(self):
        """POSIX path SIGTERMs only LISTENer PIDs (never clients) — the #43846 fix.

        Replaces the old fuser-based test: ``fuser``/bare ``lsof -i`` also
        matched client sockets sharing the port number, which closed unrelated
        processes (a browser tab on the same port). The implementation now
        resolves listeners via ``_listener_pids_on_port`` and signals only those.

        ``linux_only``: asserts the POSIX ``os.kill``/SIGTERM path, which is
        genuinely selected here without patching ``_IS_WINDOWS``.
        """
        from plugins.platforms.whatsapp import adapter as wa

        kills = []
        with patch("plugins.platforms.whatsapp.adapter._listener_pids_on_port",
                   return_value=[55555]) as mock_listeners, \
             patch("plugins.platforms.whatsapp.adapter._pid_looks_like_node_bridge",
                   return_value=True), \
             patch("plugins.platforms.whatsapp.adapter.os.kill",
                   side_effect=lambda pid, sig: kills.append((pid, sig))):
            wa._kill_port_process(3000)

        mock_listeners.assert_called_once_with(3000)
        assert kills == [(55555, signal.SIGTERM)]

    @pytest.mark.linux_only
    def test_non_bridge_listener_is_never_killed(self):
        """#89614 class: a listener that is not a node bridge is refused."""
        from plugins.platforms.whatsapp import adapter as wa

        kills = []
        with patch("plugins.platforms.whatsapp.adapter._listener_pids_on_port",
                   return_value=[55555]), \
             patch("plugins.platforms.whatsapp.adapter._pid_looks_like_node_bridge",
                   return_value=False), \
             patch("plugins.platforms.whatsapp.adapter.os.kill",
                   side_effect=lambda pid, sig: kills.append((pid, sig))):
            wa._kill_port_process(3000)

        assert kills == []


# ---------------------------------------------------------------------------
# Persistent HTTP session lifecycle
# ---------------------------------------------------------------------------

class TestHttpSessionLifecycle:
    """Verify persistent aiohttp.ClientSession is created and cleaned up."""

    @pytest.mark.asyncio
    @pytest.mark.windows_only
    async def test_disconnect_uses_taskkill_tree_on_windows(self):
        """Windows disconnect should target the bridge process tree, not just the parent PID.

        ``windows_only``: ``taskkill /T`` is the Windows tree-kill primitive;
        on Linux the branch was reachable only by faking ``_IS_WINDOWS``.
        """
        adapter = _make_adapter()
        mock_proc = MagicMock()
        mock_proc.pid = 12345
        mock_proc.poll.side_effect = [0]
        adapter._bridge_process = mock_proc
        adapter._poll_task = None
        adapter._http_session = None
        adapter._running = True
        adapter._session_lock_identity = None

        with patch("plugins.platforms.whatsapp.adapter.subprocess.run", return_value=MagicMock(returncode=0)) as mock_run, \
             patch("plugins.platforms.whatsapp.adapter.asyncio.sleep", new_callable=AsyncMock):
            await adapter.disconnect()

        mock_run.assert_called_once_with(
            ["taskkill", "/PID", "12345", "/T"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=10,
        )
        mock_proc.terminate.assert_not_called()
        mock_proc.kill.assert_not_called()

    @pytest.mark.asyncio
    async def test_session_closed_on_disconnect(self):
        """disconnect() should close self._http_session."""
        adapter = _make_adapter()
        mock_session = AsyncMock()
        mock_session.closed = False
        adapter._http_session = mock_session
        adapter._poll_task = None
        adapter._bridge_process = None
        adapter._running = True
        adapter._session_lock_identity = None

        await adapter.disconnect()

        mock_session.close.assert_called_once()
        assert adapter._http_session is None


# ---------------------------------------------------------------------------
# Pre-flight: refuse to start the bridge when creds.json is missing
# ---------------------------------------------------------------------------


class TestNoCredsPreflight:
    """Verify ``connect()`` fast-fails as non-retryable when WhatsApp is
    enabled but the user never finished pairing (no ``creds.json``).

    Without this guard, every gateway boot:
      • spawned the bridge subprocess (npm install if needed)
      • waited 30s for status:connected (never happens without creds)
      • queued WhatsApp for indefinite retries that would just repeat
    With the guard, ``connect()`` returns False immediately with a
    non-retryable fatal error so the reconnect watcher drops the platform
    and the gateway gets a single clear log line telling the user to run
    ``hermes whatsapp``.
    """


    @pytest.mark.asyncio
    async def test_connect_proceeds_when_creds_present(self, tmp_path):
        """When creds.json exists, the preflight check is bypassed and
        connect() proceeds to the bridge bootstrap path. We don't fully
        simulate the bridge here — we just verify no fast-fail occurs.
        """
        from plugins.platforms.whatsapp.adapter import WhatsAppAdapter

        adapter = WhatsAppAdapter.__new__(WhatsAppAdapter)
        adapter.platform = Platform.WHATSAPP
        adapter.config = MagicMock()
        adapter._bridge_port = 19877
        bridge = tmp_path / "bridge.js"
        bridge.write_text("// stub")
        adapter._bridge_script = str(bridge)
        session_dir = tmp_path / "session"
        session_dir.mkdir()
        (session_dir / "creds.json").write_text("{}")
        adapter._session_path = session_dir
        adapter._bridge_log_fh = None
        adapter._fatal_error_code = None
        adapter._fatal_error_message = None
        adapter._fatal_error_retryable = True
        # Stub _acquire_platform_lock to return False so connect() exits
        # cleanly *after* the preflight, without spawning subprocesses.
        adapter._acquire_platform_lock = MagicMock(return_value=False)

        with patch(
            "plugins.platforms.whatsapp.adapter.check_whatsapp_requirements",
            return_value=True,
        ):
            result = await adapter.connect()

        # Preflight passed — exits because we faked lock acquisition,
        # but the fatal-error code is NOT the "not paired" one.
        assert result is False
        assert adapter._fatal_error_code != "whatsapp_not_paired"


# ---------------------------------------------------------------------------
# Port-free wait (gateway restart / :3000 race)
# ---------------------------------------------------------------------------

class TestWaitPortFree:
    """The new-gateway vs leftover-bridge race on the WhatsApp port."""

    def test_windows_probe_reads_netstat_listeners(self):
        from plugins.platforms.whatsapp.adapter import _windows_listener_pids

        netstat_output = (
            "  Proto  Local Address          Foreign Address        State           PID\n"
            "  TCP    127.0.0.1:3000         0.0.0.0:0              LISTENING       12345\n"
            "  TCP    127.0.0.1:3001         0.0.0.0:0              LISTENING       99999\n"
        )
        with patch(
            "plugins.platforms.whatsapp.adapter.subprocess.run",
            return_value=MagicMock(stdout=netstat_output),
        ):
            pids = _windows_listener_pids(3000)

        assert pids == [12345]

    def test_bind_probe_reports_occupied_port(self):
        import socket

        from plugins.platforms.whatsapp.adapter import _port_is_bindable

        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
            listener.bind(("127.0.0.1", 0))
            port = listener.getsockname()[1]
            listener.listen(1)

            assert _port_is_bindable(port) is False

    @pytest.mark.linux_only
    def test_bind_probe_tolerates_time_wait_residue(self):
        """A killed bridge leaves TIME_WAIT sockets on the port.
        The Node bridge binds with SO_REUSEADDR (Node's default).  After
        disconnect() SIGTERMs it, its accepted connections linger in
        TIME_WAIT for ~60s; a plain bind() then fails EADDRINUSE while a
        SO_REUSEADDR bind succeeds.  The probe must mirror the bridge's bind
        semantics, or every ordinary restart misfires as "port busy".
        An active LISTEN socket still defeats a SO_REUSEADDR bind, so the
        duplicate-bridge protection is unaffected.
        """
        import socket
        import time

        from plugins.platforms.whatsapp.adapter import _port_is_bindable

        listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
        listener.listen(5)
        client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client.settimeout(2)
        client.connect(("127.0.0.1", port))
        conn, _ = listener.accept()
        # Server-side active close puts this tuple into TIME_WAIT.
        conn.close()
        client.close()
        listener.close()

        # Wait until a plain bind hits EADDRINUSE, proving the residue exists
        # (otherwise the test would pass vacuously on a free port).
        deadline = time.monotonic() + 2.0
        residue_seen = False
        while time.monotonic() < deadline:
            probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            try:
                try:
                    probe.bind(("127.0.0.1", port))
                except OSError:
                    residue_seen = True
                    break
            finally:
                probe.close()
            time.sleep(0.05)
        if not residue_seen:
            pytest.skip("could not produce TIME_WAIT residue on this host")

        assert _port_is_bindable(port) is True

    @pytest.mark.asyncio
    async def test_probe_failure_is_not_reported_as_free(self):
        from plugins.platforms.whatsapp.adapter import _wait_port_free

        with patch("plugins.platforms.whatsapp.adapter._port_is_bindable", return_value=None), \
             patch("plugins.platforms.whatsapp.adapter.asyncio.sleep", new_callable=AsyncMock):
            result = await _wait_port_free(3000, timeout=0.3, interval=0.1)

        assert result is False

    @pytest.mark.asyncio
    async def test_stops_when_monotonic_deadline_passes(self):
        from plugins.platforms.whatsapp.adapter import _wait_port_free

        fake_loop = MagicMock()
        fake_loop.time.side_effect = [100.0, 111.0]
        with patch("plugins.platforms.whatsapp.adapter.asyncio.get_running_loop", return_value=fake_loop), \
             patch("plugins.platforms.whatsapp.adapter.asyncio.to_thread", new_callable=AsyncMock, return_value=False) as probe, \
             patch("plugins.platforms.whatsapp.adapter.asyncio.sleep", new_callable=AsyncMock) as sleep:
            result = await _wait_port_free(3000, timeout=10.0, interval=0.2)

        assert result is False
        probe.assert_awaited_once()
        sleep.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_returns_immediately_when_port_is_free(self):
        from plugins.platforms.whatsapp.adapter import _wait_port_free

        with patch("plugins.platforms.whatsapp.adapter._port_is_bindable", return_value=True):
            ok = await _wait_port_free(3000, timeout=2.0, interval=0.2)
        assert ok is True

    @pytest.mark.asyncio
    async def test_gives_up_after_bounded_rounds_when_busy(self):
        from plugins.platforms.whatsapp.adapter import _wait_port_free

        sleeps = []

        async def fake_sleep(seconds):
            sleeps.append(seconds)

        with patch("plugins.platforms.whatsapp.adapter._port_is_bindable", return_value=False), \
             patch("plugins.platforms.whatsapp.adapter.asyncio.sleep", side_effect=fake_sleep):
            ok = await _wait_port_free(3000, timeout=1.0, interval=0.2)
        assert ok is False
        assert len(sleeps) == 5

    @pytest.mark.asyncio
    async def test_poll_loop_stops_quietly_while_shutting_down(self):
        adapter = _make_adapter()
        adapter._running = True
        adapter._shutting_down = True
        adapter._bridge_process = None
        adapter._http_session = MagicMock()
        adapter._http_session.get = MagicMock(
            side_effect=AssertionError("poll must not hit the bridge while shutting down")
        )

        await adapter._poll_messages()
