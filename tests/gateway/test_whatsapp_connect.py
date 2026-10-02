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
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gateway.config import Platform
from gateway.platforms.base import SendResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _pm_node(monkeypatch):
    """Stand-in for PM's Node/npm; the user's PATH copy is never picked up."""
    from plugins.platforms.whatsapp import adapter as whatsapp_adapter
    monkeypatch.setattr(whatsapp_adapter, "find_node_executable", lambda name: f"/pm/{name}")


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


def _connect_patches(mock_proc, mock_fh):
    """Return common patches needed to reach the health-check loop."""
    base = [
        patch("plugins.platforms.whatsapp.adapter.check_whatsapp_requirements", return_value=True),
        patch.object(Path, "exists", return_value=True),
        patch.object(Path, "mkdir", return_value=None),
        patch("subprocess.run", return_value=MagicMock(returncode=0)),
        patch("subprocess.Popen", return_value=mock_proc),
        patch("builtins.open", return_value=mock_fh),
        patch("plugins.platforms.whatsapp.adapter.asyncio.sleep", new_callable=AsyncMock),
        patch("plugins.platforms.whatsapp.adapter.asyncio.create_task"),
    ]
    return base


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "windows,winerror,recovered",
    [(True, 5, True), (True, 2, False), (False, 5, False)],
    ids=["windows-job-denied", "other-windows-error", "non-windows"],
)
async def test_bridge_spawn_retries_only_windows_job_denial(windows, winerror, recovered):
    """A job that rejects breakaway must still permit the managed bridge to start."""
    adapter = _make_adapter()
    proc, log = MagicMock(), MagicMock()
    denied = PermissionError("process creation denied")
    denied.winerror = winerror
    with ExitStack() as stack:
        for item in _connect_patches(proc, log):
            stack.enter_context(item)
        stack.enter_context(patch("plugins.platforms.whatsapp.adapter._IS_WINDOWS", windows))
        stack.enter_context(patch.object(adapter, "_acquire_platform_lock", return_value=True))
        stack.enter_context(patch.object(adapter, "_ensure_bridge_deps", return_value=True))
        stack.enter_context(patch.object(adapter, "_reuse_running_bridge", new_callable=AsyncMock, return_value=False))
        stack.enter_context(patch.object(adapter, "_bridge_env", return_value={"PUBLIC_TEST": "1"}))
        stack.enter_context(patch.object(adapter, "_wait_for_bridge", new_callable=AsyncMock, return_value=True))
        stack.enter_context(patch.object(adapter, "_attach_to_bridge"))
        stack.enter_context(patch.object(adapter, "_wire_plugin_handlers"))
        for name in ("_kill_stale_bridge_by_pidfile", "_kill_port_process", "_write_bridge_pidfile"):
            stack.enter_context(patch(f"plugins.platforms.whatsapp.adapter.{name}"))
        stack.enter_context(patch("plugins.platforms.whatsapp.adapter.windows_detach_popen_kwargs", return_value={"creationflags": 0x09000200}))
        spawn = stack.enter_context(patch("subprocess.Popen", side_effect=[denied, proc]))
        result = await adapter.connect()
    assert result is recovered
    assert spawn.call_count == (2 if recovered else 1)
    if recovered:
        first, second = spawn.call_args_list
        assert first.args == second.args
        assert {k:v for k,v in first.kwargs.items() if k != "creationflags"} == {k:v for k,v in second.kwargs.items() if k != "creationflags"}
        assert first.kwargs["creationflags"] & 0x01000000
        assert not second.kwargs["creationflags"] & 0x01000000
        assert adapter._bridge_process is proc


@pytest.mark.asyncio
async def test_bridge_spawn_failed_fallback_releases_lock_and_log():
    """Failed fallback remains a real connection failure and closes its resources."""
    adapter = _make_adapter()
    proc, log = MagicMock(), MagicMock()
    denied = PermissionError("process creation denied")
    denied.winerror = 5
    with ExitStack() as stack:
        for item in _connect_patches(proc, log):
            stack.enter_context(item)
        stack.enter_context(patch("plugins.platforms.whatsapp.adapter._IS_WINDOWS", True))
        stack.enter_context(patch.object(adapter, "_acquire_platform_lock", return_value=True))
        release = stack.enter_context(patch.object(adapter, "_release_platform_lock"))
        stack.enter_context(patch.object(adapter, "_ensure_bridge_deps", return_value=True))
        stack.enter_context(patch.object(adapter, "_reuse_running_bridge", new_callable=AsyncMock, return_value=False))
        stack.enter_context(patch.object(adapter, "_bridge_env", return_value={"PUBLIC_TEST": "1"}))
        for name in ("_kill_stale_bridge_by_pidfile", "_kill_port_process"):
            stack.enter_context(patch(f"plugins.platforms.whatsapp.adapter.{name}"))
        spawn = stack.enter_context(patch("subprocess.Popen", side_effect=[denied, OSError("bridge unavailable")]))
        result = await adapter.connect()
    assert result is False
    assert spawn.call_count == 2
    log.close.assert_called_once()
    release.assert_called_once()


# ---------------------------------------------------------------------------
# _close_bridge_log() unit tests
# ---------------------------------------------------------------------------



# ---------------------------------------------------------------------------
# data variable initialization
# ---------------------------------------------------------------------------

class TestDataInitialized:
    """Verify an unparseable health response cannot leave polling state unbound."""

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

        adapter._bridge_process = mock_proc

        # _probe_bridge_health normalizes an unparseable HTTP 200 body to
        # (True, None). Exercise the polling contract without requiring the
        # optional messaging transport in the base test environment.
        with patch.object(
            adapter,
            "_probe_bridge_health",
            new_callable=AsyncMock,
            return_value=(True, None),
        ), patch(
            "plugins.platforms.whatsapp.adapter.asyncio.sleep",
            new_callable=AsyncMock,
        ):
            result = await adapter._wait_for_bridge()

        # The bridge HTTP server is up, so timeout uses the warn-and-proceed path.
        assert result is True


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
             patches[5], patches[6], patches[7]:
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
        assert adapter.fatal_error_code == "whatsapp_bridge_exited"
        assert adapter.fatal_error_retryable is True
        fatal_handler.assert_awaited_once()
        mock_fh.close.assert_called_once()
        assert adapter._bridge_log_fh is None

    @pytest.mark.asyncio
    @pytest.mark.parametrize("returncode", [-15, -2], ids=["sigterm", "sigint"])
    async def test_signal_exit_during_gateway_signal_shutdown_is_not_fatal(self, returncode):
        """A -15/-2 bridge exit races the stop flow: the signal handler flags the runner
        long before disconnect() flips ``_shutting_down`` (#127047)."""
        from types import SimpleNamespace

        adapter = _make_adapter()
        fatal_handler = AsyncMock()
        adapter.set_fatal_error_handler(fatal_handler)
        adapter._running = True
        adapter._shutting_down = False  # disconnect() has NOT run yet — this is the race
        adapter.gateway_runner = SimpleNamespace(_stop_requested_by_signal=True)
        adapter._bridge_log_fh = MagicMock()

        mock_proc = MagicMock()
        mock_proc.poll.return_value = returncode
        adapter._bridge_process = mock_proc

        assert await adapter._check_managed_bridge_exit() is None
        assert adapter.fatal_error_code is None
        fatal_handler.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_sigterm_exit_without_gateway_shutdown_stays_fatal(self):
        """The runner flag must not mask a genuine crash: a bridge SIGTERMed while the
        gateway keeps running still queues the retryable reconnect."""
        from types import SimpleNamespace

        adapter = _make_adapter()
        fatal_handler = AsyncMock()
        adapter.set_fatal_error_handler(fatal_handler)
        adapter._running = True
        adapter._shutting_down = False
        adapter.gateway_runner = SimpleNamespace(_stop_requested_by_signal=False)
        adapter._bridge_log_fh = MagicMock()

        mock_proc = MagicMock()
        mock_proc.poll.return_value = -15
        adapter._bridge_process = mock_proc

        assert await adapter._check_managed_bridge_exit() is not None
        assert adapter.fatal_error_code == "whatsapp_bridge_exited"
        assert adapter.fatal_error_retryable is True
        fatal_handler.assert_awaited_once()

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

        adapter._http_session = MagicMock()

        with patch.object(
            adapter,
            "_post_bridge_message",
            new_callable=AsyncMock,
            return_value=SendResult(success=True, message_id="msg-1"),
        ) as mock_post:
            result = await adapter.send("+15551234567", "hello")

        assert result.success is True
        mock_post.assert_awaited_once_with(
            "send",
            {"chatId": "15551234567@s.whatsapp.net", "message": "hello"},
            timeout=30,
        )


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

        mock_fh = MagicMock()
        adapter._bridge_process = mock_proc
        adapter._bridge_log_fh = mock_fh

        # A healthy HTTP endpoint with status != connected exhausts Phase 1.
        # The managed bridge then dies on the first Phase 2 poll.
        with patch.object(
            adapter,
            "_probe_bridge_health",
            new_callable=AsyncMock,
            return_value=(True, {"status": "disconnected"}),
        ), patch(
            "plugins.platforms.whatsapp.adapter.asyncio.sleep",
            new_callable=AsyncMock,
        ):
            result = await adapter._wait_for_bridge()

        assert result is False
        mock_fh.close.assert_called_once()
        assert adapter._bridge_log_fh is None


# ---------------------------------------------------------------------------
# _kill_port_process() cross-platform tests
# ---------------------------------------------------------------------------

class TestKillPortProcess:
    """Verify _kill_port_process uses platform-appropriate commands."""

    @pytest.mark.platforms("windows")
    def test_uses_netstat_and_taskkill_on_windows(self):
        """``platforms("windows")``: netstat/taskkill are Windows binaries. The old
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

    @pytest.mark.platforms("windows")
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


    @pytest.mark.platforms("linux")
    def test_kills_only_listeners_on_linux(self):
        """POSIX path SIGTERMs only LISTENer PIDs (never clients) — the #43846 fix.

        Replaces the old fuser-based test: ``fuser``/bare ``lsof -i`` also
        matched client sockets sharing the port number, which closed unrelated
        processes (a browser tab on the same port). The implementation now
        resolves listeners via ``_listener_pids_on_port`` and signals only those.

        ``platforms("linux")``: asserts the POSIX ``os.kill``/SIGTERM path, which is
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

    @pytest.mark.platforms("linux")
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
    @pytest.mark.platforms("windows")
    async def test_disconnect_uses_taskkill_tree_on_windows(self):
        """Windows disconnect should target the bridge process tree, not just the parent PID.

        ``platforms("windows")``: ``taskkill /T`` is the Windows tree-kill primitive;
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

        mock_run.assert_called_once()
        assert mock_run.call_args.args[0] == ["taskkill", "/PID", "12345", "/T"]
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
        bridge.write_text("// stub", encoding="utf-8")
        adapter._bridge_script = str(bridge)
        session_dir = tmp_path / "session"
        session_dir.mkdir()
        (session_dir / "creds.json").write_text("{}", encoding="utf-8")
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
