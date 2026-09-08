"""Tests for /update gateway slash command.

Tests both the _handle_update_command handler (spawns update process) and
the _send_update_notification startup hook (sends results after restart).
"""

import hashlib
import asyncio
import concurrent.futures
import json
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from unittest.mock import patch, MagicMock, AsyncMock

import pytest

from gateway.config import Platform
from gateway.platforms.event import MessageEvent
from gateway.session import SessionSource
from gateway.update_contract import UpdateHelperLock, acquire_update_helper_probe


def _make_event(text="/update", platform=Platform.TELEGRAM,
                user_id="12345", chat_id="67890", thread_id=None):
    """Build a MessageEvent for testing."""
    source = SessionSource(
        platform=platform,
        user_id=user_id,
        chat_id=chat_id,
        user_name="testuser",
        thread_id=thread_id,
    )
    return MessageEvent(text=text, source=source)


def _make_runner():
    """Create a bare GatewayRunner without calling __init__."""
    from gateway.run import GatewayRunner
    runner = object.__new__(GatewayRunner)
    runner.adapters = {}
    runner._voice_mode = {}
    runner._update_prompt_pending = {}
    return runner


# ---------------------------------------------------------------------------
# _handle_update_command
# ---------------------------------------------------------------------------


class TestHandleUpdateCommand:
    """Tests for GatewayRunner._handle_update_command."""

    @pytest.mark.asyncio
    async def test_no_git_directory(self, tmp_path):
        """Returns an error when .git does not exist."""
        runner = _make_runner()
        event = _make_event()
        # Point _hermes_home to tmp_path and project_root to a dir without .git
        fake_root = tmp_path / "project"
        fake_root.mkdir()
        with patch("gateway.run._hermes_home", tmp_path), \
             patch("gateway.run.Path") as MockPath:
            # Path(__file__).parent.parent.resolve() -> fake_root
            MockPath.return_value = MagicMock()
            MockPath.__truediv__ = Path.__truediv__
            # Easier: just patch the __file__ resolution in the method
            pass

        # Simpler approach — mock at method level using a wrapper
        runner = _make_runner()

        with patch("gateway.run._hermes_home", tmp_path):
            # The handler does Path(__file__).parent.parent.resolve()
            # We need to make project_root / '.git' not exist.
            # Since Path(__file__) resolves to the real gateway/run.py,
            # project_root will be the real hermes-agent dir (which HAS .git).
            # Patch Path to control this.
            original_path = Path

            class FakePath(type(Path())):
                pass

            # Actually, simplest: just patch the specific file attr.
            # The _handle_update_command handler lives in gateway/slash_commands.py
            # (extracted from run.py in the god-file decomposition); it resolves
            # project_root via Path(__file__).parent.parent, so fake that file.
            fake_file = str(fake_root / "gateway" / "slash_commands.py")
            (fake_root / "gateway").mkdir(parents=True)
            (fake_root / "gateway" / "slash_commands.py").touch()

            with patch("gateway.slash_commands.__file__", fake_file):
                result = await runner._handle_update_command(event)

        assert "Not a git repository" in result


    @pytest.mark.asyncio
    async def test_resolve_hermes_bin_fallback(self):
        """_resolve_hermes_bin falls back to sys.executable argv when which fails."""
        import sys
        from gateway.run import _resolve_hermes_bin

        fake_spec = MagicMock()
        with patch("shutil.which", return_value=None), \
             patch("importlib.util.find_spec", return_value=fake_spec):
            result = _resolve_hermes_bin()

        assert result == [sys.executable, "-m", "hermes_cli.main"]


    @pytest.mark.asyncio
    async def test_writes_pending_marker(self, tmp_path):
        """Writes .update_pending.json with correct platform and chat info."""
        runner = _make_runner()
        event = _make_event(platform=Platform.TELEGRAM, chat_id="99999")
        event.message_id = "m-update"

        fake_root = tmp_path / "project"
        fake_root.mkdir()
        (fake_root / ".git").mkdir()
        (fake_root / "gateway").mkdir()
        (fake_root / "gateway" / "run.py").touch()
        fake_file = str(fake_root / "gateway" / "run.py")
        hermes_home = tmp_path / "hermes"
        hermes_home.mkdir()

        with patch("gateway.run._hermes_home", hermes_home), \
             patch("gateway.run.__file__", fake_file), \
             patch("gateway.slash_commands.sys.platform", "linux"), \
             patch("shutil.which", side_effect=lambda x: "/usr/bin/hermes" if x == "hermes" else "/usr/bin/setsid"), \
             patch("subprocess.Popen") as mock_popen:
            result = await runner._handle_update_command(event)

        pending_path = hermes_home / ".update_pending.json"
        assert pending_path.exists()
        data = json.loads(pending_path.read_text())
        assert data["platform"] == "telegram"
        assert data["chat_id"] == "99999"
        assert data["chat_type"] == "dm"
        assert data["message_id"] == "m-update"
        assert "timestamp" in data
        assert len(data["handoff_token_sha256"]) == 64
        assert not (hermes_home / ".update_exit_code").exists()

        popen_command = mock_popen.call_args[0][0]
        popen_kwargs = mock_popen.call_args.kwargs
        assert all("handoff_token" not in str(arg).lower() for arg in popen_command)
        assert "HERMES_UPDATE_HANDOFF_TOKEN" not in str(popen_kwargs)
        assert popen_kwargs["stdin"] is subprocess.PIPE
        helper_stdin = mock_popen.return_value.stdin
        written = helper_stdin.write.call_args.args[0]
        assert isinstance(written, bytes)
        assert written.endswith(b"\n")
        helper_stdin.close.assert_called_once_with()


    @pytest.mark.asyncio
    async def test_windows_handoff_passes_token_to_helper(self, tmp_path):
        runner = _make_runner()
        event = _make_event()
        fake_root = tmp_path / "project"
        (fake_root / ".git").mkdir(parents=True)
        fake_file = str(fake_root / "gateway" / "run.py")
        hermes_home = tmp_path / "hermes"
        hermes_home.mkdir()

        with patch("gateway.run._hermes_home", hermes_home), \
             patch("gateway.run.__file__", fake_file), \
             patch("gateway.slash_commands.sys.platform", "win32"), \
             patch("shutil.which", return_value="C:/hermes.exe"), \
             patch("subprocess.Popen") as mock_popen:
            await runner._handle_update_command(event)

        command = mock_popen.call_args[0][0]
        popen_kwargs = mock_popen.call_args.kwargs
        pending = json.loads((hermes_home / ".update_pending.json").read_text())
        assert all("handoff_token" not in str(arg).lower() for arg in command)
        assert "HERMES_UPDATE_HANDOFF_TOKEN" not in str(popen_kwargs)
        assert popen_kwargs["stdin"] is subprocess.PIPE
        assert len(pending["handoff_token_sha256"]) == 64

    @pytest.mark.asyncio
    @pytest.mark.live_system_guard_bypass
    async def test_real_helper_lock_done_release_and_watcher_cleanup(self, tmp_path):
        runner = _make_runner()
        runner._schedule_update_notification_watch = MagicMock()
        adapter = AsyncMock()
        runner.adapters = {Platform.TELEGRAM: adapter}
        event = _make_event(chat_id="integration-chat")
        fake_root = tmp_path / "project"
        (fake_root / ".git").mkdir(parents=True)
        fake_file = str(fake_root / "gateway" / "run.py")
        hermes_home = tmp_path / "hermes"
        hermes_home.mkdir()

        with patch("gateway.run._hermes_home", hermes_home), \
             patch("gateway.run.__file__", fake_file), \
             patch("gateway.slash_commands.sys.platform", "win32"), \
             patch("shutil.which", return_value="C:/hermes.exe"), \
             patch("subprocess.Popen") as mocked_popen:
            await runner._handle_update_command(event)

        pending = json.loads(
            (hermes_home / ".update_pending.json").read_text(encoding="utf-8")
        )
        helper_source = mocked_popen.call_args.args[0][2]
        lock_path = hermes_home / ".update_helper.lock"
        done_path = hermes_home / ".update_helper_done.json"
        output_path = hermes_home / ".update_output.txt"
        exit_path = hermes_home / ".update_exit_code"
        helper = subprocess.Popen(
            [
                sys.executable,
                "-c",
                helper_source,
                str(output_path),
                str(exit_path),
                str(lock_path),
                str(done_path),
                pending["request_id"],
                "5",
                sys.executable,
                "-c",
                "import time; print('fake update child'); time.sleep(0.5)",
            ],
            stdin=subprocess.PIPE,
        )
        assert helper.stdin is not None
        helper.stdin.write(b"integration-token\n")
        helper.stdin.close()

        deadline = time.monotonic() + 5
        lock_was_held = False
        while time.monotonic() < deadline:
            probe = acquire_update_helper_probe(lock_path)
            if probe is None:
                lock_was_held = True
                break
            probe.release()
            time.sleep(0.01)
        assert lock_was_held
        assert not done_path.exists()
        assert helper.wait(timeout=5) == 0

        done = json.loads(done_path.read_text(encoding="utf-8"))
        assert done["request_id"] == pending["request_id"]
        assert done["exit_code"] == 0
        assert done["status"] == "completed"
        assert not list(hermes_home.glob(".update_helper_done.json.*.tmp"))
        probe = acquire_update_helper_probe(lock_path)
        assert probe is not None
        probe.release()

        with patch("gateway.run._hermes_home", hermes_home):
            await runner._watch_update_progress(
                poll_interval=0.001,
                stream_interval=0.001,
                timeout=1.0,
            )

        adapter.send.assert_called_once()
        assert not (hermes_home / ".update_pending.json").exists()
        assert not done_path.exists()
        assert not output_path.exists()
        assert not exit_path.exists()

    @pytest.mark.asyncio
    @pytest.mark.live_system_guard_bypass
    async def test_helper_atomically_writes_pid_and_start_time_identity(self, tmp_path):
        runner = _make_runner()
        event = _make_event()
        fake_root = tmp_path / "project"
        (fake_root / ".git").mkdir(parents=True)
        fake_file = str(fake_root / "gateway" / "run.py")
        hermes_home = tmp_path / "hermes"
        hermes_home.mkdir()

        with patch("gateway.run._hermes_home", hermes_home), \
             patch("gateway.run.__file__", fake_file), \
             patch("gateway.slash_commands.sys.platform", "win32"), \
             patch("shutil.which", return_value="C:/hermes.exe"), \
             patch("subprocess.Popen") as mock_popen:
            await runner._handle_update_command(event)

        helper_command = mock_popen.call_args.args[0]
        helper_source = helper_command[2]
        assert "os.replace(" in helper_source
        assert 'def atomic_write(path, data):' in helper_source

        output_path = tmp_path / "helper-output.bin"
        exit_code_path = tmp_path / "helper-exit.txt"
        helper_lock_path = tmp_path / "helper.lock"
        done_path = tmp_path / "helper-done.json"
        identity_path = tmp_path / ".update_helper_pid"
        helper = subprocess.Popen(
            [
                sys.executable, "-c", helper_source,
                str(output_path), str(exit_code_path), str(helper_lock_path),
                str(done_path), "identity-request", "5",
                sys.executable, "-c", "raise SystemExit(0)",
            ],
            stdin=subprocess.PIPE,
        )
        try:
            deadline = time.monotonic() + 5
            while not identity_path.exists() and time.monotonic() < deadline:
                time.sleep(0.01)
            assert identity_path.exists()

            from gateway.status import get_process_start_time

            identity = json.loads(identity_path.read_bytes())
            assert set(identity) == {"pid", "start_time"}
            assert identity["start_time"] == get_process_start_time(identity["pid"])
            expected = json.dumps(identity, separators=(",", ":")).encode("utf-8")
            assert identity_path.read_bytes() == expected
            assert not list(tmp_path.glob(".update_helper_pid.*.tmp"))
        finally:
            assert helper.stdin is not None
            helper.stdin.write(b"test-token\n")
            helper.stdin.close()
            assert helper.wait(timeout=5) == 0

    @pytest.mark.asyncio
    @pytest.mark.live_system_guard_bypass
    @pytest.mark.parametrize(
        ("child_args", "helper_exit", "child_exit", "status"),
        [
            (["definitely-not-an-executable"], 125, 125, "helper_exception"),
            (
                [sys.executable, "-c", "import time; time.sleep(1)"],
                0,
                124,
                "timed_out",
            ),
        ],
    )
    async def test_real_helper_records_failure_before_unlock(
        self, tmp_path, child_args, helper_exit, child_exit, status
    ):
        runner = _make_runner()
        fake_root = tmp_path / "project"
        (fake_root / ".git").mkdir(parents=True)
        hermes_home = tmp_path / "hermes"
        hermes_home.mkdir()
        with patch("gateway.run._hermes_home", hermes_home), \
             patch("gateway.run.__file__", str(fake_root / "gateway" / "run.py")), \
             patch("gateway.slash_commands.sys.platform", "win32"), \
             patch("shutil.which", return_value="C:/hermes.exe"), \
             patch("subprocess.Popen") as mocked_popen:
            await runner._handle_update_command(_make_event())

        helper_source = mocked_popen.call_args.args[0][2]
        request_id = json.loads(
            (hermes_home / ".update_pending.json").read_text(encoding="utf-8")
        )["request_id"]
        output_path = hermes_home / ".update_output.txt"
        exit_path = hermes_home / ".update_exit_code"
        lock_path = hermes_home / ".update_helper.lock"
        done_path = hermes_home / ".update_helper_done.json"
        helper = subprocess.Popen(
            [
                sys.executable, "-c", helper_source,
                str(output_path), str(exit_path), str(lock_path), str(done_path),
                request_id, "0.05", *child_args,
            ],
            stdin=subprocess.PIPE,
        )
        assert helper.stdin is not None
        helper.stdin.write(b"test-token\n")
        helper.stdin.close()
        assert helper.wait(timeout=5) == helper_exit

        assert exit_path.read_text(encoding="utf-8") == str(child_exit)
        assert json.loads(done_path.read_text(encoding="utf-8")) == {
            "request_id": request_id,
            "exit_code": child_exit,
            "status": status,
        }
        probe = acquire_update_helper_probe(lock_path)
        assert probe is not None
        probe.release()


    @pytest.mark.asyncio
    @pytest.mark.live_system_guard_bypass
    async def test_helper_reaps_live_child_after_general_communicate_exception(
        self, tmp_path
    ):
        runner = _make_runner()
        fake_root = tmp_path / "project"
        (fake_root / ".git").mkdir(parents=True)
        hermes_home = tmp_path / "hermes"
        hermes_home.mkdir()
        with patch("gateway.run._hermes_home", hermes_home), \
             patch("gateway.run.__file__", str(fake_root / "gateway" / "run.py")), \
             patch("gateway.slash_commands.sys.platform", "win32"), \
             patch("shutil.which", return_value="C:/hermes.exe"), \
             patch("subprocess.Popen") as mocked_popen:
            await runner._handle_update_command(_make_event())

        helper_source = mocked_popen.call_args.args[0][2]
        helper_source = helper_source.replace(
            "from gateway.update_contract import UpdateHelperLock",
            "from gateway.update_contract import UpdateHelperLock\n"
            "import helper_injection\nhelper_injection.install(subprocess)",
        )
        request_id = json.loads(
            (hermes_home / ".update_pending.json").read_text(encoding="utf-8")
        )["request_id"]
        injection = tmp_path / "injection"
        injection.mkdir()
        child_pid_path = tmp_path / "child.pid"
        events_path = tmp_path / "events.txt"
        (injection / "helper_injection.py").write_text(
            """import os, subprocess
_real_popen = subprocess.Popen
_pid_path = os.environ["HELPER_TEST_CHILD_PID"]
_events_path = os.environ["HELPER_TEST_EVENTS"]
def _event(value):
    with open(_events_path, "a", encoding="utf-8") as stream:
        stream.write(value + "\\n")
        stream.flush()
        os.fsync(stream.fileno())
class InjectedPopen:
    def __init__(self, *args, **kwargs):
        self._proc = _real_popen(*args, **kwargs)
        with open(_pid_path, "w", encoding="ascii") as stream:
            stream.write(str(self._proc.pid))
            stream.flush()
            os.fsync(stream.fileno())
        self._first = True
    def communicate(self, *args, **kwargs):
        if self._first and kwargs.get("timeout") is not None:
            self._first = False
            _event("communicate_error")
            raise KeyboardInterrupt()
        _event("communicate_reap")
        return self._proc.communicate(*args, **kwargs)
    def kill(self):
        _event("kill")
        return self._proc.kill()
    def terminate(self):
        _event("terminate")
        return self._proc.terminate()
    def wait(self, *args, **kwargs):
        _event("wait")
        return self._proc.wait(*args, **kwargs)
    def poll(self):
        return self._proc.poll()
    def __getattr__(self, name):
        return getattr(self._proc, name)
def injected_popen(*args, **kwargs):
    if hasattr(kwargs.get("stdout"), "write") and kwargs.get("stderr") is subprocess.STDOUT:
        return InjectedPopen(*args, **kwargs)
    return _real_popen(*args, **kwargs)
def install(module):
    module.Popen = injected_popen
""",
            encoding="utf-8",
        )
        env = dict(os.environ)
        env["PYTHONPATH"] = str(injection) + os.pathsep + env.get("PYTHONPATH", "")
        env["HELPER_TEST_CHILD_PID"] = str(child_pid_path)
        env["HELPER_TEST_EVENTS"] = str(events_path)
        lock_path = hermes_home / ".update_helper.lock"
        done_path = hermes_home / ".update_helper_done.json"
        helper = subprocess.Popen(
            [
                sys.executable, "-c", helper_source,
                str(hermes_home / ".update_output.txt"),
                str(hermes_home / ".update_exit_code"),
                str(lock_path), str(done_path), request_id, "5",
                sys.executable, "-c", "import time; time.sleep(30)",
            ],
            stdin=subprocess.PIPE,
            env=env,
        )
        assert helper.stdin is not None
        helper.stdin.write(b"test-token\n")
        helper.stdin.close()
        child_pid = None
        try:
            assert helper.wait(timeout=8) == 125
            child_pid = int(child_pid_path.read_text(encoding="ascii"))
            from gateway.status import _pid_exists

            assert _pid_exists(child_pid) is False
            assert events_path.read_text(encoding="utf-8").splitlines()[:3] == [
                "communicate_error", "kill", "communicate_reap",
            ]
            assert done_path.stat().st_mtime_ns >= events_path.stat().st_mtime_ns
            assert json.loads(done_path.read_text(encoding="utf-8")) == {
                "request_id": request_id,
                "exit_code": 125,
                "status": "helper_exception",
            }
            probe = acquire_update_helper_probe(lock_path)
            assert probe is not None
            probe.release()
        finally:
            if child_pid is not None:
                try:
                    os.kill(child_pid, signal.SIGTERM)
                except OSError:
                    pass

    @pytest.mark.asyncio
    @pytest.mark.live_system_guard_bypass
    async def test_helper_keeps_lock_and_withholds_done_when_reap_cannot_be_verified(
        self, tmp_path
    ):
        runner = _make_runner()
        fake_root = tmp_path / "project"
        (fake_root / ".git").mkdir(parents=True)
        hermes_home = tmp_path / "hermes"
        hermes_home.mkdir()
        with patch("gateway.run._hermes_home", hermes_home), \
             patch("gateway.run.__file__", str(fake_root / "gateway" / "run.py")), \
             patch("gateway.slash_commands.sys.platform", "win32"), \
             patch("shutil.which", return_value="C:/hermes.exe"), \
             patch("subprocess.Popen") as mocked_popen:
            await runner._handle_update_command(_make_event())

        helper_source = mocked_popen.call_args.args[0][2]
        helper_source = helper_source.replace(
            "from gateway.update_contract import UpdateHelperLock",
            "from gateway.update_contract import UpdateHelperLock\n"
            "import helper_injection\nhelper_injection.install(subprocess)",
        )
        request_id = json.loads(
            (hermes_home / ".update_pending.json").read_text(encoding="utf-8")
        )["request_id"]
        injection = tmp_path / "injection"
        injection.mkdir()
        child_pid_path = tmp_path / "child.pid"
        (injection / "helper_injection.py").write_text(
            """import os, subprocess
_real_popen = subprocess.Popen
_pid_path = os.environ["HELPER_TEST_CHILD_PID"]
class InjectedPopen:
    def __init__(self, *args, **kwargs):
        self._proc = _real_popen(*args, **kwargs)
        with open(_pid_path, "w", encoding="ascii") as stream:
            stream.write(str(self._proc.pid))
            stream.flush()
            os.fsync(stream.fileno())
    def communicate(self, *args, **kwargs):
        raise OSError("injected communicate failure")
    def kill(self):
        raise OSError("injected kill failure")
    def terminate(self):
        raise OSError("injected terminate failure")
    def wait(self, *args, **kwargs):
        raise OSError("injected wait failure")
    def poll(self):
        raise OSError("injected poll failure")
    def __getattr__(self, name):
        return getattr(self._proc, name)
def injected_popen(*args, **kwargs):
    if hasattr(kwargs.get("stdout"), "write") and kwargs.get("stderr") is subprocess.STDOUT:
        return InjectedPopen(*args, **kwargs)
    return _real_popen(*args, **kwargs)
def install(module):
    module.Popen = injected_popen
""",
            encoding="utf-8",
        )
        env = dict(os.environ)
        env["PYTHONPATH"] = str(injection) + os.pathsep + env.get("PYTHONPATH", "")
        env["HELPER_TEST_CHILD_PID"] = str(child_pid_path)
        lock_path = hermes_home / ".update_helper.lock"
        done_path = hermes_home / ".update_helper_done.json"
        helper = subprocess.Popen(
            [
                sys.executable, "-c", helper_source,
                str(hermes_home / ".update_output.txt"),
                str(hermes_home / ".update_exit_code"),
                str(lock_path), str(done_path), request_id, "5",
                sys.executable, "-c", "import time; time.sleep(30)",
            ],
            stdin=subprocess.PIPE,
            env=env,
        )
        assert helper.stdin is not None
        helper.stdin.write(b"test-token\n")
        helper.stdin.close()
        child_pid = None
        try:
            deadline = time.monotonic() + 5
            while not child_pid_path.exists() and time.monotonic() < deadline:
                time.sleep(0.01)
            child_pid = int(child_pid_path.read_text(encoding="ascii"))
            time.sleep(0.2)
            assert helper.poll() is None
            assert not done_path.exists()
            assert not (hermes_home / ".update_exit_code").exists()
            assert acquire_update_helper_probe(lock_path) is None
        finally:
            helper.kill()
            helper.wait(timeout=5)
            if child_pid is not None:
                try:
                    os.kill(child_pid, signal.SIGTERM)
                except OSError:
                    pass


    @pytest.mark.asyncio
    async def test_duplicate_update_is_rejected_without_overwriting_files(self, tmp_path):
        runner = _make_runner()
        event = _make_event()
        fake_root = tmp_path / "project"
        (fake_root / ".git").mkdir(parents=True)
        fake_file = str(fake_root / "gateway" / "run.py")
        hermes_home = tmp_path / "hermes"
        hermes_home.mkdir()
        pending_path = hermes_home / ".update_pending.json"
        output_path = hermes_home / ".update_output.txt"
        exit_path = hermes_home / ".update_exit_code"
        original_pending = json.dumps({"platform": "telegram", "chat_id": "old"})
        pending_path.write_text(original_pending, encoding="utf-8")
        output_path.write_text("original output", encoding="utf-8")
        exit_path.write_text("0", encoding="utf-8")

        with patch("gateway.run._hermes_home", hermes_home), \
             patch("gateway.run.__file__", fake_file), \
             patch("shutil.which", return_value="C:/hermes.exe"), \
             patch("subprocess.Popen") as mock_popen:
            result = await runner._handle_update_command(event)

        assert "already" in result.lower()
        assert pending_path.read_text(encoding="utf-8") == original_pending
        assert output_path.read_text(encoding="utf-8") == "original output"
        assert exit_path.read_text(encoding="utf-8") == "0"
        mock_popen.assert_not_called()

    def test_concurrent_update_requests_have_one_winner(self, tmp_path):
        """Two real concurrent handlers preserve one request and reject the other."""
        hermes_home = tmp_path / "hermes"
        hermes_home.mkdir()
        barrier_reached = threading.Event()
        start_barrier = threading.Barrier(2, action=barrier_reached.set)

        def synchronize_admission():
            start_barrier.wait(timeout=5)
            return False

        def invoke(chat_id):
            runner = _make_runner()
            return asyncio.run(runner._handle_update_command(_make_event(chat_id=chat_id)))

        with patch("gateway.run._hermes_home", hermes_home), \
             patch("hermes_cli.config.is_managed", side_effect=synchronize_admission), \
             patch("shutil.which", return_value="C:/hermes.exe"), \
             patch("subprocess.Popen") as popen, \
             concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            results = list(executor.map(invoke, ("first", "second")))

        assert barrier_reached.is_set()
        assert sum("starting hermes update" in result.lower() for result in results) == 1
        assert sum("already" in result.lower() for result in results) == 1
        assert popen.call_count == 1
        pending = json.loads(
            (hermes_home / ".update_pending.json").read_text(encoding="utf-8")
        )
        winner_index = next(
            index for index, result in enumerate(results)
            if "starting hermes update" in result.lower()
        )
        assert pending["chat_id"] == ("first", "second")[winner_index]


    @pytest.mark.asyncio
    async def test_fallback_when_no_setsid(self, tmp_path):
        """Falls back to start_new_session=True when setsid is not available."""
        runner = _make_runner()
        event = _make_event()

        fake_root = tmp_path / "project"
        fake_root.mkdir()
        (fake_root / ".git").mkdir()
        (fake_root / "gateway").mkdir()
        (fake_root / "gateway" / "run.py").touch()
        fake_file = str(fake_root / "gateway" / "run.py")
        hermes_home = tmp_path / "hermes"
        hermes_home.mkdir()

        mock_popen = MagicMock()

        def which_no_setsid(x):
            if x == "hermes":
                return "/usr/bin/hermes"
            if x == "setsid":
                return None
            return None

        with patch("gateway.run._hermes_home", hermes_home), \
             patch("gateway.run.__file__", fake_file), \
             patch("gateway.slash_commands.sys.platform", "linux"), \
             patch("shutil.which", side_effect=which_no_setsid), \
             patch("subprocess.Popen", mock_popen):
            result = await runner._handle_update_command(event)

        # Verify the Python helper detaches without the optional setsid binary.
        call_args = mock_popen.call_args[0][0]
        assert call_args[0] == sys.executable
        assert ".update_exit_code" in call_args[4]
        call_kwargs = mock_popen.call_args[1]
        assert call_kwargs.get("start_new_session") is True
        assert call_kwargs.get("stdin") is subprocess.PIPE
        assert "Starting Hermes update" in result


# ---------------------------------------------------------------------------
# Platform allowlist gate
# ---------------------------------------------------------------------------


class TestUpdateCommandPlatformGate:
    """Tests for the platform-allowlist gate at the top of
    ``_handle_update_command``.  Built-in messaging platforms are listed in
    ``_UPDATE_ALLOWED_PLATFORMS``; plugin-migrated platforms (discord,
    mattermost, teams, …) are NOT in the frozenset and rely on the
    registry's ``allow_update_command=True`` fallback.  Programmatic
    interfaces (ACP, API server, webhooks) must be blocked.
    """


    @pytest.mark.asyncio
    async def test_allows_plugin_platform_via_registry_fallback(self, monkeypatch):
        """A plugin-migrated platform (DISCORD) is no longer in
        ``_UPDATE_ALLOWED_PLATFORMS`` but must still pass the gate via
        the registry's ``allow_update_command=True`` flag.

        This test is the empirical guarantee that removing DISCORD from
        the hardcoded frozenset does not regress the /update command for
        Discord users.
        """
        from gateway.run import GatewayRunner

        # Precondition: DISCORD is NOT in the hardcoded set anymore.
        assert Platform.DISCORD not in GatewayRunner._UPDATE_ALLOWED_PLATFORMS

        # Make sure the plugin registry is populated so the fallback fires.
        from hermes_cli.plugins import PluginManager
        PluginManager().discover_and_load(force=True)
        from gateway.platform_registry import platform_registry
        discord_entry = platform_registry.get("discord")
        assert discord_entry is not None
        assert discord_entry.allow_update_command is True

        runner = _make_runner()
        event = _make_event(platform=Platform.DISCORD)
        monkeypatch.setenv("HERMES_MANAGED", "")

        with patch("subprocess.Popen"):
            result = await runner._handle_update_command(event)

        # The gate must NOT have rejected us — anything other than the
        # ``platform_not_messaging`` rejection string is acceptable here.
        # Later steps may legitimately return success ("Starting Hermes
        # update…") or fail for environment reasons.
        assert "only available from messaging platforms" not in result


    @pytest.mark.asyncio
    async def test_allows_homeassistant_via_registry_fallback(self, monkeypatch):
        """Same as DISCORD/MATTERMOST: HOMEASSISTANT is now plugin-migrated
        (PR #40709) and not in the hardcoded frozenset; the registry must
        keep /update working via ``allow_update_command=True``.
        """
        from gateway.run import GatewayRunner

        assert Platform.HOMEASSISTANT not in GatewayRunner._UPDATE_ALLOWED_PLATFORMS

        from hermes_cli.plugins import PluginManager
        PluginManager().discover_and_load(force=True)
        from gateway.platform_registry import platform_registry
        ha_entry = platform_registry.get("homeassistant")
        assert ha_entry is not None
        assert ha_entry.allow_update_command is True

        runner = _make_runner()
        event = _make_event(platform=Platform.HOMEASSISTANT)
        monkeypatch.setenv("HERMES_MANAGED", "")

        with patch("subprocess.Popen"):
            result = await runner._handle_update_command(event)

        assert "only available from messaging platforms" not in result


# ---------------------------------------------------------------------------
# _send_update_notification
# ---------------------------------------------------------------------------


class TestSendUpdateNotification:
    """Tests for GatewayRunner._send_update_notification."""

    @pytest.mark.asyncio
    async def test_upgrade_handoff_delivers_legacy_exit_once_and_cleans_legacy_markers(
        self, tmp_path
    ):
        """A new watcher completes an update started by the pre-request-id helper."""
        runner = _make_runner()
        hermes_home = tmp_path / "hermes"
        hermes_home.mkdir()
        legacy_pending = {
            "platform": "telegram",
            "chat_id": "legacy-chat",
            "session_key": "telegram:legacy-chat",
            "timestamp": "2026-09-07T12:00:00",
        }
        (hermes_home / ".update_pending.json").write_text(
            json.dumps(legacy_pending), encoding="utf-8"
        )
        (hermes_home / ".update_output.txt").write_text(
            "legacy helper completed", encoding="utf-8"
        )
        (hermes_home / ".update_exit_code").write_text("0", encoding="utf-8")
        (hermes_home / ".update_helper_pid").write_text("999999", encoding="utf-8")
        (hermes_home / ".update_prompt.json").write_text("{}", encoding="utf-8")
        (hermes_home / ".update_response").write_text("yes", encoding="utf-8")
        adapter = AsyncMock()
        runner.adapters = {Platform.TELEGRAM: adapter}

        with patch("gateway.run._hermes_home", hermes_home):
            await runner._watch_update_progress(
                poll_interval=0.001, stream_interval=0.001, timeout=0.1
            )

        adapter.send.assert_called_once()
        assert adapter.send.call_args.args[0] == "legacy-chat"
        assert "finished" in adapter.send.call_args.args[1].lower()
        for name in (
            ".update_pending.json",
            ".update_pending.claimed.json",
            ".update_output.txt",
            ".update_exit_code",
            ".update_helper_pid",
            ".update_prompt.json",
            ".update_response",
        ):
            assert not (hermes_home / name).exists()

    @pytest.mark.asyncio
    async def test_concurrent_watchers_deliver_legacy_exit_exactly_once(self, tmp_path):
        runner = _make_runner()
        hermes_home = tmp_path / "hermes"
        hermes_home.mkdir()
        (hermes_home / ".update_pending.json").write_text(json.dumps({
            "platform": "telegram",
            "chat_id": "legacy-chat",
            "session_key": "telegram:legacy-chat",
            "timestamp": "2026-09-07T12:00:00",
        }), encoding="utf-8")
        (hermes_home / ".update_exit_code").write_text("1", encoding="utf-8")
        adapter = AsyncMock()

        async def slow_send(*args, **kwargs):
            await asyncio.sleep(0.05)

        adapter.send.side_effect = slow_send
        runner.adapters = {Platform.TELEGRAM: adapter}

        with patch("gateway.run._hermes_home", hermes_home):
            results = await asyncio.gather(
                runner._send_update_notification(),
                runner._send_update_notification(),
            )

        assert adapter.send.call_count == 1
        assert results.count(True) == 1

    @pytest.mark.asyncio
    async def test_legacy_orphan_waits_for_mtime_deadline_then_reports_exit_125_once(
        self, tmp_path
    ):
        runner = _make_runner()
        hermes_home = tmp_path / "hermes"
        hermes_home.mkdir()
        pending_path = hermes_home / ".update_pending.json"
        exit_path = hermes_home / ".update_exit_code"
        pending_path.write_text(json.dumps({
            "platform": "telegram",
            "chat_id": "legacy-chat",
            "session_key": "telegram:legacy-chat",
        }), encoding="utf-8")
        os.utime(pending_path, (1000.0, 1000.0))
        observed_exit_codes = []
        adapter = AsyncMock()

        async def observe_exit(*args, **kwargs):
            observed_exit_codes.append(exit_path.read_text(encoding="utf-8"))

        adapter.send.side_effect = observe_exit
        runner.adapters = {Platform.TELEGRAM: adapter}

        with patch("gateway.run._hermes_home", hermes_home), \
             patch("gateway.run.time.time", return_value=4629.0):
            assert await runner._send_update_notification() is False
        assert pending_path.exists()
        assert not exit_path.exists()
        adapter.send.assert_not_called()

        with patch("gateway.run._hermes_home", hermes_home), \
             patch("gateway.run.time.time", return_value=4631.0):
            assert await runner._send_update_notification() is True

        adapter.send.assert_called_once()
        assert observed_exit_codes == ["125"]
        assert not pending_path.exists()
        assert not exit_path.exists()

    @pytest.mark.asyncio
    async def test_legacy_cleanup_never_deletes_concurrently_published_request_state(
        self, tmp_path
    ):
        runner = _make_runner()
        hermes_home = tmp_path / "hermes"
        hermes_home.mkdir()
        pending_path = hermes_home / ".update_pending.json"
        done_path = hermes_home / ".update_helper_done.json"
        exit_path = hermes_home / ".update_exit_code"
        output_path = hermes_home / ".update_output.txt"
        pending_path.write_text(json.dumps({
            "platform": "telegram", "chat_id": "legacy-chat",
        }), encoding="utf-8")
        exit_path.write_text("0", encoding="utf-8")
        output_path.write_text("legacy output", encoding="utf-8")
        adapter = AsyncMock()
        new_pending = {
            "request_id": "new-request",
            "platform": "telegram",
            "chat_id": "new-chat",
            "deadline_epoch": time.time() + 60,
            "orphan_grace_seconds": 30,
        }
        new_done = {"request_id": "new-request", "exit_code": 0}

        async def publish_new_request(*args, **kwargs):
            pending_path.write_text(json.dumps(new_pending), encoding="utf-8")
            done_path.write_text(json.dumps(new_done), encoding="utf-8")
            exit_path.write_text("0", encoding="utf-8")
            output_path.write_text("new output", encoding="utf-8")

        adapter.send.side_effect = publish_new_request
        runner.adapters = {Platform.TELEGRAM: adapter}

        with patch("gateway.run._hermes_home", hermes_home):
            assert await runner._send_update_notification() is True

        assert json.loads(pending_path.read_text(encoding="utf-8")) == new_pending
        assert json.loads(done_path.read_text(encoding="utf-8")) == new_done
        assert exit_path.read_text(encoding="utf-8") == "0"
        assert output_path.read_text(encoding="utf-8") == "new output"

    @pytest.mark.asyncio
    @pytest.mark.parametrize("done_bytes", [
        json.dumps({"request_id": "another-request", "exit_code": 0}),
        '{"request_id":"current-request",',
    ])
    async def test_mismatched_or_partial_done_is_preserved(self, tmp_path, done_bytes):
        runner = _make_runner()
        hermes_home = tmp_path / "hermes"
        hermes_home.mkdir()
        pending_path = hermes_home / ".update_pending.json"
        done_path = hermes_home / ".update_helper_done.json"
        exit_path = hermes_home / ".update_exit_code"
        output_path = hermes_home / ".update_output.txt"
        pending_path.write_text(json.dumps({
            "request_id": "current-request",
            "platform": "telegram",
            "chat_id": "111",
            "deadline_epoch": time.time() + 60,
            "orphan_grace_seconds": 30,
        }), encoding="utf-8")
        done_path.write_text(done_bytes, encoding="utf-8")
        exit_path.write_text("0", encoding="utf-8")
        output_path.write_text("request output", encoding="utf-8")
        adapter = AsyncMock()
        runner.adapters = {Platform.TELEGRAM: adapter}

        with patch("gateway.run._hermes_home", hermes_home):
            assert await runner._send_update_notification() is False

        adapter.send.assert_not_called()
        assert pending_path.exists()
        assert done_path.read_text(encoding="utf-8") == done_bytes
        assert exit_path.exists()
        assert output_path.exists()
        assert not (hermes_home / ".update_pending.claimed.json").exists()

    @pytest.mark.asyncio
    async def test_partial_pending_json_is_preserved_for_atomic_writer_retry(self, tmp_path):
        runner = _make_runner()
        hermes_home = tmp_path / "hermes"
        hermes_home.mkdir()
        pending_path = hermes_home / ".update_pending.json"
        pending_bytes = '{"request_id":"not-finished",'
        pending_path.write_text(pending_bytes, encoding="utf-8")

        with patch("gateway.run._hermes_home", hermes_home):
            assert await runner._send_update_notification() is False

        assert pending_path.read_text(encoding="utf-8") == pending_bytes
        assert not (hermes_home / ".update_pending.claimed.json").exists()

    @pytest.mark.asyncio
    async def test_partial_exit_code_is_preserved_for_atomic_writer_retry(self, tmp_path):
        runner = _make_runner()
        hermes_home = tmp_path / "hermes"
        hermes_home.mkdir()
        request_id = "partial-exit"
        pending_path = hermes_home / ".update_pending.json"
        done_path = hermes_home / ".update_helper_done.json"
        exit_path = hermes_home / ".update_exit_code"
        pending_path.write_text(json.dumps({
            "request_id": request_id,
            "platform": "telegram",
            "chat_id": "111",
            "deadline_epoch": time.time() + 60,
            "orphan_grace_seconds": 30,
        }), encoding="utf-8")
        done_path.write_text(json.dumps({
            "request_id": request_id,
            "exit_code": 0,
        }), encoding="utf-8")
        exit_path.write_text("-", encoding="utf-8")
        adapter = AsyncMock()
        runner.adapters = {Platform.TELEGRAM: adapter}

        with patch("gateway.run._hermes_home", hermes_home):
            assert await runner._send_update_notification() is False

        adapter.send.assert_not_called()
        assert pending_path.exists()
        assert done_path.exists()
        assert exit_path.read_text(encoding="utf-8") == "-"

    @pytest.mark.asyncio
    async def test_concurrent_watchers_deliver_matching_request_once(self, tmp_path):
        runner = _make_runner()
        hermes_home = tmp_path / "hermes"
        hermes_home.mkdir()
        request_id = "one-delivery"
        (hermes_home / ".update_pending.json").write_text(json.dumps({
            "request_id": request_id,
            "platform": "telegram",
            "chat_id": "111",
            "deadline_epoch": time.time() + 60,
            "orphan_grace_seconds": 30,
        }), encoding="utf-8")
        (hermes_home / ".update_helper_done.json").write_text(json.dumps({
            "request_id": request_id,
            "exit_code": 0,
        }), encoding="utf-8")
        (hermes_home / ".update_exit_code").write_text("0", encoding="utf-8")
        adapter = AsyncMock()

        async def slow_send(*args, **kwargs):
            await asyncio.sleep(0.05)

        adapter.send.side_effect = slow_send
        runner.adapters = {Platform.TELEGRAM: adapter}

        with patch("gateway.run._hermes_home", hermes_home):
            results = await asyncio.gather(
                runner._send_update_notification(),
                runner._send_update_notification(),
            )

        assert adapter.send.call_count == 1
        assert results.count(True) == 1

    @pytest.mark.asyncio
    async def test_helper_lock_blocks_cleanup_and_second_request(self, tmp_path):
        runner = _make_runner()
        hermes_home = tmp_path / "hermes"
        hermes_home.mkdir()
        request_id = "request-one"
        pending_path = hermes_home / ".update_pending.json"
        done_path = hermes_home / ".update_helper_done.json"
        output_path = hermes_home / ".update_output.txt"
        exit_path = hermes_home / ".update_exit_code"
        helper_lock_path = hermes_home / ".update_helper.lock"
        pending_path.write_text(json.dumps({
            "request_id": request_id,
            "platform": "telegram",
            "chat_id": "111",
            "deadline_epoch": time.time() + 60,
            "orphan_grace_seconds": 30,
        }), encoding="utf-8")
        done_path.write_text(json.dumps({
            "request_id": request_id, "exit_code": 0,
        }), encoding="utf-8")
        output_path.write_text("not fully flushed", encoding="utf-8")
        exit_path.write_text("0", encoding="utf-8")
        adapter = AsyncMock()
        runner.adapters = {Platform.TELEGRAM: adapter}

        fake_root = tmp_path / "project"
        (fake_root / ".git").mkdir(parents=True)
        fake_file = str(fake_root / "gateway" / "run.py")
        with UpdateHelperLock(helper_lock_path):
            with patch("gateway.run._hermes_home", hermes_home):
                delivered = await runner._send_update_notification()
            with patch("gateway.run._hermes_home", hermes_home), \
                 patch("gateway.run.__file__", fake_file), \
                 patch("shutil.which", return_value="C:/hermes.exe"), \
                 patch("subprocess.Popen") as popen:
                duplicate_result = await runner._handle_update_command(
                    _make_event(chat_id="222")
                )

        assert delivered is False
        adapter.send.assert_not_called()
        assert pending_path.exists()
        assert done_path.exists()
        assert output_path.read_text(encoding="utf-8") == "not fully flushed"
        assert exit_path.exists()
        assert "already" in duplicate_result.lower()
        popen.assert_not_called()

    @pytest.mark.asyncio
    async def test_matching_done_and_unlocked_notifies_then_cleans_request(self, tmp_path):
        runner = _make_runner()
        hermes_home = tmp_path / "hermes"
        hermes_home.mkdir()
        request_id = "request-complete"
        paths = {
            "pending": hermes_home / ".update_pending.json",
            "done": hermes_home / ".update_helper_done.json",
            "output": hermes_home / ".update_output.txt",
            "exit": hermes_home / ".update_exit_code",
        }
        paths["pending"].write_text(json.dumps({
            "request_id": request_id,
            "platform": "telegram",
            "chat_id": "111",
            "deadline_epoch": time.time() + 60,
            "orphan_grace_seconds": 30,
        }), encoding="utf-8")
        paths["done"].write_text(json.dumps({
            "request_id": request_id,
            "exit_code": 0,
        }), encoding="utf-8")
        paths["output"].write_text("complete output", encoding="utf-8")
        paths["exit"].write_text("0", encoding="utf-8")
        adapter = AsyncMock()
        runner.adapters = {Platform.TELEGRAM: adapter}

        with patch("gateway.run._hermes_home", hermes_home):
            delivered = await runner._send_update_notification()

        assert delivered is True
        adapter.send.assert_called_once()
        assert all(not path.exists() for path in paths.values())

    @pytest.mark.asyncio
    async def test_crashed_helper_is_preserved_until_deadline_then_failed_until_notified(
        self, tmp_path
    ):
        runner = _make_runner()
        hermes_home = tmp_path / "hermes"
        hermes_home.mkdir()
        request_id = "request-orphan"
        pending_path = hermes_home / ".update_pending.json"
        done_path = hermes_home / ".update_helper_done.json"
        exit_path = hermes_home / ".update_exit_code"
        output_path = hermes_home / ".update_output.txt"
        pending_path.write_text(json.dumps({
            "request_id": request_id,
            "platform": "telegram",
            "chat_id": "111",
            "deadline_epoch": 1000.0,
            "orphan_grace_seconds": 30.0,
        }), encoding="utf-8")
        output_path.write_text("partial child output", encoding="utf-8")

        with patch("gateway.run._hermes_home", hermes_home), \
             patch("gateway.run.time.time", return_value=1029.0):
            assert await runner._send_update_notification() is False

        assert pending_path.exists()
        assert not done_path.exists()
        assert not exit_path.exists()
        assert output_path.read_text(encoding="utf-8") == "partial child output"

        with patch("gateway.run._hermes_home", hermes_home), \
             patch("gateway.run.time.time", return_value=1031.0):
            assert await runner._send_update_notification() is False

        assert pending_path.exists()
        assert json.loads(done_path.read_text(encoding="utf-8")) == {
            "request_id": request_id,
            "exit_code": 125,
            "status": "helper_crashed",
        }
        assert exit_path.read_text(encoding="utf-8") == "125"
        assert output_path.exists()

        adapter = AsyncMock()
        runner.adapters = {Platform.TELEGRAM: adapter}
        with patch("gateway.run._hermes_home", hermes_home), \
             patch("gateway.run.time.time", return_value=1032.0):
            assert await runner._send_update_notification() is True

        adapter.send.assert_called_once()
        assert not pending_path.exists()
        assert not done_path.exists()
        assert not exit_path.exists()
        assert not output_path.exists()

    @pytest.mark.asyncio
    async def test_helper_lock_probe_os_error_preserves_every_marker(self, tmp_path):
        runner = _make_runner()
        hermes_home = tmp_path / "hermes"
        hermes_home.mkdir()
        paths = [
            hermes_home / ".update_pending.json",
            hermes_home / ".update_helper_done.json",
            hermes_home / ".update_output.txt",
            hermes_home / ".update_exit_code",
        ]
        paths[0].write_text(json.dumps({
            "request_id": "request-lock-error",
            "platform": "telegram",
            "chat_id": "111",
        }), encoding="utf-8")
        paths[1].write_text(json.dumps({
            "request_id": "request-lock-error", "exit_code": 0,
        }), encoding="utf-8")
        paths[2].write_text("done", encoding="utf-8")
        paths[3].write_text("0", encoding="utf-8")
        adapter = AsyncMock()
        runner.adapters = {Platform.TELEGRAM: adapter}

        with patch("gateway.run._hermes_home", hermes_home), \
             patch(
                 "gateway.run.acquire_update_helper_probe",
                 side_effect=OSError("lock filesystem unavailable"),
             ):
            delivered = await runner._send_update_notification()

        assert delivered is False
        adapter.send.assert_not_called()
        assert all(path.exists() for path in paths)

    @pytest.mark.asyncio
    @pytest.mark.parametrize("legacy_identity", [
        "4242",
        json.dumps({"pid": 4242, "start_time": 100}),
        json.dumps({"pid": 4242, "start_time": None}),
    ])
    async def test_pid_metadata_cannot_extend_expired_orphan_forever(
        self, tmp_path, legacy_identity
    ):
        runner = _make_runner()
        hermes_home = tmp_path / "hermes"
        hermes_home.mkdir()
        pending_path = hermes_home / ".update_pending.json"
        helper_pid_path = hermes_home / ".update_helper_pid"
        pending_path.write_text(json.dumps({
            "request_id": "request-expired",
            "platform": "telegram",
            "chat_id": "111",
            "session_key": "telegram:111",
            "deadline_epoch": 1.0,
            "orphan_grace_seconds": 0.0,
        }), encoding="utf-8")
        helper_pid_path.write_text(legacy_identity, encoding="utf-8")
        adapter = AsyncMock()
        runner.adapters = {Platform.TELEGRAM: adapter}

        with patch("gateway.run._hermes_home", hermes_home), \
             patch("gateway.run.time.time", return_value=2.0), \
             patch(
                 "gateway.status._pid_exists",
                 side_effect=AssertionError("PID must not be cleanup authority"),
             ), \
             patch(
                 "gateway.status.get_process_start_time",
                 side_effect=AssertionError("start time must not be cleanup authority"),
             ):
            await runner._watch_update_progress(
                poll_interval=0.001,
                stream_interval=0.001,
                timeout=0.0,
            )

        adapter.send.assert_called_once()
        assert not pending_path.exists()
        assert not helper_pid_path.exists()
        assert not (hermes_home / ".update_helper_done.json").exists()

    @pytest.mark.asyncio
    async def test_defers_notification_while_update_still_running(self, tmp_path):
        """Returns False and keeps marker files when the update has not exited yet."""
        runner = _make_runner()
        hermes_home = tmp_path / "hermes"
        hermes_home.mkdir()

        pending_path = hermes_home / ".update_pending.json"
        pending_path.write_text(json.dumps({
            "platform": "telegram", "chat_id": "67890", "user_id": "12345",
        }))
        (hermes_home / ".update_output.txt").write_text("still running")

        mock_adapter = AsyncMock()
        runner.adapters = {Platform.TELEGRAM: mock_adapter}

        with patch("gateway.run._hermes_home", hermes_home):
            result = await runner._send_update_notification()

        assert result is False
        mock_adapter.send.assert_not_called()
        assert pending_path.exists()

    @pytest.mark.asyncio
    async def test_recovers_from_claimed_pending_file(self, tmp_path):
        """A claimed pending file from a crashed notifier is still deliverable."""
        runner = _make_runner()
        hermes_home = tmp_path / "hermes"
        hermes_home.mkdir()

        claimed_path = hermes_home / ".update_pending.claimed.json"
        request_id = "claimed-request"
        claimed_path.write_text(json.dumps({
            "request_id": request_id,
            "platform": "telegram", "chat_id": "67890", "user_id": "12345",
        }))
        (hermes_home / ".update_output.txt").write_text("done")
        (hermes_home / ".update_exit_code").write_text("0")
        (hermes_home / ".update_helper_done.json").write_text(json.dumps({
            "request_id": request_id, "exit_code": 0,
        }))

        mock_adapter = AsyncMock()
        runner.adapters = {Platform.TELEGRAM: mock_adapter}

        with patch("gateway.run._hermes_home", hermes_home):
            result = await runner._send_update_notification()

        assert result is True
        mock_adapter.send.assert_called_once()
        assert not claimed_path.exists()

    @pytest.mark.asyncio
    async def test_sends_notification_with_output(self, tmp_path):
        """Sends update output to the correct platform and chat."""
        runner = _make_runner()
        hermes_home = tmp_path / "hermes"
        hermes_home.mkdir()

        # Write pending marker
        pending = {
            "request_id": "output-request",
            "platform": "telegram",
            "chat_id": "67890",
            "user_id": "12345",
            "timestamp": "2026-03-04T21:00:00",
        }
        (hermes_home / ".update_pending.json").write_text(json.dumps(pending))
        (hermes_home / ".update_output.txt").write_text(
            "→ Found 3 new commit(s)\n✓ Code updated!\n✓ Update complete!"
        )
        (hermes_home / ".update_exit_code").write_text("0")
        (hermes_home / ".update_helper_done.json").write_text(json.dumps({
            "request_id": "output-request", "exit_code": 0,
        }))

        # Mock the adapter
        mock_adapter = AsyncMock()
        mock_adapter.send = AsyncMock()
        runner.adapters = {Platform.TELEGRAM: mock_adapter}

        with patch("gateway.run._hermes_home", hermes_home):
            await runner._send_update_notification()

        mock_adapter.send.assert_called_once()
        call_args = mock_adapter.send.call_args
        assert call_args[0][0] == "67890"  # chat_id
        assert "Update complete" in call_args[0][1] or "update finished" in call_args[0][1].lower()


    @pytest.mark.asyncio
    async def test_preserves_request_on_notification_error(self, tmp_path):
        """Files remain retryable if notification delivery fails."""
        runner = _make_runner()
        hermes_home = tmp_path / "hermes"
        hermes_home.mkdir()

        pending_path = hermes_home / ".update_pending.json"
        output_path = hermes_home / ".update_output.txt"
        exit_code_path = hermes_home / ".update_exit_code"
        done_path = hermes_home / ".update_helper_done.json"
        pending_path.write_text(json.dumps({
            "request_id": "send-failure",
            "platform": "telegram", "chat_id": "111", "user_id": "222",
        }))
        output_path.write_text("✓ Done")
        exit_code_path.write_text("0")
        done_path.write_text(json.dumps({
            "request_id": "send-failure", "exit_code": 0,
        }))

        # Adapter send raises
        mock_adapter = AsyncMock()
        mock_adapter.send.side_effect = RuntimeError("network error")
        runner.adapters = {Platform.TELEGRAM: mock_adapter}

        with patch("gateway.run._hermes_home", hermes_home):
            await runner._send_update_notification()

        assert pending_path.exists()
        assert output_path.exists()
        assert exit_code_path.exists()
        assert done_path.exists()

    @pytest.mark.asyncio
    @pytest.mark.parametrize("legacy", [False, True])
    async def test_preserves_request_when_notification_returns_unsuccessful(
        self, tmp_path, legacy
    ):
        """A non-raising SendResult failure must remain retryable."""
        runner = _make_runner()
        hermes_home = tmp_path / "hermes"
        hermes_home.mkdir()

        pending_path = hermes_home / ".update_pending.json"
        output_path = hermes_home / ".update_output.txt"
        exit_code_path = hermes_home / ".update_exit_code"
        done_path = hermes_home / ".update_helper_done.json"
        pending = {
            "platform": "telegram",
            "chat_id": "111",
            "user_id": "222",
        }
        if not legacy:
            pending["request_id"] = "unsuccessful-send"
        pending_path.write_text(json.dumps(pending), encoding="utf-8")
        output_path.write_text("done", encoding="utf-8")
        exit_code_path.write_text("0", encoding="utf-8")
        if not legacy:
            done_path.write_text(
                json.dumps({"request_id": "unsuccessful-send", "exit_code": 0}),
                encoding="utf-8",
            )

        mock_adapter = AsyncMock()
        mock_adapter.send.return_value = MagicMock(
            success=False, error="delivery rejected"
        )
        runner.adapters = {Platform.TELEGRAM: mock_adapter}

        with patch("gateway.run._hermes_home", hermes_home):
            delivered = await runner._send_update_notification()

        assert delivered is False
        assert pending_path.exists()
        assert output_path.exists()
        assert exit_code_path.exists()
        if not legacy:
            assert done_path.exists()


    @pytest.mark.asyncio
    async def test_no_adapter_for_platform_preserves_markers(self, tmp_path):
        """A finished update whose platform is offline keeps its markers.

        When the target platform's adapter has not reconnected yet, dropping
        the completion markers would silently lose the notification. Instead the
        call defers (returns False) and leaves every marker on disk so a later
        retry can deliver once the platform is back.
        """
        runner = _make_runner()
        hermes_home = tmp_path / "hermes"
        hermes_home.mkdir()

        pending = {"platform": "discord", "chat_id": "111", "user_id": "222"}
        pending_path = hermes_home / ".update_pending.json"
        output_path = hermes_home / ".update_output.txt"
        exit_code_path = hermes_home / ".update_exit_code"
        pending_path.write_text(json.dumps(pending))
        output_path.write_text("Done")
        exit_code_path.write_text("0")

        # Only telegram adapter available, but pending says discord
        mock_adapter = AsyncMock()
        runner.adapters = {Platform.TELEGRAM: mock_adapter}

        with patch("gateway.run._hermes_home", hermes_home):
            result = await runner._send_update_notification()

        # No send (wrong platform offline) and the result is deferred.
        assert result is False
        mock_adapter.send.assert_not_called()
        # Markers are preserved for a later retry — NOT cleaned up.
        assert pending_path.exists()
        assert output_path.exists()
        assert exit_code_path.exists()
        # The marker stays in its canonical pending location (claim restored).
        assert not (hermes_home / ".update_pending.claimed.json").exists()

    @pytest.mark.asyncio
    async def test_deferred_notification_delivers_after_reconnect(self, tmp_path):
        """A deferred completion is delivered once the platform reconnects.

        Regression for the late-reconnect /update bug: the update finishes while
        the target platform is offline, the markers survive the deferral, and
        the next call (after the adapter is registered) delivers the result and
        cleans up — exactly once.
        """
        runner = _make_runner()
        hermes_home = tmp_path / "hermes"
        hermes_home.mkdir()

        pending = {
            "request_id": "reconnect-request",
            "platform": "discord", "chat_id": "111", "user_id": "222",
        }
        pending_path = hermes_home / ".update_pending.json"
        output_path = hermes_home / ".update_output.txt"
        exit_code_path = hermes_home / ".update_exit_code"
        pending_path.write_text(json.dumps(pending))
        output_path.write_text("✓ Update complete!")
        exit_code_path.write_text("0")
        (hermes_home / ".update_helper_done.json").write_text(json.dumps({
            "request_id": "reconnect-request", "exit_code": 0,
        }))

        # First pass: target platform (discord) is still offline → defer.
        with patch("gateway.run._hermes_home", hermes_home):
            first = await runner._send_update_notification()

        assert first is False
        assert pending_path.exists()

        # Platform reconnects: the reconnect watcher adds the adapter back.
        mock_adapter = AsyncMock()
        runner.adapters = {Platform.DISCORD: mock_adapter}

        with patch("gateway.run._hermes_home", hermes_home):
            second = await runner._send_update_notification()

        assert second is True
        mock_adapter.send.assert_called_once()
        sent_text = mock_adapter.send.call_args[0][1]
        assert "Update complete" in sent_text
        # Now everything is cleaned up — no duplicate deliveries possible.
        assert not pending_path.exists()
        assert not output_path.exists()
        assert not exit_code_path.exists()
        assert not (hermes_home / ".update_pending.claimed.json").exists()

    @pytest.mark.asyncio
    async def test_completion_notification_tolerates_invalid_utf8_output(self, tmp_path):
        """Completion-only update notifications must not crash on bad bytes."""
        runner = _make_runner()
        hermes_home = tmp_path / "hermes"
        hermes_home.mkdir()

        pending = {
            "request_id": "invalid-utf8-request",
            "platform": "discord", "chat_id": "111", "user_id": "222",
        }
        pending_path = hermes_home / ".update_pending.json"
        output_path = hermes_home / ".update_output.txt"
        exit_code_path = hermes_home / ".update_exit_code"
        pending_path.write_text(json.dumps(pending))
        output_path.write_bytes(b"ok before\ninvalid byte: \x96\ncontinued after\n")
        exit_code_path.write_text("0")
        (hermes_home / ".update_helper_done.json").write_text(json.dumps({
            "request_id": "invalid-utf8-request", "exit_code": 0,
        }))

        mock_adapter = AsyncMock()
        runner.adapters = {Platform.DISCORD: mock_adapter}

        with patch("gateway.run._hermes_home", hermes_home):
            delivered = await runner._send_update_notification()

        assert delivered is True
        mock_adapter.send.assert_called_once()
        sent_text = mock_adapter.send.call_args[0][1]
        assert "ok before" in sent_text
        assert "invalid byte" in sent_text
        assert "continued after" in sent_text
        assert "Hermes update finished" in sent_text
        assert not pending_path.exists()
        assert not output_path.exists()
        assert not exit_code_path.exists()


# ---------------------------------------------------------------------------
# /update in help and known_commands
# ---------------------------------------------------------------------------


class TestUpdateInHelp:
    """Verify /update appears in help text and known commands set."""


    def test_update_is_known_command(self):
        """/update dispatches through the gateway's plain-command handler table.

        (Was an inspect.getsource() check for the literal '"update"' in
        _handle_message — a banned source-reading test. The if-chain was
        replaced by _gateway_plain_command_handlers(), so assert the real
        dispatch contract: the table maps "update" to the update handler.)
        """
        from gateway.run import GatewayRunner

        runner = object.__new__(GatewayRunner)
        handlers = runner._gateway_plain_command_handlers()
        assert handlers.get("update") == runner._handle_update_command

class TestWatchUpdateProgress:
    @pytest.mark.asyncio
    async def test_invalid_utf8_update_output_does_not_crash_watcher(self, tmp_path):
        runner = _make_runner()
        hermes_home = tmp_path / "hermes"
        hermes_home.mkdir()

        (hermes_home / ".update_pending.json").write_text(json.dumps({
            "request_id": "watch-invalid-utf8",
            "platform": "telegram",
            "chat_id": "67890",
            "user_id": "12345",
        }))
        (hermes_home / ".update_output.txt").write_bytes(
            b"ok before\n\xe2\x9c invalid-continuation: \x96\ncontinued after\n"
        )
        (hermes_home / ".update_exit_code").write_text("0")
        (hermes_home / ".update_helper_done.json").write_text(json.dumps({
            "request_id": "watch-invalid-utf8", "exit_code": 0,
        }))

        mock_adapter = AsyncMock()
        runner.adapters = {Platform.TELEGRAM: mock_adapter}

        with patch("gateway.run._hermes_home", hermes_home):
            await runner._watch_update_progress(poll_interval=0.01, stream_interval=0.01, timeout=1.0)

        sent = "\n".join(call.args[1] for call in mock_adapter.send.call_args_list)
        assert "ok before" in sent
        assert "continued after" in sent
        assert "Hermes update finished" in sent
        assert not (hermes_home / ".update_pending.json").exists()
