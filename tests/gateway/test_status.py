"""Tests for gateway runtime status tracking."""

import json
import os
from pathlib import Path
from types import SimpleNamespace

from gateway import status


class TestGatewayPidState:
    def test_write_pid_file_records_gateway_metadata(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))

        status.write_pid_file()

        payload = json.loads((tmp_path / "gateway.pid").read_text())
        assert payload["pid"] == os.getpid()
        assert payload["kind"] == "hermes-gateway"
        assert isinstance(payload["argv"], list)
        assert payload["argv"]

    def test_write_pid_file_is_atomic_against_concurrent_writers(self, tmp_path, monkeypatch):
        """Regression: two concurrent --replace invocations must not both win.

        Without O_CREAT|O_EXCL, two processes racing through start_gateway()'s
        termination-wait would both write to gateway.pid, silently overwriting
        each other and leaving multiple gateway instances alive (#11718).
        """
        import pytest

        monkeypatch.setenv("HERMES_HOME", str(tmp_path))

        # First write wins.
        status.write_pid_file()
        assert (tmp_path / "gateway.pid").exists()

        # Second write (simulating a racing --replace that missed the earlier
        # guards) must raise FileExistsError rather than clobber the record.
        with pytest.raises(FileExistsError):
            status.write_pid_file()

        # Original record is preserved.
        payload = json.loads((tmp_path / "gateway.pid").read_text())
        assert payload["pid"] == os.getpid()

    def test_get_running_pid_rejects_live_non_gateway_pid(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        pid_path = tmp_path / "gateway.pid"
        pid_path.write_text(str(os.getpid()))

        assert status.get_running_pid() is None
        assert not pid_path.exists()

    def test_get_running_pid_cleans_stale_record_from_dead_process(self, tmp_path, monkeypatch):
        # Simulates the aftermath of a crash: the PID file still points at a
        # process that no longer exists. The next gateway startup must be
        # able to unlink it so ``write_pid_file``'s O_EXCL create succeeds —
        # otherwise systemd's restart loop hits "PID file race lost" forever.
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        pid_path = tmp_path / "gateway.pid"
        dead_pid = 999999  # not our pid, and below we simulate it's dead
        pid_path.write_text(json.dumps({
            "pid": dead_pid,
            "kind": "hermes-gateway",
            "argv": ["python", "-m", "hermes_cli.main", "gateway", "run"],
            "start_time": 111,
        }))

        def _dead_process(pid, sig):
            raise ProcessLookupError

        monkeypatch.setattr(status.os, "kill", _dead_process)

        assert status.get_running_pid() is None
        assert not pid_path.exists()

    def test_get_running_pid_accepts_gateway_metadata_when_cmdline_unavailable(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        pid_path = tmp_path / "gateway.pid"
        pid_path.write_text(json.dumps({
            "pid": os.getpid(),
            "kind": "hermes-gateway",
            "argv": ["python", "-m", "hermes_cli.main", "gateway"],
            "start_time": 123,
        }))

        monkeypatch.setattr(status.os, "kill", lambda pid, sig: None)
        monkeypatch.setattr(status, "_get_process_start_time", lambda pid: 123)
        monkeypatch.setattr(status, "_read_process_cmdline", lambda pid: None)

        assert status.acquire_gateway_runtime_lock() is True
        try:
            assert status.get_running_pid() == os.getpid()
        finally:
            status.release_gateway_runtime_lock()

    def test_get_running_pid_accepts_script_style_gateway_cmdline(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        pid_path = tmp_path / "gateway.pid"
        pid_path.write_text(json.dumps({
            "pid": os.getpid(),
            "kind": "hermes-gateway",
            "argv": ["/venv/bin/python", "/repo/hermes_cli/main.py", "gateway", "run", "--replace"],
            "start_time": 123,
        }))

        monkeypatch.setattr(status.os, "kill", lambda pid, sig: None)
        monkeypatch.setattr(status, "_get_process_start_time", lambda pid: 123)
        monkeypatch.setattr(
            status,
            "_read_process_cmdline",
            lambda pid: "/venv/bin/python /repo/hermes_cli/main.py gateway run --replace",
        )

        assert status.acquire_gateway_runtime_lock() is True
        try:
            assert status.get_running_pid() == os.getpid()
        finally:
            status.release_gateway_runtime_lock()

    def test_get_running_pid_accepts_explicit_pid_path_without_cleanup(self, tmp_path, monkeypatch):
        other_home = tmp_path / "profile-home"
        other_home.mkdir()
        pid_path = other_home / "gateway.pid"
        pid_path.write_text(json.dumps({
            "pid": os.getpid(),
            "kind": "hermes-gateway",
            "argv": ["python", "-m", "hermes_cli.main", "gateway"],
            "start_time": 123,
        }))

        monkeypatch.setattr(status.os, "kill", lambda pid, sig: None)
        monkeypatch.setattr(status, "_get_process_start_time", lambda pid: 123)
        monkeypatch.setattr(status, "_read_process_cmdline", lambda pid: None)

        lock_path = other_home / "gateway.lock"
        lock_path.write_text(json.dumps({
            "pid": os.getpid(),
            "kind": "hermes-gateway",
            "argv": ["python", "-m", "hermes_cli.main", "gateway"],
            "start_time": 123,
        }))
        monkeypatch.setattr(status, "is_gateway_runtime_lock_active", lambda lock_path=None: True)

        assert status.get_running_pid(pid_path, cleanup_stale=False) == os.getpid()
        assert pid_path.exists()

    def test_runtime_lock_claims_and_releases_liveness(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))

        assert status.is_gateway_runtime_lock_active() is False
        assert status.acquire_gateway_runtime_lock() is True
        assert status.is_gateway_runtime_lock_active() is True

        status.release_gateway_runtime_lock()

        assert status.is_gateway_runtime_lock_active() is False

    def test_get_running_pid_treats_pid_file_as_stale_without_runtime_lock(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        pid_path = tmp_path / "gateway.pid"
        pid_path.write_text(json.dumps({
            "pid": os.getpid(),
            "kind": "hermes-gateway",
            "argv": ["python", "-m", "hermes_cli.main", "gateway"],
            "start_time": 123,
        }))

        monkeypatch.setattr(status.os, "kill", lambda pid, sig: None)
        monkeypatch.setattr(status, "_get_process_start_time", lambda pid: 123)
        monkeypatch.setattr(status, "_read_process_cmdline", lambda pid: None)

        assert status.get_running_pid() is None
        assert not pid_path.exists()

    def test_get_running_pid_accepts_no_supervisor_restart_runtime(self, tmp_path, monkeypatch):
        """WSL/no-systemd restart fallback runs the gateway in a restart argv process."""
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        pid_path = tmp_path / "gateway.pid"
        record = {
            "pid": os.getpid(),
            "kind": "hermes-gateway",
            "argv": ["python", "-m", "hermes_cli.main", "gateway", "restart"],
            "start_time": 123,
        }
        pid_path.write_text(json.dumps(record))

        monkeypatch.setattr(status.os, "kill", lambda pid, sig: None)
        monkeypatch.setattr(status, "_get_process_start_time", lambda pid: 123)
        monkeypatch.setattr(
            status,
            "_read_process_cmdline",
            lambda pid: "python -m hermes_cli.main gateway restart",
        )

        assert status.acquire_gateway_runtime_lock() is True
        try:
            assert status.get_running_pid() == os.getpid()
        finally:
            status.release_gateway_runtime_lock()

    def test_get_running_pid_falls_back_to_no_supervisor_runtime_state(self, tmp_path, monkeypatch):
        """A live gateway_state.json PID should keep status accurate without a pidfile."""
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        state_path = tmp_path / "gateway_state.json"
        state_path.write_text(json.dumps({
            "gateway_state": "running",
            "pid": os.getpid(),
            "kind": "hermes-gateway",
            "argv": ["python", "-m", "hermes_cli.main", "gateway", "restart"],
            "start_time": 123,
        }))

        monkeypatch.setattr(status.os, "kill", lambda pid, sig: None)
        monkeypatch.setattr(status, "_get_process_start_time", lambda pid: 123)
        monkeypatch.setattr(
            status,
            "_read_process_cmdline",
            lambda pid: "python -m hermes_cli.main gateway restart",
        )

        assert status.get_running_pid() == os.getpid()

    def test_get_running_pid_cleans_stale_metadata_from_dead_foreign_pid(self, tmp_path, monkeypatch):
        """Stale PID file from a *different* PID (crashed process) must still be cleaned.

        Regression for: ``remove_pid_file()`` defensively refuses to delete a
        PID file whose pid != ``os.getpid()`` to protect ``--replace``
        handoffs.  Stale-cleanup must not go through that path or real
        crashed-process PID files never get removed.
        """
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        pid_path = tmp_path / "gateway.pid"
        lock_path = tmp_path / "gateway.lock"

        # PID that is guaranteed not alive and not our own.
        dead_foreign_pid = 999999
        assert dead_foreign_pid != os.getpid()

        pid_path.write_text(json.dumps({
            "pid": dead_foreign_pid,
            "kind": "hermes-gateway",
            "argv": ["python", "-m", "hermes_cli.main", "gateway"],
            "start_time": 123,
        }))
        lock_path.write_text(json.dumps({
            "pid": dead_foreign_pid,
            "kind": "hermes-gateway",
            "argv": ["python", "-m", "hermes_cli.main", "gateway"],
            "start_time": 123,
        }))

        # No live lock holder → get_running_pid should clean both files.
        assert status.get_running_pid() is None
        assert not pid_path.exists()
        assert not lock_path.exists()

    def test_get_running_pid_falls_back_to_live_lock_record(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        pid_path = tmp_path / "gateway.pid"
        pid_path.write_text(json.dumps({
            "pid": 99999,
            "kind": "hermes-gateway",
            "argv": ["python", "-m", "hermes_cli.main", "gateway"],
            "start_time": 123,
        }))

        monkeypatch.setattr(status, "_get_process_start_time", lambda pid: 123)
        monkeypatch.setattr(status, "_read_process_cmdline", lambda pid: None)
        monkeypatch.setattr(
            status,
            "_build_pid_record",
            lambda: {
                "pid": os.getpid(),
                "kind": "hermes-gateway",
                "argv": ["python", "-m", "hermes_cli.main", "gateway"],
                "start_time": 123,
            },
        )
        assert status.acquire_gateway_runtime_lock() is True

        def fake_kill(pid, sig):
            if pid == 99999:
                raise ProcessLookupError
            return None

        monkeypatch.setattr(status.os, "kill", fake_kill)

        try:
            assert status.get_running_pid() == os.getpid()
        finally:
            status.release_gateway_runtime_lock()


class TestGatewayRuntimeStatus:
    def test_write_json_file_uses_atomic_json_write(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        calls = []

        def _fake_atomic_json_write(path, payload, **kwargs):
            calls.append((Path(path), payload, kwargs))

        monkeypatch.setattr(status, "atomic_json_write", _fake_atomic_json_write)

        payload = {"gateway_state": "running"}
        target = tmp_path / "gateway_state.json"
        status._write_json_file(target, payload)

        assert calls == [
            (
                target,
                payload,
                {"indent": None, "separators": (",", ":")},
            )
        ]

    def test_write_runtime_status_overwrites_stale_pid_on_restart(self, tmp_path, monkeypatch):
        """Regression: setdefault() preserved stale PID from previous process (#1631)."""
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))

        # Simulate a previous gateway run that left a state file with a stale PID
        state_path = tmp_path / "gateway_state.json"
        state_path.write_text(json.dumps({
            "pid": 99999,
            "start_time": 1000.0,
            "kind": "hermes-gateway",
            "platforms": {},
            "updated_at": "2025-01-01T00:00:00Z",
        }))

        status.write_runtime_status(gateway_state="running")

        payload = status.read_runtime_status()
        assert payload["pid"] == os.getpid(), "PID should be overwritten, not preserved via setdefault"
        assert payload["start_time"] != 1000.0, "start_time should be overwritten on restart"

    def test_write_runtime_status_overwrites_stale_argv_on_restart(self, tmp_path, monkeypatch):
        """Regression: gateway_state.json must not keep the previous launch argv."""
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))

        state_path = tmp_path / "gateway_state.json"
        state_path.write_text(json.dumps({
            "pid": 99999,
            "start_time": 1000.0,
            "kind": "hermes-gateway",
            "argv": ["/old/path/hermes", "gateway", "run"],
            "platforms": {},
            "updated_at": "2025-01-01T00:00:00Z",
        }))

        monkeypatch.setattr(status.sys, "argv", ["/new/path/hermes", "gateway", "run"])
        monkeypatch.setattr(status, "_get_process_start_time", lambda pid: 2000)

        status.write_runtime_status(gateway_state="running")

        payload = status.read_runtime_status()
        assert payload["argv"] == ["/new/path/hermes", "gateway", "run"]
        assert payload["pid"] == os.getpid()
        assert payload["start_time"] == 2000

    def test_runtime_status_running_pid_rejects_stale_record_for_supervisor_pid(self, monkeypatch):
        """Regression: stale profile runtime state must not mark s6 supervisors live.

        Docker per-profile supervision can leave a named profile with
        ``gateway_state=running`` metadata while the real gateway process is gone
        and the recorded PID now belongs to ``s6-supervise`` or ``s6-log``.  If
        the live command line is readable, it wins over the stale record argv.
        """
        payload = {
            "pid": 132,
            "start_time": 123,
            "gateway_state": "running",
            "kind": "hermes-gateway",
            "argv": ["/opt/hermes/.venv/bin/hermes", "gateway", "run", "--replace"],
        }

        monkeypatch.setattr(status, "_pid_exists", lambda pid: True)
        monkeypatch.setattr(status, "_get_process_start_time", lambda pid: 123)
        monkeypatch.setattr(status, "_read_process_cmdline", lambda pid: "s6-supervise gateway-coder")

        assert status.get_runtime_status_running_pid(payload) is None

    def test_runtime_status_running_pid_uses_record_when_cmdline_unreadable(self, monkeypatch):
        """Keep the cross-platform fallback for hosts where cmdline is unavailable."""
        payload = {
            "pid": 132,
            "start_time": 123,
            "gateway_state": "running",
            "kind": "hermes-gateway",
            "argv": ["/opt/hermes/.venv/bin/hermes", "gateway", "run", "--replace"],
        }

        monkeypatch.setattr(status, "_pid_exists", lambda pid: True)
        monkeypatch.setattr(status, "_get_process_start_time", lambda pid: 123)
        monkeypatch.setattr(status, "_read_process_cmdline", lambda pid: None)

        assert status.get_runtime_status_running_pid(payload) == 132

    def test_runtime_status_running_pid_rejects_pid_reused_by_other_profile(self, monkeypatch):
        """Regression (user report): a stale profile's recycled PID must not be
        reported running just because it now hosts a DIFFERENT profile's gateway.

        Per-profile Docker supervision: ``coder``'s gateway died leaving a
        ``gateway_state=running`` record at PID 139.  The OS then recycled 139
        onto the live *default* gateway (``hermes gateway run``).  The recorded
        ``start_time`` is absent (older state file), so the start-time PID-reuse
        guard does not catch it.  Without the profile scope the live command
        line still ``looks_like_gateway`` and ``coder`` is wrongly reported up.
        """
        payload = {
            "pid": 139,
            "gateway_state": "running",
            "kind": "hermes-gateway",
            "argv": ["hermes", "gateway", "run"],
        }
        coder_home = Path("/opt/data/profiles/coder")

        monkeypatch.setattr(status, "_pid_exists", lambda pid: True)
        monkeypatch.setattr(status, "_get_process_start_time", lambda pid: None)
        # PID 139 is now the live DEFAULT gateway (bare, no -p coder).
        monkeypatch.setattr(
            status, "_read_process_cmdline", lambda pid: "hermes gateway run --replace"
        )

        assert (
            status.get_runtime_status_running_pid(payload, expected_home=coder_home)
            is None
        )

    def test_runtime_status_running_pid_accepts_matching_profile_cmdline(self, monkeypatch):
        """A genuinely-live named gateway carries ``-p <profile>`` / ``--profile``
        on its command line and must be reported running for that profile."""
        payload = {
            "pid": 139,
            "gateway_state": "running",
            "kind": "hermes-gateway",
            "argv": ["hermes", "gateway", "run"],
            "start_time": 1000,
        }
        coder_home = Path("/opt/data/profiles/coder")

        monkeypatch.setattr(status, "_pid_exists", lambda pid: True)
        monkeypatch.setattr(status, "_get_process_start_time", lambda pid: 1000)
        for cmdline in (
            "hermes -p coder gateway run --replace",
            "/opt/hermes/.venv/bin/hermes --profile coder gateway run --replace",
            "hermes_home=/opt/data/profiles/coder hermes gateway run --replace",
        ):
            monkeypatch.setattr(status, "_read_process_cmdline", lambda pid, c=cmdline: c)
            assert (
                status.get_runtime_status_running_pid(payload, expected_home=coder_home)
                == 139
            ), cmdline

    def test_runtime_status_running_pid_default_profile_rejects_named_cmdline(self, monkeypatch):
        """The default/root profile runs a bare gateway (no profile flag).  A
        recycled PID now hosting a *named* profile gateway must not be reported
        running for the default profile."""
        payload = {
            "pid": 139,
            "gateway_state": "running",
            "kind": "hermes-gateway",
            "argv": ["hermes", "gateway", "run"],
        }
        default_home = Path("/opt/data")

        monkeypatch.setattr(status, "_pid_exists", lambda pid: True)
        monkeypatch.setattr(status, "_get_process_start_time", lambda pid: None)
        monkeypatch.setattr(
            status, "_read_process_cmdline", lambda pid: "hermes -p coder gateway run --replace"
        )

        assert (
            status.get_runtime_status_running_pid(payload, expected_home=default_home)
            is None
        )

    def test_runtime_status_running_pid_default_profile_accepts_bare_cmdline(self, monkeypatch):
        """The default/root gateway (bare ``hermes gateway run``) is reported
        running for the default profile."""
        payload = {
            "pid": 139,
            "gateway_state": "running",
            "kind": "hermes-gateway",
            "argv": ["hermes", "gateway", "run"],
            "start_time": 1000,
        }
        default_home = Path("/opt/data")

        monkeypatch.setattr(status, "_pid_exists", lambda pid: True)
        monkeypatch.setattr(status, "_get_process_start_time", lambda pid: 1000)
        monkeypatch.setattr(
            status, "_read_process_cmdline", lambda pid: "hermes gateway run --replace"
        )

        assert (
            status.get_runtime_status_running_pid(payload, expected_home=default_home)
            == 139
        )

    def test_runtime_status_running_pid_profile_scope_falls_back_when_cmdline_unreadable(self, monkeypatch):
        """When the live command line is unreadable (Windows/permission), the
        profile scope cannot apply — fall back to the persisted record so the
        cross-platform behavior is preserved."""
        payload = {
            "pid": 139,
            "gateway_state": "running",
            "kind": "hermes-gateway",
            "argv": ["hermes", "gateway", "run"],
            "start_time": 1000,
        }
        coder_home = Path("/opt/data/profiles/coder")

        monkeypatch.setattr(status, "_pid_exists", lambda pid: True)
        monkeypatch.setattr(status, "_get_process_start_time", lambda pid: 1000)
        monkeypatch.setattr(status, "_read_process_cmdline", lambda pid: None)

        assert (
            status.get_runtime_status_running_pid(payload, expected_home=coder_home)
            == 139
        )

    def test_write_runtime_status_records_platform_failure(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))

        status.write_runtime_status(
            gateway_state="startup_failed",
            exit_reason="telegram conflict",
            platform="telegram",
            platform_state="fatal",
            error_code="telegram_polling_conflict",
            error_message="another poller is active",
        )

        payload = status.read_runtime_status()
        assert payload["gateway_state"] == "startup_failed"
        assert payload["exit_reason"] == "telegram conflict"
        assert payload["platforms"]["telegram"]["state"] == "fatal"
        assert payload["platforms"]["telegram"]["error_code"] == "telegram_polling_conflict"
        assert payload["platforms"]["telegram"]["error_message"] == "another poller is active"

    def test_write_runtime_status_explicit_none_clears_stale_fields(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))

        status.write_runtime_status(
            gateway_state="startup_failed",
            exit_reason="stale error",
            platform="discord",
            platform_state="fatal",
            error_code="discord_timeout",
            error_message="stale platform error",
        )

        status.write_runtime_status(
            gateway_state="running",
            exit_reason=None,
            platform="discord",
            platform_state="connected",
            error_code=None,
            error_message=None,
        )

        payload = status.read_runtime_status()
        assert payload["gateway_state"] == "running"
        assert payload["exit_reason"] is None
        assert payload["platforms"]["discord"]["state"] == "connected"
        assert payload["platforms"]["discord"]["error_code"] is None
        assert payload["platforms"]["discord"]["error_message"] is None


class TestGetProcessStartTime:
    """Start-time fingerprint backing the PID-reuse guard (#43846 / #50468).

    Must be stable across repeated reads of the same live process and degrade to
    a cross-platform psutil fallback when /proc is unavailable (macOS/Windows),
    so the guard isn't a Linux-only no-op.
    """

    def test_live_process_is_stable_int(self):
        import subprocess
        import time
        p = subprocess.Popen(["sleep", "20"])
        try:
            a = status._get_process_start_time(p.pid)
            time.sleep(0.2)
            b = status._get_process_start_time(p.pid)
            assert a is not None and isinstance(a, int)
            assert a == b  # same process → identical fingerprint
        finally:
            p.kill()
            p.wait()

    def test_dead_pid_returns_none(self):
        assert status._get_process_start_time(999999999) is None

    def test_psutil_fallback_when_no_proc(self, monkeypatch):
        """When /proc is missing (macOS/Windows), psutil supplies a stable int."""
        import subprocess
        orig_read_text = Path.read_text

        def no_proc(self, *args, **kwargs):
            if str(self).startswith("/proc/"):
                raise FileNotFoundError
            return orig_read_text(self, *args, **kwargs)

        monkeypatch.setattr(Path, "read_text", no_proc)
        p = subprocess.Popen(["sleep", "20"])
        try:
            a = status._get_process_start_time(p.pid)
            b = status._get_process_start_time(p.pid)
            assert a is not None and isinstance(a, int)
            assert a == b  # fallback is stable across reads
        finally:
            p.kill()
            p.wait()


class TestTerminatePid:
    def test_force_uses_taskkill_on_windows(self, monkeypatch):
        calls = []
        monkeypatch.setattr(status, "_IS_WINDOWS", True)

        def fake_run(cmd, capture_output=False, text=False, timeout=None, creationflags=0):
            calls.append((cmd, capture_output, text, timeout, creationflags))
            return SimpleNamespace(returncode=0, stdout="", stderr="")

        monkeypatch.setattr(status.subprocess, "run", fake_run)

        status.terminate_pid(123, force=True)

        # taskkill is spawned with the no-window flag so the windowless
        # pythonw.exe backend doesn't flash a conhost window on force-kill.
        # windows_hide_flags() is 0 on the POSIX test host (a valid no-op
        # creationflags value); on real Windows it is CREATE_NO_WINDOW.
        from hermes_cli._subprocess_compat import windows_hide_flags

        assert calls == [
            (["taskkill", "/PID", "123", "/T", "/F"], True, True, 10, windows_hide_flags())
        ]

    def test_force_falls_back_to_sigterm_when_taskkill_missing(self, monkeypatch):
        calls = []
        monkeypatch.setattr(status, "_IS_WINDOWS", True)

        def fake_run(*args, **kwargs):
            raise FileNotFoundError

        def fake_kill(pid, sig):
            calls.append((pid, sig))

        monkeypatch.setattr(status.subprocess, "run", fake_run)
        monkeypatch.setattr(status.os, "kill", fake_kill)

        status.terminate_pid(456, force=True)

        assert calls == [(456, status.signal.SIGTERM)]


class TestScopedLocks:
    def test_windows_file_lock_uses_high_offset(self, tmp_path, monkeypatch):
        lock_path = tmp_path / "gateway.lock"
        handle = open(lock_path, "a+", encoding="utf-8")
        fd = handle.fileno()
        calls = []

        def fake_locking(fd, mode, size):
            calls.append((fd, mode, size, handle.tell()))

        monkeypatch.setattr(status, "_IS_WINDOWS", True)
        monkeypatch.setattr(
            status,
            "msvcrt",
            SimpleNamespace(LK_NBLCK=1, LK_UNLCK=2, locking=fake_locking),
            raising=False,
        )

        try:
            assert status._try_acquire_file_lock(handle) is True
            status._release_file_lock(handle)
        finally:
            handle.close()

        assert calls == [
            (fd, 1, 1, status._WINDOWS_LOCK_OFFSET),
            (fd, 2, 1, status._WINDOWS_LOCK_OFFSET),
        ]
        assert lock_path.read_text(encoding="utf-8") == "\n"

    def test_acquire_scoped_lock_rejects_live_other_process(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_GATEWAY_LOCK_DIR", str(tmp_path / "locks"))
        lock_path = tmp_path / "locks" / "telegram-bot-token-2bb80d537b1da3e3.lock"
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        lock_path.write_text(json.dumps({
            "pid": 99999,
            "start_time": 123,
            "kind": "hermes-gateway",
        }))

        # Post-#21561 the liveness probe routes through
        # ``gateway.status._pid_exists`` (psutil-first, safe on Windows).
        monkeypatch.setattr(status, "_pid_exists", lambda pid: True)
        monkeypatch.setattr(status, "_get_process_start_time", lambda pid: 123)

        acquired, existing = status.acquire_scoped_lock("telegram-bot-token", "secret", metadata={"platform": "telegram"})

        assert acquired is False
        assert existing["pid"] == 99999

    def test_acquire_scoped_lock_replaces_pid_reused_by_unrelated_process(self, tmp_path, monkeypatch):
        """macOS regression: PID reused by an unrelated process with start_time=None.

        On macOS /proc is unavailable, so both the lock record and the live
        process report start_time=None.  The live PID is alive (os.kill
        succeeds) but belongs to a completely different program.  The lock
        must be treated as stale.
        """
        monkeypatch.setenv("HERMES_GATEWAY_LOCK_DIR", str(tmp_path / "locks"))
        lock_path = tmp_path / "locks" / "telegram-bot-token-2bb80d537b1da3e3.lock"
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        lock_path.write_text(json.dumps({
            "pid": 873,
            "start_time": None,
            "kind": "hermes-gateway",
            "argv": ["/Users/user/.hermes/hermes-agent/hermes_cli/main.py", "gateway", "run", "--replace"],
        }))

        # Post-#21561 the liveness probe routes through
        # ``gateway.status._pid_exists`` (psutil-first, safe on Windows),
        # not ``os.kill``.
        monkeypatch.setattr(status, "_pid_exists", lambda pid: True)
        monkeypatch.setattr(status, "_get_process_start_time", lambda pid: None)
        monkeypatch.setattr(status, "_looks_like_gateway_process", lambda pid: False)
        # On macOS ``ps`` is available, so _read_process_cmdline returns the
        # unrelated process's name.  This confirms the PID was reused.
        monkeypatch.setattr(status, "_read_process_cmdline", lambda pid: "/usr/libexec/bluetoothuserd")

        acquired, existing = status.acquire_scoped_lock("telegram-bot-token", "secret", metadata={"platform": "telegram"})

        assert acquired is True
        payload = json.loads(lock_path.read_text())
        assert payload["pid"] == os.getpid()
        assert payload["metadata"]["platform"] == "telegram"

    def test_acquire_scoped_lock_keeps_lock_when_cmdline_unreadable_but_record_is_gateway(self, tmp_path, monkeypatch):
        """Windows regression: ps unavailable so cmdline cannot be read.

        When start_time is None on both sides and _looks_like_gateway_process
        returns False because ps is missing (not because the PID belongs to an
        unrelated process), the stale check must not delete a valid gateway
        lock.  Fall back to the lock record's own argv — written by the
        gateway at startup — before declaring the lock stale.
        """
        monkeypatch.setenv("HERMES_GATEWAY_LOCK_DIR", str(tmp_path / "locks"))
        lock_path = tmp_path / "locks" / "telegram-bot-token-2bb80d537b1da3e3.lock"
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        lock_path.write_text(json.dumps({
            "pid": 99999,
            "start_time": None,
            "kind": "hermes-gateway",
            "argv": ["hermes_cli/main.py", "gateway", "run"],
        }))

        monkeypatch.setattr(status, "_pid_exists", lambda pid: True)
        monkeypatch.setattr(status, "_get_process_start_time", lambda pid: None)
        # Windows: ps not available, so _read_process_cmdline returns None
        # and _looks_like_gateway_process returns False for every process.
        monkeypatch.setattr(status, "_looks_like_gateway_process", lambda pid: False)
        monkeypatch.setattr(status, "_read_process_cmdline", lambda pid: None)

        acquired, existing = status.acquire_scoped_lock("telegram-bot-token", "secret", metadata={"platform": "telegram"})

        assert acquired is False
        assert existing["pid"] == 99999

    def test_acquire_scoped_lock_keeps_lock_when_pid_reused_by_gateway(self, tmp_path, monkeypatch):
        """When start_time is None but the live PID still looks like a gateway, keep the lock."""
        monkeypatch.setenv("HERMES_GATEWAY_LOCK_DIR", str(tmp_path / "locks"))
        lock_path = tmp_path / "locks" / "telegram-bot-token-2bb80d537b1da3e3.lock"
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        lock_path.write_text(json.dumps({
            "pid": 99999,
            "start_time": None,
            "kind": "hermes-gateway",
            "argv": ["/Users/user/.hermes/hermes-agent/hermes_cli/main.py", "gateway", "run", "--replace"],
        }))

        monkeypatch.setattr(status, "_pid_exists", lambda pid: True)
        monkeypatch.setattr(status, "_get_process_start_time", lambda pid: None)
        monkeypatch.setattr(status, "_looks_like_gateway_process", lambda pid: True)

        acquired, existing = status.acquire_scoped_lock("telegram-bot-token", "secret", metadata={"platform": "telegram"})

        assert acquired is False
        assert existing["pid"] == 99999

    def test_acquire_scoped_lock_replaces_stale_record(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_GATEWAY_LOCK_DIR", str(tmp_path / "locks"))
        lock_path = tmp_path / "locks" / "telegram-bot-token-2bb80d537b1da3e3.lock"
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        lock_path.write_text(json.dumps({
            "pid": 99999,
            "start_time": 123,
            "kind": "hermes-gateway",
        }))

        # Post-#21561: simulate "PID gone" via _pid_exists returning False.
        monkeypatch.setattr(status, "_pid_exists", lambda pid: False)

        acquired, existing = status.acquire_scoped_lock("telegram-bot-token", "secret", metadata={"platform": "telegram"})

        assert acquired is True
        payload = json.loads(lock_path.read_text())
        assert payload["pid"] == os.getpid()
        assert payload["metadata"]["platform"] == "telegram"

    def test_acquire_scoped_lock_recovers_empty_lock_file(self, tmp_path, monkeypatch):
        """Empty lock file (0 bytes) left by a crashed process should be treated as stale."""
        monkeypatch.setenv("HERMES_GATEWAY_LOCK_DIR", str(tmp_path / "locks"))
        lock_path = tmp_path / "locks" / "slack-app-token-2bb80d537b1da3e3.lock"
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        lock_path.write_text("")  # simulate crash between O_CREAT and json.dump

        acquired, existing = status.acquire_scoped_lock("slack-app-token", "secret", metadata={"platform": "slack"})

        assert acquired is True
        payload = json.loads(lock_path.read_text())
        assert payload["pid"] == os.getpid()
        assert payload["metadata"]["platform"] == "slack"

    def test_acquire_scoped_lock_recovers_corrupt_lock_file(self, tmp_path, monkeypatch):
        """Lock file with invalid JSON should be treated as stale."""
        monkeypatch.setenv("HERMES_GATEWAY_LOCK_DIR", str(tmp_path / "locks"))
        lock_path = tmp_path / "locks" / "slack-app-token-2bb80d537b1da3e3.lock"
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        lock_path.write_text("{truncated")  # simulate partial write

        acquired, existing = status.acquire_scoped_lock("slack-app-token", "secret", metadata={"platform": "slack"})

        assert acquired is True
        payload = json.loads(lock_path.read_text())
        assert payload["pid"] == os.getpid()

    def test_release_scoped_lock_only_removes_current_owner(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_GATEWAY_LOCK_DIR", str(tmp_path / "locks"))

        acquired, _ = status.acquire_scoped_lock("telegram-bot-token", "secret", metadata={"platform": "telegram"})
        assert acquired is True
        lock_path = tmp_path / "locks" / "telegram-bot-token-2bb80d537b1da3e3.lock"
        assert lock_path.exists()

        status.release_scoped_lock("telegram-bot-token", "secret")
        assert not lock_path.exists()

    def test_release_all_scoped_locks_can_target_single_owner(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_GATEWAY_LOCK_DIR", str(tmp_path / "locks"))
        lock_dir = tmp_path / "locks"
        lock_dir.mkdir(parents=True, exist_ok=True)

        target_lock = lock_dir / "telegram-bot-token-target.lock"
        other_lock = lock_dir / "slack-app-token-other.lock"
        target_lock.write_text(json.dumps({
            "pid": 111,
            "start_time": 222,
            "kind": "hermes-gateway",
        }))
        other_lock.write_text(json.dumps({
            "pid": 999,
            "start_time": 333,
            "kind": "hermes-gateway",
        }))

        removed = status.release_all_scoped_locks(
            owner_pid=111,
            owner_start_time=222,
        )

        assert removed == 1
        assert not target_lock.exists()
        assert other_lock.exists()

    def test_release_all_scoped_locks_skips_pid_reuse_mismatch(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_GATEWAY_LOCK_DIR", str(tmp_path / "locks"))
        lock_dir = tmp_path / "locks"
        lock_dir.mkdir(parents=True, exist_ok=True)

        reused_pid_lock = lock_dir / "telegram-bot-token-reused.lock"
        reused_pid_lock.write_text(json.dumps({
            "pid": 111,
            "start_time": 999,
            "kind": "hermes-gateway",
        }))

        removed = status.release_all_scoped_locks(
            owner_pid=111,
            owner_start_time=222,
        )

        assert removed == 0
        assert reused_pid_lock.exists()

    def test_acquire_scoped_lock_replaces_reused_pid_even_with_matching_start_time(self, tmp_path, monkeypatch):
        """Regression: boot-time PID+start_time collision must not block gateway startup.

        On Linux, systemd assigns PIDs and jiffy start_times deterministically
        across reboots. A core service (e.g. cron) can land on the exact same
        PID and start_time as a previous gateway. The start_time check passes,
        but the live process is not a gateway — the lock must be evicted.
        """
        monkeypatch.setenv("HERMES_GATEWAY_LOCK_DIR", str(tmp_path / "locks"))
        lock_path = tmp_path / "locks" / "telegram-bot-token-2bb80d537b1da3e3.lock"
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        lock_path.write_text(json.dumps({
            "pid": 840,
            "start_time": 123,
            "kind": "hermes-gateway",
            "argv": ["/usr/bin/python", "-m", "hermes_cli.main", "gateway", "run"],
        }))

        monkeypatch.setattr(status, "_pid_exists", lambda pid: True)
        monkeypatch.setattr(status, "_get_process_start_time", lambda pid: 123)
        monkeypatch.setattr(status, "_looks_like_gateway_process", lambda pid: False)
        monkeypatch.setattr(status, "_read_process_cmdline", lambda pid: "/usr/sbin/nginx")

        acquired, existing = status.acquire_scoped_lock("telegram-bot-token", "secret", metadata={"platform": "telegram"})

        assert acquired is True
        payload = json.loads(lock_path.read_text())
        assert payload["pid"] == os.getpid()
        assert payload["metadata"]["platform"] == "telegram"


class TestTakeoverMarker:
    """Tests for the --replace takeover marker.

    The marker breaks the post-#5646 flap loop between two gateway services
    fighting for the same bot token. The replacer writes a file naming the
    target PID + start_time; the target's shutdown handler sees it and exits
    0 instead of 1, so systemd's Restart=on-failure doesn't revive it.
    """

    def test_write_marker_records_target_identity(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        monkeypatch.setattr(status, "_get_process_start_time", lambda pid: 42)

        ok = status.write_takeover_marker(target_pid=12345)

        assert ok is True
        marker = tmp_path / ".gateway-takeover.json"
        assert marker.exists()
        payload = json.loads(marker.read_text())
        assert payload["target_pid"] == 12345
        assert payload["target_start_time"] == 42
        assert payload["replacer_pid"] == os.getpid()
        assert "written_at" in payload

    def test_consume_returns_true_when_marker_names_self(self, tmp_path, monkeypatch):
        """Primary happy path: planned takeover is recognised."""
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        # Mark THIS process as the target
        monkeypatch.setattr(status, "_get_process_start_time", lambda pid: 100)
        ok = status.write_takeover_marker(target_pid=os.getpid())
        assert ok is True

        # Call consume as if this process just got SIGTERMed
        result = status.consume_takeover_marker_for_self()

        assert result is True
        # Marker must be unlinked after consumption
        assert not (tmp_path / ".gateway-takeover.json").exists()

    def test_consume_returns_false_for_different_pid(self, tmp_path, monkeypatch):
        """A marker naming a DIFFERENT process must not be consumed as ours."""
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        monkeypatch.setattr(status, "_get_process_start_time", lambda pid: 100)
        # Marker names a different PID
        other_pid = os.getpid() + 9999
        ok = status.write_takeover_marker(target_pid=other_pid)
        assert ok is True

        result = status.consume_takeover_marker_for_self()

        assert result is False
        # Marker IS unlinked even on non-match (the record has been consumed
        # and isn't relevant to us — leaving it around would grief a later
        # legitimate check).
        assert not (tmp_path / ".gateway-takeover.json").exists()

    def test_consume_returns_false_on_start_time_mismatch(self, tmp_path, monkeypatch):
        """PID reuse defence: old marker's start_time mismatches current process."""
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        # Marker says target started at time 100 with our PID
        monkeypatch.setattr(status, "_get_process_start_time", lambda pid: 100)
        status.write_takeover_marker(target_pid=os.getpid())

        # Now change the reported start_time to simulate PID reuse
        monkeypatch.setattr(status, "_get_process_start_time", lambda pid: 9999)

        result = status.consume_takeover_marker_for_self()

        assert result is False

    def test_consume_returns_true_on_windows_when_start_time_unavailable(
        self, tmp_path, monkeypatch
    ):
        """Takeover consume must also recognise a self-marker on platforms
        without ``/proc`` (macOS / native Windows).

        ``consume_takeover_marker_for_self`` shares ``_consume_pid_marker_for_self``
        with the planned-stop path, so the same start_time fallback applies:
        a ``--replace`` SIGTERM on Windows (where start_time is None on both
        sides) must be recognised as a planned takeover and exit 0, not be
        misclassified as an unexpected UNKNOWN exit. With start_time
        unavailable we fall back to PID equality alone, bounded by the TTL.
        """
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        # Simulate Windows: no start_time available for any PID.
        monkeypatch.setattr(status, "_get_process_start_time", lambda pid: None)

        ok = status.write_takeover_marker(target_pid=os.getpid())
        assert ok is True
        payload = json.loads((tmp_path / ".gateway-takeover.json").read_text())
        assert payload["target_start_time"] is None

        result = status.consume_takeover_marker_for_self()

        assert result is True
        assert not (tmp_path / ".gateway-takeover.json").exists()

    def test_consume_returns_false_when_marker_missing(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))

        result = status.consume_takeover_marker_for_self()

        assert result is False

    def test_consume_returns_false_for_stale_marker(self, tmp_path, monkeypatch):
        """A marker older than 60s must be ignored."""
        from datetime import datetime, timezone, timedelta

        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        marker_path = tmp_path / ".gateway-takeover.json"
        # Hand-craft a marker written 2 minutes ago
        stale_time = (datetime.now(timezone.utc) - timedelta(minutes=2)).isoformat()
        marker_path.write_text(json.dumps({
            "target_pid": os.getpid(),
            "target_start_time": 123,
            "replacer_pid": 99999,
            "written_at": stale_time,
        }))
        monkeypatch.setattr(status, "_get_process_start_time", lambda pid: 123)

        result = status.consume_takeover_marker_for_self()

        assert result is False
        # Stale markers are unlinked so a later legit shutdown isn't griefed
        assert not marker_path.exists()

    def test_consume_handles_malformed_marker_gracefully(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        marker_path = tmp_path / ".gateway-takeover.json"
        marker_path.write_text("not valid json{")

        # Must not raise
        result = status.consume_takeover_marker_for_self()

        assert result is False

    def test_consume_handles_marker_with_missing_fields(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        marker_path = tmp_path / ".gateway-takeover.json"
        marker_path.write_text(json.dumps({"only_replacer_pid": 99999}))

        result = status.consume_takeover_marker_for_self()

        assert result is False
        # Malformed marker should be cleaned up
        assert not marker_path.exists()

    def test_clear_takeover_marker_is_idempotent(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))

        # Nothing to clear — must not raise
        status.clear_takeover_marker()

        # Write then clear
        monkeypatch.setattr(status, "_get_process_start_time", lambda pid: 100)
        status.write_takeover_marker(target_pid=12345)
        assert (tmp_path / ".gateway-takeover.json").exists()

        status.clear_takeover_marker()
        assert not (tmp_path / ".gateway-takeover.json").exists()

        # Clear again — still no error
        status.clear_takeover_marker()

    def test_write_marker_returns_false_on_write_failure(self, tmp_path, monkeypatch):
        """write_takeover_marker is best-effort; returns False but doesn't raise."""
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))

        def raise_oserror(*args, **kwargs):
            raise OSError("simulated write failure")

        monkeypatch.setattr(status, "_write_json_file", raise_oserror)

        ok = status.write_takeover_marker(target_pid=12345)

        assert ok is False

    def test_consume_ignores_marker_for_different_process_and_prevents_stale_grief(
        self, tmp_path, monkeypatch
    ):
        """Regression: a stale marker from a dead replacer naming a dead
        target must not accidentally cause an unrelated future gateway to
        exit 0 on legitimate SIGTERM.

        The distinguishing check is ``target_pid == our_pid AND
        target_start_time == our_start_time``. Different PID always wins.
        """
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        marker_path = tmp_path / ".gateway-takeover.json"
        # Fresh marker (timestamp is recent) but names a totally different PID
        from datetime import datetime, timezone
        marker_path.write_text(json.dumps({
            "target_pid": os.getpid() + 10000,
            "target_start_time": 42,
            "replacer_pid": 99999,
            "written_at": datetime.now(timezone.utc).isoformat(),
        }))
        monkeypatch.setattr(status, "_get_process_start_time", lambda pid: 42)

        result = status.consume_takeover_marker_for_self()

        # We are not the target — must NOT consume as planned
        assert result is False

    def test_write_marker_records_replacer_hermes_home(self, tmp_path, monkeypatch):
        """The marker stamps the replacer's HERMES_HOME for cross-profile guard (#29092)."""
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        monkeypatch.setattr(status, "_get_process_start_time", lambda pid: 42)

        status.write_takeover_marker(target_pid=12345)

        payload = json.loads((tmp_path / ".gateway-takeover.json").read_text())
        assert payload["replacer_hermes_home"] == str(tmp_path)

    def test_consume_rejects_marker_from_different_profile(self, tmp_path, monkeypatch):
        """Regression (#29092): a marker written by a gateway under a DIFFERENT
        HERMES_HOME must be rejected even when PID + start_time coincidentally
        match — otherwise two profile services sharing a default ~/.hermes flap
        each other in an infinite SIGTERM/Restart loop. The mismatched marker is
        left in place so the profile it was actually meant for can consume it.
        """
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        monkeypatch.setattr(status, "_get_process_start_time", lambda pid: 100)
        marker_path = tmp_path / ".gateway-takeover.json"
        from datetime import datetime, timezone
        # Marker names OUR pid + start_time (the coincidental match the bug
        # relied on) but was written by a gateway in a different profile.
        marker_path.write_text(json.dumps({
            "target_pid": os.getpid(),
            "target_start_time": 100,
            "replacer_pid": 99999,
            "replacer_hermes_home": str(tmp_path / "profiles" / "other"),
            "written_at": datetime.now(timezone.utc).isoformat(),
        }))

        result = status.consume_takeover_marker_for_self()

        assert result is False
        # Left in place for the correct profile, not griefed away.
        assert marker_path.exists()

    def test_consume_accepts_legacy_marker_without_hermes_home(self, tmp_path, monkeypatch):
        """Back-compat (#29092): markers written by older Hermes versions have no
        ``replacer_hermes_home`` field; an absent field is treated as same-home so
        single-profile setups and mixed old/new deployments keep working.
        """
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        monkeypatch.setattr(status, "_get_process_start_time", lambda pid: 100)
        marker_path = tmp_path / ".gateway-takeover.json"
        from datetime import datetime, timezone
        marker_path.write_text(json.dumps({
            "target_pid": os.getpid(),
            "target_start_time": 100,
            "replacer_pid": 99999,
            "written_at": datetime.now(timezone.utc).isoformat(),
        }))

        result = status.consume_takeover_marker_for_self()

        assert result is True
        assert not marker_path.exists()


class TestPlannedStopMarker:
    """Tests for intentional service/manual gateway stop markers."""

    def test_write_marker_records_target_identity(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        monkeypatch.setattr(status, "_get_process_start_time", lambda pid: 42)

        ok = status.write_planned_stop_marker(target_pid=12345)

        assert ok is True
        marker = tmp_path / ".gateway-planned-stop.json"
        assert marker.exists()
        payload = json.loads(marker.read_text())
        assert payload["target_pid"] == 12345
        assert payload["target_start_time"] == 42
        assert payload["stopper_pid"] == os.getpid()
        assert "written_at" in payload

    def test_consume_returns_true_when_marker_names_self(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        monkeypatch.setattr(status, "_get_process_start_time", lambda pid: 100)
        ok = status.write_planned_stop_marker(target_pid=os.getpid())
        assert ok is True

        result = status.consume_planned_stop_marker_for_self()

        assert result is True
        assert not (tmp_path / ".gateway-planned-stop.json").exists()

    def test_consume_returns_false_for_different_pid(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        monkeypatch.setattr(status, "_get_process_start_time", lambda pid: 100)
        ok = status.write_planned_stop_marker(target_pid=os.getpid() + 9999)
        assert ok is True

        result = status.consume_planned_stop_marker_for_self()

        assert result is False
        assert not (tmp_path / ".gateway-planned-stop.json").exists()

    def test_consume_returns_false_for_stale_marker(self, tmp_path, monkeypatch):
        from datetime import datetime, timezone, timedelta

        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        marker_path = tmp_path / ".gateway-planned-stop.json"
        stale_time = (datetime.now(timezone.utc) - timedelta(minutes=2)).isoformat()
        marker_path.write_text(json.dumps({
            "target_pid": os.getpid(),
            "target_start_time": 123,
            "stopper_pid": 99999,
            "written_at": stale_time,
        }))
        monkeypatch.setattr(status, "_get_process_start_time", lambda pid: 123)

        result = status.consume_planned_stop_marker_for_self()

        assert result is False
        assert not marker_path.exists()

    def test_clear_planned_stop_marker_is_idempotent(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        monkeypatch.setattr(status, "_get_process_start_time", lambda pid: 100)

        status.clear_planned_stop_marker()
        status.write_planned_stop_marker(target_pid=12345)
        assert (tmp_path / ".gateway-planned-stop.json").exists()

        status.clear_planned_stop_marker()

        assert not (tmp_path / ".gateway-planned-stop.json").exists()
        status.clear_planned_stop_marker()

    def test_write_marker_returns_false_on_write_failure(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))

        def raise_oserror(*args, **kwargs):
            raise OSError("simulated write failure")

        monkeypatch.setattr(status, "_write_json_file", raise_oserror)

        ok = status.write_planned_stop_marker(target_pid=12345)

        assert ok is False

    def test_consume_returns_true_on_windows_when_start_time_unavailable(
        self, tmp_path, monkeypatch
    ):
        """Regression for #34597: a legitimate stop must be recognised on
        platforms without ``/proc``.

        ``_get_process_start_time`` returns None on macOS / native Windows
        (no ``/proc/<pid>/stat``). The planned-stop watcher only runs there,
        so if the authoritative consume required a non-None start_time match
        it would always return False — and ``hermes gateway stop`` would be
        misclassified as an unexpected ``UNKNOWN`` exit, exit 1, and revived
        by the service manager (the very crash loop #34597 set out to fix).
        With start_time unavailable on BOTH sides we fall back to PID
        equality alone, bounded by the marker TTL.
        """
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        # Simulate Windows: no start_time available for any PID.
        monkeypatch.setattr(status, "_get_process_start_time", lambda pid: None)

        ok = status.write_planned_stop_marker(target_pid=os.getpid())
        assert ok is True
        # Marker carries a null start_time, exactly as written on Windows.
        payload = json.loads((tmp_path / ".gateway-planned-stop.json").read_text())
        assert payload["target_start_time"] is None

        result = status.consume_planned_stop_marker_for_self()

        assert result is True
        assert not (tmp_path / ".gateway-planned-stop.json").exists()

    def test_consume_still_rejects_foreign_pid_when_start_time_unavailable(
        self, tmp_path, monkeypatch
    ):
        """The PID-only fallback must NOT match a marker naming another PID.

        Falling back to PID equality when start_time is unknown must remain
        a PID check — a marker for a different process is never ours.
        """
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        monkeypatch.setattr(status, "_get_process_start_time", lambda pid: None)

        ok = status.write_planned_stop_marker(target_pid=os.getpid() + 9999)
        assert ok is True

        result = status.consume_planned_stop_marker_for_self()

        assert result is False

    def test_consume_still_rejects_start_time_mismatch_when_both_known(
        self, tmp_path, monkeypatch
    ):
        """PID-reuse defence is preserved when BOTH start_times are present.

        The Windows fallback only relaxes matching when a start_time is
        unavailable. When both sides report one (Linux), a mismatch must
        still reject — otherwise PID reuse could resurrect a stale marker.
        """
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        monkeypatch.setattr(status, "_get_process_start_time", lambda pid: 100)
        status.write_planned_stop_marker(target_pid=os.getpid())

        # Simulate PID reuse: same PID, different start_time.
        monkeypatch.setattr(status, "_get_process_start_time", lambda pid: 9999)

        result = status.consume_planned_stop_marker_for_self()

        assert result is False


class TestReadProcessCmdlinePsFallback:
    """Tests for _read_process_cmdline falling back to ps on non-Linux."""

    def test_ps_fallback_when_proc_unavailable(self, monkeypatch):
        monkeypatch.setattr(status.Path, "read_bytes", lambda self: (_ for _ in ()).throw(FileNotFoundError))
        monkeypatch.setattr(
            status.subprocess, "run",
            lambda args, **kwargs: SimpleNamespace(returncode=0, stdout="/usr/libexec/bluetoothuserd\n"),
        )
        result = status._read_process_cmdline(873)
        assert result == "/usr/libexec/bluetoothuserd"

    def test_ps_fallback_returns_none_on_failure(self, monkeypatch):
        monkeypatch.setattr(status.Path, "read_bytes", lambda self: (_ for _ in ()).throw(FileNotFoundError))
        monkeypatch.setattr(
            status.subprocess, "run",
            lambda args, **kwargs: SimpleNamespace(returncode=1, stdout=""),
        )
        result = status._read_process_cmdline(99999)
        assert result is None

    def test_proc_cmdline_takes_priority_over_ps(self, monkeypatch):
        calls = []

        def fake_read_bytes(self):
            calls.append("proc")
            return b"python\x00hermes_cli/main.py\x00gateway\x00"

        monkeypatch.setattr(status.Path, "read_bytes", fake_read_bytes)
        result = status._read_process_cmdline(12345)
        assert "hermes_cli/main.py" in result
        assert calls == ["proc"]

    def test_ps_fallback_used_when_proc_returns_empty(self, monkeypatch):
        monkeypatch.setattr(status.Path, "read_bytes", lambda self: b"")
        monkeypatch.setattr(
            status.subprocess, "run",
            lambda args, **kwargs: SimpleNamespace(returncode=0, stdout="python hermes_cli/main.py gateway run\n"),
        )
        result = status._read_process_cmdline(12345)
        assert "hermes_cli/main.py" in result


class TestCorruptStatusFiles:
    """A status / pid file holding non-UTF-8 (binary) bytes must read as
    None, not crash the gateway status path with UnicodeDecodeError."""

    def test_read_json_file_returns_none_on_binary_garbage(self, tmp_path):
        p = tmp_path / "runtime.json"
        p.write_bytes(b"\xff\xfe\x00\x80not utf-8\x81")
        assert status._read_json_file(p) is None

    def test_read_json_file_still_parses_valid_json(self, tmp_path):
        p = tmp_path / "runtime.json"
        p.write_text(json.dumps({"pid": 7}), encoding="utf-8")
        assert status._read_json_file(p) == {"pid": 7}

    def test_read_pid_record_returns_none_on_binary_garbage(self, tmp_path):
        p = tmp_path / "gateway.pid"
        p.write_bytes(b"\xff\xfe\x00\x80\x81")
        assert status._read_pid_record(p) is None

    def test_read_pid_record_still_parses_bare_pid(self, tmp_path):
        p = tmp_path / "gateway.pid"
        p.write_text("4242", encoding="utf-8")
        assert status._read_pid_record(p) == {"pid": 4242}


class TestParseActiveAgents:
    """The shared read-side coercion used by BOTH HTTP surfaces (/api/status
    and /health/detailed) so the exposed active_agents field is consistent and
    never negative regardless of what the status file holds."""

    def test_valid_int_passthrough(self):
        assert status.parse_active_agents(3) == 3

    def test_zero(self):
        assert status.parse_active_agents(0) == 0

    def test_numeric_string_coerced(self):
        assert status.parse_active_agents("5") == 5

    def test_negative_clamped_to_zero(self):
        assert status.parse_active_agents(-3) == 0

    def test_none_degrades_to_zero(self):
        assert status.parse_active_agents(None) == 0

    def test_garbage_string_degrades_to_zero(self):
        assert status.parse_active_agents("garbage") == 0

    def test_float_truncates(self):
        # int() truncation, then clamp — never raises.
        assert status.parse_active_agents(2.9) == 2


class TestActiveAgentsTurnBoundaryWrite:
    """The load-bearing Phase 1a contract: writing the in-flight count at a
    turn boundary must PRESERVE the lifecycle gateway_state. The whole readout
    depends on active_agents being refreshed per-turn while gateway_state is
    only touched by lifecycle transitions — so an active_agents-only write must
    not clobber it."""

    def test_active_agents_only_write_preserves_gateway_state(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))

        # Lifecycle transition sets running.
        status.write_runtime_status(gateway_state="running", active_agents=0)
        assert status.read_runtime_status()["gateway_state"] == "running"

        # Turn-boundary write: ONLY active_agents (gateway_state left _UNSET).
        status.write_runtime_status(active_agents=2)

        rec = status.read_runtime_status()
        assert rec["active_agents"] == 2
        # The state must survive the per-turn write — this is what makes the
        # _persist_active_agents helper safe to call on every turn.
        assert rec["gateway_state"] == "running"

    def test_active_agents_only_write_preserves_draining_state(self, tmp_path, monkeypatch):
        """Same invariant while draining — a turn finishing mid-drain (count
        falling) must not flip the state back to running."""
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))

        status.write_runtime_status(gateway_state="draining", active_agents=3)
        status.write_runtime_status(active_agents=2)

        rec = status.read_runtime_status()
        assert rec["active_agents"] == 2
        assert rec["gateway_state"] == "draining"

    def test_active_agents_clamped_non_negative(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        status.write_runtime_status(gateway_state="running", active_agents=-5)
        assert status.read_runtime_status()["active_agents"] == 0
class TestGatewayBusyDerivation:
    """Pure contract for derive_gateway_busy / derive_gateway_drainable — the
    single shared definition both /api/status and /health/detailed consume."""

    def test_busy_requires_running_state_and_positive_count(self):
        assert status.derive_gateway_busy(
            gateway_running=True, gateway_state="running", active_agents=1
        ) is True
        assert status.derive_gateway_busy(
            gateway_running=True, gateway_state="running", active_agents=0
        ) is False

    def test_busy_false_when_not_live_even_if_file_says_active(self):
        # Liveness wins: gateway_running False ⇒ never busy, regardless of count.
        assert status.derive_gateway_busy(
            gateway_running=False, gateway_state="running", active_agents=9
        ) is False

    def test_busy_false_for_non_running_states(self):
        for state in ("draining", "stopping", "stopped", "startup_failed", None):
            assert status.derive_gateway_busy(
                gateway_running=True, gateway_state=state, active_agents=5
            ) is False, state

    def test_busy_degrades_on_unparseable_count(self):
        for bad in (None, "garbage", object()):
            assert status.derive_gateway_busy(
                gateway_running=True, gateway_state="running", active_agents=bad
            ) is False

    def test_drainable_is_running_and_live_independent_of_count(self):
        # Idle running gateway is drainable but NOT busy.
        assert status.derive_gateway_drainable(
            gateway_running=True, gateway_state="running"
        ) is True
        assert status.derive_gateway_busy(
            gateway_running=True, gateway_state="running", active_agents=0
        ) is False

    def test_drainable_false_when_down_or_not_running(self):
        assert status.derive_gateway_drainable(
            gateway_running=False, gateway_state="running"
        ) is False
        for state in ("draining", "stopped", None):
            assert status.derive_gateway_drainable(
                gateway_running=True, gateway_state=state
            ) is False, state
