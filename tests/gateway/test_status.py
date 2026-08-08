"""Tests for gateway runtime status tracking."""

import json
import os
import sys
import time
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


    def test_runtime_lock_claims_and_releases_liveness(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))

        assert status.is_gateway_runtime_lock_active() is False
        assert status.acquire_gateway_runtime_lock() is True
        assert status.is_gateway_runtime_lock_active() is True

        status.release_gateway_runtime_lock()

        assert status.is_gateway_runtime_lock_active() is False


    def test_get_running_pid_cached_invalidates_when_pid_file_changes(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        status._clear_running_pid_cache()

        pid_path = tmp_path / "gateway.pid"

        def _write_record(pid: int, start_time: int) -> None:
            record = {
                "pid": pid,
                "kind": "hermes-gateway",
                "argv": ["python", "-m", "hermes_cli.main", "gateway"],
                "start_time": start_time,
            }
            pid_path.write_text(json.dumps(record))
            (tmp_path / "gateway.lock").write_text(json.dumps(record))

        _write_record(111, 123)

        calls = {"lock_active": 0}

        def _lock_active(lock_path=None):
            calls["lock_active"] += 1
            return True

        monkeypatch.setattr(status, "is_gateway_runtime_lock_active", _lock_active)
        monkeypatch.setattr(status, "_pid_exists", lambda pid: True)
        monkeypatch.setattr(status, "_get_process_start_time", lambda pid: 123 if pid == 111 else 456)
        monkeypatch.setattr(status, "_read_process_cmdline", lambda pid: None)

        assert status.get_running_pid_cached(ttl_seconds=60) == 111

        _write_record(2222, 456)

        assert status.get_running_pid_cached(ttl_seconds=60) == 2222
        assert calls["lock_active"] == 2


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

    def test_gateway_identity_files_use_process_home_not_context_override(
        self, tmp_path, monkeypatch
    ):
        """Regression: pid/lock/state files must use process-level HERMES_HOME.

        When a profile context override is active (e.g., during session dispatch
        for a named profile), gateway identity files should still be written to
        the process-level HERMES_HOME, not the profile's directory.  See #56986.
        """
        from hermes_constants import set_hermes_home_override, reset_hermes_home_override

        process_home = tmp_path / "default"
        process_home.mkdir()
        profile_home = tmp_path / "profiles" / "cfo"
        profile_home.mkdir(parents=True)
        monkeypatch.setenv("HERMES_HOME", str(process_home))

        # Simulate a profile context override being active during write.
        token = set_hermes_home_override(str(profile_home))
        try:
            status.write_pid_file()
        finally:
            reset_hermes_home_override(token)

        # PID file must land in the process-level home, not the profile home.
        assert (process_home / "gateway.pid").exists()
        assert not (profile_home / "gateway.pid").exists()

        payload = json.loads((process_home / "gateway.pid").read_text())
        assert payload["pid"] == os.getpid()

        # Cleanup for atexit hooks.
        monkeypatch.setenv("HERMES_HOME", str(process_home))
        (process_home / "gateway.pid").unlink(missing_ok=True)


class TestGatewayRuntimeStatus:

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


    def test_command_line_belongs_to_profile_normalizes_separators(self):
        """A Windows argv renders HERMES_HOME with backslashes while the
        profile's Path may carry forward slashes (and, on Windows, vice
        versa).  The separator difference must not defeat the match."""
        home = Path("c:/opt/data/profiles/coder")
        cmdline = r"hermes_home=c:\opt\data\profiles\coder hermes gateway run --replace"
        assert status._command_line_belongs_to_profile(cmdline, home) is True


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


class TestTerminatePid:
    def test_force_uses_taskkill_on_windows(self, monkeypatch):
        calls = []
        monkeypatch.setattr(status, "_IS_WINDOWS", True)

        def fake_run(cmd, capture_output=False, text=False, timeout=None, creationflags=0, **kwargs):
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

    def test_acquire_scoped_lock_self_reacquires_when_disk_start_time_null(
        self, tmp_path, monkeypatch
    ):
        """#81468: same PID with on-disk start_time=null must self-reacquire.

        After Discord 503 reconnect, the scoped lock still names this process
        but may carry ``start_time: null`` (psutil miss at first write). The
        freshly built record has a real fingerprint. Requiring equality made
        the gateway report its own PID as a foreign token squatter.
        """
        monkeypatch.setenv("HERMES_GATEWAY_LOCK_DIR", str(tmp_path / "locks"))
        lock_path = tmp_path / "locks" / "discord-bot-token-2bb80d537b1da3e3.lock"
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        lock_path.write_text(json.dumps({
            "pid": os.getpid(),
            "start_time": None,
            "kind": "hermes-gateway",
            "argv": ["hermes_cli/main.py", "--profile", "milena", "gateway", "run", "--replace"],
            "scope": "discord-bot-token",
        }))

        # Live process can resolve start_time; disk cannot — the mismatch
        # that previously failed the self-reacquire short-circuit.
        monkeypatch.setattr(status, "_get_process_start_time", lambda pid: 987654321)
        # If we wrongly fall through to staleness, a gateway-looking self PID
        # would be treated as a live foreign holder (return False).
        monkeypatch.setattr(status, "_pid_exists", lambda pid: True)
        monkeypatch.setattr(status, "_looks_like_gateway_process", lambda pid: True)

        acquired, existing = status.acquire_scoped_lock(
            "discord-bot-token", "secret", metadata={"platform": "discord"}
        )

        assert acquired is True
        assert existing["pid"] == os.getpid()
        payload = json.loads(lock_path.read_text())
        assert payload["pid"] == os.getpid()
        assert payload["start_time"] == 987654321
        assert payload["metadata"]["platform"] == "discord"

    def test_release_scoped_lock_allows_null_disk_start_time(self, tmp_path, monkeypatch):
        """#81468: release must not no-op when disk start_time is null."""
        monkeypatch.setenv("HERMES_GATEWAY_LOCK_DIR", str(tmp_path / "locks"))
        lock_path = tmp_path / "locks" / "discord-bot-token-2bb80d537b1da3e3.lock"
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        lock_path.write_text(json.dumps({
            "pid": os.getpid(),
            "start_time": None,
            "kind": "hermes-gateway",
        }))
        monkeypatch.setattr(status, "_get_process_start_time", lambda pid: 987654321)

        status.release_scoped_lock("discord-bot-token", "secret")

        assert not lock_path.exists()

    def test_acquire_scoped_lock_self_reacquires_when_start_times_differ(
        self, tmp_path, monkeypatch
    ):
        """Same PID with two known, differing start_time values must self-reacquire.

        The on-disk record may carry a stale integer (from a previous run or a
        psutil transient) while the live process now reports a different value.
        Since the PID is ours, start_time equality is not required.
        """
        monkeypatch.setenv("HERMES_GATEWAY_LOCK_DIR", str(tmp_path / "locks"))
        lock_path = tmp_path / "locks" / "discord-bot-token-2bb80d537b1da3e3.lock"
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        lock_path.write_text(json.dumps({
            "pid": os.getpid(),
            "start_time": 111,
            "kind": "hermes-gateway",
            "argv": ["hermes_cli/main.py", "gateway", "run", "--replace"],
            "scope": "discord-bot-token",
        }))

        monkeypatch.setattr(status, "_get_process_start_time", lambda pid: 987654321)

        acquired, existing = status.acquire_scoped_lock(
            "discord-bot-token", "secret", metadata={"platform": "discord"}
        )

        assert acquired is True
        assert existing["pid"] == os.getpid()
        payload = json.loads(lock_path.read_text())
        assert payload["pid"] == os.getpid()
        assert payload["start_time"] == 987654321

    def test_acquire_scoped_lock_race_second_acquirer_loses(self, tmp_path, monkeypatch):
        """Two racing starters both observe the same stale lock. The loser's
        os.replace() hits FileNotFoundError (winner already claimed the stale
        file) and the winner's FRESH lock must survive: the loser must fall
        through to O_EXCL and lose, not clobber it like the old unlink() did."""
        monkeypatch.setenv("HERMES_GATEWAY_LOCK_DIR", str(tmp_path / "locks"))
        lock_path = tmp_path / "locks" / "telegram-bot-token-2bb80d537b1da3e3.lock"
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        stale_record = {
            "pid": 99999,
            "start_time": 123,
            "kind": "hermes-gateway",
        }
        lock_path.write_text(json.dumps(stale_record))
        monkeypatch.setattr(status, "_pid_exists", lambda pid: False)

        winner_record = {
            "pid": 424242,
            "start_time": 456,
            "kind": "hermes-gateway",
            "scope": "telegram-bot-token",
        }
        real_replace = os.replace

        def racing_replace(src, dst, *args, **kwargs):
            if str(src) == str(lock_path):
                # Simulate the winner completing removal + O_EXCL create
                # between our staleness check and our removal attempt.
                lock_path.write_text(json.dumps(winner_record))
                raise FileNotFoundError(2, "No such file or directory", str(src))
            return real_replace(src, dst, *args, **kwargs)

        monkeypatch.setattr(status.os, "replace", racing_replace)

        acquired, existing = status.acquire_scoped_lock("telegram-bot-token", "secret", metadata={"platform": "telegram"})

        assert acquired is False
        assert existing is not None
        assert existing["pid"] == 424242
        # The winner's fresh lock must be untouched on disk.
        assert json.loads(lock_path.read_text())["pid"] == 424242


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


class TestScopedLockTakeover:
    """Cross-home takeover requires explicit, corroborated process identity."""

    @staticmethod
    def _owner_record(target_home: Path, *, pid: int = 4242, start_time: int = 123):
        target_home.mkdir(parents=True, exist_ok=True)
        record = {
            "pid": pid,
            "kind": "hermes-gateway",
            "argv": ["python", "-m", "hermes_cli.main", "gateway", "run"],
            "start_time": start_time,
            "hermes_home": str(target_home),
        }
        (target_home / "gateway.pid").write_text(json.dumps(record))
        return record

    def test_verified_distinct_home_handoff_marks_target_before_sigterm(
        self, tmp_path, monkeypatch
    ):
        replacer_home = tmp_path / "replacer"
        target_home = tmp_path / "target"
        replacer_home.mkdir()
        monkeypatch.setenv("HERMES_HOME", str(replacer_home))
        record = self._owner_record(target_home)

        alive = iter([True, True, False])
        monkeypatch.setattr(status, "_pid_exists", lambda _pid: next(alive))
        monkeypatch.setattr(status, "_get_process_start_time", lambda _pid: 123)
        monkeypatch.setattr(
            status,
            "_read_process_cmdline",
            lambda _pid: "python -m hermes_cli.main gateway run",
        )
        calls = []

        def terminate(pid, *, force=False):
            marker_path = target_home / ".gateway-takeover.json"
            assert marker_path.exists()
            payload = json.loads(marker_path.read_text())
            assert payload["target_hermes_home"] == str(target_home)
            assert payload["replacer_hermes_home"] == str(replacer_home)
            calls.append((pid, force))

        monkeypatch.setattr(status, "terminate_pid", terminate)

        owner_pid = status.take_over_scoped_lock_holder(
            record, graceful_attempts=1
        )

        assert owner_pid == 4242
        assert calls == [(4242, False)]
        assert not (target_home / ".gateway-takeover.json").exists()
        assert not (replacer_home / ".gateway-takeover.json").exists()

    def test_handoff_rejects_uncorroborated_target_home(self, tmp_path, monkeypatch):
        target_home = tmp_path / "target"
        record = self._owner_record(target_home)
        # The lock claims target_home, but that home's PID record names a
        # different process identity.
        bad_pid_record = dict(record, pid=9999)
        (target_home / "gateway.pid").write_text(json.dumps(bad_pid_record))

        monkeypatch.setattr(status, "_pid_exists", lambda _pid: True)
        monkeypatch.setattr(status, "_get_process_start_time", lambda _pid: 123)
        monkeypatch.setattr(
            status,
            "_read_process_cmdline",
            lambda _pid: "python -m hermes_cli.main gateway run",
        )
        calls = []
        monkeypatch.setattr(
            status, "terminate_pid", lambda *args, **kwargs: calls.append(args)
        )

        assert status.take_over_scoped_lock_holder(record) is None
        assert calls == []
        assert not (target_home / ".gateway-takeover.json").exists()


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
        monkeypatch.setattr(status, "_IS_WINDOWS", False)
        monkeypatch.setattr(
            status.subprocess, "run",
            lambda args, **kwargs: SimpleNamespace(returncode=0, stdout="/usr/libexec/bluetoothuserd\n"),
        )
        result = status._read_process_cmdline(873)
        assert result == "/usr/libexec/bluetoothuserd"


    def test_proc_cmdline_takes_priority_over_ps(self, monkeypatch):
        calls = []

        def fake_read_bytes(self):
            calls.append("proc")
            return b"python\x00hermes_cli/main.py\x00gateway\x00"

        monkeypatch.setattr(status.Path, "read_bytes", fake_read_bytes)
        result = status._read_process_cmdline(12345)
        assert "hermes_cli/main.py" in result
        assert calls == ["proc"]


class TestCorruptStatusFiles:
    """A status / pid file holding non-UTF-8 (binary) bytes must read as
    None, not crash the gateway status path with UnicodeDecodeError."""

    def test_read_json_file_returns_none_on_binary_garbage(self, tmp_path):
        p = tmp_path / "runtime.json"
        p.write_bytes(b"\xff\xfe\x00\x80not utf-8\x81")
        assert status._read_json_file(p) is None


class TestParseActiveAgents:
    """The shared read-side coercion used by BOTH HTTP surfaces (/api/status
    and /health/detailed) so the exposed active_agents field is consistent and
    never negative regardless of what the status file holds."""

    def test_valid_int_passthrough(self):
        assert status.parse_active_agents(3) == 3

    def test_zero(self):
        assert status.parse_active_agents(0) == 0


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


    def test_drainable_is_running_and_live_independent_of_count(self):
        # Idle running gateway is drainable but NOT busy.
        assert status.derive_gateway_drainable(
            gateway_running=True, gateway_state="running"
        ) is True
        assert status.derive_gateway_busy(
            gateway_running=True, gateway_state="running", active_agents=0
        ) is False


class TestRespawnStormBreaker:
    def test_no_storm_under_threshold(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        for _ in range(5):
            result = status.record_start_and_check_storm(
                max_starts=5, window_s=120.0
            )
            assert result is None


class TestLaunchdPlistRespawnGovernance:
    def test_plist_has_throttle_interval(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        from hermes_cli.gateway import generate_launchd_plist

        plist = generate_launchd_plist()
        assert "<key>ThrottleInterval</key>" in plist
        assert "<key>ExitTimeOut</key>" in plist
        assert "<key>KeepAlive</key>" in plist


class TestPermissionErrorOnLockFile:
    """Stale root-owned lock files from launchd Background sessions must not
    crash the gateway on restart (issue #42685)."""

    def test_permission_error_on_lock_file_returns_false_and_removes(self, tmp_path, monkeypatch):
        """When the lock file is not writable (root-owned), the function should
        remove the stale file and report the lock as inactive."""
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        lock_path = tmp_path / "gateway.lock"
        lock_path.write_text("stale", encoding="utf-8")

        real_open = open

        def deny_write(path, *args, **kwargs):
            if str(path) == str(lock_path):
                raise PermissionError(13, "Permission denied", str(path))
            return real_open(path, *args, **kwargs)

        monkeypatch.setattr("builtins.open", deny_write)

        result = status.is_gateway_runtime_lock_active(lock_path)
        assert result is False
        assert not lock_path.exists(), "stale root-owned lock file should be removed"

    def test_permission_error_unlink_failure_still_returns_false(self, tmp_path, monkeypatch):
        """Even if unlinking the stale lock file fails (e.g. directory not writable),
        the function should still return False to allow startup."""
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        lock_path = tmp_path / "gateway.lock"
        lock_path.write_text("stale", encoding="utf-8")

        real_open = open

        def deny_write(path, *args, **kwargs):
            if str(path) == str(lock_path):
                raise PermissionError(13, "Permission denied", str(path))
            return real_open(path, *args, **kwargs)

        real_unlink = Path.unlink

        def deny_unlink(self, *args, **kwargs):
            if str(self) == str(lock_path):
                raise OSError(13, "Permission denied", str(self))
            return real_unlink(self, *args, **kwargs)

        monkeypatch.setattr("builtins.open", deny_write)
        monkeypatch.setattr(Path, "unlink", deny_unlink)

        result = status.is_gateway_runtime_lock_active(lock_path)
        assert result is False

    def test_acquire_gateway_runtime_lock_recovers_from_permission_error(self, tmp_path, monkeypatch):
        """acquire_gateway_runtime_lock must survive a stale root-owned lock
        file: unlink it and retry with a fresh file instead of crashing."""
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        lock_path = status._get_gateway_lock_path()
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        lock_path.write_text("stale", encoding="utf-8")

        real_open = open

        def deny_write(path, *args, **kwargs):
            # Simulate a root-owned file: opening fails while the ORIGINAL
            # stale file is still on disk; after unlink, the fresh file the
            # retry creates opens fine.
            if (
                str(path) == str(lock_path)
                and lock_path.exists()
                and lock_path.read_text(encoding="utf-8") == "stale"
            ):
                raise PermissionError(13, "Permission denied", str(path))
            return real_open(path, *args, **kwargs)

        monkeypatch.setattr("builtins.open", deny_write)

        try:
            assert status.acquire_gateway_runtime_lock() is True
        finally:
            status.release_gateway_runtime_lock()


class TestNormalizeUpdatedAt:
    """Unit tests for the updated_at RFC3339|None normalization funnel."""

    def test_epoch_int_converts_to_utc_iso(self):
        from datetime import datetime, timezone

        result = status.normalize_updated_at(1750000000)
        assert isinstance(result, str)
        parsed = datetime.fromisoformat(result)
        assert parsed.tzinfo is not None
        assert parsed == datetime.fromtimestamp(1750000000, tz=timezone.utc)


    def test_iso_with_z_suffix_accepted(self):
        from datetime import datetime, timezone

        result = status.normalize_updated_at("2026-07-21T12:00:00Z")
        assert result is not None
        parsed = datetime.fromisoformat(result)
        assert parsed.tzinfo is not None
        assert parsed == datetime(2026, 7, 21, 12, 0, 0, tzinfo=timezone.utc)

    def test_naive_iso_coerced_to_utc(self):
        from datetime import datetime, timezone

        result = status.normalize_updated_at("2026-07-21T12:00:00")
        assert result is not None
        parsed = datetime.fromisoformat(result)
        assert parsed.tzinfo is not None
        assert parsed.utcoffset().total_seconds() == 0
        assert parsed == datetime(2026, 7, 21, 12, 0, 0, tzinfo=timezone.utc)

    def test_offset_aware_iso_round_trips_canonically(self):
        canonical = "2026-07-21T12:00:00+00:00"
        assert status.normalize_updated_at(canonical) == canonical


    def test_epoch_before_2000_rejected(self):
        assert status.normalize_updated_at(0) is None
        assert status.normalize_updated_at(946684799) is None  # 1999-12-31T23:59:59Z
        assert status.normalize_updated_at(-1750000000) is None


class TestRuntimeStatusUpdatedAtContract:
    def test_write_then_read_updated_at_parses_tz_aware(self, tmp_path, monkeypatch):
        """write_runtime_status persists an updated_at that fromisoformat
        parses as a timezone-aware datetime — the writer side of the
        string|null contract every emit surface relies on."""
        from datetime import datetime

        monkeypatch.setenv("HERMES_HOME", str(tmp_path))

        status.write_runtime_status(gateway_state="running")

        payload = status.read_runtime_status()
        updated_at = payload["updated_at"]
        assert isinstance(updated_at, str)
        parsed = datetime.fromisoformat(updated_at)
        assert parsed.tzinfo is not None
        # And it survives the normalization funnel unchanged (canonical form).
        assert status.normalize_updated_at(updated_at) == updated_at


class TestResolveGatewayLiveness:
    """The single liveness ladder both dashboard status surfaces share.

    Before this existed, /api/status and /api/messaging/platforms each
    open-coded their own ladder and disagreed on the same page load — the
    sidebar read "running" while the Channels page rendered "The gateway is
    not running." These pin the ladder's ordering and its fail-open signal.
    """

    def test_pid_rung_wins_and_skips_later_rungs(self):
        calls = {"health": 0, "runtime_pid": 0}

        def _health():
            calls["health"] += 1
            return True, {"pid": 999}

        def _runtime_pid(runtime, **kw):
            calls["runtime_pid"] += 1
            return 888

        result = status.resolve_gateway_liveness(
            runtime=None,
            health_probe=_health,
            pid_probe=lambda *a, **k: 4242,
            runtime_pid_probe=_runtime_pid,
        )

        assert result.running is True
        assert result.pid == 4242
        assert result.source == "pid"
        # The authoritative rung answered: no lower rung should have run.
        assert calls == {"health": 0, "runtime_pid": 0}


    def test_probe_exception_degrades_instead_of_raising(self):
        """A raising rung must fall through, never propagate.

        Status endpoints poll this constantly; an exotic /proc or a
        permissions error must not turn into a 500.
        """
        def _boom(*a, **k):
            raise RuntimeError("probe exploded")

        result = status.resolve_gateway_liveness(
            runtime=None,
            health_probe=None,
            pid_probe=_boom,
            runtime_pid_probe=lambda *a, **k: None,
        )

        assert result.running is False
        # probe_error distinguishes "down" from "couldn't tell" for callers
        # that must fail OPEN (the kanban dispatcher warning).
        assert result.probe_error is True


    def test_profile_dir_scopes_every_read_to_that_profile(self, tmp_path):
        """Gateway identity files live in the per-profile home.

        The status readers resolve process-level paths and deliberately
        ignore the HERMES_HOME contextvar override (#56986), so the profile
        directory must be threaded through explicitly or a scoped request
        silently reports a DIFFERENT profile's gateway (#71211).
        """
        profile_dir = tmp_path / "profiles" / "worker"
        profile_dir.mkdir(parents=True)
        seen = {}

        def _pid(pid_path=None, **kw):
            seen["pid_path"] = pid_path
            return None

        def _reader(path=None):
            seen["status_path"] = path
            return None

        def _runtime_pid(runtime, *, expected_home=None):
            seen["expected_home"] = expected_home
            return None

        status.resolve_gateway_liveness(
            profile_dir=profile_dir,
            health_probe=None,
            pid_probe=_pid,
            runtime_reader=_reader,
            runtime_pid_probe=_runtime_pid,
        )

        assert seen["pid_path"] == profile_dir / "gateway.pid"
        assert seen["status_path"] == profile_dir / "gateway_state.json"
        # expected_home is what stops a recycled PID belonging to another
        # profile's live gateway from being reported as this profile's.
        assert seen["expected_home"] == profile_dir

