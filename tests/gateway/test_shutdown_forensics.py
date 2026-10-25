"""Tests for gateway.shutdown_forensics — fast snapshot + async diag spawn."""

from __future__ import annotations

import io
import json
import os
import signal
import subprocess
import sys
import time

import pytest

from gateway import shutdown_forensics as sf

# ---------------------------------------------------------------------------
# _signal_name
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# snapshot_shutdown_context
# ---------------------------------------------------------------------------

class TestSnapshotShutdownContext:

    def test_detects_takeover_marker_for_self(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        marker = tmp_path / ".gateway-takeover.json"
        marker.write_text(
            f'{{"target_pid": {os.getpid()}, "replacer_pid": 99999}}',
            encoding="utf-8",
        )
        ctx = sf.snapshot_shutdown_context(signal.SIGTERM)
        assert "takeover_marker" in ctx
        assert ctx["takeover_marker_for_self"] is True

# ---------------------------------------------------------------------------
# format_context_for_log / context_as_json
# ---------------------------------------------------------------------------

class TestFormatters:

    def test_context_as_json_handles_unserialisable_values(self):
        ctx = {"signal": "SIGTERM", "weird": object()}
        payload = sf.context_as_json(ctx)
        # default=str means objects get repr'd, JSON stays valid
        decoded = json.loads(payload)
        assert decoded["signal"] == "SIGTERM"
        assert "weird" in decoded

# ---------------------------------------------------------------------------
# persisted snapshots must never include process argv (#112459)
# ---------------------------------------------------------------------------

_ARGV_CANARY = "lin_api_CANARY_SHUTDOWN_FORENSICS_9f3a2c"

@pytest.fixture
def child_with_secret_argv():
    """A live child whose argv carries a token-shaped value, like ``docker exec -e KEY=...``."""
    proc = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(30)", f"--token={_ARGV_CANARY}"],
        stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        yield proc
    finally:
        proc.kill()
        proc.wait()

class TestArgvFreePersistence:

    @pytest.mark.platforms("linux")
    def test_snapshot_and_log_line_identify_process_without_argv(self, child_with_secret_argv):
        """/proc-backed summaries keep pid/name/ppid/state but never the command line, so neither
        the JSON snapshot nor the warning line can carry a credential from a parent's argv."""
        summary = sf._proc_summary(child_with_secret_argv.pid)
        assert summary["pid"] == child_with_secret_argv.pid
        assert summary["name"]  # identity survives
        assert "cmdline" not in summary

        ctx = sf.snapshot_shutdown_context(signal.SIGTERM)
        ctx["parent"] = summary
        line = sf.format_context_for_log(ctx)
        assert _ARGV_CANARY not in line and _ARGV_CANARY not in sf.context_as_json(ctx)
        assert f"parent_pid={child_with_secret_argv.pid}" in line

# ---------------------------------------------------------------------------
# spawn_async_diagnostic
# ---------------------------------------------------------------------------

class TestSpawnAsyncDiagnostic:
    @pytest.mark.platforms("linux")
    def test_spawns_subprocess_and_writes_output(self, tmp_path):
        self._assert_diagnostic_written(tmp_path)

    @pytest.mark.platforms("macos")
    def test_spawns_without_gnu_timeout_on_macos(self, tmp_path):
        """Stock macOS has no ``timeout`` binary and BSD ``ps``; the diagnostic still lands."""
        self._assert_diagnostic_written(tmp_path)

    @staticmethod
    def _assert_diagnostic_written(tmp_path):
        log_path = tmp_path / "diag.log"
        pid = sf.spawn_async_diagnostic(log_path, "SIGTERM", timeout_seconds=3.0)
        assert pid is not None and pid > 0

        # Wait briefly for the subprocess to write — bounded by its own timeout.
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            if log_path.exists() and log_path.stat().st_size > 0:
                # Wait a touch longer for the script to finish writing
                time.sleep(0.2)
                break
            time.sleep(0.1)

        # Reap the subprocess so it doesn't show up as a zombie.
        try:
            os.waitpid(pid, 0)
        except (ChildProcessError, OSError):
            pass

        assert log_path.exists()
        contents = log_path.read_text(encoding="utf-8", errors="replace")
        assert "shutdown diagnostic" in contents
        assert "SIGTERM" in contents
        lines = contents.splitlines()
        ps_section = lines[lines.index("--- ps (top 60 by cpu, comm only) ---") + 1:]
        assert ps_section and ps_section[0].split()[:2] == ["PID", "PPID"], \
            "ps column header must lead the listing, not sort as a 0.0-cpu row"

    @pytest.mark.platforms("linux")
    def test_diagnostic_log_omits_child_argv_and_is_owner_only(self, tmp_path, child_with_secret_argv):
        """The detached ps/pstree walk must not write any process's argv to disk, and the log
        (even one created 0644 by an earlier release) ends up owner-only."""
        log_path = tmp_path / "diag.log"
        log_path.write_text("prior\n", encoding="utf-8")
        os.chmod(log_path, 0o644)

        pid = sf.spawn_async_diagnostic(log_path, "SIGTERM", timeout_seconds=5.0)
        assert pid is not None
        deadline = time.monotonic() + 8.0
        while time.monotonic() < deadline:
            try:
                if os.waitpid(pid, os.WNOHANG)[0] == pid:
                    break
            except ChildProcessError:
                break
            time.sleep(0.1)

        contents = log_path.read_text(encoding="utf-8", errors="replace")
        assert "shutdown diagnostic" in contents
        assert _ARGV_CANARY not in contents
        assert (log_path.stat().st_mode & 0o777) == 0o600

# ---------------------------------------------------------------------------
# parse_systemd_duration_to_us
# ---------------------------------------------------------------------------

class TestParseSystemdDuration:
    def test_seconds(self):
        assert sf.parse_systemd_duration_to_us("90s") == 90 * 1_000_000

    def test_minutes(self):
        assert sf.parse_systemd_duration_to_us("3min") == 180 * 1_000_000

# ---------------------------------------------------------------------------
# check_systemd_timing_alignment
# ---------------------------------------------------------------------------

class TestCheckSystemdTimingAlignment:
    @pytest.mark.parametrize(
        "cgroup, user_timeout, system_timeout, expected_timeout, mismatch, user_manager",
        [
            ("0::/system.slice/hermes-gateway.service\n", "90s", "3min 30s", 210.0, False, False),
            ("0::/system.slice/hermes-gateway.service\n", "210s", "90s", 90.0, True, False),
            ("2:cpu:/\n1:name=systemd:/system.slice/hermes-gateway.service\n",
             "90s", "210000000", 210.0, False, False),
            ("0::/user.slice/user-1000.slice/user@1000.service/app.slice/hermes-gateway.service\n",
             "3min 30s", "90s", 210.0, False, True),
            ("0::/user.slice/user-1000.slice/user@1000.service/app.slice/hermes-gateway.service\n",
             "90s", "210s", 90.0, True, True),
            ("0::/user.slice/user-1000.slice/user@1000.service/system.slice/hermes-gateway.service\n",
             "210s", "90s", 210.0, False, True),
        ],
    )
    def test_timeout_comes_from_running_cgroup_manager(
        self, monkeypatch, cgroup, user_timeout, system_timeout,
        expected_timeout, mismatch, user_manager,
    ):
        """An inactive same-named unit in the other manager cannot mask the running unit."""
        monkeypatch.setenv("INVOCATION_ID", "fixture-invocation")
        monkeypatch.setattr(sf, "open", lambda *args, **kwargs: io.StringIO(cgroup), raising=False)
        calls = []

        def fake_run(cmd, **kwargs):
            calls.append(cmd)
            assert cmd == ["systemctl", *(["--user"] if "--user" in cmd else []),
                           "show", "hermes-gateway.service", "--property=TimeoutStopUSec"]
            timeout = user_timeout if "--user" in cmd else system_timeout
            return subprocess.CompletedProcess(cmd, 0, f"TimeoutStopUSec={timeout}\n")

        monkeypatch.setattr(sf.subprocess, "run", fake_run)
        result = sf.check_systemd_timing_alignment(180.0)

        assert result == {
            "unit": "hermes-gateway.service", "timeout_stop_sec": expected_timeout,
            "drain_timeout": 180.0, "cron_drain_timeout": 30.0,
            "expected_min": float(sf.resolve_systemd_timeout_stop_sec(180.0, 30.0)),
            "mismatch": mismatch,
        }
        assert len(calls) == 1
        assert ("--user" in calls[0]) is user_manager

    @pytest.mark.parametrize("user_manager", [False, True])
    @pytest.mark.parametrize("failure", ["exit", "oserror", "timeout", "unparseable"])
    def test_unavailable_preferred_manager_keeps_existing_fallback(
        self, monkeypatch, user_manager, failure,
    ):
        monkeypatch.setenv("INVOCATION_ID", "fixture-invocation")
        cgroup_path = ("/user.slice/user-1000.slice/user@1000.service/app.slice"
                       if user_manager else "/system.slice")
        monkeypatch.setattr(
            sf, "open", lambda *args, **kwargs: io.StringIO(
                f"0::{cgroup_path}/hermes-gateway.service\n"), raising=False,
        )
        calls = []

        def fake_run(cmd, **kwargs):
            calls.append(cmd)
            if ("--user" in cmd) is user_manager:
                if failure == "oserror":
                    raise OSError("fixture manager unavailable")
                if failure == "timeout":
                    raise subprocess.TimeoutExpired(cmd, kwargs["timeout"])
                return subprocess.CompletedProcess(
                    cmd, 1 if failure == "exit" else 0, "TimeoutStopUSec=unknown\n",
                )
            return subprocess.CompletedProcess(cmd, 0, "TimeoutStopUSec=3min 30s\n")

        monkeypatch.setattr(sf.subprocess, "run", fake_run)
        result = sf.check_systemd_timing_alignment(180.0)

        assert result is not None
        assert result["timeout_stop_sec"] == 210.0
        assert result["mismatch"] is False
        assert len(calls) == 2
        assert ("--user" in calls[0]) is user_manager
        assert ("--user" in calls[1]) is not user_manager
