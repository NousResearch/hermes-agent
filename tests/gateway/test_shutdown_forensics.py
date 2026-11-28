"""Tests for gateway.shutdown_forensics — fast snapshot + async diag spawn."""

from __future__ import annotations

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

    @pytest.mark.parametrize("env, expected", [
        ({}, "under_systemd=no supervisor=none"),  # a macOS launchd orphan: ppid 1, nothing revives it
        ({"INVOCATION_ID": "abc"}, "under_systemd=yes supervisor=systemd"),
        ({"HERMES_LAUNCHD_LABEL": "ai.hermes.gateway"}, "under_systemd=no supervisor=launchd"),
    ])
    def test_supervisor_comes_from_its_stamp_not_from_ppid_1(self, monkeypatch, env, expected):
        for key in ("INVOCATION_ID", "XPC_SERVICE_NAME", "HERMES_LAUNCHD_LABEL"):
            monkeypatch.delenv(key, raising=False)
        for key, value in env.items():
            monkeypatch.setenv(key, value)
        monkeypatch.setattr(sf.os, "getppid", lambda: 1)
        line = sf.format_context_for_log(sf.snapshot_shutdown_context(signal.SIGTERM))
        assert expected in line

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

# ---------------------------------------------------------------------------
# check_systemd_timing_alignment — systemd ignores the manager that does not own the unit
# ---------------------------------------------------------------------------

class TestParseCgroupUnitScope:
    """Regression for #85117: the cgroup slice names the manager that owns the unit."""

    def test_system_slice_is_system_scope(self):
        assert sf._parse_cgroup_unit_scope("0::/system.slice/hermes-gateway.service\n") == (
            "hermes-gateway.service", "system")

    def test_user_slice_is_user_scope(self):
        assert sf._parse_cgroup_unit_scope(
            "0::/user.slice/user-1000.slice/user@1000.service/hermes-gateway.service\n"
        ) == ("hermes-gateway.service", "user")

    def test_cgroup_v1_lines_keep_the_scope(self):
        assert sf._parse_cgroup_unit_scope(
            "12:pids:/system.slice/hermes-gateway.service\n1:name=systemd:/system.slice/hermes-gateway.service\n"
        ) == ("hermes-gateway.service", "system")

    def test_unknown_slice_leaves_scope_unset(self):
        assert sf._parse_cgroup_unit_scope("0::/hermes-gateway.service\n") == (
            "hermes-gateway.service", None)

    def test_no_service_unit(self):
        assert sf._parse_cgroup_unit_scope("0::/init.scope\n") == (None, None)


class _FakeSystemctl:
    """Answers ``systemctl [--user] show <unit> --property=LoadState --property=TimeoutStopUSec``.

    Like the real thing: a manager that has not loaded the unit still exits 0 and reports its
    own default, which is the whole failure mode in #85117.
    """

    def __init__(self, answers, failing=()):
        self.answers = answers
        self.failing = set(failing)
        self.calls = []

    def __call__(self, argv, **kwargs):
        scope = "user" if "--user" in argv else "system"
        unit = argv[argv.index("show") + 1]
        self.calls.append((scope, tuple(argv)))
        if (scope, unit) in self.failing:
            return subprocess.CompletedProcess(argv, 1, "", "Failed to connect to bus")
        load_state, value = self.answers.get((scope, unit), ("not-found", "1min 30s"))
        stdout = f"LoadState={load_state}\nTimeoutStopUSec={value}\n"
        return subprocess.CompletedProcess(argv, 0, stdout, "")


def _probe(monkeypatch, scope, answers, failing=()):
    monkeypatch.setenv("INVOCATION_ID", "test-invocation")
    monkeypatch.setattr(sf, "_systemd_unit_and_scope", lambda: ("hermes-gateway.service", scope))
    fake = _FakeSystemctl(answers, failing)
    monkeypatch.setattr(sf.subprocess, "run", fake)
    return fake


class TestSystemdTimingAlignmentScope:
    """Regression for #85117: a system unit's value comes from the system manager."""

    def test_system_unit_reads_the_system_manager_not_the_user_default(self, monkeypatch):
        fake = _probe(monkeypatch, "system", {
            ("user", "hermes-gateway.service"): ("not-found", "1min 30s"),
            ("system", "hermes-gateway.service"): ("loaded", "3min 30s"),
        })
        result = sf.check_systemd_timing_alignment(180.0, 30.0)
        assert result["timeout_stop_sec"] == 210.0
        assert result["mismatch"] is False
        assert result["scope"] == "system"
        assert fake.calls[0][0] == "system", "the owning manager is asked first"

    def test_attribution_is_requested_from_systemctl(self, monkeypatch):
        fake = _probe(monkeypatch, "system", {})
        sf.check_systemd_timing_alignment(180.0, 30.0)
        assert "--property=LoadState" in fake.calls[0][1]

    def test_no_manager_owns_the_unit_means_no_verdict(self, monkeypatch):
        _probe(monkeypatch, "system", {})  # both managers answer not-found with a default
        assert sf.check_systemd_timing_alignment(180.0, 30.0) is None

    def test_user_owned_unit_still_measured_and_a_real_mismatch_still_fires(self, monkeypatch):
        # Positive control: a user unit keeps being read from the user manager, and 60s < 210s
        # must still warn — the guard must not turn the check into a no-op.
        fake = _probe(monkeypatch, "user", {
            ("user", "hermes-gateway.service"): ("loaded", "1min"),
            ("system", "hermes-gateway.service"): ("loaded", "3min 30s"),
        })
        result = sf.check_systemd_timing_alignment(180.0, 30.0)
        assert result["scope"] == "user"
        assert result["timeout_stop_sec"] == 60.0
        assert result["mismatch"] is True
        assert fake.calls[0][0] == "user", "a user unit is read from the user manager first"

    def test_user_manager_unreachable_falls_through_to_the_system_manager(self, monkeypatch):
        _probe(monkeypatch, "user", {
            ("system", "hermes-gateway.service"): ("loaded", "3min 30s"),
        }, failing=[("user", "hermes-gateway.service")])
        result = sf.check_systemd_timing_alignment(180.0, 30.0)
        assert result["timeout_stop_sec"] == 210.0
        assert result["mismatch"] is False
