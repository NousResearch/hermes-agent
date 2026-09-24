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

    @pytest.mark.linux_only
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
    @pytest.mark.linux_only
    def test_spawns_subprocess_and_writes_output(self, tmp_path):
        self._assert_diagnostic_written(tmp_path)

    @pytest.mark.macos_only
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

    @pytest.mark.linux_only
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


class _Show:
    def __init__(self, stdout: str, returncode: int = 0) -> None:
        self.stdout = stdout
        self.returncode = returncode
        self.stderr = ""


def test_not_found_user_unit_does_not_hide_the_loaded_system_unit(monkeypatch):
    """A missing user unit still prints the compiled 90s default. That must not win."""
    seen = []

    def fake_run(cmd, **kwargs):
        seen.append(list(cmd))
        if "--user" in cmd:
            return _Show(
                "LoadState=not-found\nFragmentPath=\nTimeoutStopUSec=1min 30s\n"
            )
        return _Show(
            "LoadState=loaded\n"
            "FragmentPath=/etc/systemd/system/hermes-gateway.service\n"
            "TimeoutStopUSec=3min 30s\n"
        )

    monkeypatch.setattr(sf.subprocess, "run", fake_run)

    assert sf._systemd_timeout_stop_us("hermes-gateway.service") == 210 * 1_000_000
    assert any("--user" in cmd for cmd in seen)
    assert any("--user" not in cmd for cmd in seen)


def test_unloaded_unit_on_both_managers_is_undeterminable(monkeypatch):
    def fake_run(cmd, **kwargs):
        return _Show("LoadState=not-found\nFragmentPath=\nTimeoutStopUSec=1min 30s\n")

    monkeypatch.setattr(sf.subprocess, "run", fake_run)

    assert sf._systemd_timeout_stop_us("hermes-gateway.service") is None


def test_loaded_user_unit_still_wins(monkeypatch):
    def fake_run(cmd, **kwargs):
        if "--user" in cmd:
            return _Show(
                "LoadState=loaded\n"
                "FragmentPath=/home/me/.config/systemd/user/hermes-gateway.service\n"
                "TimeoutStopUSec=4min\n"
            )
        raise AssertionError("system manager must not be queried when the user unit is loaded")

    monkeypatch.setattr(sf.subprocess, "run", fake_run)

    assert sf._systemd_timeout_stop_us("hermes-gateway.service") == 240 * 1_000_000

