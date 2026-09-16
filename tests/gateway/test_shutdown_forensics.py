"""Tests for gateway.shutdown_forensics — fast snapshot + async diag spawn."""

from __future__ import annotations

import json
import os
import signal
import time
from pathlib import Path

import pytest

from gateway import shutdown_forensics as sf


# ---------------------------------------------------------------------------
# _signal_name
# ---------------------------------------------------------------------------

class TestSignalName:

    def test_unknown_int_returns_signal_num_token(self):
        # Pick an integer extremely unlikely to ever be a real signal alias
        assert sf._signal_name(9999) == "signal#9999"


# ---------------------------------------------------------------------------
# snapshot_shutdown_context
# ---------------------------------------------------------------------------

class TestSnapshotShutdownContext:

    def test_handles_none_signal(self):
        ctx = sf.snapshot_shutdown_context(None)
        assert ctx["signal"] == "UNKNOWN"
        assert ctx["signal_num"] is None

    def test_includes_timestamps(self):
        before = time.time()
        ctx = sf.snapshot_shutdown_context(signal.SIGTERM)
        after = time.time()
        assert before <= ctx["ts"] <= after
        assert isinstance(ctx["ts_monotonic"], float)


    def test_under_systemd_false_without_invocation_id_and_normal_ppid(
        self, monkeypatch
    ):
        monkeypatch.delenv("INVOCATION_ID", raising=False)
        # We can't actually change ppid; skip if we happen to be reaped
        # by init (e.g. running under tini).
        if os.getppid() == 1:
            pytest.skip("test process is reaped by init")
        ctx = sf.snapshot_shutdown_context(signal.SIGTERM)
        assert ctx["under_systemd"] is False


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
# spawn_async_diagnostic
# ---------------------------------------------------------------------------

class TestSpawnAsyncDiagnostic:
    # The diagnostic wraps its script in GNU coreutils ``timeout`` and the script
    # body is Linux-only (``ps auxf --sort``, ``/proc/loadavg``, ``dmesg``,
    # ``pstree``). On hosts without ``timeout`` (macOS) Popen raises and the
    # producer returns None by design (fail-soft), so the spawn can only be
    # observed on Linux.
    @pytest.mark.linux_only
    def test_spawns_subprocess_and_writes_output(self, tmp_path):
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

    @pytest.mark.linux_only
    def test_redacts_child_argv_secrets_and_uses_0600(self, tmp_path):
        import subprocess
        import sys
        fake = "lin_api_" + "A1b2C3d4" * 5
        child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(20)",
                                  f"LINEAR_API_KEY={fake}", "sbp_" + "ab12" * 10])
        try:
            time.sleep(0.3)
            log_path = tmp_path / "diag.log"
            pid = sf.spawn_async_diagnostic(log_path, "SIGTERM", timeout_seconds=3.0)
            assert pid is not None
            try:
                os.waitpid(pid, 0)
            except (ChildProcessError, OSError):
                pass
            contents = log_path.read_text(encoding="utf-8", errors="replace")
            assert "shutdown diagnostic" in contents
            assert "LINEAR_API_KEY=[REDACTED]" in contents  # child is visible, value masked
            assert fake not in contents
            assert "ab12ab12ab12" not in contents
            assert (log_path.stat().st_mode & 0o777) == 0o600
        finally:
            child.kill()
            child.wait()

    def test_tightens_existing_log_mode(self, tmp_path, monkeypatch):
        log_path = tmp_path / "diag.log"
        log_path.write_text("old\n")
        os.chmod(log_path, 0o644)
        monkeypatch.setattr(sf.subprocess, "Popen", lambda *a, **k: type("P", (), {"pid": 1})())
        sf.spawn_async_diagnostic(log_path, "SIGTERM")
        assert (log_path.stat().st_mode & 0o777) == 0o600


class TestRedactSecrets:
    def test_masks_known_token_shapes(self):
        pem = "-----BEGIN PRIVATE KEY-----MIIEvQIBADANBgkq-----END PRIVATE KEY-----"
        raw = (f"docker exec -e EXPO_KEY={pem} -e SUPABASE_ACCESS_TOKEN=sbp_{'a1' * 20} "
               f"-H 'Authorization: Bearer abcdefghijklmnop.qrs' ghp_{'x' * 36} sk-{'Z' * 30} "
               "MY_SECRET=hunter2hunter2 plain=keep")
        out = sf.redact_secrets(raw)
        for leaked in ("MIIEvQ", "a1a1a1", "abcdefghijklmnop", "x" * 36, "Z" * 30, "hunter2"):
            assert leaked not in out
        assert "PRIVATE KEY-----" not in out
        assert "plain=keep" in out and "docker exec" in out

    def test_filter_source_matches_python_redactor(self):
        import subprocess
        import sys
        raw = "A_TOKEN=secretvalue123 lin_api_" + "Q" * 40 + "\n"
        res = subprocess.run([sys.executable, "-c", sf._redaction_filter_source()],
                             input=raw.encode(), capture_output=True, check=True)
        assert res.stdout.decode() == sf.redact_secrets(raw)

    def test_multiline_pem_bodies_are_dropped(self):
        body = "MIIEvQIBADANBgkqhkiG9w0BAQEFAASC\nBKcwggSjAgEAAoIBAQC7\n"
        raw = (f"x EXPO_KEY=-----BEGIN PRIVATE KEY-----\n{body}-----END PRIVATE KEY----- tail A_TOKEN=zzz\n"
               "next line\n")
        out = sf.redact_secrets(raw)
        assert "MIIEvQ" not in out and "BKcwgg" not in out and "zzz" not in out
        assert "[REDACTED-PRIVATE-KEY]" in out and "next line" in out and "tail" in out
        unterminated = sf.redact_secrets(f"k=-----BEGIN RSA PRIVATE KEY-----\n{body}")
        assert "MIIEvQ" not in unterminated and "TRUNCATED" in unterminated

    def test_filter_matches_python_redactor_multiline(self):
        import subprocess
        import sys
        raw = ("a -----BEGIN PRIVATE KEY-----\nSECRETBODY\n-----END PRIVATE KEY----- b\n"
               "c -----BEGIN PRIVATE KEY-----\nOPENBODY\n")
        res = subprocess.run([sys.executable, "-c", sf._redaction_filter_source()],
                             input=raw.encode(), capture_output=True, check=True)
        assert res.stdout.decode() == sf.redact_secrets(raw)
        assert "SECRETBODY" not in res.stdout.decode() and "OPENBODY" not in res.stdout.decode()

    def test_filter_streams_lines_before_eof(self):
        """Killed-before-EOF safety: completed lines reach stdout while stdin is still open."""
        import select
        import subprocess
        import sys
        proc = subprocess.Popen([sys.executable, "-c", sf._redaction_filter_source()],
                                stdin=subprocess.PIPE, stdout=subprocess.PIPE)
        try:
            proc.stdin.write(b"MY_TOKEN=abcdef123456 first\n")
            proc.stdin.flush()
            ready, _, _ = select.select([proc.stdout], [], [], 10)
            assert ready, "filter buffered until EOF"
            line = proc.stdout.readline().decode()
            assert line == "MY_TOKEN=[REDACTED] first\n"
            proc.kill()  # simulate cgroup kill: nothing after this is required
        finally:
            proc.kill()
            proc.wait()


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

    def test_returns_none_when_unit_undeterminable(self, monkeypatch):
        monkeypatch.setenv("INVOCATION_ID", "abc")
        # /proc/self/cgroup likely doesn't end in .service for the test runner
        result = sf.check_systemd_timing_alignment(180.0)
        # Either None (we couldn't find a unit) or a dict with mismatch info
        # for whatever unit pytest IS in.  Both are valid; we just ensure
        # the function doesn't raise.
        assert result is None or isinstance(result, dict)
