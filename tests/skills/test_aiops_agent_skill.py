"""Tests for optional-skills/devops/aiops-agent/scripts/log_triage.py"""

import json
import sys
from pathlib import Path
from unittest import mock

import pytest

SCRIPTS_DIR = (
    Path(__file__).resolve().parents[2]
    / "optional-skills"
    / "devops"
    / "aiops-agent"
    / "scripts"
)
sys.path.insert(0, str(SCRIPTS_DIR))

import log_triage as lt


class TestClassifyLevel:
    def test_error_keyword(self):
        assert lt.classify_level("2026-07-08T10:00:00Z ERROR request failed") == "ERROR"

    def test_critical_keyword_outranks_error(self):
        # A line mentioning both should be classified by the level-keyword search order.
        assert lt.classify_level("CRITICAL: out of memory") == "CRITICAL"

    def test_warn_keyword(self):
        assert lt.classify_level("WARNING disk usage high") == "WARN"

    def test_info_keyword(self):
        assert lt.classify_level("INFO service started") == "INFO"

    def test_no_keyword_is_other(self):
        assert lt.classify_level("plain text with no level marker") == "OTHER"

    def test_err_does_not_match_inside_error(self):
        # "ERR" must not falsely match as a substring of "ERROR" via bad word boundaries.
        assert lt.classify_level("a line about ERRORS in general") == "ERROR"

    def test_plural_warnings_matches_warn(self):
        assert lt.classify_level("multiple warnings detected during startup") == "WARN"

    def test_plural_exceptions_matches_error(self):
        assert lt.classify_level("3 unhandled exceptions found") == "ERROR"


class TestFingerprint:
    def test_numbers_normalized(self):
        a = lt.fingerprint("request 123 failed for user 45")
        b = lt.fingerprint("request 999 failed for user 2")
        assert a == b

    def test_ip_normalized(self):
        a = lt.fingerprint("connection from 10.0.0.5 refused")
        b = lt.fingerprint("connection from 192.168.1.20 refused")
        assert a == b

    def test_uuid_normalized(self):
        a = lt.fingerprint("job 550e8400-e29b-41d4-a716-446655440000 failed")
        b = lt.fingerprint("job 123e4567-e89b-12d3-a456-426614174000 failed")
        assert a == b

    def test_timestamp_normalized(self):
        a = lt.fingerprint("2026-07-08T10:00:00Z ERROR boom")
        b = lt.fingerprint("2026-07-08T11:30:05Z ERROR boom")
        assert a == b

    def test_distinct_messages_stay_distinct(self):
        a = lt.fingerprint("ERROR disk full")
        b = lt.fingerprint("ERROR connection refused")
        assert a != b


class TestAnalyze:
    LOG = "\n".join(
        [
            "2026-07-08T10:00:00Z INFO service started",
            "2026-07-08T10:15:00Z WARNING disk usage at 80% on /data",
            "2026-07-08T10:30:00Z ERROR request 123 failed for user 45 at 10.0.0.5",
            "2026-07-08T10:30:10Z ERROR request 124 failed for user 46 at 10.0.0.6",
        ]
    )

    def test_total_lines(self):
        assert lt.analyze(self.LOG)["total_lines"] == 4

    def test_counts(self):
        counts = lt.analyze(self.LOG)["counts"]
        assert counts["INFO"] == 1
        assert counts["WARN"] == 1
        assert counts["ERROR"] == 2

    def test_top_signature_groups_similar_errors(self):
        sigs = lt.analyze(self.LOG)["top_signatures"]
        assert len(sigs) == 1
        assert sigs[0]["count"] == 2

    def test_blank_lines_ignored(self):
        assert lt.analyze("\n\n  \n").get("total_lines") == 0

    def test_signature_level_marked_critical(self):
        log = "2026-07-08T10:00:00Z CRITICAL out of memory, restarting"
        sigs = lt.analyze(log)["top_signatures"]
        assert sigs[0]["level"] == "CRITICAL"


class TestSpikeDetection:
    def test_burst_of_recent_errors_detected(self):
        result = lt.analyze(lt._SELFTEST_LOG)
        assert result["spike"]["detected"] is True

    def test_no_spike_with_too_few_errors(self):
        log = "2026-07-08T10:00:00Z ERROR only one error here"
        assert lt.analyze(log)["spike"]["detected"] is False

    def test_steady_low_rate_no_spike(self):
        lines = [
            f"2026-07-08T10:{minute:02d}:00Z ERROR steady failure {minute}"
            for minute in range(0, 60, 10)
        ]
        result = lt.analyze("\n".join(lines), window_minutes=5)
        assert result["spike"]["detected"] is False

    def test_no_timestamps_no_crash(self):
        log = "ERROR something broke\nERROR something broke again"
        result = lt.analyze(log)
        assert result["spike"]["detected"] is False


class TestSelftest:
    def test_selftest_exits_zero(self):
        assert lt.main(["--selftest"]) == 0

    def test_selftest_prints_ok(self, capsys):
        lt.main(["--selftest"])
        out = json.loads(capsys.readouterr().out)
        assert out == {"selftest": "ok"}


class TestCli:
    def test_stdin_with_errors_exits_one(self, capsys):
        with mock.patch.object(sys.stdin, "read", return_value=TestAnalyze.LOG):
            rc = lt.main([])
        out = json.loads(capsys.readouterr().out)
        assert rc == 1
        assert out["counts"]["ERROR"] == 2

    def test_stdin_clean_log_exits_zero(self, capsys):
        clean = "2026-07-08T10:00:00Z INFO all good\n2026-07-08T10:01:00Z INFO still good"
        with mock.patch.object(sys.stdin, "read", return_value=clean):
            rc = lt.main([])
        out = json.loads(capsys.readouterr().out)
        assert rc == 0
        assert out["counts"].get("ERROR", 0) == 0

    def test_file_argument(self, tmp_path, capsys):
        log_file = tmp_path / "app.log"
        log_file.write_text(TestAnalyze.LOG, encoding="utf-8")
        rc = lt.main(["--file", str(log_file)])
        out = json.loads(capsys.readouterr().out)
        assert rc == 1
        assert out["total_lines"] == 4

    def test_top_flag_limits_signatures(self, capsys):
        # Distinct non-numeric words keep each line a separate signature after
        # fingerprint() normalizes away the timestamp (numbers are normalized too).
        words = ["alpha", "beta", "gamma", "delta", "epsilon"]
        lines = [f"2026-07-08T10:00:00Z ERROR distinct failure {word}" for word in words]
        with mock.patch.object(sys.stdin, "read", return_value="\n".join(lines)):
            lt.main(["--top", "2"])
        out = json.loads(capsys.readouterr().out)
        assert len(out["top_signatures"]) <= 2


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
