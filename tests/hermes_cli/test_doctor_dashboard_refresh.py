"""Dashboard-auth refresh health check for `hermes doctor` (Refs #98338, Defect 4).

The "Nous Portal auth (logged in)" row reports CLI-credential *presence* — a
3 req/s rejection storm once ran invisible beneath that green tick. The new
check reads the dashboard-auth audit log passively (never triggering a
refresh) and reports the recent REFRESH_FAILURE rate: aggregates only, so no
token material can leak into doctor output.

Run: scripts/run_tests.sh tests/hermes_cli/test_doctor_dashboard_refresh.py
"""

from __future__ import annotations

import datetime as dt
import json

import pytest

import hermes_cli.doctor as doctor_mod
from hermes_cli.dashboard_auth import audit as audit_mod


def _line(event, *, reason="all_providers_rejected_rt", age_s=60):
    ts = (dt.datetime.now(dt.timezone.utc) - dt.timedelta(seconds=age_s)).isoformat()
    return json.dumps({"ts": ts, "event": event, "reason": reason}) + "\n"


def _now():
    return dt.datetime.now(dt.timezone.utc)


class TestSummarizeRefreshFailures:
    def test_counts_failures_by_reason_in_window(self):
        lines = [
            _line("refresh_failure", reason="all_providers_rejected_rt"),
            _line("refresh_failure", reason="all_providers_rejected_rt"),
            _line("refresh_failure", reason="rate_limited"),
            _line("refresh_success"),
        ]
        total, by_reason, latest = doctor_mod._summarize_refresh_failures(
            lines, window_s=3600.0, now=_now()
        )
        assert total == 3
        assert by_reason == {"all_providers_rejected_rt": 2, "rate_limited": 1}
        assert latest is not None

    def test_old_failures_outside_window_ignored(self):
        lines = [
            _line("refresh_failure", age_s=7200),
            _line("refresh_failure", age_s=60),
        ]
        total, _, _ = doctor_mod._summarize_refresh_failures(
            lines, window_s=3600.0, now=_now()
        )
        assert total == 1

    def test_malformed_lines_skipped(self):
        lines = [
            "not json\n",
            json.dumps({"event": "refresh_failure"}) + "\n",
            json.dumps({"ts": "garbage", "event": "refresh_failure"}) + "\n",
            json.dumps(["refresh_failure"]) + "\n",
        ]
        total, by_reason, latest = doctor_mod._summarize_refresh_failures(
            lines, window_s=3600.0, now=_now()
        )
        assert (total, by_reason, latest) == (0, {}, None)

    def test_empty_log_is_clean(self):
        assert doctor_mod._summarize_refresh_failures(
            [], window_s=3600.0, now=_now()
        ) == (0, {}, None)


class TestResolveLogPath:
    def test_points_at_dashboard_auth_log(self):
        assert audit_mod.resolve_log_path().name == "dashboard-auth.log"
        assert audit_mod.resolve_log_path().parent.name == "logs"


def _write_log(path, lines):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(lines), encoding="utf-8")


class TestDashboardAuthRefreshCheck:
    def test_registered_in_doctor_checks(self):
        assert doctor_mod._check_dashboard_auth_refresh in [
            c for _, c in doctor_mod.DOCTOR_CHECKS
        ]

    def test_storm_fails_with_manual_issue(self, tmp_path, monkeypatch, capsys):
        log = tmp_path / "logs" / "dashboard-auth.log"
        _write_log(log, [_line("refresh_failure") for _ in range(25)])
        monkeypatch.setattr(audit_mod, "resolve_log_path", lambda: log)
        finding = doctor_mod._check_dashboard_auth_refresh(False)
        assert len(finding.manual_issues) == 1
        assert "25" in finding.manual_issues[0]
        assert "✗" in capsys.readouterr().out

    def test_large_log_reads_bounded_tail_only(self, tmp_path, monkeypatch):
        # A storm-grown log must not force a full-file read: entries older than
        # the byte-tail window are ignored even though the file is huge.
        import json as _json

        log = tmp_path / "logs" / "dashboard-auth.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        filler = _json.dumps({"event": "old_filler"}) + "x" * 200 + "\n"
        log.write_text(filler * 5000, encoding="utf-8")  # ~1 MB of old noise
        with log.open("a", encoding="utf-8") as fh:
            fh.writelines([_line("refresh_failure") for _ in range(3)])
        monkeypatch.setattr(audit_mod, "resolve_log_path", lambda: log)
        finding = doctor_mod._check_dashboard_auth_refresh(False)
        assert finding.manual_issues == []
        assert finding.issues == []

    def test_few_failures_warn_without_issue(self, tmp_path, monkeypatch, capsys):
        log = tmp_path / "logs" / "dashboard-auth.log"
        _write_log(log, [_line("refresh_failure") for _ in range(3)])
        monkeypatch.setattr(audit_mod, "resolve_log_path", lambda: log)
        finding = doctor_mod._check_dashboard_auth_refresh(False)
        assert finding.issues == []
        assert "⚠" in capsys.readouterr().out

    def test_clean_log_passes(self, tmp_path, monkeypatch, capsys):
        log = tmp_path / "logs" / "dashboard-auth.log"
        _write_log(log, [_line("refresh_success") for _ in range(5)])
        monkeypatch.setattr(audit_mod, "resolve_log_path", lambda: log)
        finding = doctor_mod._check_dashboard_auth_refresh(False)
        assert finding.issues == []
        assert "✓" in capsys.readouterr().out


    def test_line_cap_bounds_the_tail_independently_of_the_byte_cap(self, tmp_path, monkeypatch):
        """The LINE cap is a second, independent bound: the whole file fits inside the
        512 KB byte window, so only ``[-_REFRESH_LOG_TAIL_LINES:]`` can exclude a storm sitting
        at the OLDEST lines. Removing that slice turns this RED."""
        import json as _json

        log = tmp_path / "logs" / "dashboard-auth.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        storm = "".join(_line("refresh_failure") for _ in range(25))
        filler = _json.dumps({"event": "old_filler"}) + "\n"
        # Failures first (oldest), then enough short filler lines to pass the 5000-line cap
        # while the whole file stays well inside the 512 KB byte window.
        log.write_text(storm + filler * 5100, encoding="utf-8")
        monkeypatch.setattr(audit_mod, "resolve_log_path", lambda: log)
        finding = doctor_mod._check_dashboard_auth_refresh(False)
        assert finding.manual_issues == [], (
            "line cap did not bound the tail: a storm older than the cap was still counted"
        )

    def test_storm_threshold_boundary_is_exactly_at_the_limit(self, tmp_path, monkeypatch, capsys):
        """``total >= 20`` — pin the boundary itself, since a strict ``>`` would also pass
        every other case in this file (the suite only uses 3 and 25 failures)."""
        log = tmp_path / "logs" / "dashboard-auth.log"
        _write_log(log, [_line("refresh_failure") for _ in range(doctor_mod._REFRESH_STORM_ISSUE_COUNT)])
        monkeypatch.setattr(audit_mod, "resolve_log_path", lambda: log)
        finding = doctor_mod._check_dashboard_auth_refresh(False)
        assert "✗" in capsys.readouterr().out, "exactly-at-the-limit must FAIL, not warn"
        assert len(finding.manual_issues) == 1

    def test_byte_cap_bounds_the_tail_independently_of_the_line_cap(self, tmp_path, monkeypatch):
        """The BYTE cap is the other independent bound, and nothing else pins it.

        25 failures at the OLDEST offsets, then 4900 filler lines of ~110 B: the file is
        ~541 KB (> the 512 KB byte window) but only 4925 lines (< the 5000-line cap). The storm
        is therefore inside the last 5000 lines — the LINE cap keeps it — yet before the start
        of the byte window, so only _REFRESH_LOG_TAIL_BYTES can exclude it."""
        import json as _json

        log = tmp_path / "logs" / "dashboard-auth.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        storm = "".join(_line("refresh_failure") for _ in range(25))
        filler = _json.dumps({"event": "old_filler"}) + "y" * 84 + "\n"
        assert len(storm) + 4900 * len(filler) > doctor_mod._REFRESH_LOG_TAIL_BYTES
        assert 25 + 4900 < doctor_mod._REFRESH_LOG_TAIL_LINES
        log.write_text(storm + filler * 4900, encoding="utf-8")
        monkeypatch.setattr(audit_mod, "resolve_log_path", lambda: log)
        finding = doctor_mod._check_dashboard_auth_refresh(False)
        assert finding.manual_issues == [], (
            "byte cap did not bound the tail: a storm older than the byte window was counted"
        )
