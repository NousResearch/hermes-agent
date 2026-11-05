"""Tests for cron cadence fallback logging in cron/jobs.py."""

import logging

import cron.jobs as jobs_mod


class TestComputeGraceSeconds:
    def test_uncomputable_cron_expr_logs_once_and_falls_back(self, monkeypatch, caplog):
        """A cron expr whose cadence croniter can't compute keeps the minimum grace but is
        logged with its cause, once per expr rather than on every tick."""

        def raising_croniter(*args, **kwargs):
            raise RuntimeError("cadence boom")

        monkeypatch.setattr(jobs_mod, "HAS_CRONITER", True)
        monkeypatch.setattr(jobs_mod, "croniter", raising_croniter)
        monkeypatch.setattr(jobs_mod, "_cron_cadence_cache", {})
        schedule = {"kind": "cron", "expr": "0 9 * * *"}

        with caplog.at_level(logging.WARNING, logger="cron.jobs"):
            first = jobs_mod._compute_grace_seconds(schedule)
            second = jobs_mod._compute_grace_seconds(schedule)

        assert first == second == jobs_mod._MIN_GRACE_SECONDS
        warnings = [r for r in caplog.records if "0 9 * * *" in r.getMessage()]
        assert len(warnings) == 1
        assert warnings[0].levelno == logging.WARNING
        assert "cadence boom" in warnings[0].getMessage()
        assert warnings[0].exc_info is not None
