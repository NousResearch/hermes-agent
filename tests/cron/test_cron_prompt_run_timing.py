"""Run provenance in the assembled cron prompt (#99879 follow-up).

The due scan already persists ``last_dispatch`` (``scheduled_at`` / ``dispatched_at`` /
``lateness_seconds`` / ``kind``) so the CLI can surface a late catch-up, but the model — which
decides what "today's close" or "tomorrow" means — never saw it. A fire that landed hours late
reasoned against a world it believed was the scheduled moment, and a collapsed backlog hid that
earlier occurrences were skipped outright. Injected ``context_from`` output had the same gap: the
heading said "most recent output" with no age, so an eight-hour-old snapshot read as current
state. These tests pin the two prompt-side frames that close it.
"""

import os
import time
from datetime import UTC, datetime, timedelta

import pytest

from cron.scheduler_prompt import (
    _build_job_prompt,
    _dispatch_lateness_block,
    _humanize_age,
)

JOB_ID = "a1b2c3d4"


@pytest.fixture
def cron_env(tmp_path, monkeypatch):
    """Isolated cron store so ``context_from`` reads this tmp output tree, not the real one."""
    hermes_home = tmp_path / ".hermes"
    (hermes_home / "cron" / "output").mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    import cron.jobs as jobs_mod
    monkeypatch.setattr(jobs_mod, "HERMES_DIR", hermes_home)
    monkeypatch.setattr(jobs_mod, "CRON_DIR", hermes_home / "cron")
    monkeypatch.setattr(jobs_mod, "JOBS_FILE", hermes_home / "cron" / "jobs.json")
    monkeypatch.setattr(jobs_mod, "OUTPUT_DIR", hermes_home / "cron" / "output")
    return hermes_home


def _job(**extra):
    return {"id": JOB_ID, "name": "late job", "prompt": "summarize the close",
            "schedule": {}, **extra}


def _stamp(lateness, kind="late"):
    due = datetime.now(UTC) - timedelta(seconds=lateness)
    return {"scheduled_at": due.isoformat(), "dispatched_at": datetime.now(UTC).isoformat(),
            "lateness_seconds": lateness, "kind": kind}


def _write_archive(output_dir, job_id, body="old but usable body", age_seconds=0.0):
    """One run archive whose mtime is ``age_seconds`` in the past."""
    target = output_dir / job_id
    target.mkdir(parents=True, exist_ok=True)
    archive = target / "2026-11-20_08-00-00.md"
    archive.write_text(
        "# Cron Job: upstream\n\n**Run Time:** 2026-11-20 08:00:00\n\n---\n\n"
        f"## Response\n{body}\n",
        encoding="utf-8",
    )
    if age_seconds:
        stamp = time.time() - age_seconds
        os.utime(archive, (stamp, stamp))
    return archive


class TestDispatchLatenessBlock:
    """The block is silent unless the gap is worth reasoning about."""

    def test_ticker_slip_stays_silent(self):
        assert _dispatch_lateness_block(_job(last_dispatch=_stamp(42))) is None

    def test_a_run_without_a_usable_stamp_has_no_block(self):
        assert _dispatch_lateness_block(_job()) is None
        assert _dispatch_lateness_block(_job(last_dispatch="garbage")) is None
        assert _dispatch_lateness_block(_job(last_dispatch={"lateness_seconds": "n/a"})) is None

    def test_an_operator_triggered_run_has_no_block(self):
        assert _dispatch_lateness_block(
            _job(last_dispatch=_stamp(7200), manual_run_at="2026-11-20T22:00:00")) is None

    def test_a_late_run_states_both_moments_and_the_gap(self):
        block = _dispatch_lateness_block(_job(last_dispatch=_stamp(19200)))
        assert block is not None
        assert block.startswith("## Run Timing\n")
        assert "5.3h late" in block
        # Trimmed to minutes: the raw stamps carry microseconds the model has no use for.
        scheduled = block.split("scheduled for ", 1)[1].split(" but started at ", 1)[0]
        assert len(scheduled) == len("2026-11-20 17:36")
        assert "T" not in scheduled
        # A plain late fire did not collapse a backlog, so it must not claim it did.
        assert "catch-up" not in block
        assert "skipped" not in block
        assert "unobserved time" in block

    def test_a_catch_up_says_the_backlog_was_collapsed(self):
        block = _dispatch_lateness_block(_job(last_dispatch=_stamp(25200, kind="catch_up")))
        assert block is not None
        assert "(catch-up after downtime)" in block
        assert "one or more scheduled occurrences were skipped" in block


class TestPromptAssembly:
    def test_the_block_sits_after_the_run_time_line_and_before_the_job_prompt(self):
        prompt = _build_job_prompt(_job(last_dispatch=_stamp(19200)))
        assert (prompt.index("[CURRENT RUN TIME:")
                < prompt.index("## Run Timing")
                < prompt.index("summarize the close"))

    def test_an_on_time_run_carries_no_timing_block(self):
        prompt = _build_job_prompt(_job(last_dispatch=_stamp(30)))
        assert "[CURRENT RUN TIME:" in prompt
        assert "## Run Timing" not in prompt


class TestContextAge:
    def test_upstream_context_heading_states_the_age(self, cron_env):
        _write_archive(cron_env / "cron" / "output", "abcdef12", age_seconds=3 * 3600 + 12 * 60)
        prompt = _build_job_prompt(_job(context_from="abcdef12"))
        assert "## Output from job 'abcdef12' (written 3.2h ago)" in prompt
        assert "treat an output older than this run as stale history" in prompt

    def test_self_context_heading_states_the_age(self, cron_env):
        _write_archive(cron_env / "cron" / "output", JOB_ID, age_seconds=90 * 60)
        prompt = _build_job_prompt(_job(context_from="self"))
        assert "## Your previous run's output (written 1.5h ago)" in prompt

    def test_a_silent_archive_still_has_no_age_heading(self, cron_env):
        _write_archive(cron_env / "cron" / "output", "abcdef12", body="[SILENT]")
        prompt = _build_job_prompt(_job(context_from="abcdef12"))
        assert "written" not in prompt


@pytest.mark.parametrize(("seconds", "label"), [
    (45, "45s"),
    (700, "11m"),
    (12000, "3.3h"),
    (190000, "2.2d"),
])
def test_humanize_age_stays_compact(seconds, label):
    assert _humanize_age(seconds) == label
