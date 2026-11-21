"""Older continuity reports must disclose newer runs without reusable output."""

import os

import pytest

from cron import jobs
from cron.scheduler_prompt import _inject_context_from


@pytest.mark.parametrize("source", ["self", "abcdef", "feedab"])
def test_repeated_silent_runs_label_historical_blocker(tmp_path, source):
    with jobs.use_cron_store(tmp_path):
        directory = jobs.get_cron_output_dir() / ("feedab" if source == "feedab" else "abcdef")
        directory.mkdir(parents=True)
        records = ["## Response\n\nBlocked awaiting an artifact.\n"]
        records.extend("## Response\n\n[SILENT]\n" for _ in range(3))
        for index, record in enumerate(records):
            path = directory / f"{index}.md"
            path.write_text(record, encoding="utf-8")
            os.utime(path, (index + 1, index + 1))

        prompt, injected = _inject_context_from(
            {"id": "abcdef", "context_from": [source]}, "Continue the unfinished task")

        assert injected
        assert "Blocked awaiting an artifact." in prompt
        assert "skipped 3 newer archives with no reusable output" in prompt
        assert "Recheck any historical blocker" in prompt
        assert prompt.count("Continuity freshness:") == 1
        assert "[SILENT]" not in prompt


def test_current_output_and_unconfigured_jobs_have_no_freshness_notice(tmp_path):
    with jobs.use_cron_store(tmp_path):
        directory = jobs.get_cron_output_dir() / "abcdef"
        directory.mkdir(parents=True)
        old = directory / "silent.md"
        old.write_text("## Response\n\n[SILENT]\n", encoding="utf-8")
        os.utime(old, (1, 1))
        latest = directory / "run.md"
        latest.write_text("## Response\n\nArtifact ready.\n", encoding="utf-8")
        os.utime(latest, (2, 2))
        prompt, injected = _inject_context_from(
            {"id": "abcdef", "context_from": ["self"]}, "Continue")
        assert injected and "Artifact ready." in prompt
        assert "Continuity freshness:" not in prompt
        assert _inject_context_from({"id": "abcdef"}, "Continue") == ("Continue", False)
