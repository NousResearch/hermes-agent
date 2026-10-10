"""Imported jobs with safe, human-readable ids can chain their saved output."""

from cron import jobs
from cron.job_definition import import_job_definitions
from cron.scheduler import _build_job_prompt
from hermes_constants import reset_hermes_home_override, set_hermes_home_override


def test_imported_named_jobs_can_read_upstream_and_self_context(tmp_path):
    token = set_hermes_home_override(tmp_path / "home")
    try:
        import_job_definitions({
            "daily-brief": {"prompt": "Find news", "schedule": "every 1h"},
            "daily_summary": {"prompt": "Summarize", "schedule": "every 2h",
                              "context_from": ["daily-brief", "self"]},
        }, paused_reason="Review imported jobs")
        for job_id, output in (("daily-brief", "Fresh upstream news"),
                               ("daily_summary", "Previous summary")):
            folder = jobs._job_output_dir(job_id)
            folder.mkdir(parents=True)
            (folder / "latest.md").write_text(output, encoding="utf-8")

        prompt = _build_job_prompt(jobs.get_job("daily_summary"))
        assert "Fresh upstream news" in prompt
        assert "Previous summary" in prompt
        assert "Your previous run's output" in prompt
    finally:
        reset_hermes_home_override(token)
