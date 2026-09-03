"""Continuity must carry the previous *response*, not the whole run file.

Regression tests for #101623: ``context_from`` injected the entire previous
output file, which begins with the augmented ``## Prompt``. Each run therefore
nested inside the next, and the 8,000-character cap truncated away the response
the injection exists to carry.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


@pytest.fixture
def cron_env(tmp_path, monkeypatch):
    """Isolated cron environment with temp HERMES_HOME."""
    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir()
    (hermes_home / "cron").mkdir()
    (hermes_home / "cron" / "output").mkdir()
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    import cron.jobs as jobs_mod
    monkeypatch.setattr(jobs_mod, "HERMES_DIR", hermes_home)
    monkeypatch.setattr(jobs_mod, "CRON_DIR", hermes_home / "cron")
    monkeypatch.setattr(jobs_mod, "JOBS_FILE", hermes_home / "cron" / "jobs.json")
    monkeypatch.setattr(jobs_mod, "OUTPUT_DIR", hermes_home / "cron" / "output")

    return hermes_home


def _run_file(prompt: str, response: str, job_name: str = "nest") -> str:
    """Render an output file the way the scheduler writes one."""
    return (
        f"# Cron Job: {job_name}\n\n"
        "**Job ID:** abcdef123456\n"
        "**Run Time:** 2026-09-02 10:00:00\n"
        "**Schedule:** */5 * * * *\n\n"
        "## Prompt\n\n"
        f"{prompt}\n\n"
        "## Response\n\n"
        f"{response}\n"
    )


class TestExtractCronOutputBody:
    def test_returns_response_section(self):
        from cron.scheduler import _extract_cron_output_body

        text = _run_file("Say one new fact.", "Otters hold hands.")
        assert _extract_cron_output_body(text) == "Otters hold hands."

    def test_ignores_prompt_section_entirely(self):
        from cron.scheduler import _extract_cron_output_body

        text = _run_file("SECRET_PROMPT_MARKER", "the answer")
        assert "SECRET_PROMPT_MARKER" not in _extract_cron_output_body(text)

    def test_recovers_response_from_an_already_nested_file(self):
        """Files written before the fix embed prior runs in ## Prompt."""
        from cron.scheduler import _extract_cron_output_body

        inner = _run_file("original prompt", "FIRST RESPONSE")
        nested_prompt = (
            "## Your previous run's output\n"
            f"```\n{inner}\n```\n\n"
            "original prompt"
        )
        outer = _run_file(nested_prompt, "SECOND RESPONSE")

        # The last ## Response is this run's own, not the embedded one.
        assert _extract_cron_output_body(outer) == "SECOND RESPONSE"

    def test_falls_back_to_error_section(self):
        from cron.scheduler import _extract_cron_output_body

        text = (
            "# Cron Job: nest (FAILED)\n\n"
            "## Prompt\n\nsome prompt\n\n"
            "## Error\n\n```\nboom\n```\n"
        )
        assert "boom" in _extract_cron_output_body(text)
        assert "some prompt" not in _extract_cron_output_body(text)

    def test_unknown_format_returned_unchanged(self):
        """Plain-text output files predate the sectioned format."""
        from cron.scheduler import _extract_cron_output_body

        assert _extract_cron_output_body("  just text  ") == "just text"

    def test_empty_input(self):
        from cron.scheduler import _extract_cron_output_body

        assert _extract_cron_output_body("") == ""


class TestContinuityInjection:
    def test_self_context_injects_response_not_prompt(self, cron_env):
        from cron.jobs import create_job, OUTPUT_DIR
        from cron.scheduler import _build_job_prompt

        job = create_job(prompt="Say one new fact.", schedule="every 1h",
                         context_from="self")
        out = OUTPUT_DIR / job["id"]
        out.mkdir(parents=True, exist_ok=True)
        (out / "2026-09-02_10-00-00.md").write_text(
            _run_file("Say one new fact.", "Otters hold hands."), encoding="utf-8"
        )

        result = _build_job_prompt(job)

        assert "Otters hold hands." in result
        assert "## Prompt" not in result
        assert "**Job ID:**" not in result

    def test_does_not_nest_across_three_runs(self, cron_env):
        """The reported symptom: run 3 must still see run 2's response."""
        from cron.jobs import create_job, OUTPUT_DIR
        from cron.scheduler import _build_job_prompt

        job = create_job(prompt="Say one new fact.", schedule="every 1h",
                         context_from="self")
        out = OUTPUT_DIR / job["id"]
        out.mkdir(parents=True, exist_ok=True)

        # Run 1 — no prior context. A realistically sized report: the issue
        # reports 10-11 KB run files, which is what pushes the next run's
        # ## Response past the 8,000-character cap.
        fact_one = "FACT ONE. " + ("detail line for the weekly report. " * 300)
        (out / "2026-09-02_10-00-00.md").write_text(
            _run_file("Say one new fact.", fact_one), encoding="utf-8"
        )

        # Run 2 — built from run 1, then recorded with its augmented prompt.
        run2_prompt = _build_job_prompt(job)
        assert "FACT ONE" in run2_prompt
        assert len(run2_prompt) > 8000  # the next read would truncate mid-prompt
        f2 = out / "2026-09-02_10-05-00.md"
        f2.write_text(_run_file(run2_prompt, "FACT TWO"), encoding="utf-8")
        import os, time
        os.utime(f2, (time.time() + 10, time.time() + 10))

        # Run 3 must carry FACT TWO — before the fix this was truncated away.
        run3_prompt = _build_job_prompt(job)
        assert "FACT TWO" in run3_prompt

    def test_injected_context_stays_small(self, cron_env):
        """Without the fix each run's file grows by the whole previous file."""
        from cron.jobs import create_job, OUTPUT_DIR
        from cron.scheduler import _build_job_prompt

        job = create_job(prompt="p", schedule="every 1h", context_from="self")
        out = OUTPUT_DIR / job["id"]
        out.mkdir(parents=True, exist_ok=True)

        bulky_prompt = "PRIOR CONTEXT " * 800  # ~11 KB of prompt history
        (out / "2026-09-02_10-00-00.md").write_text(
            _run_file(bulky_prompt, "short answer"), encoding="utf-8"
        )

        result = _build_job_prompt(job)

        assert "short answer" in result
        assert "PRIOR CONTEXT" not in result
        assert "[... output truncated ...]" not in result
