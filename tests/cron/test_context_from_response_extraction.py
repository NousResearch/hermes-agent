"""context_from must inject the previous run's answer, not its prompt (#117290).

Stored agent-run archives are ``# Cron Job`` / ``## Prompt`` / ``## Response``
documents; skill-bearing prompts routinely exceed the 8000-char injection
budget, so head-truncation amputated the ``## Response`` section and
self-continuity silently became a no-op.
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


def _write_archive(cron_env, job_id: str, filename: str, body: str) -> None:
    from cron.jobs import OUTPUT_DIR

    out_dir = OUTPUT_DIR / job_id
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / filename).write_text(body, encoding="utf-8")


class TestResponseSurvivesLongPrompt:
    """The answer, not the prompt, is the part continuity needs."""

    def test_long_prompt_archive_keeps_response(self, cron_env):
        from cron.jobs import create_job
        from cron.scheduler import _build_job_prompt

        job = create_job(prompt="Run the daily check", schedule="0 8 * * *", context_from="self")
        _write_archive(
            cron_env, job["id"], "2026-09-19_08-00-00.md",
            "# Cron Job: probe\n\n## Prompt\n\n" + "SKILL LINE\n" * 1800 +
            "\n\n## Response\n\nCONCLUSION-MARKER-42\n",
        )

        prompt = _build_job_prompt(job)

        assert "CONCLUSION-MARKER-42" in prompt
        assert "SKILL LINE" not in prompt  # the prompt half is dropped, not the answer


class TestUnusableAnswersFallThrough:
    """A [SILENT] or blank response is not usable continuity."""

    def test_silent_response_falls_through_to_older_archive(self, cron_env):
        from cron.jobs import create_job
        from cron.scheduler_prompt import _inject_context_from

        job = create_job(prompt="Report", schedule="0 8 * * *", context_from="self")
        _write_archive(
            cron_env, job["id"], "2026-09-18_08-00-00.md",
            "# Cron Job: probe\n\n## Prompt\n\ndo the thing\n\n## Response\n\nOLDER-REAL-ANSWER\n",
        )
        _write_archive(
            cron_env, job["id"], "2026-09-19_08-00-00.md",
            "# Cron Job: probe\n\n## Prompt\n\ndo the thing\n\n## Response\n\n[SILENT]\n",
        )

        prompt, injected = _inject_context_from(job, "Report")

        assert injected is True
        assert "OLDER-REAL-ANSWER" in prompt
        assert "[SILENT]" not in prompt

class TestScriptModeArchives:
    """Archives without a ## Response heading (script-mode) stay whole-document."""

    def test_headingless_archive_injects_whole_document(self, cron_env):
        from cron.jobs import create_job
        from cron.scheduler_prompt import _inject_context_from

        job = create_job(prompt="Report", schedule="0 8 * * *", context_from="self")
        _write_archive(cron_env, job["id"], "2026-09-19_08-00-00.md", "plain script payload\nline two")

        prompt, injected = _inject_context_from(job, "Report")

        assert injected is True
        assert "plain script payload" in prompt
        assert "line two" in prompt

class TestMalformedContextFrom:
    """Hand-edited jobs.json fields must warn-skip, not crash the fire."""

    def test_non_list_context_from_is_ignored(self, cron_env):
        from cron.jobs import create_job
        from cron.scheduler_prompt import _inject_context_from

        job = create_job(prompt="Report", schedule="0 8 * * *")
        for bad in (12345, 3.14, True, {"a": 1}):
            job["context_from"] = bad
            prompt, injected = _inject_context_from(job, "Report")
            assert prompt == "Report"
            assert injected is False

    def test_non_string_entry_is_skipped(self, cron_env):
        from cron.jobs import create_job
        from cron.scheduler_prompt import _inject_context_from

        job = create_job(prompt="Report", schedule="0 8 * * *")
        for bad in ([12345], [True], [{"a": 1}], [["deadbeef"]]):
            job["context_from"] = bad
            prompt, injected = _inject_context_from(job, "Report")
            assert prompt == "Report"
            assert injected is False

    def test_bad_entry_does_not_block_valid_sibling(self, cron_env):
        from cron.jobs import create_job
        from cron.scheduler_prompt import _inject_context_from

        source = create_job(prompt="Upstream", schedule="0 8 * * *")
        _write_archive(
            cron_env, source["id"], "2026-09-19_08-00-00.md",
            "# Cron Job: up\n\n## Prompt\n\ngo\n\n## Response\n\nUPSTREAM-ANSWER\n",
        )
        job = create_job(prompt="Report", schedule="0 8 * * *")
        job["context_from"] = [12345, source["id"]]

        prompt, injected = _inject_context_from(job, "Report")

        assert injected is True
        assert "UPSTREAM-ANSWER" in prompt

    def test_build_job_prompt_survives_poisoned_context_from(self, cron_env):
        """The path run_job actually takes: a poisoned context_from must not
        escape _build_job_prompt, which sits outside run_job's try/except."""
        from cron.jobs import create_job
        from cron.scheduler_prompt import _build_job_prompt

        job = create_job(prompt="Report", schedule="0 8 * * *")
        job["context_from"] = 12345

        prompt = _build_job_prompt(job)

        assert "Report" in prompt

    def test_non_iterable_skills_is_ignored(self, cron_env):
        """Hand-edited jobs.json `skills: 5` must not crash prompt assembly."""
        from cron.scheduler_prompt import _job_skill_names

        for bad in (5, 3.14, True, {"a": 1}):
            assert _job_skill_names({"skills": bad}) == []

    def test_build_job_prompt_survives_fully_poisoned_job(self, cron_env):
        """One job carrying every malformed field still builds a prompt:
        script error is reported in-band, the rest are warn-skipped."""
        from cron.jobs import create_job
        from cron.scheduler_prompt import _build_job_prompt

        job = create_job(prompt="Report", schedule="0 8 * * *")
        job["context_from"] = 12345
        job["skills"] = 7
        job["workdir"] = {"x": 1}
        job["script"] = 12345

        prompt = _build_job_prompt(job)

        assert "Report" in prompt
        assert "Script Error" in prompt
