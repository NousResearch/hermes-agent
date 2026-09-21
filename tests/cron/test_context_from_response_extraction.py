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


class TestUnreadableArchives:
    """A corrupt archive file must be skipped like a silent one, not kill the run."""

    def test_corrupt_newest_archive_falls_through_to_older(self, cron_env):
        """Newest .md holds non-UTF-8 bytes: skip it and use the older good one."""
        import os
        from cron.jobs import OUTPUT_DIR, create_job
        from cron.scheduler_prompt import _inject_context_from

        source = create_job(prompt="Upstream", schedule="0 8 * * *")
        _write_archive(
            cron_env, source["id"], "2026-09-18_08-00-00.md",
            "# Cron Job: up\n\n## Prompt\n\ngo\n\n## Response\n\nOLDER-GOOD\n",
        )
        corrupt = OUTPUT_DIR / source["id"] / "2026-09-19_08-00-00.md"
        corrupt.write_bytes(b"\xff\xfe\x00corrupt")
        os.utime(corrupt, (2_000_000_000, 2_000_000_000))

        job = create_job(prompt="Report", schedule="0 8 * * *")
        job["context_from"] = [source["id"]]

        prompt, injected = _inject_context_from(job, "Report")

        assert injected is True
        assert "OLDER-GOOD" in prompt

    def test_corrupt_only_archive_skips_source(self, cron_env):
        """A source whose only archive is undecodable is skipped, not fatal."""
        from cron.jobs import OUTPUT_DIR, create_job
        from cron.scheduler_prompt import _inject_context_from

        source = create_job(prompt="Upstream", schedule="0 8 * * *")
        out_dir = OUTPUT_DIR / source["id"]
        out_dir.mkdir(parents=True)
        (out_dir / "2026-09-19_08-00-00.md").write_bytes(b"\xff\xfe\x00corrupt")

        job = create_job(prompt="Report", schedule="0 8 * * *")
        job["context_from"] = [source["id"]]

        assert _inject_context_from(job, "Report") == ("Report", False)

    def test_build_job_prompt_survives_corrupt_archive(self, cron_env):
        """E2E through the real run path: _build_job_prompt must not raise on a
        corrupt archive - the job fires with no injected context."""
        from cron.jobs import OUTPUT_DIR, create_job
        from cron.scheduler_prompt import _build_job_prompt

        source = create_job(prompt="Upstream", schedule="0 8 * * *")
        out_dir = OUTPUT_DIR / source["id"]
        out_dir.mkdir(parents=True)
        (out_dir / "2026-09-19_08-00-00.md").write_bytes(b"\xff\xfe\x00corrupt")

        job = create_job(prompt="Report", schedule="0 8 * * *")
        job["context_from"] = [source["id"]]

        prompt = _build_job_prompt(job)

        assert "Report" in prompt
        assert "Output from job" not in prompt
