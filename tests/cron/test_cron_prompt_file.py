"""Tests for the cron job prompt_file feature.

Covers:
- _load_prompt_file() path resolution (relative, absolute)
- _load_prompt_file() path traversal guard
- _load_prompt_file() missing file and unreadable file errors
- _build_job_prompt() with prompt_file (wins over inline prompt)
- _build_job_prompt() with prompt_file that fails to load
- _build_job_prompt() without prompt_file (existing behaviour unchanged)
"""

import sys
import textwrap
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


@pytest.fixture()
def cron_env(tmp_path, monkeypatch):
    """Isolated Hermes home with cron, scripts, and cron-jobs directories."""
    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir()
    (hermes_home / "cron").mkdir()
    (hermes_home / "cron" / "output").mkdir()
    (hermes_home / "scripts").mkdir()
    (hermes_home / "cron-jobs").mkdir()
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    import cron.jobs as jobs_mod
    monkeypatch.setattr(jobs_mod, "HERMES_DIR", hermes_home)
    monkeypatch.setattr(jobs_mod, "CRON_DIR", hermes_home / "cron")
    monkeypatch.setattr(jobs_mod, "JOBS_FILE", hermes_home / "cron" / "jobs.json")
    monkeypatch.setattr(jobs_mod, "OUTPUT_DIR", hermes_home / "cron" / "output")

    return hermes_home


# ---------------------------------------------------------------------------
# _load_prompt_file — unit tests
# ---------------------------------------------------------------------------


class TestLoadPromptFile:
    def test_relative_path_resolved_against_cron_jobs_dir(self, cron_env):
        from cron.scheduler import _load_prompt_file

        prompt_path = cron_env / "cron-jobs" / "morning.md"
        prompt_path.write_text("Summarise the overnight events.\n")

        ok, content = _load_prompt_file("morning.md")
        assert ok is True
        assert content == "Summarise the overnight events."

    def test_absolute_path_within_cron_jobs_dir(self, cron_env):
        from cron.scheduler import _load_prompt_file

        prompt_path = cron_env / "cron-jobs" / "sub" / "briefing.md"
        prompt_path.parent.mkdir(parents=True)
        prompt_path.write_text("Daily briefing prompt.\n")

        ok, content = _load_prompt_file(str(prompt_path))
        assert ok is True
        assert content == "Daily briefing prompt."

    def test_file_not_found(self, cron_env):
        from cron.scheduler import _load_prompt_file

        ok, msg = _load_prompt_file("nonexistent.md")
        assert ok is False
        assert "not found" in msg.lower()

    def test_path_traversal_blocked(self, cron_env):
        from cron.scheduler import _load_prompt_file

        # Try to escape cron-jobs/ via ../
        ok, msg = _load_prompt_file("../scripts/evil.sh")
        assert ok is False
        assert "Blocked" in msg

    def test_absolute_path_outside_cron_jobs_blocked(self, cron_env):
        from cron.scheduler import _load_prompt_file

        outside = cron_env.parent / "secret.txt"
        outside.write_text("sensitive data")

        ok, msg = _load_prompt_file(str(outside))
        assert ok is False
        assert "Blocked" in msg

    def test_unicode_content_preserved(self, cron_env):
        from cron.scheduler import _load_prompt_file

        prompt_path = cron_env / "cron-jobs" / "intl.md"
        prompt_path.write_text("日次レポート — Günlük rapor.\n", encoding="utf-8")

        ok, content = _load_prompt_file("intl.md")
        assert ok is True
        assert "日次" in content
        assert "Günlük" in content

    def test_trailing_whitespace_stripped(self, cron_env):
        from cron.scheduler import _load_prompt_file

        prompt_path = cron_env / "cron-jobs" / "padded.md"
        prompt_path.write_text("\n\nDo the thing.\n\n")

        ok, content = _load_prompt_file("padded.md")
        assert ok is True
        assert content == "Do the thing."

    def test_cron_jobs_dir_created_if_missing(self, cron_env, monkeypatch):
        """_load_prompt_file creates cron-jobs/ on first access."""
        from cron.scheduler import _load_prompt_file

        cron_jobs_dir = cron_env / "cron-jobs"
        cron_jobs_dir.rmdir()
        assert not cron_jobs_dir.exists()

        ok, _ = _load_prompt_file("nofile.md")
        assert not ok  # file still missing, but no crash
        assert cron_jobs_dir.exists()  # directory was created


# ---------------------------------------------------------------------------
# _build_job_prompt — prompt_file integration
# ---------------------------------------------------------------------------


class TestBuildJobPromptWithPromptFile:
    def test_prompt_file_used_as_prompt(self, cron_env):
        from cron.scheduler import _build_job_prompt

        prompt_path = cron_env / "cron-jobs" / "report.md"
        prompt_path.write_text("Generate the weekly status report.")

        job = {"prompt_file": "report.md"}
        result = _build_job_prompt(job)
        assert "Generate the weekly status report." in result

    def test_prompt_file_wins_over_inline_prompt(self, cron_env):
        """When both prompt and prompt_file are set, prompt_file takes precedence."""
        from cron.scheduler import _build_job_prompt

        prompt_path = cron_env / "cron-jobs" / "file_prompt.md"
        prompt_path.write_text("Prompt from file.")

        job = {
            "prompt": "Inline prompt — should be ignored.",
            "prompt_file": "file_prompt.md",
        }
        result = _build_job_prompt(job)
        assert "Prompt from file." in result
        assert "Inline prompt" not in result

    def test_missing_prompt_file_injects_error(self, cron_env):
        from cron.scheduler import _build_job_prompt

        job = {"prompt_file": "nonexistent.md"}
        result = _build_job_prompt(job)
        assert "## Prompt File Error" in result
        assert "not found" in result.lower()

    def test_traversal_prompt_file_injects_error(self, cron_env):
        from cron.scheduler import _build_job_prompt

        job = {"prompt_file": "../../etc/passwd"}
        result = _build_job_prompt(job)
        assert "## Prompt File Error" in result
        assert "Blocked" in result

    def test_without_prompt_file_inline_prompt_unchanged(self, cron_env):
        """Existing behaviour: no prompt_file → inline prompt used as before."""
        from cron.scheduler import _build_job_prompt

        job = {"prompt": "Plain inline job."}
        result = _build_job_prompt(job)
        assert "Plain inline job." in result
        assert "## Prompt File Error" not in result

    def test_prompt_file_combined_with_script_output(self, cron_env):
        """Script output is still prepended in front of the file-sourced prompt."""
        from cron.scheduler import _build_job_prompt

        prompt_path = cron_env / "cron-jobs" / "analysis.md"
        prompt_path.write_text("Analyse the data above.")

        script = cron_env / "scripts" / "data.py"
        script.write_text('print("metric_value=42")\n')

        job = {
            "prompt_file": "analysis.md",
            "script": str(script),
        }
        result = _build_job_prompt(job)
        assert "metric_value=42" in result
        assert "Analyse the data above." in result
        assert result.index("metric_value=42") < result.index("Analyse the data above.")
