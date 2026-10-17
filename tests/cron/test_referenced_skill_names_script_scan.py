"""Tests for cron.jobs.referenced_skill_names — the script-file scan leg.

Bug this fixes (upstream #136030): ``referenced_skill_names()`` only read the
job's declared ``skill``/``skills`` fields. A ``no_agent`` script job that
hard-codes a ``skills/<category>/<name>`` path inside its script file is
invisible to the curator's protection set, never gets ``bump_use`` (script
jobs never build a prompt), ages past the inactivity cutoff, and is archived —
physically moved into ``.archive/`` so the production script's next run breaks.

The function's own docstring already claims "referenced by ANY cron job": a
job references a skill through its script, so the scan must read the script.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Ensure project root is importable
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


@pytest.fixture
def cron_env(tmp_path, monkeypatch):
    """Isolated cron environment with temp HERMES_HOME (same shape as
    test_rewrite_skill_refs.py — the neighboring curator-integration suite)."""
    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir()
    (hermes_home / "cron").mkdir()
    (hermes_home / "cron" / "output").mkdir()
    (hermes_home / "scripts").mkdir()
    (hermes_home / "skills").mkdir()
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    import cron.jobs as jobs_mod
    monkeypatch.setattr(jobs_mod, "HERMES_DIR", hermes_home)
    monkeypatch.setattr(jobs_mod, "CRON_DIR", hermes_home / "cron")
    monkeypatch.setattr(jobs_mod, "JOBS_FILE", hermes_home / "cron" / "jobs.json")
    monkeypatch.setattr(jobs_mod, "OUTPUT_DIR", hermes_home / "cron" / "output")

    return hermes_home


def _write_script(hermes_home: Path, name: str, body: str) -> Path:
    scripts_dir = hermes_home / "scripts"
    path = scripts_dir / name
    path.write_text(body, encoding="utf-8")
    return path


class TestScriptFileReferenceScan:
    """A script job references a skill through its script file — the scan leg."""

    def test_script_hardcoded_skill_path_is_protected(self, cron_env):
        """The issue #136030 production shape: no_agent job, ``skills: []``,
        script hard-codes ``skills/media/movie-fetcher``. The scan must pick
        the skill name out of the script file so the curator spares it."""
        from cron.jobs import create_job, referenced_skill_names

        skill_dir = cron_env / "skills" / "media" / "movie-fetcher"
        skill_dir.mkdir(parents=True)
        _write_script(
            cron_env, "weekly-movies.sh",
            "#!/bin/bash\nSKILL_DIR=/Users/x/.hermes/skills/media/movie-fetcher\n",
        )
        create_job(prompt="", schedule="every 1h", script="weekly-movies.sh",
                   no_agent=True)

        refs = referenced_skill_names()

        assert refs == {"movie-fetcher"}

    def test_missing_script_file_fail_open(self, cron_env):
        """Script unresolvable/missing → that job contributes nothing and the
        function must not raise (contract: corrupt store yields an empty set,
        never a crash — fail-open, same as the declared-fields leg)."""
        from cron.jobs import create_job, referenced_skill_names

        create_job(prompt="", schedule="every 1h",
                   script="never-written.sh", no_agent=True)

        refs = referenced_skill_names()

        assert refs == set()

    def test_script_path_outside_scripts_dir_is_skipped(self, cron_env):
        """A job whose ``script`` value cannot resolve inside the scripts
        directory would be rejected by the scheduler at run time — the scan
        must skip it rather than guess at a path that will never run."""
        from cron.jobs import create_job, referenced_skill_names

        outside = cron_env / "elsewhere"
        outside.mkdir()
        (outside / "renegade.sh").write_text(
            "#!/bin/bash\ncat skills/media/movie-fetcher/SKILL.md\n",
            encoding="utf-8")
        create_job(prompt="", schedule="every 1h",
                   script=str(outside / "renegade.sh"), no_agent=True)

        refs = referenced_skill_names()

        assert refs == set()

    def test_declared_skills_regression(self, cron_env):
        """Existing declared-fields behavior is byte-for-byte unchanged:
        the scan leg is additive only."""
        from cron.jobs import create_job, referenced_skill_names

        create_job(prompt="", schedule="every 1h", skills=["legacy-skill"])

        refs = referenced_skill_names()

        assert "legacy-skill" in refs

    def test_script_url_mention_does_not_false_positive(self, cron_env):
        """Narrow-domain guard: a bare word or URL fragment must not count —
        only ``skills/`` path-shaped references resolving to a real directory
        under the skills root count (double guard: narrow regex + is_dir)."""
        from cron.jobs import create_job, referenced_skill_names

        _write_script(
            cron_env, "plain.sh",
            "#!/bin/bash\ncurl -s https://skills.example.com/media/movie-fetcher\n"
            "curl -s https://example.com/skills/media/movie-fetcher\n"
            "echo movie-fetcher\n",
        )
        create_job(prompt="", schedule="every 1h", script="plain.sh",
                   no_agent=True)

        refs = referenced_skill_names()

        assert refs == set()

    def test_url_token_skipped_even_when_dir_exists(self, cron_env):
        """URL guard is not just the is_dir intersection: a URL token is
        skipped even when its trailing segments happen to name a REAL skill
        directory (would-be false positive without the :// guard)."""
        from cron.jobs import create_job, referenced_skill_names

        skill_dir = cron_env / "skills" / "media" / "movie-fetcher"
        skill_dir.mkdir(parents=True)
        _write_script(
            cron_env, "with-url.sh",
            "#!/bin/bash\ncurl -s https://example.com/skills/media/movie-fetcher\n",
        )
        create_job(prompt="", schedule="every 1h", script="with-url.sh",
                   no_agent=True)

        refs = referenced_skill_names()

        assert refs == set()

    def test_inline_path_without_leading_delimiter_matches(self, cron_env):
        """``skills`` must count as a complete path segment even without a
        leading delimiter (assignment form: ``SKILL_DIR=$HOME/skills/media/x``)
        — but a dotted module form (``vendor.skills/x``) must not."""
        from cron.jobs import create_job, referenced_skill_names

        skill_dir = cron_env / "skills" / "media" / "movie-fetcher"
        skill_dir.mkdir(parents=True)
        _write_script(
            cron_env, "inline.sh",
            "#!/bin/bash\n"
            "SKILL_DIR=\"$HOME/skills/media/movie-fetcher\"\n"
            "ls vendor.skills/other-thing\n",
        )
        create_job(prompt="", schedule="every 1h", script="inline.sh",
                   no_agent=True)

        refs = referenced_skill_names()

        assert refs == {"movie-fetcher"}
