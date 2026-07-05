"""Tests for cron.jobs.rewrite_skill_refs — the curator integration that
keeps scheduled cron jobs pointing at the right skill names after a
consolidation / pruning pass.

Bug this fixes: when the curator consolidates skill X into umbrella Y,
any cron job whose ``skills`` list contains X would silently fail to
load X at run time (the scheduler logs a warning and skips it), so the
job runs without the instructions it was scheduled to follow.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Ensure project root is importable
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


class TestRewriteSkillRefsNoop:
    """No jobs, no rewrites, no map — every combination of empty inputs."""

    def test_empty_map_and_no_jobs(self, cron_env):
        from cron.jobs import rewrite_skill_refs

        report = rewrite_skill_refs(consolidated={}, pruned=[])
        assert report == {"rewrites": [], "jobs_updated": 0, "jobs_scanned": 0}

    def test_jobs_exist_but_map_empty(self, cron_env):
        from cron.jobs import create_job, rewrite_skill_refs

        create_job(prompt="", schedule="every 1h", skills=["foo"])
        report = rewrite_skill_refs(consolidated={}, pruned=[])
        assert report["jobs_updated"] == 0
        # Early return: we don't even scan when there's nothing to apply.
        assert report["jobs_scanned"] == 0

    def test_jobs_exist_but_no_match(self, cron_env):
        from cron.jobs import create_job, get_job, rewrite_skill_refs

        job = create_job(prompt="", schedule="every 1h", skills=["foo"])
        report = rewrite_skill_refs(
            consolidated={"unrelated": "umbrella"},
            pruned=["other"],
        )
        assert report["jobs_updated"] == 0
        assert report["jobs_scanned"] == 1
        # Job untouched
        loaded = get_job(job["id"])
        assert loaded["skills"] == ["foo"]


class TestRewriteSkillRefsConsolidation:
    """Consolidated skills should be replaced with their umbrella target."""

    def test_single_skill_replaced(self, cron_env):
        from cron.jobs import create_job, get_job, rewrite_skill_refs

        job = create_job(prompt="", schedule="every 1h", skills=["legacy-skill"])
        report = rewrite_skill_refs(
            consolidated={"legacy-skill": "umbrella-skill"},
            pruned=[],
        )

        assert report["jobs_updated"] == 1
        loaded = get_job(job["id"])
        assert loaded["skills"] == ["umbrella-skill"]
        # Legacy ``skill`` field realigned
        assert loaded["skill"] == "umbrella-skill"

    def test_multiple_skills_one_consolidated(self, cron_env):
        from cron.jobs import create_job, get_job, rewrite_skill_refs

        job = create_job(
            prompt="",
            schedule="every 1h",
            skills=["keep-a", "legacy", "keep-b"],
        )
        rewrite_skill_refs(consolidated={"legacy": "umbrella"}, pruned=[])

        loaded = get_job(job["id"])
        # Ordering preserved, legacy replaced in-place
        assert loaded["skills"] == ["keep-a", "umbrella", "keep-b"]

    def test_umbrella_already_in_list_dedupes(self, cron_env):
        from cron.jobs import create_job, get_job, rewrite_skill_refs

        # Job already loads the umbrella AND the legacy sub-skill
        job = create_job(
            prompt="",
            schedule="every 1h",
            skills=["umbrella", "legacy"],
        )
        rewrite_skill_refs(consolidated={"legacy": "umbrella"}, pruned=[])

        loaded = get_job(job["id"])
        # No duplicate — the umbrella stays exactly once
        assert loaded["skills"] == ["umbrella"]

    def test_rewrite_report_records_mapping(self, cron_env):
        from cron.jobs import create_job, rewrite_skill_refs

        job = create_job(
            prompt="",
            schedule="every 1h",
            skills=["a", "b"],
            name="my-job",
        )
        report = rewrite_skill_refs(
            consolidated={"a": "umbrella-a", "b": "umbrella-b"},
            pruned=[],
        )

        assert len(report["rewrites"]) == 1
        entry = report["rewrites"][0]
        assert entry["job_id"] == job["id"]
        assert entry["job_name"] == "my-job"
        assert entry["before"] == ["a", "b"]
        assert entry["after"] == ["umbrella-a", "umbrella-b"]
        assert entry["mapped"] == {"a": "umbrella-a", "b": "umbrella-b"}
        assert entry["dropped"] == []


class TestRewriteSkillRefsPruning:
    """Pruned skills should be dropped outright (no forwarding target)."""

    def test_pruned_skill_dropped(self, cron_env):
        from cron.jobs import create_job, get_job, rewrite_skill_refs

        job = create_job(
            prompt="",
            schedule="every 1h",
            skills=["keep", "stale"],
        )
        report = rewrite_skill_refs(consolidated={}, pruned=["stale"])

        assert report["jobs_updated"] == 1
        loaded = get_job(job["id"])
        assert loaded["skills"] == ["keep"]
        assert loaded["skill"] == "keep"

    def test_all_skills_pruned_leaves_empty_list(self, cron_env):
        from cron.jobs import create_job, get_job, rewrite_skill_refs

        job = create_job(prompt="", schedule="every 1h", skills=["gone"])
        rewrite_skill_refs(consolidated={}, pruned=["gone"])

        loaded = get_job(job["id"])
        assert loaded["skills"] == []
        assert loaded["skill"] is None

    def test_pruned_report_records_drops(self, cron_env):
        from cron.jobs import create_job, rewrite_skill_refs

        create_job(prompt="", schedule="every 1h", skills=["keep", "stale"])
        report = rewrite_skill_refs(consolidated={}, pruned=["stale"])

        entry = report["rewrites"][0]
        assert entry["dropped"] == ["stale"]
        assert entry["mapped"] == {}


class TestRewriteSkillRefsMixed:
    """Consolidation + pruning in the same pass."""

    def test_mixed_consolidation_and_pruning(self, cron_env):
        from cron.jobs import create_job, get_job, rewrite_skill_refs

        job = create_job(
            prompt="",
            schedule="every 1h",
            skills=["keep", "legacy", "stale"],
        )
        rewrite_skill_refs(
            consolidated={"legacy": "umbrella"},
            pruned=["stale"],
        )

        loaded = get_job(job["id"])
        assert loaded["skills"] == ["keep", "umbrella"]

    def test_skill_in_both_maps_wins_as_consolidated(self, cron_env):
        """Defensive: if a skill appears in both lists (shouldn't happen
        in practice), prefer consolidation — it has a forwarding target,
        which is the more useful outcome."""
        from cron.jobs import create_job, get_job, rewrite_skill_refs

        job = create_job(prompt="", schedule="every 1h", skills=["ambiguous"])
        rewrite_skill_refs(
            consolidated={"ambiguous": "umbrella"},
            pruned=["ambiguous"],
        )

        loaded = get_job(job["id"])
        assert loaded["skills"] == ["umbrella"]


class TestRewriteSkillRefsMultipleJobs:
    """Multiple jobs, some affected, some not."""

    def test_only_affected_jobs_reported(self, cron_env):
        from cron.jobs import create_job, get_job, rewrite_skill_refs

        j1 = create_job(prompt="", schedule="every 1h", skills=["legacy"])
        j2 = create_job(prompt="", schedule="every 1h", skills=["untouched"])
        j3 = create_job(prompt="", schedule="every 1h", skills=[])

        report = rewrite_skill_refs(
            consolidated={"legacy": "umbrella"},
            pruned=[],
        )

        assert report["jobs_updated"] == 1
        assert report["jobs_scanned"] == 3
        assert len(report["rewrites"]) == 1
        assert report["rewrites"][0]["job_id"] == j1["id"]

        # Untouched jobs stay put
        assert get_job(j2["id"])["skills"] == ["untouched"]
        assert get_job(j3["id"])["skills"] == []

    def test_legacy_skill_field_also_rewritten(self, cron_env):
        """Old jobs may have the legacy single-skill ``skill`` field
        set instead of ``skills``. Both paths should be rewritten."""
        from cron.jobs import create_job, get_job, rewrite_skill_refs

        # Create via the legacy ``skill`` argument
        job = create_job(
            prompt="",
            schedule="every 1h",
            skill="legacy",
        )
        rewrite_skill_refs(consolidated={"legacy": "umbrella"}, pruned=[])

        loaded = get_job(job["id"])
        assert loaded["skills"] == ["umbrella"]
        assert loaded["skill"] == "umbrella"


class TestRewriteSkillRefsPersistence:
    """Rewrites persist to disk and survive a reload."""

    def test_changes_persist_across_reload(self, cron_env):
        import json
        from cron.jobs import create_job, rewrite_skill_refs, JOBS_FILE

        create_job(prompt="", schedule="every 1h", skills=["legacy"])
        rewrite_skill_refs(consolidated={"legacy": "umbrella"}, pruned=[])

        # Read raw file contents
        data = json.loads(JOBS_FILE.read_text())
        assert data["jobs"][0]["skills"] == ["umbrella"]
        assert data["jobs"][0]["skill"] == "umbrella"

    def test_noop_does_not_rewrite_file(self, cron_env):
        from cron.jobs import create_job, rewrite_skill_refs, JOBS_FILE

        create_job(prompt="", schedule="every 1h", skills=["keep"])
        mtime_before = JOBS_FILE.stat().st_mtime_ns

        # Nothing in the map matches
        report = rewrite_skill_refs(
            consolidated={"unrelated": "umbrella"},
            pruned=["other"],
        )

        assert report["jobs_updated"] == 0
        # File untouched — no pointless disk write
        assert JOBS_FILE.stat().st_mtime_ns == mtime_before
