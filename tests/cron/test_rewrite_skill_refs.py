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
        assert report == {
            "rewrites": [],
            "jobs_updated": 0,
            "prompt_rewrite_jobs": 0,
            "prompt_warning_jobs": 0,
            "jobs_scanned": 0,
        }

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


class TestRewriteSkillRefsPromptText:
    """Token-level mentions in a job's freeform ``prompt`` are handled
    two ways: consolidated mentions (target known) are auto-rewritten
    in place; pruned mentions (no target) become warnings. See #23398.
    """

    def test_consolidated_skill_in_prompt_is_auto_rewritten(self, cron_env):
        from cron.jobs import create_job, get_job, rewrite_skill_refs

        job = create_job(
            prompt="Follow `legacy-skill` as the canonical spec.",
            schedule="every 1h",
            skills=["legacy-skill"],
        )
        report = rewrite_skill_refs(
            consolidated={"legacy-skill": "umbrella-skill"},
            pruned=[],
        )

        assert report["jobs_updated"] == 1
        assert report["prompt_rewrite_jobs"] == 1
        assert report["prompt_warning_jobs"] == 0
        entry = report["rewrites"][0]
        assert entry["after"] == ["umbrella-skill"]
        # Prompt IS auto-rewritten — the token-level mention becomes the
        # forwarding target, surrounding punctuation untouched.
        loaded = get_job(job["id"])
        assert "legacy-skill" not in loaded["prompt"]
        assert loaded["prompt"] == "Follow `umbrella-skill` as the canonical spec."
        # Auto-rewrite is reported (not a warning).
        assert entry["prompt_rewrites"] == [
            {"name": "legacy-skill", "target": "umbrella-skill", "count": 1},
        ]
        assert entry["prompt_warnings"] == []

    def test_pruned_skill_in_prompt_emits_warning_with_no_target(self, cron_env):
        from cron.jobs import create_job, get_job, rewrite_skill_refs

        job = create_job(
            prompt="The `stale` skill explains the canonical flow.",
            schedule="every 1h",
            skills=["stale"],
        )
        report = rewrite_skill_refs(consolidated={}, pruned=["stale"])

        assert report["prompt_warning_jobs"] == 1
        assert report["prompt_rewrite_jobs"] == 0
        entry = report["rewrites"][0]
        # No target to rewrite to — prompt left as-is, warning recorded.
        loaded = get_job(job["id"])
        assert "stale" in loaded["prompt"]
        assert entry["prompt_warnings"] == [
            {"name": "stale", "target": None, "count": 1},
        ]
        assert entry["prompt_rewrites"] == []

    def test_auto_rewrite_when_structured_field_already_clean(self, cron_env):
        """Job has no structural reference to the old skill, but its
        prompt does. The prompt mention is still auto-rewritten — the
        job's record gets saved purely for the prompt change.
        """
        from cron.jobs import create_job, get_job, rewrite_skill_refs

        job = create_job(
            prompt="Use `legacy-skill` for parsing — the old behavior.",
            schedule="every 1h",
            skills=["umbrella-skill"],
        )
        report = rewrite_skill_refs(
            consolidated={"legacy-skill": "umbrella-skill"},
            pruned=[],
        )

        assert report["jobs_updated"] == 1
        assert report["prompt_rewrite_jobs"] == 1
        assert report["prompt_warning_jobs"] == 0
        assert len(report["rewrites"]) == 1
        entry = report["rewrites"][0]
        # No structural change — skills field already pointed at umbrella.
        assert entry["mapped"] == {}
        assert entry["dropped"] == []
        assert entry["before"] == ["umbrella-skill"]
        assert entry["after"] == ["umbrella-skill"]
        # But the prompt was rewritten and the job's record persisted.
        loaded = get_job(job["id"])
        assert loaded["prompt"] == "Use `umbrella-skill` for parsing — the old behavior."
        assert entry["prompt_rewrites"] == [
            {"name": "legacy-skill", "target": "umbrella-skill", "count": 1},
        ]

    def test_word_boundary_does_not_false_match_substring(self, cron_env):
        """``legacy`` should NOT match inside ``my-legacy-skill`` or
        ``legacy_v2`` — those are different tokens that just share a
        prefix/embedded substring."""
        from cron.jobs import create_job, get_job, rewrite_skill_refs

        job = create_job(
            prompt="See my-legacy-skill and legacy_v2 for details.",
            schedule="every 1h",
            skills=["unrelated"],
        )
        report = rewrite_skill_refs(consolidated={"legacy": "umbrella"}, pruned=[])

        assert report["prompt_rewrite_jobs"] == 0
        assert report["prompt_warning_jobs"] == 0
        assert report["rewrites"] == []
        # Prompt untouched.
        loaded = get_job(job["id"])
        assert loaded["prompt"] == "See my-legacy-skill and legacy_v2 for details."

    def test_word_boundary_matches_tokenized_forms(self, cron_env):
        """Bare, backtick-wrapped, quoted, and trailing-punctuation forms
        should all count as a real mention and be rewritten."""
        from cron.jobs import create_job, get_job, rewrite_skill_refs

        prompt = (
            "legacy is the old way. "
            "Follow `legacy` as the canonical spec. "
            "Quote 'legacy' for backwards compat. "
            "End with legacy."
        )
        job = create_job(prompt=prompt, schedule="every 1h", skills=[])
        report = rewrite_skill_refs(consolidated={"legacy": "umbrella"}, pruned=[])

        assert report["prompt_rewrite_jobs"] == 1
        entry = report["rewrites"][0]
        assert entry["prompt_rewrites"] == [
            {"name": "legacy", "target": "umbrella", "count": 4},
        ]
        loaded = get_job(job["id"])
        assert loaded["prompt"] == (
            "umbrella is the old way. "
            "Follow `umbrella` as the canonical spec. "
            "Quote 'umbrella' for backwards compat. "
            "End with umbrella."
        )

    def test_mixed_consolidated_and_pruned_in_same_prompt(self, cron_env):
        """Consolidated mentions get rewritten; pruned mentions stay and
        become warnings — both reported in the same entry."""
        from cron.jobs import create_job, get_job, rewrite_skill_refs

        job = create_job(
            prompt="Use `old-a` first, then fall back to `old-b`. Avoid `gone`.",
            schedule="every 1h",
            skills=["old-a", "old-b"],
        )
        report = rewrite_skill_refs(
            consolidated={"old-a": "umbrella", "old-b": "umbrella"},
            pruned=["gone"],
        )

        entry = report["rewrites"][0]
        # Two consolidated names → both auto-rewritten, sorted by name.
        assert entry["prompt_rewrites"] == [
            {"name": "old-a", "target": "umbrella", "count": 1},
            {"name": "old-b", "target": "umbrella", "count": 1},
        ]
        # ``gone`` stays in the prompt and is warned about.
        assert entry["prompt_warnings"] == [
            {"name": "gone", "target": None, "count": 1},
        ]
        loaded = get_job(job["id"])
        assert loaded["prompt"] == (
            "Use `umbrella` first, then fall back to `umbrella`. Avoid `gone`."
        )

    def test_rewrite_is_single_pass_no_chain_through_targets(self, cron_env):
        """If a target is itself a key in ``consolidated``, the rewritten
        text is NOT re-rewritten — matches the single-pass behaviour of
        the structural skill-list rewrite (``a`` → ``b`` stays ``b``,
        even when ``b`` → ``c`` is also in the map)."""
        from cron.jobs import create_job, get_job, rewrite_skill_refs

        job = create_job(
            prompt="Use `a` as the canonical name.",
            schedule="every 1h",
            skills=[],
        )
        rewrite_skill_refs(
            consolidated={"a": "b", "b": "c"},
            pruned=[],
        )

        loaded = get_job(job["id"])
        # ``a`` → ``b``, but ``b`` (now in prompt) is not re-rewritten to ``c``.
        assert loaded["prompt"] == "Use `b` as the canonical name."


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
