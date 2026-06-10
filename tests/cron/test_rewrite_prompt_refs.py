"""Tests for prompt-text skill reference rewriting in rewrite_skill_refs.

Bug: #23398 — when the curator consolidates a skill, rewrite_skill_refs
updates the structured ``skills`` / ``skill`` fields but leaves stale skill
names inside the freeform ``prompt`` text.  This produces a silent consistency
gap: the job loads one skill while the prompt still names the old one.
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


class TestPromptTextRewriteConsolidation:
    """Consolidated skill names in prompt text should be rewritten."""

    def test_backtick_wrapped_reference_rewritten(self, cron_env):
        """Backtick-wrapped skill names should be rewritten automatically."""
        from cron.jobs import create_job, get_job, rewrite_skill_refs

        job = create_job(
            prompt="Follow `old-skill` as the canonical spec.",
            schedule="every 1h",
            skills=["old-skill"],
        )
        rewrite_skill_refs(
            consolidated={"old-skill": "umbrella-skill"},
            pruned=[],
        )

        loaded = get_job(job["id"])
        assert loaded["skills"] == ["umbrella-skill"]
        assert "`umbrella-skill`" in loaded["prompt"]
        assert "`old-skill`" not in loaded["prompt"]

    def test_multiple_backtick_refs_all_rewritten(self, cron_env):
        """Multiple backtick-wrapped references to the same old skill."""
        from cron.jobs import create_job, get_job, rewrite_skill_refs

        job = create_job(
            prompt="See `old-skill` for details. Always follow `old-skill`.",
            schedule="every 1h",
            skills=["old-skill"],
        )
        rewrite_skill_refs(
            consolidated={"old-skill": "new-skill"},
            pruned=[],
        )

        loaded = get_job(job["id"])
        assert "`new-skill`" in loaded["prompt"]
        assert "`old-skill`" not in loaded["prompt"]
        # Both occurrences replaced
        assert loaded["prompt"].count("`new-skill`") == 2

    def test_bare_word_reference_rewritten(self, cron_env):
        """A bare skill name that is not backtick-wrapped but matches exactly."""
        from cron.jobs import create_job, get_job, rewrite_skill_refs

        job = create_job(
            prompt="Load old-skill and execute it.",
            schedule="every 1h",
            skills=["old-skill"],
        )
        rewrite_skill_refs(
            consolidated={"old-skill": "new-skill"},
            pruned=[],
        )

        loaded = get_job(job["id"])
        assert "new-skill" in loaded["prompt"]
        assert "old-skill" not in loaded["prompt"]

    def test_partial_name_not_rewritten(self, cron_env):
        """A skill name that is a substring of a longer word should NOT be rewritten."""
        from cron.jobs import create_job, get_job, rewrite_skill_refs

        job = create_job(
            prompt="Follow `old-skill-v2` as the spec.",
            schedule="every 1h",
            skills=["old-skill"],
        )
        rewrite_skill_refs(
            consolidated={"old-skill": "new-skill"},
            pruned=[],
        )

        loaded = get_job(job["id"])
        # The skills list IS rewritten
        assert loaded["skills"] == ["new-skill"]
        # But the prompt reference `old-skill-v2` is NOT rewritten because
        # it's a different name (substring match should not rewrite)
        assert "`old-skill-v2`" in loaded["prompt"]

    def test_prompt_none_handled_gracefully(self, cron_env):
        """Jobs with no prompt should not crash."""
        from cron.jobs import create_job, get_job, rewrite_skill_refs

        job = create_job(
            prompt=None,
            schedule="every 1h",
            skills=["old-skill"],
        )
        rewrite_skill_refs(
            consolidated={"old-skill": "new-skill"},
            pruned=[],
        )

        loaded = get_job(job["id"])
        assert loaded["skills"] == ["new-skill"]


class TestPromptTextRewritePruning:
    """Pruned skill names in prompt text should be warned about."""

    def test_pruned_skill_in_prompt_generates_warning(self, cron_env, caplog):
        """When a pruned skill appears in prompt, report should note it."""
        import logging

        from cron.jobs import create_job, get_job, rewrite_skill_refs

        job = create_job(
            prompt="Follow `stale-skill` as the spec.",
            schedule="every 1h",
            skills=["stale-skill"],
        )
        with caplog.at_level(logging.WARNING, logger="cron.jobs"):
            report = rewrite_skill_refs(
                consolidated={},
                pruned=["stale-skill"],
            )

        loaded = get_job(job["id"])
        assert loaded["skills"] == []
        # Prompt text is NOT auto-rewritten for pruned skills
        # (there's no target to rewrite to), but the report should note it
        assert len(report["rewrites"]) == 1
        entry = report["rewrites"][0]
        assert "stale-skill" in entry.get("dropped", [])
        assert entry["prompt_warnings"] == [
            "prompt still references pruned skill 'stale-skill' (no umbrella target)"
        ]
        assert "prompt still references pruned skill 'stale-skill'" in caplog.text

    def test_pruned_skill_partial_name_does_not_warn(self, cron_env, caplog):
        """Pruned skill warnings should not fire for longer skill-name substrings."""
        import logging

        from cron.jobs import create_job, rewrite_skill_refs

        create_job(
            prompt="Follow `stale-skill-v2` as the spec.",
            schedule="every 1h",
            skills=["stale-skill"],
        )

        with caplog.at_level(logging.WARNING, logger="cron.jobs"):
            report = rewrite_skill_refs(
                consolidated={},
                pruned=["stale-skill"],
            )

        entry = report["rewrites"][0]
        assert entry["prompt_warnings"] == []
        assert "prompt still references pruned skill" not in caplog.text


class TestPromptTextRewriteReport:
    """Prompt rewrites should be recorded in the report."""

    def test_report_includes_prompt_rewrites(self, cron_env):
        from cron.jobs import create_job, rewrite_skill_refs

        job = create_job(
            prompt="Follow `old-skill` as the spec.",
            schedule="every 1h",
            skills=["old-skill"],
        )
        report = rewrite_skill_refs(
            consolidated={"old-skill": "umbrella-skill"},
            pruned=[],
        )

        assert len(report["rewrites"]) == 1
        entry = report["rewrites"][0]
        assert entry.get("prompt_rewritten") is True

    def test_report_no_prompt_rewrite_when_no_prompt_refs(self, cron_env):
        """Job has consolidated skill but prompt doesn't reference it."""
        from cron.jobs import create_job, rewrite_skill_refs

        job = create_job(
            prompt="Do something unrelated.",
            schedule="every 1h",
            skills=["old-skill"],
        )
        report = rewrite_skill_refs(
            consolidated={"old-skill": "umbrella-skill"},
            pruned=[],
        )

        entry = report["rewrites"][0]
        assert entry.get("prompt_rewritten") is False


class TestPromptTextRewritePersistence:
    """Prompt rewrites persist to disk."""

    def test_prompt_rewrite_persists_to_disk(self, cron_env):
        import json
        from cron.jobs import create_job, rewrite_skill_refs, JOBS_FILE

        create_job(
            prompt="Follow `old-skill` as the canonical spec.",
            schedule="every 1h",
            skills=["old-skill"],
        )
        rewrite_skill_refs(
            consolidated={"old-skill": "umbrella-skill"},
            pruned=[],
        )

        data = json.loads(JOBS_FILE.read_text())
        assert data["jobs"][0]["skills"] == ["umbrella-skill"]
        assert "`umbrella-skill`" in data["jobs"][0]["prompt"]
        assert "`old-skill`" not in data["jobs"][0]["prompt"]
