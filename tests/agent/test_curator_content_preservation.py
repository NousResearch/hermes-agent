"""Tests for curator content-preservation rule and lossy-merge audit (issue #80791).

Covers:
- Prompt-level content preservation invariant in CURATOR_REVIEW_PROMPT and CURATOR_DRY_RUN_BANNER
- Deterministic byte-ratio audit comparing absorbed skill bytes vs umbrella added bytes
- run.json and REPORT.md reporting of lossy merges and store-health metrics
- hermes curator status output for store health
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest


@pytest.fixture
def curator_env(tmp_path, monkeypatch):
    """Isolated HERMES_HOME with skills/ and logs/ dirs + reloaded curator."""
    home = tmp_path / ".hermes"
    home.mkdir()
    (home / "skills").mkdir()
    (home / "skills" / ".archive").mkdir()
    (home / "logs").mkdir()
    (home / "logs" / "curator").mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)

    import importlib
    import hermes_constants
    importlib.reload(hermes_constants)
    from agent import curator
    importlib.reload(curator)
    from tools import skill_usage
    importlib.reload(skill_usage)

    return {"home": home, "curator": curator, "skill_usage": skill_usage}


def _make_llm_meta(**overrides):
    base = {
        "final": "short summary",
        "summary": "short summary",
        "model": "test-model",
        "provider": "test-provider",
        "tool_calls": [],
        "error": None,
    }
    base.update(overrides)
    return base


def test_curator_prompt_has_content_preservation_rule(curator_env):
    """Prompt must contain the mandatory content preservation rule from arXiv:2607.26637."""
    curator = curator_env["curator"]

    assert "Content preservation — not optional:" in curator.CURATOR_REVIEW_PROMPT
    assert "Consolidation must be lossless" in curator.CURATOR_REVIEW_PROMPT
    assert "arXiv:2607.26637" in curator.CURATOR_REVIEW_PROMPT
    assert "A one-line summary is NOT absorption" in curator.CURATOR_REVIEW_PROMPT
    assert "When in doubt, demote to `references/` verbatim" in curator.CURATOR_REVIEW_PROMPT

    # Hard rule #6
    assert "6. Consolidation MUST be lossless" in curator.CURATOR_REVIEW_PROMPT

    # Dry-run banner
    assert "Content-preservation rule still applies" in curator.CURATOR_DRY_RUN_BANNER


def test_lossy_merge_audit_flags_under_threshold(curator_env):
    """If added bytes to the umbrella are less than 40% of absorbed bytes, flag as lossy."""
    curator = curator_env["curator"]

    consolidated = [{"name": "detailed-docker", "into": "docker-umbrella"}]
    before_sizes = {
        "detailed-docker": 10000,
        "docker-umbrella": 2000,
    }

    # Simulate docker-umbrella on disk having 3000 bytes (added_bytes = 1000, which is 10% of 10000)
    home = curator_env["home"]
    umbrella_dir = home / "skills" / "docker-umbrella"
    umbrella_dir.mkdir(parents=True, exist_ok=True)
    (umbrella_dir / "SKILL.md").write_text("x" * 3000, encoding="utf-8")

    flagged = curator._audit_lossy_merges(consolidated, before_sizes=before_sizes, threshold=0.40)
    assert len(flagged) == 1
    assert consolidated[0]["lossy"] is True
    assert "possible lossy merge — review .archive/detailed-docker" in consolidated[0]["lossy_warning"]
    assert consolidated[0]["absorbed_bytes"] == 10000
    assert consolidated[0]["added_bytes"] == 1000
    assert consolidated[0]["retention_ratio"] == 0.1


def test_lossy_merge_audit_passes_lossless(curator_env):
    """If added bytes exceed the 40% threshold, it is not flagged."""
    curator = curator_env["curator"]

    consolidated = [{"name": "detailed-docker", "into": "docker-umbrella"}]
    before_sizes = {
        "detailed-docker": 1000,
        "docker-umbrella": 500,
    }

    # Simulate docker-umbrella on disk having 1400 bytes (added_bytes = 900, 90% >= 40%)
    home = curator_env["home"]
    umbrella_dir = home / "skills" / "docker-umbrella"
    umbrella_dir.mkdir(parents=True, exist_ok=True)
    (umbrella_dir / "SKILL.md").write_text("x" * 1400, encoding="utf-8")

    flagged = curator._audit_lossy_merges(consolidated, before_sizes=before_sizes, threshold=0.40)
    assert len(flagged) == 0
    assert consolidated[0]["lossy"] is False
    assert consolidated[0]["absorbed_bytes"] == 1000
    assert consolidated[0]["added_bytes"] == 900
    assert consolidated[0]["retention_ratio"] == 0.9


def test_lossy_merge_audit_from_archive_dir(curator_env):
    """When before_sizes is not provided, audit looks up archived skill in .archive/."""
    curator = curator_env["curator"]
    home = curator_env["home"]

    # Place archived package in .archive/python-tricks
    archived_dir = home / "skills" / ".archive" / "python-tricks"
    archived_dir.mkdir(parents=True, exist_ok=True)
    (archived_dir / "SKILL.md").write_text("y" * 5000, encoding="utf-8")
    refs_dir = archived_dir / "references"
    refs_dir.mkdir()
    (refs_dir / "tricks.md").write_text("z" * 5000, encoding="utf-8")
    # Total absorbed bytes = 10,000

    # Umbrella has only 500 bytes
    umbrella_dir = home / "skills" / "python-master"
    umbrella_dir.mkdir(parents=True, exist_ok=True)
    (umbrella_dir / "SKILL.md").write_text("w" * 500, encoding="utf-8")

    consolidated = [{"name": "python-tricks", "into": "python-master"}]
    flagged = curator._audit_lossy_merges(consolidated, before_sizes=None)

    assert len(flagged) == 1
    assert consolidated[0]["lossy"] is True
    assert consolidated[0]["absorbed_bytes"] == 10000
    assert consolidated[0]["added_bytes"] == 500
    assert consolidated[0]["retention_ratio"] == 0.05


def test_write_run_report_includes_lossy_merges_and_store_health(curator_env):
    """_write_run_report records lossy_merges and store_health in run.json and REPORT.md."""
    curator = curator_env["curator"]
    home = curator_env["home"]

    # Set up archived skill and tiny umbrella
    archived = home / "skills" / ".archive" / "git-rebase-wizard"
    archived.mkdir(parents=True, exist_ok=True)
    (archived / "SKILL.md").write_text("A" * 4000, encoding="utf-8")

    umbrella = home / "skills" / "git-umbrella"
    umbrella.mkdir(parents=True, exist_ok=True)
    (umbrella / "SKILL.md").write_text("B" * 200, encoding="utf-8")

    run_dir = curator._write_run_report(
        started_at=datetime.now(timezone.utc),
        elapsed_seconds=5.0,
        auto_counts={"checked": 1, "marked_stale": 0, "archived": 0, "reactivated": 0},
        auto_summary="no changes",
        before_report=[{"name": "git-rebase-wizard", "state": "active"}],
        before_names={"git-rebase-wizard"},
        after_report=[{"name": "git-umbrella", "state": "active"}],
        llm_meta=_make_llm_meta(
            final="```yaml\nconsolidations:\n  - from: git-rebase-wizard\n    into: git-umbrella\n    reason: merged\nprunings: []\n```"
        ),
        before_sizes={"git-rebase-wizard": 4000},
    )

    assert run_dir is not None
    run_json = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))

    # counts
    assert run_json["counts"]["consolidated_this_run"] == 1
    assert run_json["counts"]["lossy_merges"] == 1

    # store_health
    assert "store_health" in run_json
    health = run_json["store_health"]
    assert health["skills_absorbed"] == 1
    assert health["bytes_archived"] == 4000
    assert health["bytes_retained"] == 200
    assert health["lossy_merges_count"] == 1

    # lossy_merges detail
    assert len(run_json["lossy_merges"]) == 1
    assert run_json["lossy_merges"][0]["from"] == "git-rebase-wizard"
    assert run_json["lossy_merges"][0]["into"] == "git-umbrella"

    # REPORT.md contains callout and warnings
    report_md = (run_dir / "REPORT.md").read_text(encoding="utf-8")
    assert "Lossy-merge audit (arXiv:2607.26637)" in report_md
    assert "flagged as possible lossy merge" in report_md
    assert "possible lossy merge — review .archive/git-rebase-wizard" in report_md


def test_build_rename_summary_lossy_warning(curator_env):
    """_build_rename_summary includes warning about lossy merges."""
    curator = curator_env["curator"]
    home = curator_env["home"]

    umbrella = home / "skills" / "u"
    umbrella.mkdir(parents=True, exist_ok=True)
    (umbrella / "SKILL.md").write_text("x" * 10, encoding="utf-8")

    archived = home / "skills" / ".archive" / "s"
    archived.mkdir(parents=True, exist_ok=True)
    (archived / "SKILL.md").write_text("y" * 1000, encoding="utf-8")

    summary = curator._build_rename_summary(
        before_names={"s"},
        after_report=[{"name": "u", "state": "active"}],
        tool_calls=[],
        model_final="```yaml\nconsolidations:\n  - from: s\n    into: u\n    reason: test\nprunings: []\n```",
        before_sizes={"s": 1000},
    )

    assert "possible lossy merge(s) — review .archive/" in summary
