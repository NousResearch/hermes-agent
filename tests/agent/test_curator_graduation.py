"""Regression tests for detect_graduation_candidates() -- Stage 1 (kn79130).

Fixture layout:
  - Signature A (reid::code): 3 success runs (the qualifying signature)
  - Signature B (cal::research): 2 unrelated runs (1 success, 1 failure -- fails stability)
  - Signature C (grant::review): 3 success runs but name matches existing skill "grant-review"
    (bigram similarity >= 0.7 -- should be filtered out)

Only signature A should be returned.
"""

from __future__ import annotations

import importlib
import json
import sqlite3
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_ale_db(path: Path) -> None:
    """Create a minimal ALE database with synthetic fixture rows."""
    conn = sqlite3.connect(str(path))
    conn.execute("""
        CREATE TABLE runs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id TEXT UNIQUE NOT NULL,
            agent TEXT NOT NULL,
            model TEXT NOT NULL DEFAULT 'test-model',
            task_type TEXT,
            task_summary TEXT NOT NULL DEFAULT '',
            task_prompt_hash TEXT,
            skill_code TEXT,
            started_at TEXT NOT NULL,
            completed_at TEXT,
            duration_seconds INTEGER,
            status TEXT NOT NULL DEFAULT 'running',
            tokens_in INTEGER DEFAULT 0,
            tokens_out INTEGER DEFAULT 0,
            score REAL,
            reviewer TEXT,
            review_notes TEXT,
            error_type TEXT,
            files_read TEXT,
            files_written TEXT,
            convex_deployment TEXT,
            git_commit TEXT,
            learnings TEXT,
            prompt_adjustments TEXT,
            created_at TEXT DEFAULT (datetime('now'))
        )
    """)

    rows = [
        # Signature A: reid::code -- 3 successes => qualifying
        ("run-a1", "reid", "code", "success", "2026-05-01T10:00:00Z"),
        ("run-a2", "reid", "code", "success", "2026-05-10T10:00:00Z"),
        ("run-a3", "reid", "code", "success", "2026-05-20T10:00:00Z"),
        # Signature B: cal::research -- 1 success + 1 failure
        # Last run (most recent) is failure => fails stability_window check
        ("run-b1", "cal", "research", "success", "2026-05-05T10:00:00Z"),
        ("run-b2", "cal", "research", "failure", "2026-05-15T10:00:00Z"),
        # Signature C: grant::review -- 3 successes but existing skill matches name
        ("run-c1", "grant", "review", "success", "2026-05-02T10:00:00Z"),
        ("run-c2", "grant", "review", "success", "2026-05-12T10:00:00Z"),
        ("run-c3", "grant", "review", "success", "2026-05-22T10:00:00Z"),
    ]

    conn.executemany(
        "INSERT INTO runs (run_id, agent, task_type, status, started_at) VALUES (?,?,?,?,?)",
        rows,
    )
    conn.commit()
    conn.close()


def _make_skills_dir(path: Path) -> None:
    """Create a skills directory with one SKILL.md that matches grant-review."""
    skill_dir = path / "grant-review"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        "---\nname: grant-review\ndescription: Grant review workflow\n---\n",
        encoding="utf-8",
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def graduation_env(tmp_path, monkeypatch):
    """Isolated environment for graduation detector tests."""
    home = tmp_path / "home"
    home.mkdir()

    # Patch Path.home() so hermes_constants resolves to our tmp dir
    monkeypatch.setattr(Path, "home", lambda: home)
    monkeypatch.setenv("HERMES_HOME", str(home / ".hermes"))

    # Create ALE db
    ale_dir = home / ".hermes" / "workspace" / "agents" / "ale"
    ale_dir.mkdir(parents=True)
    ale_db = ale_dir / "ale.db"
    _make_ale_db(ale_db)

    # Create skills dir with one matching skill
    skills_root = home / ".hermes" / "skills"
    skills_root.mkdir(parents=True)
    _make_skills_dir(skills_root)

    # Reload modules so they pick up patched Path.home()
    import hermes_constants
    importlib.reload(hermes_constants)
    import agent.curator as curator
    importlib.reload(curator)

    return {
        "curator": curator,
        "ale_db": ale_db,
        "skills_root": skills_root,
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_only_qualifying_signature_returned(graduation_env):
    """Only reid::code satisfies all three conditions."""
    curator = graduation_env["curator"]
    ale_db = graduation_env["ale_db"]
    skills_root = graduation_env["skills_root"]

    results = curator.detect_graduation_candidates(
        ale_db_path=ale_db,
        skills_root=skills_root,
        min_runs=2,
        stability_window=3,
        similarity_threshold=0.7,
    )

    assert len(results) == 1, f"Expected 1 candidate, got {len(results)}: {results}"
    r = results[0]
    assert r["workflow_signature"] == "reid::code"
    assert r["run_count"] == 3
    assert set(r["last_3_run_ids"]) == {"run-a1", "run-a2", "run-a3"}
    # Chronological (ascending) order
    assert r["last_3_run_ids"] == ["run-a1", "run-a2", "run-a3"]
    assert isinstance(r["candidate_skill_name"], str)
    assert isinstance(r["detection_ts"], str)


def test_signature_b_excluded_due_to_failure(graduation_env):
    """cal::research has a failure in its most recent runs -- must not appear."""
    curator = graduation_env["curator"]
    ale_db = graduation_env["ale_db"]
    skills_root = graduation_env["skills_root"]

    results = curator.detect_graduation_candidates(
        ale_db_path=ale_db,
        skills_root=skills_root,
        min_runs=2,
        stability_window=3,
        similarity_threshold=0.7,
    )
    sigs = [r["workflow_signature"] for r in results]
    assert "cal::research" not in sigs


def test_signature_c_excluded_due_to_existing_skill(graduation_env):
    """grant::review has 3 successes but bigram similarity >= 0.7 to 'grant-review' skill."""
    curator = graduation_env["curator"]
    ale_db = graduation_env["ale_db"]
    skills_root = graduation_env["skills_root"]

    # Verify the similarity calculation itself is >= 0.7
    sim = curator._bigram_similarity("grant-review", "grant-review")
    assert sim >= 0.7, f"Expected >= 0.7, got {sim}"

    results = curator.detect_graduation_candidates(
        ale_db_path=ale_db,
        skills_root=skills_root,
        min_runs=2,
        stability_window=3,
        similarity_threshold=0.7,
    )
    sigs = [r["workflow_signature"] for r in results]
    assert "grant::review" not in sigs


def test_missing_ale_db_returns_empty(graduation_env):
    """When ale.db doesn't exist, return empty list gracefully."""
    curator = graduation_env["curator"]
    fake_path = Path("/nonexistent/ale.db")

    results = curator.detect_graduation_candidates(ale_db_path=fake_path)
    assert results == []


def test_idempotent_same_output(graduation_env):
    """Two calls with same input must produce identical results."""
    curator = graduation_env["curator"]
    ale_db = graduation_env["ale_db"]
    skills_root = graduation_env["skills_root"]

    r1 = curator.detect_graduation_candidates(
        ale_db_path=ale_db, skills_root=skills_root
    )
    r2 = curator.detect_graduation_candidates(
        ale_db_path=ale_db, skills_root=skills_root
    )

    # Strip detection_ts for comparison (timestamp varies between calls)
    def strip_ts(lst):
        return [{k: v for k, v in d.items() if k != "detection_ts"} for d in lst]

    assert strip_ts(r1) == strip_ts(r2)


def test_write_graduation_queue(graduation_env, tmp_path, monkeypatch):
    """write_graduation_queue writes valid JSONL and returns path."""
    curator = graduation_env["curator"]
    ale_db = graduation_env["ale_db"]
    skills_root = graduation_env["skills_root"]

    candidates = curator.detect_graduation_candidates(
        ale_db_path=ale_db, skills_root=skills_root
    )

    # Redirect the module-level output path to our tmp dir
    out_path = tmp_path / "graduation-queue.jsonl"
    monkeypatch.setattr(curator, "_GRADUATION_QUEUE_PATH", out_path)

    result_path = curator.write_graduation_queue(candidates)

    assert result_path == out_path
    assert out_path.exists()
    lines = out_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == len(candidates)
    for line in lines:
        obj = json.loads(line)
        assert "workflow_signature" in obj
        assert "run_count" in obj
        assert "last_3_run_ids" in obj
        assert "candidate_skill_name" in obj
        assert "detection_ts" in obj
