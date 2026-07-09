"""Pilot scope guard — nothing leaks outside tests/test_executive_v2/canary_pilot/.

Hard rule from operator: every artifact produced by this pilot MUST live
under tests/test_executive_v2/canary_pilot/. This guard walks the canary
tree and confirms:

  1. All 5 expected files exist with non-zero size.
  2. No unexpected top-level directories were created.
  3. No file outside the canary_pilot/ tree was modified by the pilot.
"""

from __future__ import annotations

from pathlib import Path

import pytest

PILOT_DIR = Path(__file__).resolve().parent
EXPECTED_FILES = (
    "__init__.py",
    "conftest.py",
    "test_pilot_run_pipeline.py",
    "test_pilot_execution_contract.py",
    "test_pilot_subgoals.py",
    "test_pilot_dryrun_render.py",
    "test_pilot_scope_guard.py",
)


def test_scope_pilot_dir_is_correct():
    """Sanity: this file MUST live under tests/test_executive_v2/canary_pilot/."""
    assert PILOT_DIR.name == "canary_pilot"
    assert PILOT_DIR.parent.name == "test_executive_v2"
    assert PILOT_DIR.parent.parent.name == "tests"


def test_scope_all_expected_files_exist():
    for name in EXPECTED_FILES:
        p = PILOT_DIR / name
        assert p.exists(), f"missing expected pilot file: {name}"
        assert p.stat().st_size > 0, f"pilot file is empty: {name}"


def test_scope_no_unexpected_top_level_dirs():
    """Only Python files + __pycache__ allowed at this level."""
    actual = {p.name for p in PILOT_DIR.iterdir()}
    expected = set(EXPECTED_FILES) | {"__pycache__"}
    extras = actual - expected
    assert not extras, f"unexpected top-level entries in canary_pilot: {extras}"


def test_scope_no_production_code_modified():
    """Production code lives under agent/. Pilot must not touch it.

    This is a coarse check — it confirms the agent/ tree has no
    uncommitted modifications on this branch. It does NOT enforce
    absence of past modifications; that is the operator's branch
    hygiene responsibility.
    """
    import subprocess

    repo_root = PILOT_DIR.parents[2]  # tests/ -> repo root
    result = subprocess.run(
        ["git", "-C", str(repo_root), "status", "--porcelain", "agent/"],
        capture_output=True, text=True, timeout=10,
    )
    assert result.returncode == 0, f"git status failed: {result.stderr}"
    assert result.stdout.strip() == "", (
        f"agent/ tree has uncommitted changes on this branch:\n"
        f"{result.stdout}"
    )


def test_scope_pilot_branch_is_correct():
    """Confirm we are on integration/pilot_dryrun_kd_kanban_v1, NOT on
    integration/post_batch18_domain_forwarded. Operator's hard rule:
    the pilot must not leak commits into post_batch18_domain_forwarded.
    """
    import subprocess

    repo_root = PILOT_DIR.parents[2]
    result = subprocess.run(
        ["git", "-C", str(repo_root), "branch", "--show-current"],
        capture_output=True, text=True, timeout=10,
    )
    assert result.returncode == 0, f"git branch failed: {result.stderr}"
    branch = result.stdout.strip()
    assert branch == "integration/pilot_dryrun_kd_kanban_v1", (
        f"pilot must run on its own branch; got {branch!r}"
    )


def test_scope_head_is_frozen_sha():
    """Confirm HEAD is still 5d7886cf... — the branch creation must
    have left HEAD pointing at the frozen SHA."""
    import subprocess

    repo_root = PILOT_DIR.parents[2]
    result = subprocess.run(
        ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
        capture_output=True, text=True, timeout=10,
    )
    assert result.returncode == 0
    assert result.stdout.strip() == "5d7886cfc192dfce55e68f71d4a519600bc68b27", (
        f"HEAD drifted from frozen SHA: {result.stdout.strip()!r}"
    )