"""Optional self-report-pr skill: draft on the fork only; never merge origin."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SKILL_PATH = (
    ROOT
    / "optional-skills"
    / "software-development"
    / "self-report-pr"
    / "SKILL.md"
)
FIXTURE = ROOT / "tests" / "fixtures" / "issues-tiny.json"
CLUSTER = ROOT / ".verified-oss-loop" / "scripts" / "cluster-similar-issues.py"


def _body() -> str:
    return SKILL_PATH.read_text(encoding="utf-8")


def test_skill_file_exists():
    assert SKILL_PATH.is_file()


def test_frontmatter_name():
    content = _body()
    assert content.startswith("---")
    assert "name: self-report-pr" in content


def test_refuses_gh_pr_merge():
    body = _body()
    assert "gh pr merge" in body
    assert "Refuse" in body or "never merge" in body.lower()


def test_refuses_origin_nousresearch():
    body = _body()
    assert "NousResearch/hermes-agent" in body
    assert "github_writes=0" in body
    assert "kvnloo/hermes-agent" in body


def test_draft_only():
    body = _body().lower()
    assert "draft" in body
    assert "never merge" in body


def test_tiny_fixture_size():
    issues = json.loads(FIXTURE.read_text(encoding="utf-8"))
    assert len(issues) == 8
    assert len(issues) <= 64


def test_cluster_tiny_fixture():
    proc = subprocess.run(
        [sys.executable, str(CLUSTER), "--json", str(FIXTURE)],
        check=True,
        capture_output=True,
        text=True,
    )
    clusters = [set(c) for c in json.loads(proc.stdout)["clusters"]]
    assert any({101, 102, 107} <= s for s in clusters)
    assert any({103, 104} <= s for s in clusters)
    assert any(s == {108} for s in clusters)
    assert "cap 64" in _body()
