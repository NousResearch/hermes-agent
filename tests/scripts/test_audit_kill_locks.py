"""Regression tests for the KILL LOCK audit (scripts/audit_kill_locks.py).

The audit is pure — it takes dicts, returns a report. These tests exercise
every invariant offline with fixtures: PR→issue keyword linkage, issue→PR
thread tokens, dedup both directions, resolution convergence, and the
'Progress on is not a keyword' rule.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import audit_kill_locks as akl  # noqa: E402


def _pr(number: int, title: str, body: str, state: str = "OPEN") -> dict:
    return {"number": number, "title": title, "body": body, "state": state, "mergedAt": None}


def test_pr_keyword_bindings_recognizes_github_keywords():
    body = (
        "Part of #78647\n"
        "Part of #78631\n"
        "Fixes #12345\n"
        "Closes #67890\n"
    )
    assert akl.pr_keyword_bindings(body) == {78647, 78631, 12345, 67890}


def test_progress_on_is_not_a_keyword():
    """'Progress on #N' reads fine to humans and links NOTHING for GitHub."""
    body = "Progress on #78631 — this PR advances the shard."
    assert akl.pr_keyword_bindings(body) == set()


def test_related_footer_is_not_a_binding():
    """The native-links footer ('Related #78791 #78792') is loose, not a bind."""
    body = "<!-- native-links:v1 --> Related #78791 #78792"
    assert akl.pr_keyword_bindings(body) == set()


def test_issue_thread_tokens_extracts_literal_hash_prs():
    comments = [
        {"body": "Interlock: resolved by #79844. Part of #78647."},
        {"body": "Scoreboard posted — #79845 #79846."},
    ]
    assert akl.issue_thread_tokens(comments) == {79844, 78647, 79845, 79846}


def test_full_audit_passes_with_complete_interlock():
    shard_issues = {78631: "Shard hermes_cli/main.py"}
    prs = [
        _pr(79844, "refactor(main): slice R1", "Part of #78647\nPart of #78631"),
        _pr(79845, "refactor(main): slice R2", "Part of #78647\nPart of #78631"),
    ]
    comments = {
        78631: [{"body": "#79844 #79845 — scoreboard. Part of #78647."}],
    }
    report = akl.audit("epic", shard_issues, comments, prs)
    assert report["verdict"] == "PASS"
    assert not report["issue_to_pr_holes"]
    assert not report["unresolved_shards"]


def test_full_audit_detects_thread_token_hole():
    """Keyword binding exists but the issue thread never got the #PR token."""
    shard_issues = {78631: "Shard hermes_cli/main.py"}
    prs = [_pr(79844, "refactor(main): slice R1", "Part of #78647\nPart of #78631")]
    comments = {78631: [{"body": "Scoreboard posted (prose, no token)."}]}
    report = akl.audit("epic", shard_issues, comments, prs)
    assert report["verdict"] == "HOLES"
    assert report["issue_to_pr_holes"] == [{"issue": 78631, "missing": [79844]}]


def test_full_audit_detects_unresolved_shard():
    """A shard issue with zero binding PRs is unresolved — the GAP-a class."""
    shard_issues = {78631: "Shard hermes_cli/main.py"}
    prs: list[dict] = []
    report = akl.audit("epic", shard_issues, {}, prs)
    assert report["verdict"] == "HOLES"
    assert report["unresolved_shards"][0]["issue"] == 78631


def test_audit_detects_duplicate_prs():
    """Two PRs shipping the same title = duplicate PRs (dedup both directions)."""
    shard_issues = {78631: "Shard hermes_cli/main.py"}
    prs = [
        _pr(79844, "refactor(main): extract oneshot hard-exit", "Part of #78631"),
        _pr(79899, "refactor(main): extract oneshot hard-exit", "Part of #78631"),
    ]
    report = akl.audit("epic", shard_issues, {78631: []}, prs)
    assert report["verdict"] == "HOLES"
    assert any(a == 79844 and b == 79899 for a, b, _ in report["pr_dups"])


def test_audit_detects_duplicate_issues():
    """Two shard issues with the same title = duplicate issues."""
    shard_issues = {
        78631: "Shard hermes_cli/main.py",
        78999: "Shard hermes_cli/main.py",
    }
    report = akl.audit("epic", shard_issues, {}, [])
    assert report["verdict"] == "HOLES"
    assert any(a == 78631 and b == 78999 for a, b, _ in report["issue_dups"])


def test_audit_reports_convergence_toward_resolution():
    """Bound shards list their PRs in the converged set — the completion record."""
    shard_issues = {78631: "Shard hermes_cli/main.py"}
    prs = [_pr(79844, "refactor(main): slice R1", "Part of #78647\nPart of #78631")]
    report = akl.audit("epic", shard_issues, {78631: [{"body": "#79844"}]}, prs)
    converged = report["converged"]
    assert converged[0]["issue"] == 78631
    assert converged[0]["prs"] == [79844]
