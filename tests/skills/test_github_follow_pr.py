"""Invariants for the github skill's PR follower (event diffing only; gh is not called)."""

import importlib.util
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[2] / "skills/software-development/github/scripts/follow_pr.py"
spec = importlib.util.spec_from_file_location("follow_pr", SCRIPT)
follow_pr = importlib.util.module_from_spec(spec)
spec.loader.exec_module(follow_pr)


def _snap(rollup, comments=(), sha="abc123"):
    return {"state": "OPEN", "headRefOid": sha, "comments": list(comments), "reviews": [],
            "inline": [], "statusCheckRollup": rollup}


def test_ci_verdict_reported_once_per_head_and_only_when_settled():
    running = [{"__typename": "CheckRun", "name": "tests", "status": "IN_PROGRESS", "conclusion": ""}]
    assert follow_pr.collect_events(_snap(running), set(), "me") == []

    red = [{"__typename": "CheckRun", "name": "tests", "status": "COMPLETED", "conclusion": "FAILURE"},
           {"__typename": "StatusContext", "context": "lint", "state": "SUCCESS"}]
    events = follow_pr.collect_events(_snap(red), set(), "me")
    assert [k for k, _ in events] == ["ci:abc123"]
    assert "failed" in events[0][1] and "tests" in events[0][1]
    assert follow_pr.collect_events(_snap(red), {"ci:abc123"}, "me") == []
    assert [k for k, _ in follow_pr.collect_events(_snap(red, sha="def456"), {"ci:abc123"}, "me")] == ["ci:def456"]


def test_own_comments_and_baselined_history_never_wake():
    mine = {"id": "c1", "author": {"login": "me"}, "body": "pushed a fix"}
    old = {"id": "c2", "author": {"login": "rev"}, "body": "old"}
    new = {"id": "c3", "author": {"login": "rev"}, "body": "please rename"}
    seen = follow_pr.baseline_keys(_snap([], [old]))
    events = follow_pr.collect_events(_snap([], [mine, old, new]), seen, "me")
    assert [k for k, _ in events] == ["comment:c3"]
