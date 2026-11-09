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


def test_gitlab_mr_maps_onto_the_same_events():
    view = {"state": "opened", "sha": "f00d", "web_url": "https://gl.example/g/p/-/merge_requests/7"}
    notes = [
        {"id": 1, "system": False, "type": None, "author": {"username": "rev"}, "body": "please rename"},
        {"id": 2, "system": False, "type": "DiffNote", "author": {"username": "rev"}, "body": "typo",
         "position": {"new_path": "app.py", "new_line": 12}},
        {"id": 3, "system": True, "author": {"username": "rev"}, "body": "approved this merge request"},
        {"id": 4, "system": True, "author": {"username": "rev"}, "body": "added 1 commit"},
        {"id": 5, "system": False, "type": None, "author": {"username": "me"}, "body": "fixed"},
    ]
    manual = [{"id": 9, "status": "manual", "source": "merge_request_event"}]
    events = follow_pr.collect_events(follow_pr.normalize_gitlab(view, notes, manual), set(), "me")
    assert [k for k, _ in events] == ["comment:gl1", "review:gl3", "inline:gl2", "ci:f00d"]
    assert "app.py:12" in events[2][1] and "APPROVED" in events[1][1]
    assert "blocked" in events[3][1] and "failed" not in events[3][1]
    running = [{"id": 9, "status": "running", "source": "push"}]
    assert follow_pr.collect_events(follow_pr.normalize_gitlab(view, [], running), set(), "me") == []
