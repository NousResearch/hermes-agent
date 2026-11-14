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


def test_paginated_pages_join_into_one_list():
    assert follow_pr._json_pages('[{"id": 1}]\n[{"id": 2}][{"id": 3}]') == [{"id": 1}, {"id": 2}, {"id": 3}]
    assert follow_pr._json_pages('{"login": "me"}') == {"login": "me"}


def test_own_comments_and_baselined_history_never_wake():
    mine = {"id": "c1", "author": {"login": "me"}, "body": "pushed a fix"}
    old = {"id": "c2", "author": {"login": "rev"}, "body": "old"}
    new = {"id": "c3", "author": {"login": "rev"}, "body": "please rename"}
    seen = follow_pr.baseline_keys(_snap([], [old]))
    events = follow_pr.collect_events(_snap([], [mine, old, new]), seen, "me")
    assert [k for k, _ in events] == ["comment:c3"]


def test_gitlab_mr_maps_onto_the_same_events(monkeypatch):
    head = {"id": 9, "sha": "f00d", "status": "manual", "source": "merge_request_event"}
    view = {"state": "opened", "sha": "f00d", "web_url": "https://gl.example/g/p/-/merge_requests/7",
            "head_pipeline": head}
    notes = [
        {"id": 1, "system": False, "type": None, "author": {"username": "rev"}, "body": "please rename"},
        {"id": 2, "system": False, "type": "DiffNote", "author": {"username": "rev"}, "body": "typo",
         "position": {"new_path": "app.py", "new_line": 12}},
        {"id": 3, "system": True, "author": {"username": "rev"}, "body": "approved this merge request"},
        {"id": 4, "system": True, "author": {"username": "rev"}, "body": "added 1 commit"},
        {"id": 5, "system": False, "type": None, "author": {"username": "me"}, "body": "fixed"},
    ]
    events = follow_pr.collect_events(follow_pr.normalize_gitlab(view, notes), set(), "me")
    assert [k for k, _ in events] == ["comment:gl1", "review:gl3", "inline:gl2", "ci:f00d"]
    assert "app.py:12" in events[2][1] and "APPROVED" in events[1][1]
    assert "blocked" in events[3][1] and "failed" not in events[3][1]
    running = dict(view, head_pipeline=dict(head, status="running"))
    assert follow_pr.collect_events(follow_pr.normalize_gitlab(running, []), set(), "me") == []

    # The token belongs to one host: an MR link elsewhere never receives it.
    monkeypatch.setenv("GITLAB_TOKEN", "glpat-secret")
    monkeypatch.setenv("GITLAB_HOST", "https://gitlab.example.com/")
    assert follow_pr.gitlab_token("gitlab.example.com") == "glpat-secret"
    assert follow_pr.gitlab_token("evil.example") == ""
