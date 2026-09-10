"""Explicit PR URLs retain repository identity across the remote git route."""

import json
import threading

from fastapi import FastAPI
from fastapi.testclient import TestClient

from hermes_cli import web_git
from hermes_cli.web_routers.git import router


def test_url_only_route_hydrates_valid_github_identities_without_checkout(monkeypatch):
    urls = [f"https://github.com/owner/repo-{i}/pull/42" for i in range(9)]
    urls += ["https://github.com/actions/.github/pull/221", "https://github.com/owner/.config/pull/42/"]
    calls = []
    lock = threading.Lock()
    overlapped = threading.Event()
    active = peak = 0

    def gh(cwd, args):
        nonlocal active, peak
        assert cwd is None
        assert args[:2] == ["pr", "view"]
        with lock:
            calls.append(args)
            active += 1
            peak = max(peak, active)
            if active > 1:
                overlapped.set()
        assert overlapped.wait(5), "URL reads never overlapped"
        with lock:
            active -= 1
        return True, json.dumps({"headRefName": "feature", "isDraft": True,
                                 "number": 42, "state": "MERGED", "title": args[2], "url": args[2]})

    monkeypatch.setattr(web_git, "_gh", gh)
    app = FastAPI()
    app.include_router(router)
    with TestClient(app) as client:
        response = client.post("/api/git/review/pr-list", json={
            "path": "", "branches": [], "numbers": [], "urls": [
                *urls, urls[0], "https://evil.example/owner/repo/pull/42",
                "https://github.com.evil.example/owner/repo/pull/42",
                "https://github.com@evil.example/owner/repo/pull/42",
                "http://github.com/owner/repo/pull/42", "https://github.com/owner/repo/pull/0",
                "https://github.com/owner/./pull/42", "https://github.com/owner/../pull/42",
                "https://github.com/./repo/pull/42", "https://github.com/../repo/pull/42",
                "https://github.com/owner/%2e/pull/42", "https://github.com/owner/%2e%2e/pull/42",
                "https://github.com/owner/.github/../repo/pull/42", "https://github.com/owner/repo/pull/42\n",
            ],
        })
    assert response.status_code == 200, response.text
    result = response.json()
    assert result["ghReady"] is True
    assert [pr["url"] for pr in result["prs"]] == urls
    assert all(pr["state"] == "merged" and pr["draft"] for pr in result["prs"])
    assert sorted(args[2] for args in calls) == sorted(urls)
    assert all(args[3] == "--json" and "headRefName" in args[4] for args in calls)
    assert 1 < peak <= 4


def test_failed_url_reads_are_not_authoritative_but_legacy_results_survive(tmp_path, monkeypatch):
    known = {"headRefName": "feature", "number": 42, "state": "OPEN",
             "url": "https://github.com/owner/repo/pull/42"}
    queries = []

    def gh(cwd, args):
        if args[0] == "repo":
            return True, "owner/repo"
        if args[0] == "api":
            queries.append(args[3])
            return True, json.dumps({"data": {"repository": {"b0": {"nodes": [known]}, "n0": known}}})
        return True, "not JSON"

    monkeypatch.setattr(web_git, "_gh", gh)
    legacy = web_git.review_pr_list(str(tmp_path), ["feature"], [42])
    assert legacy["ghReady"] is True
    assert known["url"] in [pr["url"] for pr in legacy["prs"]]
    assert 'headRefName: "feature"' in "\n".join(queries)
    assert 'pullRequest(number: 42)' in "\n".join(queries)
    failed = web_git.review_pr_list(str(tmp_path), ["feature"], [42], ["https://github.com/other/repo/pull/42"])
    assert known["url"] in [pr["url"] for pr in failed["prs"]]
    assert failed["ghReady"] is False
    monkeypatch.setattr(web_git, "_gh", lambda *_: (False, "network unavailable"))
    assert web_git.review_pr_list("", [], [], [known["url"]]) == {"ghReady": False, "prs": []}
