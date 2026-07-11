import json

import pytest

from hermes_cli import web_github


def success(stdout=""):
    return {"ok": True, "kind": "success", "stdout": stdout}


def test_search_args_and_validation():
    assert web_github.search_args("created", "open")[:4] == ["search", "prs", "--author", "@me"]
    assert "closed" in web_github.search_args("created", "closed")
    assert "--review-requested" in web_github.search_args("review-requested", "open")
    assert "100" in web_github.search_args("created", "open", 999)
    with pytest.raises(ValueError): web_github.validate_filter("review-requested", "closed")
    with pytest.raises(ValueError): web_github.validate_filter("raw", "open")


@pytest.mark.parametrize("repository", ["owner", "owner/repo/extra", "owner/repo;echo", "owner/../repo"])
def test_rejects_unsafe_repository(repository):
    with pytest.raises(ValueError): web_github.validate_ref(repository, 1)


def test_list_states_and_normalization():
    assert web_github.list_pull_requests("created", "open", runner=lambda _: {"ok": False, "kind": "missing", "stdout": ""})["authState"] == "gh-missing"
    calls = []
    def runner(args):
        calls.append(args)
        return success() if args[0] == "auth" else success(json.dumps([{"repository": {"nameWithOwner": "o/r"}, "number": 1, "title": "x", "url": "u", "state": "open"}, {"bad": True}]))
    result = web_github.list_pull_requests("created", "open", runner=runner)
    assert result["authState"] == "ready" and len(result["items"]) == 1
    assert calls[1][0:2] == ["search", "prs"]


def test_malformed_and_timeout_degrade_cleanly():
    def malformed(args): return success() if args[0] == "auth" else success("{")
    assert web_github.list_pull_requests("created", "open", runner=malformed)["authState"] == "error"
    def timeout(args): return success() if args[0] == "auth" else {"ok": False, "kind": "timeout", "stdout": ""}
    assert "timed out" in web_github.list_pull_requests("created", "open", runner=timeout)["error"]


def test_detail_is_fixed_and_merged():
    calls = []
    def runner(args):
        calls.append(args)
        return success(json.dumps({"number": 2, "title": "x", "url": "u", "state": "closed", "mergedAt": "now"}))
    detail = web_github.pull_request_detail("o/r", 2, runner=runner)
    assert calls == [["pr", "view", "2", "--repo", "o/r", "--json", web_github.DETAIL_FIELDS]]
    assert detail["state"] == "MERGED"
