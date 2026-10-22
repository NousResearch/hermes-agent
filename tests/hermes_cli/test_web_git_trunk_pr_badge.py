"""The trunk is never badged with a PR whose head is the trunk.

Failure caught: a long-closed own-repo PR opened FROM `main` (main -> a baseline
branch) badged every session on `main` with "#2", linking a dead PR. By-branch
and by-number lookups both drop a PR whose head is the repository's default
branch; PRs on other branches still resolve.
"""

import json
from unittest.mock import patch

from hermes_cli import web_git


def _node(number: int, head: str, state: str = "OPEN") -> dict:
    return {"number": number, "state": state, "isDraft": False, "isCrossRepository": False,
            "title": f"PR {number}", "url": f"https://github.com/acme/widgets/pull/{number}",
            "headRefName": head}


def _fake_gh(cwd: str, args: list[str]):
    if args[:2] == ["repo", "view"]:
        return True, "acme/widgets\n", ""
    query = next(a for a in args if a.startswith("query="))
    if "n0:" in query:
        repository = {"defaultBranchRef": {"name": "main"}, "n0": _node(2, "main", "CLOSED")}
    else:
        repository = {"defaultBranchRef": {"name": "main"},
                      "b0": {"nodes": [_node(2, "main", "CLOSED")]},
                      "b1": {"nodes": [_node(7, "feature")]}}
    return True, json.dumps({"data": {"repository": repository}}), ""


def test_a_pr_opened_from_the_trunk_never_badges_the_trunk(tmp_path):
    with patch.object(web_git, "_gh", side_effect=_fake_gh):
        result = web_git.review_pr_list(str(tmp_path), ["main", "feature"], [2])

    assert [(pr["number"], pr["branch"]) for pr in result["prs"]] == [(7, "feature")]
