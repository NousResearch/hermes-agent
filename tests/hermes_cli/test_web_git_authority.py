"""Regression tests for the web-git GitHub authority boundary."""

import json

import pytest

from hermes_cli import web_git
from tools.github_capabilities import (
    GitHubAuthorizationReason,
    GitHubCapabilityError,
    GitHubOperation,
)


def _fake_gh(_cwd, args):
    if args == ["auth", "status"]:
        return True, ""
    if args == ["repo", "view", "--json", "nameWithOwner", "-q", ".nameWithOwner"]:
        return True, "owner/repository\n"
    if args == ["api", "repos/owner/repository"]:
        return True, json.dumps({
            "owner": {"login": "owner", "type": "Organization"},
            "organization": {"login": "owner"},
            "permissions": {"pull": True, "push": False},
        })
    return False, ""


def test_web_git_keeps_pr_authority_separate_from_code_write(monkeypatch):
    monkeypatch.setattr(web_git.shutil, "which", lambda _name: "gh")
    monkeypatch.setattr(web_git, "_gh", _fake_gh)

    report = web_git._github_ship_report(
        "C:/worktree",
        [GitHubOperation.PULL_REQUEST_CREATE, GitHubOperation.CONTENTS_WRITE],
    )

    assert report.for_operation(GitHubOperation.PULL_REQUEST_CREATE).effective
    contents = report.for_operation(GitHubOperation.CONTENTS_WRITE)
    assert not contents.effective
    assert contents.reason is GitHubAuthorizationReason.RESOURCE_NOT_ACCESSIBLE_UNKNOWN


def test_web_git_preflight_blocks_push_without_repository_write(monkeypatch):
    monkeypatch.setattr(web_git.shutil, "which", lambda _name: "gh")
    monkeypatch.setattr(web_git, "_gh", _fake_gh)

    with pytest.raises(GitHubCapabilityError, match="contents.write"):
        web_git._preflight_github_mutation(
            "C:/worktree",
            [GitHubOperation.CONTENTS_WRITE, GitHubOperation.GIT_DATA_REFS_WRITE],
        )
