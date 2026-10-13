"""Typed, secret-free Kanban PR-acceptance failures on the profile-scoped gh path.

Identity refusals stay ``auth`` (HTTP 401/403/404, exit 4, a GraphQL NOT_FOUND
repository). A rate-limit 403 is ``rate_limited`` even though it also names
HTTP 403. stderr never reaches the receipt. ``gh`` is still invoked with the
profile child env.
"""
import json
import subprocess

import pytest

_PR_URL = "https://github.com/acme/repo/pull/7"


def _receipt(monkeypatch, *, stdout="", stderr=""):
    import hermes_cli.kanban_pr_acceptance as acc

    calls = []

    def run(command, **kwargs):
        calls.append((command, kwargs))
        raise subprocess.CalledProcessError(returncode=1, cmd=command, output=stdout, stderr=stderr)

    monkeypatch.setattr(acc.subprocess, "run", run)
    return acc.collect_acceptance("acme/repo", _PR_URL), calls


@pytest.mark.parametrize("stderr,expected,detail_snip", [
    ("gh: HTTP 403: API rate limit exceeded for 1.2.3.4", "rate_limited", "rate limit"),
    ("To get started with GitHub CLI, please run: gh auth login", "auth", "no login"),
    ('Post "https://api.github.com/graphql": dial tcp 127.0.0.1:9: connect: connection refused', "network", "network"),
])
def test_stderr_classifies_without_persisting_stderr(monkeypatch, stderr, expected, detail_snip):
    token = "ghp_" + "q" * 36
    receipt, calls = _receipt(monkeypatch, stderr=f"{stderr}; GH_TOKEN={token}")
    blob = json.dumps(receipt)
    assert receipt["classification"] == expected
    assert detail_snip in receipt["detail"]
    assert token not in blob
    assert stderr not in blob
    assert "gh: " not in blob
    assert calls and "env" in calls[0][1]


def test_graphql_not_found_repository_is_auth(monkeypatch):
    """Real gh exits 1 for a repository the login cannot read: HTTP 200 + NOT_FOUND,
    so stderr has no ``HTTP 404``. That is an identity problem, not provider_error."""
    stderr = "gh: Could not resolve to a Repository with the name 'acme/repo'."
    body = {"data": {"repository": None},
            "errors": [{"type": "NOT_FOUND", "path": ["repository"], "message": stderr[4:]}]}
    receipt, _calls = _receipt(monkeypatch, stdout=json.dumps(body), stderr=stderr)
    assert receipt["classification"] == "auth"
    assert "acme/repo" in receipt["detail"]
    assert "the ambient gh login" in receipt["detail"]
    assert "Could not resolve" not in json.dumps(receipt)


@pytest.mark.parametrize("error_type,path,repository,expected", [
    ("NOT_FOUND", ["repository", "pullRequest"], {"pullRequest": None}, "missing"),
    ("FORBIDDEN", ["repository"], None, "auth"),
    ("BAD_CREDENTIALS", [], None, "auth"),
])
def test_graphql_target_refusals_are_actionable(monkeypatch, error_type, path, repository, expected):
    body = {"data": {"repository": repository},
            "errors": [{"type": error_type, "path": path, "message": "private provider detail"}]}
    stderr = "gh: Bad credentials" if error_type == "BAD_CREDENTIALS" else "gh: private provider detail"
    receipt, _calls = _receipt(monkeypatch, stdout=json.dumps(body), stderr=stderr)
    assert receipt["classification"] == expected
    assert "private provider detail" not in json.dumps(receipt)
