"""Typed, secret-free Kanban PR-acceptance failures on the profile-scoped gh path.

Identity refusals stay ``auth`` (HTTP 401/403/404, exit 4, GraphQL bad
credentials). A rate-limit 403 is ``rate_limited`` even though it also names
HTTP 403. stderr never reaches the receipt. ``gh`` is still invoked with the
profile child env.
"""
import json
import subprocess

import pytest

_PR_URL = "https://github.com/acme/repo/pull/7"


def _stub_run(monkeypatch, *, stderr="", exc_factory=None, returns=None):
    import hermes_cli.kanban_pr_acceptance as acc

    calls = []

    def record(command, **kwargs):
        calls.append((command, kwargs))
        if exc_factory is not None:
            raise exc_factory(command)
        if returns is not None:
            return returns
        raise subprocess.CalledProcessError(returncode=1, cmd=command, output="", stderr=stderr)

    monkeypatch.setattr(acc.subprocess, "run", record)
    return calls


def _receipt(monkeypatch, **stub):
    import hermes_cli.kanban_pr_acceptance as acc

    calls = _stub_run(monkeypatch, **stub)
    return acc.collect_acceptance("acme/repo", _PR_URL), calls


def test_missing_gh_binary_is_capability(monkeypatch):
    import hermes_cli.kanban_pr_acceptance as acc

    def missing(command, **kwargs):
        raise FileNotFoundError(command[0])

    monkeypatch.setattr(acc.subprocess, "run", missing)
    receipt = acc.collect_acceptance("acme/repo", _PR_URL)
    assert receipt["classification"] == "capability"
    assert receipt["head_sha"] is None
    assert "gh CLI" in receipt["detail"]
    assert receipt["pr_url"] == _PR_URL


@pytest.mark.parametrize("stderr,expected,detail_snip", [
    ("gh: HTTP 403: API rate limit exceeded for 1.2.3.4", "rate_limited", "rate limit"),
    ("gh: Bad credentials (HTTP 401)", "auth", "HTTP 401"),
    ("gh: Not Found (HTTP 404)", "auth", "HTTP 404"),
    ("To get started with GitHub CLI, please run: gh auth login", "auth", "no login"),
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


def test_timeout_is_network(monkeypatch):
    receipt, _calls = _receipt(
        monkeypatch,
        exc_factory=lambda cmd: subprocess.TimeoutExpired(cmd=cmd, timeout=30),
    )
    assert receipt["classification"] == "network"
    assert "timed out" in receipt["detail"]


def test_unknown_gh_exit_is_provider_error_without_stderr(monkeypatch):
    legacy = "a1b2c3d4e5f6a7b8c9d0e1f2a3b4c5d6e7f8a9b0"
    receipt, _calls = _receipt(
        monkeypatch,
        stderr=f"gh: credential {legacy} rejected for api.github.com",
    )
    blob = json.dumps(receipt)
    assert receipt["classification"] == "provider_error"
    assert receipt["detail"] == "GitHub API call failed (rc=1)."
    assert legacy not in blob
    assert "api.github.com" not in blob


def test_graphql_bad_credentials_stay_auth_without_the_body(monkeypatch):
    body = json.dumps({"data": None, "errors": [{"message": "Bad credentials ghp_secret"}]})
    receipt, _calls = _receipt(
        monkeypatch,
        returns=subprocess.CompletedProcess(["gh"], 0, stdout=body, stderr=""),
    )
    blob = json.dumps(receipt)
    assert receipt["classification"] == "auth"
    assert receipt["head_sha"] is None
    assert "credential" in receipt["detail"]
    assert "ghp_secret" not in blob
    assert "Bad credentials" not in blob


def test_malformed_gh_stdout_stays_infra(monkeypatch):
    receipt, _calls = _receipt(
        monkeypatch,
        returns=subprocess.CompletedProcess(["gh"], 0, stdout="not json", stderr=""),
    )
    assert receipt["classification"] == "infra"
    assert receipt["ok"] is False
    assert "not json" not in json.dumps(receipt)
