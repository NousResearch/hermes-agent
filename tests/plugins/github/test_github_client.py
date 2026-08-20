"""Unit tests for the GitHub connector client (plugins/github/client.py)."""

from __future__ import annotations

import json

import pytest

from plugins.github import client as gh_client
from plugins.github.client import GitHubClient, GitHubError, parse_repo


class _FakeResponse:
    def __init__(self, status_code: int, payload=None, *, text: str = ""):
        self.status_code = status_code
        self._payload = payload
        self.text = text or (json.dumps(payload) if payload is not None else "")
        self.content = self.text.encode("utf-8")

    def json(self):
        if self._payload is None:
            raise ValueError("no json")
        return self._payload


@pytest.fixture(autouse=True)
def _clean_creds(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in (
        "GITHUB_APP_ID",
        "GITHUB_APP_PRIVATE_KEY_PATH",
        "GITHUB_APP_INSTALLATION_ID",
        "GITHUB_TOKEN",
        "GH_TOKEN",
    ):
        monkeypatch.delenv(key, raising=False)


def _fake_gh_cli(monkeypatch: pytest.MonkeyPatch, token: str | None = None) -> list[str]:
    calls: list[str] = []

    def fake_run(cmd, **kwargs):
        calls.append(cmd[0] if isinstance(cmd, list) else str(cmd))
        if token is None:
            return _ProcResult(1, "", "")
        return _ProcResult(0, token, "")

    monkeypatch.setattr(gh_client.subprocess, "run", fake_run)
    return calls


class _ProcResult:
    def __init__(self, returncode: int, stdout: str, stderr: str):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


def _mock_app_auth(monkeypatch: pytest.MonkeyPatch, *, configured: bool, token: str | None = "ghs_app", slug: str | None = "jarpis-bot"):
    class _FakeAppAuth:
        def credentials_configured(self) -> bool:
            return configured

        def installation_token(self):
            return token if configured else None

        def bot_login(self):
            return f"{slug}[bot]" if (configured and slug) else None

        def app_slug(self):
            return slug if configured else None

    monkeypatch.setattr(gh_client, "GitHubAppAuth", _FakeAppAuth)


def test_parse_repo_full() -> None:
    assert parse_repo("himanusia/plus1") == ("himanusia", "plus1")


def test_parse_repo_bare() -> None:
    assert parse_repo("plus1") == ("plus1", "plus1")


def test_parse_repo_invalid() -> None:
    with pytest.raises(GitHubError):
        parse_repo("/")


def test_resolve_prefers_github_app_over_pat(monkeypatch: pytest.MonkeyPatch) -> None:
    _mock_app_auth(monkeypatch, configured=True, token="ghs_app", slug="jarpis-bot")
    monkeypatch.setenv("GITHUB_TOKEN", "gho_pat")
    client = GitHubClient()
    method, token = client._resolve_credentials()
    assert method == "github-app"
    assert token == "ghs_app"
    assert client.actor == "jarpis-bot[bot]"


def test_resolve_falls_back_to_pat(monkeypatch: pytest.MonkeyPatch) -> None:
    _mock_app_auth(monkeypatch, configured=False)
    monkeypatch.setenv("GH_TOKEN", "gho_pat")
    client = GitHubClient()
    method, token = client._resolve_credentials()
    assert method == "pat"
    assert token == "gho_pat"


def test_resolve_falls_back_to_gh_cli(monkeypatch: pytest.MonkeyPatch) -> None:
    _mock_app_auth(monkeypatch, configured=False)
    _fake_gh_cli(monkeypatch, token="gho_cli")
    client = GitHubClient()
    method, token = client._resolve_credentials()
    assert method == "gh-cli"
    assert token == "gho_cli"


def test_resolve_raises_without_any_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    _mock_app_auth(monkeypatch, configured=False)
    _fake_gh_cli(monkeypatch, token=None)
    client = GitHubClient()
    with pytest.raises(GitHubError, match="No GitHub credentials"):
        client._resolve_credentials()


def test_request_retries_once_after_401(monkeypatch: pytest.MonkeyPatch) -> None:
    _mock_app_auth(monkeypatch, configured=True, token="ghs_app")
    calls: list[str] = []

    def fake_request(method, url, headers=None, params=None, json=None, timeout=None):
        calls.append(headers["Authorization"])
        if len(calls) == 1:
            return _FakeResponse(401, {"message": "bad credentials"})
        return _FakeResponse(200, {"id": 1, "title": "ok"})

    monkeypatch.setattr("httpx.request", fake_request)

    client = GitHubClient()
    issue = client.get_issue("himanusia", "plus1", 1)
    assert issue["title"] == "ok"
    assert calls == ["Bearer ghs_app", "Bearer ghs_app"]  # re-resolved after reset


def test_attribution_included(monkeypatch: pytest.MonkeyPatch) -> None:
    _mock_app_auth(monkeypatch, configured=True, token="ghs_app", slug="jarpis-bot")
    attribution = GitHubClient().attribution()
    assert attribution["auth_method"] == "github-app"
    assert attribution["actor"] == "jarpis-bot[bot]"


def test_create_issue_payload(monkeypatch: pytest.MonkeyPatch) -> None:
    _mock_app_auth(monkeypatch, configured=True, token="ghs_app")
    captured = {}

    def fake_request(method, url, headers=None, params=None, json=None, timeout=None):
        captured.update(method=method, url=url, payload=json)
        return _FakeResponse(201, {"number": 42, "html_url": "https://github.com/himanusia/plus1/issues/42"})

    monkeypatch.setattr("httpx.request", fake_request)

    issue = GitHubClient().create_issue(
        owner="himanusia", repo="plus1", title="Test", body="Body", labels=["bug"]
    )
    assert issue["number"] == 42
    assert captured["url"] == "https://api.github.com/repos/himanusia/plus1/issues"
    assert captured["payload"]["title"] == "Test"
    assert captured["payload"]["labels"] == ["bug"]


def test_review_pr_event_validated_in_tools(monkeypatch: pytest.MonkeyPatch) -> None:
    # Tools-level validation lives in tools.py; client-level just posts.
    _mock_app_auth(monkeypatch, configured=True, token="ghs_app")
    captured = {}

    def fake_request(method, url, headers=None, params=None, json=None, timeout=None):
        captured.update(method=method, url=url, payload=json)
        return _FakeResponse(200, {"id": 7, "state": "APPROVED"})

    monkeypatch.setattr("httpx.request", fake_request)

    review = GitHubClient().review_pull_request(
        owner="himanusia", repo="plus1", number=3, event="APPROVE", body="LGTM"
    )
    assert review["state"] == "APPROVED"
    assert captured["payload"]["event"] == "APPROVE"
    assert captured["payload"]["body"] == "LGTM"


def test_merge_pr_default_squash(monkeypatch: pytest.MonkeyPatch) -> None:
    _mock_app_auth(monkeypatch, configured=True, token="ghs_app")
    captured = {}

    def fake_request(method, url, headers=None, params=None, json=None, timeout=None):
        captured.update(method=method, url=url, payload=json)
        return _FakeResponse(200, {"merged": True, "sha": "abc123"})

    monkeypatch.setattr("httpx.request", fake_request)

    result = GitHubClient().merge_pull_request(owner="himanusia", repo="plus1", number=3)
    assert result["merged"] is True
    assert captured["payload"]["merge_method"] == "squash"
