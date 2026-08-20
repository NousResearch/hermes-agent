"""Unit tests for the GitHub connector tools (plugins/github/tools.py + __init__.py)."""

from __future__ import annotations

import json

import pytest

from plugins.github import tools as gh_tools
from plugins.github import register as plugin_register
from plugins.github.client import GitHubClient


class _FakeCtx:
    """Minimal stand-in for the plugin loader's register(ctx)."""

    def __init__(self):
        self.registered = []

    def register_tool(self, **kwargs):
        self.registered.append(kwargs)


class _StubClient:
    """Client stub returning canned payloads, recording calls."""

    def __init__(self, payloads: dict | None = None):
        self.payloads = payloads or {}
        self.calls = []
        self.auth_method = "github-app"
        self.actor = "jarpis-bot[bot]"

    def attribution(self):
        return {"auth_method": self.auth_method, "actor": self.actor}

    def __getattr__(self, name):
        def _handler(**kwargs):
            self.calls.append((name, kwargs))
            return self.payloads.get(name, {})

        return _handler


def _run_handler(name: str, args: dict, client: _StubClient) -> dict:
    monkey_patch_client = _install_stub(client)
    with monkey_patch_client:
        result = gh_tools._HANDLERS[name](args)
    return json.loads(result)


def _install_stub(client: _StubClient):
    import contextlib

    import plugins.github.tools as tools_mod

    @contextlib.contextmanager
    def _patch():
        original = tools_mod._github_client
        tools_mod._github_client = lambda: client
        try:
            yield
        finally:
            tools_mod._github_client = original

    return _patch()


def test_register_registers_seven_tools() -> None:
    ctx = _FakeCtx()
    plugin_register(ctx)
    names = [t["name"] for t in ctx.registered]
    assert names == [
        "github_identity",
        "github_create_issue",
        "github_comment_issue",
        "github_list_issues",
        "github_get_issue",
        "github_review_pr",
        "github_merge_pr",
    ]
    for entry in ctx.registered:
        assert entry["toolset"] == "github"
        assert callable(entry["handler"])
        assert callable(entry["check_fn"])


def test_identity_handler_returns_attribution() -> None:
    stub = _StubClient({"verify_identity": {"auth_method": "github-app", "actor": "jarpis-bot[bot]"}})
    result = _run_handler("github_identity", {}, stub)
    assert result["actor"] == "jarpis-bot[bot]" or result["auth_method"] == "github-app"


def test_create_issue_handler_passes_repo_split() -> None:
    stub = _StubClient({"create_issue": {"number": 5, "html_url": "https://x/5", "state": "open"}})
    result = _run_handler(
        "github_create_issue",
        {"repo": "himanusia/plus1", "title": "Hello", "body": "World"},
        stub,
    )
    assert result["success"] is True
    assert result["issue_number"] == 5
    assert stub.calls[0][0] == "create_issue"
    assert stub.calls[0][1]["owner"] == "himanusia"
    assert stub.calls[0][1]["repo"] == "plus1"
    assert result["attribution"]["actor"] == "jarpis-bot[bot]"


def test_comment_handler_includes_author_and_attribution() -> None:
    stub = _StubClient({"comment_issue": {"id": 9, "html_url": "https://x/c9", "user": {"login": "jarpis-bot[bot]"}}})
    result = _run_handler("github_comment_issue", {"repo": "himanusia/plus1", "number": 5, "body": "LGTM"}, stub)
    assert result["success"] is True
    assert result["author"] == "jarpis-bot[bot]"
    assert result["attribution"]["actor"] == "jarpis-bot[bot]"


def test_review_pr_validates_event() -> None:
    stub = _StubClient({"review_pull_request": {"id": 1, "state": "APPROVED"}})
    result = _run_handler(
        "github_review_pr",
        {"repo": "himanusia/plus1", "number": 3, "event": "bogus"},
        stub,
    )
    assert "event must be one of" in result.get("error", "")
    assert stub.calls == []


def test_merge_pr_defaults_to_squash() -> None:
    stub = _StubClient({"merge_pull_request": {"merged": True, "sha": "abc"}})
    result = _run_handler("github_merge_pr", {"repo": "himanusia/plus1", "number": 3}, stub)
    assert result["merged"] is True
    assert stub.calls[0][1]["method"] == "squash"


def test_check_github_available_false_without_creds(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in ("GITHUB_APP_ID", "GITHUB_APP_PRIVATE_KEY_PATH", "GITHUB_APP_INSTALLATION_ID", "GITHUB_TOKEN", "GH_TOKEN"):
        monkeypatch.delenv(key, raising=False)

    class _NoApp:
        def credentials_configured(self):
            return False

    import shutil

    import plugins.github.tools as tools_mod
    import plugins.github.client as client_mod

    monkeypatch.setattr(client_mod, "GitHubAppAuth", _NoApp)
    monkeypatch.setattr(shutil, "which", lambda _: None)
    assert tools_mod._check_github_available() is False
