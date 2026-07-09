"""External-write approval gates for desktop/web git helpers.

The helpers shell out directly instead of going through terminal(), so these
tests stub subprocess wrappers and verify that blocked commands are never
executed.
"""

import pytest

from hermes_cli import web_git
import tools.approval as approval_module
from tools.approval import approve_session, reset_current_session_key, set_current_session_key


@pytest.fixture(autouse=True)
def _approval_session(monkeypatch):
    monkeypatch.setattr(approval_module, "_YOLO_MODE_FROZEN", False)
    monkeypatch.delenv("HERMES_INTERACTIVE", raising=False)
    monkeypatch.delenv("HERMES_GATEWAY_SESSION", raising=False)
    monkeypatch.delenv("HERMES_EXEC_ASK", raising=False)
    approval_module._permanent_approved.clear()
    approval_module.clear_session("webgit-external-test")
    token = set_current_session_key("webgit-external-test")
    try:
        yield
    finally:
        reset_current_session_key(token)
        approval_module.clear_session("webgit-external-test")
        approval_module._permanent_approved.clear()


def test_review_push_requires_external_write_approval_before_git_push(monkeypatch, tmp_path):
    git_calls = []

    def fake_git(cwd, args, *, timeout=web_git._GIT_TIMEOUT):
        if args == ["rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{u}"]:
            return 0, "origin/main\n", ""
        git_calls.append(args)
        return 0, "", ""

    monkeypatch.setattr(web_git, "_git", fake_git)

    with pytest.raises(PermissionError, match="external repo write"):
        web_git.review_push(str(tmp_path))

    assert git_calls == []


def test_review_push_uses_existing_external_repo_write_approval(monkeypatch, tmp_path):
    git_calls = []
    approve_session("webgit-external-test", "external_repo_write")

    def fake_git(cwd, args, *, timeout=web_git._GIT_TIMEOUT):
        if args == ["rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{u}"]:
            return 0, "origin/main\n", ""
        git_calls.append(args)
        return 0, "", ""

    monkeypatch.setattr(web_git, "_git", fake_git)

    assert web_git.review_push(str(tmp_path)) == {"ok": True}
    assert git_calls == [["push"]]


def test_review_create_pr_requires_approval_before_gh_pr_create(monkeypatch, tmp_path):
    gh_calls = []
    monkeypatch.setattr(web_git, "_review_push", lambda cwd: None)

    def fake_gh(cwd, args):
        gh_calls.append(args)
        return True, "https://example.invalid/pull/1\n"

    monkeypatch.setattr(web_git, "_gh", fake_gh)

    with pytest.raises(PermissionError, match="external repo write"):
        web_git.review_create_pr(str(tmp_path))

    assert gh_calls == []


def test_review_create_pr_uses_existing_external_repo_write_approval(monkeypatch, tmp_path):
    gh_calls = []
    approve_session("webgit-external-test", "external_repo_write")
    monkeypatch.setattr(web_git, "_review_push", lambda cwd: None)

    def fake_gh(cwd, args):
        gh_calls.append(args)
        return True, "https://example.invalid/pull/1\n"

    monkeypatch.setattr(web_git, "_gh", fake_gh)

    assert web_git.review_create_pr(str(tmp_path)) == {"url": "https://example.invalid/pull/1"}
    assert gh_calls == [["pr", "create", "--fill"]]
