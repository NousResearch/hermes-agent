"""Exact-head idempotency for the desktop/web pull-request writer."""

from __future__ import annotations

import json

import pytest

from hermes_cli import web_git


def _git_identity(_cwd: str, args: list[str]) -> str:
    if args == ["rev-parse", "--abbrev-ref", "HEAD"]:
        return "codex/example"
    if args == ["rev-parse", "--verify", "HEAD"]:
        return "a" * 40
    raise AssertionError(f"unexpected git read: {args}")


def test_review_create_pr_reuses_existing_exact_head(monkeypatch):
    calls: list[list[str]] = []
    monkeypatch.setattr(web_git, "_review_push", lambda _cwd: None)
    monkeypatch.setattr(web_git, "_git_out", _git_identity)

    def fake_gh(_cwd: str, args: list[str]):
        calls.append(args)
        assert args[:3] == ["pr", "list", "--state"]
        return True, json.dumps(
            [
                {
                    "number": 17,
                    "url": "https://github.com/acme/repo/pull/17",
                    "headRefName": "codex/example",
                    "headRefOid": "a" * 40,
                }
            ]
        )

    monkeypatch.setattr(web_git, "_gh", fake_gh)

    assert web_git.review_create_pr("/repo") == {
        "url": "https://github.com/acme/repo/pull/17",
        "number": 17,
        "reused": True,
    }
    assert len(calls) == 1


def test_review_create_pr_fails_closed_when_dedupe_read_fails(monkeypatch):
    monkeypatch.setattr(web_git, "_review_push", lambda _cwd: None)
    monkeypatch.setattr(web_git, "_git_out", _git_identity)
    monkeypatch.setattr(web_git, "_gh", lambda _cwd, _args: (False, "rate limited"))

    with pytest.raises(RuntimeError, match="verify existing pull requests"):
        web_git.review_create_pr("/repo")


def test_review_create_pr_creates_only_after_exact_head_dedupe(monkeypatch):
    calls: list[list[str]] = []
    monkeypatch.setattr(web_git, "_review_push", lambda _cwd: None)
    monkeypatch.setattr(web_git, "_git_out", _git_identity)

    def fake_gh(_cwd: str, args: list[str]):
        calls.append(args)
        if args[0:2] == ["pr", "list"]:
            return True, "[]"
        assert args == ["pr", "create", "--fill"]
        return True, "https://github.com/acme/repo/pull/18\n"

    monkeypatch.setattr(web_git, "_gh", fake_gh)

    assert web_git.review_create_pr("/repo") == {
        "url": "https://github.com/acme/repo/pull/18",
        "reused": False,
    }
    assert [call[:2] for call in calls] == [["pr", "list"], ["pr", "create"]]
