"""Tests for ``hermes autodevelop`` queue contract, safety, and CLI."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import pytest

from hermes_cli.autodevelop import (
    CLAIM_MARKER,
    GitHubClient,
    QueueItem,
    apply_codeowners_skip,
    bootstrap_report,
    build_oneshot_prompt,
    build_parser,
    claim_comment_body,
    claim_is_fresh,
    claim_item,
    codeowners_paths,
    draft_pr_argv,
    draft_pr_command,
    execute_argv,
    issue_from_api,
    kanban_idempotency_key,
    mark_in_progress_claims,
    open_claim_count,
    parse_queue_contract,
    persist_claim,
    skip_reason_for,
    touches_are_sensitive,
)


def _parser():
    import argparse

    parent = argparse.ArgumentParser()
    sub = parent.add_subparsers()
    build_parser(sub)
    return parent


def test_parse_queue_contract_from_body_and_labels():
    body = """
scope: small
touches: hermes_cli/autodevelop.py, website/docs
no-human-gate: true

- [ ] list agent-ready issues
- [ ] claim without colliding
"""
    contract = parse_queue_contract(body, labels=["type/feature"])
    assert contract.scope == "small"
    assert "hermes_cli/autodevelop.py" in contract.touches
    assert contract.no_human_gate is True
    assert "list agent-ready issues" in contract.acceptance


def test_parse_scope_from_label_when_body_omits_it():
    contract = parse_queue_contract("just text", labels=["scope:medium"])
    assert contract.scope == "medium"


def test_sensitive_touches_blocked_unless_allow_label():
    assert touches_are_sensitive(["hermes_cli/auth.py"], allow_sensitive=False)
    assert touches_are_sensitive([".env.local"], allow_sensitive=False)
    assert not touches_are_sensitive(["hermes_cli/autodevelop.py"], allow_sensitive=False)
    assert not touches_are_sensitive(["hermes_cli/auth.py"], allow_sensitive=True)


def test_skip_locked_assigned_large_and_human_gate():
    contract = parse_queue_contract("scope: large\nno-human-gate: false")
    assert skip_reason_for(
        locked=True, assignees=(), contract=contract, labels=(),
        include_assigned=False, include_large=False, include_human_gate=False,
    ) == "locked"
    assert skip_reason_for(
        locked=False, assignees=("alice",), contract=contract, labels=(),
        include_assigned=False, include_large=True, include_human_gate=True,
    ) == "assigned"
    small_human = parse_queue_contract("scope: small\nno-human-gate: false")
    assert skip_reason_for(
        locked=False, assignees=(), contract=small_human, labels=(),
        include_assigned=False, include_large=False, include_human_gate=False,
    ) == "needs-human-gate"
    assert skip_reason_for(
        locked=False, assignees=(), contract=contract, labels=(),
        include_assigned=False, include_large=False, include_human_gate=True,
    ) == "scope:large"


def test_issue_from_api_skips_pull_requests_payload_fields():
    raw = {
        "number": 12,
        "title": "Ready task",
        "html_url": "https://github.com/acme/repo/issues/12",
        "body": "scope: small\ntouches: docs/\nno-human-gate: true",
        "labels": [{"name": "agent-ready"}],
        "assignees": [],
        "locked": False,
    }
    item = issue_from_api(
        "acme/repo",
        raw,
        include_assigned=False,
        include_large=False,
        include_human_gate=False,
    )
    assert item.claimable
    assert item.number == 12


def test_claim_is_fresh_and_stale():
    now = datetime.now(timezone.utc)
    fresh = [{"body": CLAIM_MARKER, "created_at": now.isoformat()}]
    stale = [{"body": CLAIM_MARKER, "created_at": (now - timedelta(hours=20)).isoformat()}]
    assert claim_is_fresh(fresh, ttl_hours=8)
    assert not claim_is_fresh(stale, ttl_hours=8)
    assert not claim_is_fresh([], ttl_hours=8)


def test_claim_dry_run_does_not_post():
    posted = []

    def fake_request(method, url, **kwargs):
        class Resp:
            status_code = 200

            def json(self):
                if method == "GET" and url.endswith("/comments"):
                    return []
                if method == "POST":
                    posted.append(kwargs)
                    return {"id": 99}
                return {}

        return Resp()

    client = GitHubClient(token="x", request=fake_request)
    item = QueueItem(
        number=1, title="t", html_url="u", body="", labels=(), assignees=(),
        locked=False, repo="acme/repo", contract=parse_queue_contract(""),
    )
    result = claim_item(client, item, login="pat", ttl_hours=8, commit=False)
    assert result["reason"] == "dry-run"
    assert CLAIM_MARKER in result["comment"]
    assert posted == []


def test_claim_comment_credits_human():
    body = claim_comment_body(repo="acme/repo", number=7, login="pat")
    assert CLAIM_MARKER in body
    assert "pat" in body
    assert "human" in body.lower()


def test_oneshot_prompt_is_search_first_and_draft():
    item = QueueItem(
        number=103133,
        title="autodevelop mode",
        html_url="https://github.com/NousResearch/hermes-agent/issues/103133",
        body="scope: small",
        labels=("agent-ready",),
        assignees=(),
        locked=False,
        repo="NousResearch/hermes-agent",
        contract=parse_queue_contract("scope: small\n- [ ] ship v0"),
    )
    prompt = build_oneshot_prompt(item, draft_pr=True)
    assert "Search-first" in prompt
    assert "gh pr create" in prompt
    assert "prior art" in prompt.lower()
    assert "draft" in prompt.lower()
    assert "BYOK" in prompt
    assert "103133" in prompt


def test_kanban_idempotency_key():
    assert kanban_idempotency_key("NousResearch/hermes-agent", 103133) == (
        "github:NousResearch/hermes-agent#103133"
    )


def test_cli_help_lists_actions():
    parent = _parser()
    ns = parent.parse_args(["autodevelop", "list", "--repo", "a/b"])
    assert ns.autodevelop_action == "list"
    assert ns.repo == "a/b"
    assert ns.label == "agent-ready"


def test_cli_run_budget_flags():
    parent = _parser()
    ns = parent.parse_args(
        ["autodevelop", "run", "--repo", "a/b", "--max-issues", "3", "--budget", "2"]
    )
    assert ns.max_issues == 3
    assert ns.budget == 2
    assert ns.commit is False


def test_bootstrap_detects_project_law(tmp_path, monkeypatch):
    assert bootstrap_report(str(tmp_path))["ok"] is False
    (tmp_path / ".git").mkdir()
    (tmp_path / "CONTRIBUTING.md").write_text("hi\n")
    monkeypatch.setenv("VIRTUAL_ENV", str(tmp_path / "venv"))
    monkeypatch.setattr(
        "hermes_cli.autodevelop._git_remotes",
        lambda cwd: "origin\thttps://github.com/acme/repo.git (fetch)\n",
    )
    assert bootstrap_report(str(tmp_path))["ok"] is True
    assert "github-remote" not in bootstrap_report(str(tmp_path))["missing"]


def test_mark_in_progress_claims_skips_fresh():
    now = datetime.now(timezone.utc)

    def fake_request(method, url, **kwargs):
        class Resp:
            status_code = 200

            def json(self):
                return [{"body": CLAIM_MARKER, "created_at": now.isoformat()}]

        return Resp()

    client = GitHubClient(token="x", request=fake_request)
    item = QueueItem(
        number=9, title="t", html_url="u", body="", labels=(), assignees=(),
        locked=False, repo="acme/repo", contract=parse_queue_contract(""),
    )
    mark_in_progress_claims(client, [item], ttl_hours=8)
    assert item.skip_reason == "in-progress-claim"


def test_codeowners_and_persist(tmp_path, monkeypatch):
    owners = tmp_path / "CODEOWNERS"
    owners.write_text("hermes_cli/auth.py @core\n")
    assert "hermes_cli/auth.py" in codeowners_paths(str(tmp_path))
    item = QueueItem(
        number=1, title="t", html_url="u", body="", labels=(), assignees=(),
        locked=False, repo="acme/repo",
        contract=parse_queue_contract("touches: hermes_cli/auth.py"),
    )
    apply_codeowners_skip(item, codeowners_paths(str(tmp_path)))
    assert item.skip_reason == "codeowners-sensitive"
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
    assert persist_claim("acme/repo", "pat", 3) == 1
    assert persist_claim("acme/repo", "pat", 4) == 2
    assert open_claim_count("acme/repo", "pat") == 2


def test_draft_pr_argv_is_draft():
    item = QueueItem(
        number=2, title="t", html_url="u", body="", labels=(), assignees=(),
        locked=False, repo="acme/repo", contract=parse_queue_contract(""),
    )
    argv = draft_pr_argv(item, draft=True)
    assert argv[:4] == ["gh", "pr", "create", "--repo"]
    assert "--draft" in argv


def test_execute_argv_and_draft_pr_command():
    item = QueueItem(
        number=1, title="t", html_url="u", body="", labels=(), assignees=(),
        locked=False, repo="acme/repo", contract=parse_queue_contract(""),
    )
    argv = execute_argv("do the thing")
    assert argv[-3:] == ["chat", "-q", "do the thing"] or argv[-4:-1] == ["chat", "-q", "do the thing"]
    assert "--oneshot" in argv
    cmd = draft_pr_command(item, draft=True)
    assert "--draft" in cmd
    assert "#1" in cmd


def test_github_client_rejects_error_status():
    def boom(method, url, **kwargs):
        class Resp:
            status_code = 401

            def json(self):
                return {"message": "bad"}

        return Resp()

    client = GitHubClient(token="x", request=boom)
    with pytest.raises(Exception, match="HTTP 401"):
        client.list_issues("a/b", label="agent-ready")
