"""API tests for /api/code/github/* endpoints."""

from __future__ import annotations

import hashlib
import hmac
import json
import time

import pytest


def _signature(secret: str, body: bytes) -> str:
    return "sha256=" + hmac.new(secret.encode("utf-8"), body, hashlib.sha256).hexdigest()


@pytest.fixture()
def clients(monkeypatch, _isolate_hermes_home):
    from starlette.testclient import TestClient
    import hermes_state
    from hermes_constants import get_hermes_home
    from hermes_cli.web_server import app, _SESSION_HEADER_NAME, _SESSION_TOKEN

    monkeypatch.setattr(hermes_state, "DEFAULT_DB_PATH", get_hermes_home() / "state.db")
    auth_client = TestClient(app)
    auth_client.headers[_SESSION_HEADER_NAME] = _SESSION_TOKEN
    unauth_client = TestClient(app)
    return auth_client, unauth_client


def test_github_status_endpoint(clients):
    auth, _unauth = clients
    resp = auth.get("/api/code/github/status")
    assert resp.status_code == 200
    assert "status" in resp.json()
    assert "mode" in resp.json()["status"]


def test_github_repositories_endpoint(clients, tmp_path):
    auth, _unauth = clients
    from hermes_state import SessionDB

    db = SessionDB()
    try:
        now = time.time()
        db._conn.execute(
            """
            INSERT INTO github_repositories
                (id, installation_id, github_repo_id, owner, name, full_name, default_branch, private,
                 html_url, clone_url, ssh_url, archived, disabled, pushed_at, created_at, updated_at, last_synced_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "gr-1",
                11,
                22,
                "acme",
                "repo",
                "acme/repo",
                "main",
                0,
                "https://github.com/acme/repo",
                "https://github.com/acme/repo.git",
                "git@github.com:acme/repo.git",
                0,
                0,
                None,
                now,
                now,
                now,
            ),
        )
        db._conn.commit()
    finally:
        db.close()

    resp = auth.get("/api/code/github/repositories")
    assert resp.status_code == 200
    assert resp.json()["total"] >= 1


def test_github_repositories_sync_endpoint(monkeypatch, clients):
    auth, _unauth = clients
    from hermes_cli.code.github_sync import GitHubSyncService

    monkeypatch.setattr(
        GitHubSyncService,
        "sync_repositories",
        lambda self, installation_id=None, dry_run=False, limit=100: {"dry_run": dry_run, "synced": 1, "repositories": [{"full_name": "acme/repo"}]},
    )
    resp = auth.post("/api/code/github/repositories/sync", json={"dry_run": True, "limit": 5})
    assert resp.status_code == 200
    assert resp.json()["result"]["dry_run"] is True


def test_github_webhook_endpoint_signature_validation(monkeypatch, clients):
    _auth, unauth = clients
    monkeypatch.setenv("HERMES_GITHUB_WEBHOOK_SECRET", "secret")
    payload = {"repository": {"full_name": "acme/repo"}}
    body = json.dumps(payload).encode("utf-8")

    bad = unauth.post(
        "/api/code/github/webhooks",
        data=body,
        headers={
            "X-GitHub-Event": "push",
            "X-GitHub-Delivery": "deliv-1",
            "X-Hub-Signature-256": "sha256=deadbeef",
        },
    )
    assert bad.status_code == 401

    ok = unauth.post(
        "/api/code/github/webhooks",
        data=body,
        headers={
            "X-GitHub-Event": "push",
            "X-GitHub-Delivery": "deliv-2",
            "X-Hub-Signature-256": _signature("secret", body),
        },
    )
    assert ok.status_code == 200
    assert ok.json()["accepted"] is True


def test_github_comments_endpoint_approval_flow(monkeypatch, clients):
    auth, _unauth = clients
    from hermes_cli.code.github_integration import GitHubIntegrationService

    calls = {"count": 0}

    def _fake_post(self, repo_full_name, issue_number, body, installation_id=None):
        calls["count"] += 1
        return {"id": calls["count"], "body": body}

    monkeypatch.setattr(
        GitHubIntegrationService,
        "post_issue_comment",
        _fake_post,
    )

    pending = auth.post(
        "/api/code/github/comments",
        json={
            "repo_full_name": "acme/repo",
            "issue_number": 7,
            "body": "hello",
        },
    )
    assert pending.status_code == 200
    payload = pending.json()
    assert payload["requires_approval"] is True
    approval_id = payload["approval_id"]
    assert approval_id

    approve = auth.post(f"/api/code/approvals/{approval_id}/approve", json={"actor": "local"})
    assert approve.status_code == 200

    executed = auth.post(
        "/api/code/github/comments",
        json={
            "repo_full_name": "acme/repo",
            "issue_number": 7,
            "body": "hello",
            "approval_id": approval_id,
        },
    )
    assert executed.status_code == 200
    assert executed.json()["requires_approval"] is False
    assert executed.json()["status"] == "executed"
    assert calls["count"] == 1


def test_github_comments_rejected_expired_executed_or_mismatch_do_not_execute(monkeypatch, clients):
    auth, _unauth = clients
    from hermes_cli.code.github_integration import GitHubIntegrationService
    from hermes_state import SessionDB

    calls = {"count": 0}

    def _fake_post(self, repo_full_name, issue_number, body, installation_id=None):
        calls["count"] += 1
        return {"id": 77, "body": body}

    monkeypatch.setattr(GitHubIntegrationService, "post_issue_comment", _fake_post)

    pending = auth.post(
        "/api/code/github/comments",
        json={"repo_full_name": "acme/repo", "issue_number": 9, "body": "token=abc123"},
    )
    approval_id = pending.json()["approval_id"]

    # Rejected: should not execute
    reject = auth.post(f"/api/code/approvals/{approval_id}/reject", json={"actor": "local", "reason": "deny"})
    assert reject.status_code == 200
    rejected_call = auth.post(
        "/api/code/github/comments",
        json={"repo_full_name": "acme/repo", "issue_number": 9, "body": "token=abc123", "approval_id": approval_id},
    )
    assert rejected_call.status_code == 400
    assert calls["count"] == 0
    assert "abc123" not in str(rejected_call.json())

    # Approved + expired: should not execute
    pending2 = auth.post(
        "/api/code/github/comments",
        json={"repo_full_name": "acme/repo", "issue_number": 11, "body": "hello-expired"},
    )
    approval_id2 = pending2.json()["approval_id"]
    auth.post(f"/api/code/approvals/{approval_id2}/approve", json={"actor": "local"})
    db = SessionDB()
    try:
        db.update_code_approval_request(approval_id2, {"expires_at": time.time() - 10})
    finally:
        db.close()
    expired_call = auth.post(
        "/api/code/github/comments",
        json={"repo_full_name": "acme/repo", "issue_number": 11, "body": "hello-expired", "approval_id": approval_id2},
    )
    assert expired_call.status_code == 400
    assert calls["count"] == 0

    # Approved then executed cannot be reused
    pending3 = auth.post(
        "/api/code/github/comments",
        json={"repo_full_name": "acme/repo", "issue_number": 12, "body": "hello-reuse"},
    )
    approval_id3 = pending3.json()["approval_id"]
    auth.post(f"/api/code/approvals/{approval_id3}/approve", json={"actor": "local"})
    first_exec = auth.post(
        "/api/code/github/comments",
        json={"repo_full_name": "acme/repo", "issue_number": 12, "body": "hello-reuse", "approval_id": approval_id3},
    )
    assert first_exec.status_code == 200
    second_exec = auth.post(
        "/api/code/github/comments",
        json={"repo_full_name": "acme/repo", "issue_number": 12, "body": "hello-reuse", "approval_id": approval_id3},
    )
    assert second_exec.status_code == 400
    assert calls["count"] == 1

    # Kind/resource mismatch: approved approval for repo A cannot be replayed for repo B
    pending4 = auth.post(
        "/api/code/github/comments",
        json={"repo_full_name": "acme/repo", "issue_number": 13, "body": "hello-match"},
    )
    approval_id4 = pending4.json()["approval_id"]
    auth.post(f"/api/code/approvals/{approval_id4}/approve", json={"actor": "local"})
    mismatch = auth.post(
        "/api/code/github/comments",
        json={"repo_full_name": "acme/other", "issue_number": 13, "body": "hello-match", "approval_id": approval_id4},
    )
    assert mismatch.status_code == 400
    assert calls["count"] == 1


def test_github_pull_request_prepare_endpoint(clients):
    auth, _unauth = clients
    pending = auth.post(
        "/api/code/github/pull-requests/prepare",
        json={
            "repo_full_name": "acme/repo",
            "title": "feat: x",
            "head": "feature/x",
            "base": "main",
            "body": "desc",
        },
    )
    assert pending.status_code == 200
    payload = pending.json()
    assert payload["requires_approval"] is True
    approval_id = payload["approval_id"]
    assert approval_id

    approve = auth.post(f"/api/code/approvals/{approval_id}/approve", json={"actor": "local"})
    assert approve.status_code == 200

    resp = auth.post(
        "/api/code/github/pull-requests/prepare",
        json={
            "repo_full_name": "acme/repo",
            "title": "feat: x",
            "head": "feature/x",
            "base": "main",
            "body": "desc",
            "approval_id": approval_id,
        },
    )
    assert resp.status_code == 200
    payload = resp.json()
    assert payload["requires_approval"] is False
    assert payload["prepared"]["auto_push"] is False
    assert payload["prepared"]["auto_merge"] is False
