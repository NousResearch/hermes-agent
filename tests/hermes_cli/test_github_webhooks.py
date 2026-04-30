"""Tests for GitHub webhook verification and processing."""

from __future__ import annotations

import hashlib
import hmac
import json

import pytest

from hermes_cli.code.github_webhooks import GitHubWebhookService, WebhookSignatureError, verify_signature
from hermes_state import SessionDB


def _signature(secret: str, body: bytes) -> str:
    return "sha256=" + hmac.new(secret.encode("utf-8"), body, hashlib.sha256).hexdigest()


def test_verify_signature_valid():
    body = b'{"ok":true}'
    secret = "abc123"
    assert verify_signature(secret, body, _signature(secret, body)) is True


def test_verify_signature_missing_signature():
    with pytest.raises(WebhookSignatureError):
        verify_signature("abc", b"{}", None)


def test_verify_signature_invalid_signature():
    with pytest.raises(WebhookSignatureError):
        verify_signature("abc", b"{}", "sha256=deadbeef")


def test_process_duplicate_delivery_dedup(monkeypatch, tmp_path):
    db_path = tmp_path / "state.db"
    SessionDB(db_path=db_path).close()
    monkeypatch.setenv("HERMES_GITHUB_WEBHOOK_SECRET", "secret")
    service = GitHubWebhookService(db_path=db_path)
    payload = {
        "action": "created",
        "repository": {"full_name": "acme/repo"},
        "sender": {"login": "alice"},
        "comment": {"id": 10, "body": "@hermes plan"},
        "issue": {"number": 1},
    }
    body = json.dumps(payload).encode("utf-8")
    sig = _signature("secret", body)
    first = service.process(delivery_id="d-1", event="issue_comment", body=body, signature=sig)
    second = service.process(delivery_id="d-1", event="issue_comment", body=body, signature=sig)
    assert first["accepted"] is True
    assert second["duplicate"] is True


def test_process_issue_comment_parses_chatops(monkeypatch, tmp_path):
    db_path = tmp_path / "state.db"
    SessionDB(db_path=db_path).close()
    monkeypatch.setenv("HERMES_GITHUB_WEBHOOK_SECRET", "secret")
    service = GitHubWebhookService(db_path=db_path)
    payload = {
        "action": "created",
        "repository": {"full_name": "acme/repo"},
        "sender": {"login": "alice"},
        "comment": {"id": 10, "body": "@hermes review please"},
        "issue": {"number": 7, "pull_request": {"url": "x"}},
    }
    body = json.dumps(payload).encode("utf-8")
    result = service.process(
        delivery_id="d-2",
        event="issue_comment",
        body=body,
        signature=_signature("secret", body),
    )
    assert result["status"] == "processed"
    assert len(result["chatops_commands"]) == 1
    assert result["chatops_commands"][0]["command"] == "review"


def test_process_pull_request_event(monkeypatch, tmp_path):
    db_path = tmp_path / "state.db"
    SessionDB(db_path=db_path).close()
    monkeypatch.setenv("HERMES_GITHUB_WEBHOOK_SECRET", "secret")
    service = GitHubWebhookService(db_path=db_path)
    payload = {
        "action": "opened",
        "repository": {"full_name": "acme/repo"},
        "sender": {"login": "bob"},
        "pull_request": {"id": 3, "number": 12},
    }
    body = json.dumps(payload).encode("utf-8")
    result = service.process(
        delivery_id="d-3",
        event="pull_request",
        body=body,
        signature=_signature("secret", body),
    )
    assert result["normalized"]["pr_number"] == 12


def test_process_unsupported_event_is_safe(monkeypatch, tmp_path):
    db_path = tmp_path / "state.db"
    SessionDB(db_path=db_path).close()
    monkeypatch.setenv("HERMES_GITHUB_WEBHOOK_SECRET", "secret")
    service = GitHubWebhookService(db_path=db_path)
    payload = {"repository": {"full_name": "acme/repo"}}
    body = json.dumps(payload).encode("utf-8")
    result = service.process(
        delivery_id="d-4",
        event="fork",
        body=body,
        signature=_signature("secret", body),
    )
    assert result["status"] == "ignored"
    assert result["accepted"] is True


def test_safe_error_redacts_tokens():
    message = GitHubWebhookService.safe_error(Exception("Authorization: Bearer ghp_abcdefghijklmnopqrstuvwxyz"))
    assert "ghp_" not in message
