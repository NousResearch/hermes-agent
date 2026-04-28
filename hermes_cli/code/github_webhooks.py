#!/usr/bin/env python3
"""GitHub webhook verification and normalization for Hermes Code Mode."""

from __future__ import annotations

import hashlib
import hmac
import json
from pathlib import Path
from typing import Any, Optional

from hermes_cli.code.event_bus import get_code_event_bus
from hermes_cli.code.github_chatops import GitHubChatOpsService
from hermes_cli.code.github_integration import GitHubIntegrationStore, _env_value, redact_github_secrets

SUPPORTED_EVENTS = frozenset(
    {
        "installation",
        "installation_repositories",
        "issues",
        "issue_comment",
        "pull_request",
        "pull_request_review",
        "pull_request_review_comment",
        "check_suite",
        "check_run",
        "push",
    }
)


class WebhookSignatureError(ValueError):
    """Raised when a GitHub webhook signature is missing or invalid."""


def verify_signature(secret: str, body: bytes, signature: Optional[str]) -> bool:
    if not secret:
        raise WebhookSignatureError("GitHub webhook secret is not configured")
    if not signature or not signature.startswith("sha256="):
        raise WebhookSignatureError("Missing GitHub webhook signature")
    expected = "sha256=" + hmac.new(secret.encode("utf-8"), body, hashlib.sha256).hexdigest()
    if not hmac.compare_digest(expected, signature):
        raise WebhookSignatureError("Invalid GitHub webhook signature")
    return True


def _payload_hash(body: bytes) -> str:
    return hashlib.sha256(body).hexdigest()


def _normalize(event: str, payload: dict[str, Any]) -> dict[str, Any]:
    repo = payload.get("repository") if isinstance(payload.get("repository"), dict) else {}
    sender = payload.get("sender") if isinstance(payload.get("sender"), dict) else {}
    issue = payload.get("issue") if isinstance(payload.get("issue"), dict) else {}
    pull_request = payload.get("pull_request") if isinstance(payload.get("pull_request"), dict) else {}
    comment = payload.get("comment") if isinstance(payload.get("comment"), dict) else {}
    review = payload.get("review") if isinstance(payload.get("review"), dict) else {}

    repo_full_name = repo.get("full_name")
    issue_number = issue.get("number")
    pr_number = pull_request.get("number")
    if event in {"issue_comment", "issues"} and issue.get("pull_request"):
        pr_number = issue_number

    return {
        "event": event,
        "action": payload.get("action"),
        "repo_full_name": repo_full_name,
        "sender_login": sender.get("login"),
        "issue_number": issue_number,
        "pr_number": pr_number,
        "comment_id": comment.get("id"),
        "comment_body": comment.get("body") or "",
        "review_id": review.get("id"),
    }


class GitHubWebhookService:
    def __init__(self, db_path: Optional[Path] = None) -> None:
        self._db_path = db_path

    def _store(self) -> GitHubIntegrationStore:
        return GitHubIntegrationStore(db_path=self._db_path)

    def _emit(self, event_type: str, payload: dict[str, Any]) -> None:
        bus = get_code_event_bus(self._db_path)
        bus.publish(
            event_type,
            payload=payload,
            github_repo_full_name=(payload or {}).get("repo_full_name"),
            metadata={"source": "github_webhook"},
            source="github_webhook",
        )

    def process(
        self,
        *,
        delivery_id: str,
        event: str,
        body: bytes,
        signature: Optional[str],
    ) -> dict[str, Any]:
        verify_signature(_env_value("HERMES_GITHUB_WEBHOOK_SECRET").strip(), body, signature)
        payload_sha = _payload_hash(body)

        store = self._store()
        try:
            existing = store.get_delivery(delivery_id)
            if existing:
                return {
                    "accepted": True,
                    "duplicate": True,
                    "status": existing.get("status"),
                    "delivery": existing,
                    "chatops_commands": [],
                }

            try:
                payload = json.loads(body.decode("utf-8"))
            except json.JSONDecodeError as exc:
                store.record_webhook_delivery(
                    delivery_id=delivery_id,
                    event=event,
                    action=None,
                    repo_full_name=None,
                    sender_login=None,
                    payload_hash=payload_sha,
                    status="error",
                    error="Invalid JSON payload",
                )
                raise ValueError("Invalid GitHub webhook JSON") from exc

            normalized = _normalize(event, payload if isinstance(payload, dict) else {})
            if event not in SUPPORTED_EVENTS:
                delivery = store.record_webhook_delivery(
                    delivery_id=delivery_id,
                    event=event,
                    action=normalized.get("action"),
                    repo_full_name=normalized.get("repo_full_name"),
                    sender_login=normalized.get("sender_login"),
                    payload_hash=payload_sha,
                    status="ignored",
                )
                self._emit(
                    "github.webhook.received",
                    {"delivery_id": delivery_id, "event": event, "status": "ignored"},
                )
                return {
                    "accepted": True,
                    "duplicate": False,
                    "status": "ignored",
                    "delivery": delivery,
                    "normalized": normalized,
                    "chatops_commands": [],
                }

            delivery = store.record_webhook_delivery(
                delivery_id=delivery_id,
                event=event,
                action=normalized.get("action"),
                repo_full_name=normalized.get("repo_full_name"),
                sender_login=normalized.get("sender_login"),
                payload_hash=payload_sha,
                status="processed",
            )
        finally:
            store.close()

        chatops_commands: list[dict[str, Any]] = []
        if event in {"issue_comment", "pull_request_review_comment"} and normalized.get("comment_body"):
            chatops_commands = GitHubChatOpsService(db_path=self._db_path).create_commands_from_comment(
                delivery_id=delivery_id,
                repo_full_name=str(normalized.get("repo_full_name") or ""),
                issue_number=normalized.get("issue_number"),
                pr_number=normalized.get("pr_number"),
                comment_id=normalized.get("comment_id"),
                sender_login=normalized.get("sender_login"),
                body=str(normalized.get("comment_body") or ""),
            )

        self._emit(
            "github.webhook.received",
            {
                "delivery_id": delivery_id,
                "event": event,
                "duplicate": False,
                "repo_full_name": normalized.get("repo_full_name"),
            },
        )
        self._emit(
            "github.webhook.processed",
            {
                "delivery_id": delivery_id,
                "event": event,
                "status": "processed",
                "chatops_commands": [item.get("id") for item in chatops_commands],
            },
        )
        if chatops_commands:
            self._emit("github.chatops.detected", {"commands": chatops_commands})

        return {
            "accepted": True,
            "duplicate": False,
            "status": "processed",
            "delivery": delivery,
            "normalized": normalized,
            "chatops_commands": chatops_commands,
        }

    @staticmethod
    def safe_error(exc: Exception) -> str:
        return redact_github_secrets(str(exc))
