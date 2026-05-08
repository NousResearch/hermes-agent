"""Webhook Dispatcher — Outbound webhook delivery for task/cron completion.

Fires HTTP POST callbacks when tasks or cron jobs complete.
Supports HMAC-SHA256 signing and exponential backoff retry.

Config (config.yaml):
    webhooks:
        - url: "https://example.com/callback"
          events: ["task.completed", "cron.completed"]
          secret: "my-hmac-secret"   # optional
          headers: {}                # optional custom headers
"""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

log = logging.getLogger(__name__)


@dataclass
class WebhookConfig:
    """Configuration for a single webhook endpoint."""
    id: str
    url: str
    events: List[str]
    secret: Optional[str] = None
    headers: Dict[str, str] = field(default_factory=dict)
    enabled: bool = True
    created_at: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "url": self.url,
            "events": self.events,
            "secret": self.secret,
            "headers": self.headers,
            "enabled": self.enabled,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "WebhookConfig":
        return cls(
            id=data.get("id", ""),
            url=data.get("url", ""),
            events=data.get("events", []),
            secret=data.get("secret"),
            headers=data.get("headers", {}),
            enabled=data.get("enabled", True),
            created_at=data.get("created_at", ""),
        )


class WebhookDispatcher:
    """Outbound webhook delivery for async task/cron completion."""

    WEBHOOKS_FILE = "webhooks.json"
    MAX_RETRIES = 3
    BACKOFF_BASE = 2  # seconds

    def __init__(self, hermes_home: Path):
        self._hermes_home = hermes_home
        self._webhooks_file = hermes_home / self.WEBHOOKS_FILE
        self._webhooks: Dict[str, WebhookConfig] = {}
        self._load()

    def _load(self):
        """Load webhooks from disk."""
        if self._webhooks_file.exists():
            try:
                with open(self._webhooks_file, "r") as f:
                    data = json.load(f)
                for item in data.get("webhooks", []):
                    config = WebhookConfig.from_dict(item)
                    self._webhooks[config.id] = config
            except Exception as e:
                log.warning("Failed to load webhooks: %s", e)

    def _save(self):
        """Persist webhooks to disk."""
        data = {
            "webhooks": [w.to_dict() for w in self._webhooks.values()]
        }
        self._webhooks_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self._webhooks_file, "w") as f:
            json.dump(data, f, indent=2)

    def register(self, url: str, events: List[str], secret: Optional[str] = None,
                 headers: Optional[Dict[str, str]] = None) -> str:
        """Register a webhook endpoint. Returns webhook_id."""
        webhook_id = f"wh_{uuid.uuid4().hex[:12]}"
        config = WebhookConfig(
            id=webhook_id,
            url=url,
            events=events,
            secret=secret,
            headers=headers or {},
            created_at=time.strftime("%Y-%m-%dT%H:%M:%S"),
        )
        self._webhooks[webhook_id] = config
        self._save()
        log.info("Registered webhook %s -> %s (events: %s)", webhook_id, url, events)
        return webhook_id

    def unregister(self, webhook_id: str) -> bool:
        """Remove a webhook registration."""
        if webhook_id in self._webhooks:
            del self._webhooks[webhook_id]
            self._save()
            return True
        return False

    def list_webhooks(self) -> List[Dict[str, Any]]:
        """List all registered webhooks (secret is redacted)."""
        results = []
        for w in self._webhooks.values():
            d = w.to_dict()
            if d.get("secret"):
                d["secret"] = "***"
            results.append(d)
        return results

    def get_webhook(self, webhook_id: str) -> Optional[WebhookConfig]:
        """Get a specific webhook config."""
        return self._webhooks.get(webhook_id)

    def dispatch(self, event: str, payload: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Fire webhooks matching the event type.

        Returns list of delivery results: [{webhook_id, url, status_code, success}]
        """
        import urllib.request
        import urllib.error

        results = []
        matching = [
            w for w in self._webhooks.values()
            if w.enabled and (event in w.events or "*" in w.events)
        ]

        if not matching:
            log.debug("No webhooks registered for event: %s", event)
            return results

        body = json.dumps({
            "event": event,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "data": payload,
        }).encode("utf-8")

        for webhook in matching:
            result = self._deliver_one(webhook, body)
            results.append(result)

        return results

    def _deliver_one(self, webhook: WebhookConfig, body: bytes) -> Dict[str, Any]:
        """Deliver to a single webhook with retry."""
        import urllib.request
        import urllib.error

        headers = {
            "Content-Type": "application/json",
            "User-Agent": "Hermes-Agent-Webhook/1.0",
            **webhook.headers,
        }

        # HMAC signing
        if webhook.secret:
            signature = self._sign_payload(body, webhook.secret)
            headers["X-Hermes-Signature"] = signature

        last_error = None
        for attempt in range(self.MAX_RETRIES):
            try:
                req = urllib.request.Request(
                    webhook.url,
                    data=body,
                    headers=headers,
                    method="POST",
                )
                with urllib.request.urlopen(req, timeout=10) as resp:
                    return {
                        "webhook_id": webhook.id,
                        "url": webhook.url,
                        "status_code": resp.status,
                        "success": True,
                        "attempt": attempt + 1,
                    }
            except urllib.error.HTTPError as e:
                last_error = f"HTTP {e.code}: {e.reason}"
                log.warning("Webhook %s attempt %d failed: %s",
                           webhook.id, attempt + 1, last_error)
            except Exception as e:
                last_error = str(e)
                log.warning("Webhook %s attempt %d failed: %s",
                           webhook.id, attempt + 1, last_error)

            # Exponential backoff
            if attempt < self.MAX_RETRIES - 1:
                time.sleep(self.BACKOFF_BASE ** attempt)

        return {
            "webhook_id": webhook.id,
            "url": webhook.url,
            "status_code": 0,
            "success": False,
            "error": last_error,
            "attempt": self.MAX_RETRIES,
        }

    @staticmethod
    def _sign_payload(payload: bytes, secret: str) -> str:
        """HMAC-SHA256 signature for webhook payload verification."""
        signature = hmac.new(
            secret.encode("utf-8"),
            payload,
            hashlib.sha256,
        ).hexdigest()
        return f"sha256={signature}"

    def test_webhook(self, webhook_id: str) -> Dict[str, Any]:
        """Send a test payload to a specific webhook."""
        webhook = self._webhooks.get(webhook_id)
        if not webhook:
            return {"status": "error", "message": f"Webhook {webhook_id} not found"}

        test_payload = {
            "task_id": "test",
            "goal": "Webhook connectivity test",
            "status": "completed",
            "summary": "This is a test webhook delivery from Hermes Agent.",
            "duration_seconds": 0,
        }
        results = self.dispatch("test.ping", test_payload)
        return results[0] if results else {"status": "error", "message": "No delivery attempted"}
