"""Tests for webhook signature validation and event type extraction.

Covers:
- X-Hub-Signature (Fireflies V2) support alongside X-Hub-Signature-256
- ``event`` field in payload for event type extraction (Fireflies V2 payloads)
"""

import hashlib
import hmac
import unittest
from unittest.mock import MagicMock, patch


class _FakeRequest:
    """Minimal aiohttp-like request stub for _validate_signature."""

    def __init__(self, headers: dict | None = None):
        self.headers = headers or {}


class TestWebhookSignatureFirefliesV2(unittest.TestCase):
    """X-Hub-Signature (without -256 suffix) must be accepted."""

    def _make_adapter(self):
        """Create a minimal WebhookAdapter with _validate_signature accessible."""
        from gateway.platforms.webhook import WebhookAdapter

        adapter = object.__new__(WebhookAdapter)
        return adapter

    def test_x_hub_signature_without_256_suffix_accepted(self):
        """Fireflies V2 sends sha256=<hex> in X-Hub-Signature (no -256)."""
        adapter = self._make_adapter()
        secret = "test-secret-123"
        body = b'{"event": "meeting.completed"}'
        expected_sig = "sha256=" + hmac.new(
            secret.encode(), body, hashlib.sha256
        ).hexdigest()

        request = _FakeRequest(headers={"X-Hub-Signature": expected_sig})
        result = adapter._validate_signature(request, body, secret)
        assert result is True

    def test_x_hub_signature_256_still_works(self):
        """GitHub-style X-Hub-Signature-256 must continue to work."""
        adapter = self._make_adapter()
        secret = "test-secret-123"
        body = b'{"action": "push"}'
        expected_sig = "sha256=" + hmac.new(
            secret.encode(), body, hashlib.sha256
        ).hexdigest()

        request = _FakeRequest(headers={"X-Hub-Signature-256": expected_sig})
        result = adapter._validate_signature(request, body, secret)
        assert result is True

    def test_x_hub_signature_256_takes_precedence(self):
        """When both headers are present, X-Hub-Signature-256 wins."""
        adapter = self._make_adapter()
        secret = "test-secret-123"
        body = b'{"event": "test"}'
        correct_sig = "sha256=" + hmac.new(
            secret.encode(), body, hashlib.sha256
        ).hexdigest()
        wrong_sig = "sha256=deadbeef"

        # X-Hub-Signature-256 is correct, X-Hub-Signature is wrong
        request = _FakeRequest(headers={
            "X-Hub-Signature-256": correct_sig,
            "X-Hub-Signature": wrong_sig,
        })
        result = adapter._validate_signature(request, body, secret)
        assert result is True

    def test_x_hub_signature_wrong_value_rejected(self):
        """Wrong X-Hub-Signature value must be rejected."""
        adapter = self._make_adapter()
        secret = "test-secret-123"
        body = b'{"event": "test"}'

        request = _FakeRequest(headers={"X-Hub-Signature": "sha256=wrong"})
        result = adapter._validate_signature(request, body, secret)
        assert result is False


class TestWebhookEventTypeExtraction(unittest.TestCase):
    """Event type extraction must include ``event`` field (Fireflies V2)."""

    def test_event_field_from_payload(self):
        """Fireflies V2 uses 'event' as the field name."""
        # We test the extraction logic directly by checking the code path
        # The extraction is inline in _handle_webhook_request, so we verify
        # the code includes the right fallback chain
        import inspect
        from gateway.platforms.webhook import WebhookAdapter

        source = inspect.getsource(WebhookAdapter._handle_webhook)
        assert 'payload.get("event"' in source, (
            "Event type extraction must include payload.get('event') "
            "for Fireflies V2 compatibility"
        )

    def test_event_field_order_in_fallback_chain(self):
        """'event' must come after 'event_type' but before 'type'."""
        import inspect
        from gateway.platforms.webhook import WebhookAdapter

        source = inspect.getsource(WebhookAdapter._handle_webhook)
        # Find the event type extraction block
        idx_event_type = source.index('payload.get("event_type"')
        idx_event = source.index('payload.get("event"')
        idx_type = source.index('payload.get("type"')

        assert idx_event_type < idx_event < idx_type, (
            "Fallback order must be: event_type → event → type"
        )


if __name__ == "__main__":
    unittest.main()
