"""Regression tests for webhook security/ordering fixes (PR #104078).

Covers:
1. atexit registration guard (outbound_webhooks.py)
2. Idempotency check before rate limiting (webhook.py)
3. Event_type sanitization (webhook.py) via the handler, not a copied regex
4. Path resolution security (webhook_filters.py)
5. not_equals operator fail-closed on missing fields (webhook_filters.py)
"""
from __future__ import annotations

import asyncio
import json
import queue
import time
from unittest.mock import MagicMock, patch

from gateway.config import PlatformConfig
from gateway.platforms.webhook import WebhookAdapter, _INSECURE_NO_AUTH


def _make_adapter(**extra_kw) -> WebhookAdapter:
    extra = {"host": "127.0.0.1", "port": 0, "routes": {}, "rate_limit": 100}
    extra.update(extra_kw)
    return WebhookAdapter(PlatformConfig(enabled=True, extra=extra))


def _not_equals_op():
    from gateway.platforms.webhook_filters import _FIELD_OPERATORS

    for name, fn in _FIELD_OPERATORS:
        if name == "not_equals":
            return fn
    raise AssertionError("not_equals operator not found")


def _mock_request(event: str, delivery: str, body: bytes = b'{"ok": true}') -> MagicMock:
    req = MagicMock()
    req.headers = {
        "X-GitHub-Event": event,
        "X-GitHub-Delivery": delivery,
        "Content-Type": "application/json",
    }
    req.match_info = {"route_name": "r"}
    req.method = "POST"
    req.content_length = len(body)

    async def _read():
        return body

    req.read = _read
    return req


def _handle(adapter: WebhookAdapter, event: str, delivery: str):
    return asyncio.run(adapter._handle_webhook(_mock_request(event, delivery)))


def _json(resp) -> dict:
    return json.loads(resp.text)


# ---------------------------------------------------------------------------
# 1. atexit registration guard
# ---------------------------------------------------------------------------

class TestAtexitRegistrationGuard:
    def test_atexit_registered_only_once_across_worker_restart(self):
        import agent.outbound_webhooks as ow

        ow._worker = None
        ow._atexit_registered = False
        ow._delivery_queue = queue.Queue(maxsize=ow.QUEUE_MAX_SIZE)
        atexit_calls = []
        fake_worker = MagicMock()
        fake_worker.is_alive.return_value = True
        delivery = {
            "event": "test",
            "label": "t",
            "url": "http://127.0.0.1/unused",
            "body": b"{}",
            "headers": {},
            "timeout": 1,
        }
        with patch.object(ow.atexit, "register", lambda *a, **k: atexit_calls.append(a)), \
             patch.object(ow.threading, "Thread", lambda *a, **k: fake_worker):
            ow._enqueue(delivery)
            assert len(atexit_calls) == 1
            ow._worker = None
            fake_worker.is_alive.return_value = False
            ow._enqueue(delivery)
            assert len(atexit_calls) == 1, "worker restart must not re-register atexit"


# ---------------------------------------------------------------------------
# 2. Idempotency check before rate limiting
# ---------------------------------------------------------------------------

class TestWebhookIdempotencyOrdering:
    def test_record_delivery_id_returns_false_for_duplicate(self):
        adapter = _make_adapter()
        adapter._seen_deliveries.clear()
        now = time.time()
        assert adapter._record_delivery_id("dup-001", now) is True
        assert adapter._record_delivery_id("dup-001", now + 1) is False

    def test_record_delivery_id_returns_true_after_ttl_expires(self):
        adapter = _make_adapter()
        adapter._seen_deliveries.clear()
        now = time.time()
        adapter._record_delivery_id("old-001", now)
        ttl = getattr(adapter, "_idempotency_ttl", 300)
        assert adapter._record_delivery_id("old-001", now + ttl + 1) is True

    def test_different_delivery_ids_are_independent(self):
        adapter = _make_adapter()
        adapter._seen_deliveries.clear()
        now = time.time()
        for i in range(5):
            assert adapter._record_delivery_id(f"uniq-{i:03d}", now + i) is True

    def test_duplicate_delivery_does_not_consume_rate_limit(self):
        adapter = _make_adapter(
            routes={
                "r": {
                    "secret": _INSECURE_NO_AUTH,
                    "deliver": "log",
                    "deliver_only": True,
                    "prompt": "x",
                }
            },
            rate_limit=2,
        )
        first = _handle(adapter, "push", "same-id")
        assert first.status < 400 and first.status != 429
        dup = _handle(adapter, "push", "same-id")
        assert dup.status == 200
        assert _json(dup).get("status") == "duplicate"
        second = _handle(adapter, "push", "other-id")
        assert second.status != 429
        assert second.status < 400


# ---------------------------------------------------------------------------
# 3. Event_type sanitization
# ---------------------------------------------------------------------------

class TestEventTypeSanitization:
    def _adapter(self, **route_kw):
        route = {
            "secret": _INSECURE_NO_AUTH,
            "deliver": "log",
            "deliver_only": True,
            "prompt": "x",
        }
        route.update(route_kw)
        return _make_adapter(routes={"r": route})

    def test_newline_in_event_header_is_stripped(self):
        adapter = self._adapter(events=["push"])
        data = _json(_handle(adapter, "push\nContent-Length: 0", "evt-nl"))
        event = data.get("event") or ""
        assert "\n" not in event and "\r" not in event

    def test_long_event_type_is_capped_at_128(self):
        adapter = self._adapter(events=["push"])
        data = _json(_handle(adapter, "x" * 300, "evt-long"))
        assert len(data.get("event") or "") <= 128

    def test_normal_event_type_passes_through(self):
        captured: list[str] = []
        adapter = self._adapter()
        orig = adapter._render_prompt

        def _wrap(template, payload, event_type, route_name):
            captured.append(event_type)
            return orig(template, payload, event_type, route_name)

        adapter._render_prompt = _wrap
        resp = _handle(adapter, "push", "evt-ok")
        assert resp.status < 400
        assert captured == ["push"]


# ---------------------------------------------------------------------------
# 4. Path resolution security
# ---------------------------------------------------------------------------

class TestWebhookFilterPathResolution:
    def test_absolute_path_outside_hermes_home_is_rejected(self):
        from gateway.platforms.webhook_filters import _resolve_profile_path

        assert _resolve_profile_path("/etc/passwd") is None

    def test_relative_path_resolves_inside_hermes_home(self):
        from gateway.platforms.webhook_filters import _resolve_profile_path
        from hermes_constants import get_hermes_home

        result = _resolve_profile_path("some/file.json")
        assert result is not None
        assert result.is_relative_to(get_hermes_home().resolve())

    def test_tilde_hermes_path_resolves_correctly(self):
        from gateway.platforms.webhook_filters import _resolve_profile_path
        from hermes_constants import get_hermes_home

        result = _resolve_profile_path("~/.hermes/some/file.json")
        assert result is not None
        assert result == get_hermes_home() / "some/file.json"


# ---------------------------------------------------------------------------
# 5. not_equals operator
# ---------------------------------------------------------------------------

class TestWebhookFilterNotEquals:
    def test_not_equals_missing_field_returns_false(self):
        from gateway.platforms.webhook_filters import _MISSING

        assert _not_equals_op()(_MISSING, "anything") is False

    def test_not_equals_present_different_returns_true(self):
        assert _not_equals_op()("apple", "banana") is True

    def test_not_equals_present_same_returns_false(self):
        assert _not_equals_op()("same", "same") is False
