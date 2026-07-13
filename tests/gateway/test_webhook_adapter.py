"""Unit tests for the generic webhook platform adapter.

Covers:
- HMAC signature validation (GitHub, GitLab, generic)
- Prompt rendering with dot-notation template variables
- Event type filtering
- HTTP handler behaviour (404, 202, health)
- Idempotency cache (duplicate delivery IDs)
- Rate limiting (fixed-window, per route)
- Body size limits
- INSECURE_NO_AUTH bypass
- Session isolation for concurrent webhooks
- Delivery info cleanup after send()
- connect / disconnect lifecycle
"""

import asyncio
import base64
import hashlib
import hmac
import json
import time
from collections import deque
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import SendResult
from gateway.platforms.webhook import (
    WebhookAdapter,
    _INSECURE_NO_AUTH,
    check_webhook_requirements,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_config(
    routes=None,
    secret="",
    rate_limit=30,
    max_body_bytes=1_048_576,
    host="0.0.0.0",
    port=0,  # let OS pick a free port in tests
):
    """Build a PlatformConfig suitable for WebhookAdapter."""
    extra = {
        "host": host,
        "port": port,
        "routes": routes or {},
        "rate_limit": rate_limit,
        "max_body_bytes": max_body_bytes,
    }
    if secret:
        extra["secret"] = secret
    return PlatformConfig(enabled=True, extra=extra)


def _make_adapter(routes=None, **kwargs):
    """Create a WebhookAdapter with sensible defaults for testing."""
    config = _make_config(routes=routes, **kwargs)
    return WebhookAdapter(config)


def _create_app(adapter: WebhookAdapter) -> web.Application:
    """Build the aiohttp Application from the adapter (without starting a full server)."""
    # Mirror connect(): client_max_size enforces the cap on chunked bodies.
    app = web.Application(client_max_size=adapter._max_body_bytes)
    app.router.add_get("/health", adapter._handle_health)
    app.router.add_post("/webhooks/{route_name}", adapter._handle_webhook)
    return app


def _mock_request(headers=None, body=b"", content_length=None, match_info=None):
    """Build a lightweight mock aiohttp request for non-HTTP tests."""
    req = MagicMock()
    req.headers = headers or {}
    req.content_length = content_length if content_length is not None else len(body)
    req.match_info = match_info or {}
    req.method = "POST"

    async def _read():
        return body

    req.read = _read
    return req


def _github_signature(body: bytes, secret: str) -> str:
    """Compute X-Hub-Signature-256 for *body* using *secret*."""
    return "sha256=" + hmac.new(
        secret.encode(), body, hashlib.sha256
    ).hexdigest()


def _generic_signature(body: bytes, secret: str) -> str:
    """Compute X-Webhook-Signature (plain HMAC-SHA256 hex) for *body*."""
    return hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()


def _generic_v2_signature(body: bytes, secret: str, timestamp: str) -> str:
    """Compute X-Webhook-Signature-V2 (HMAC-SHA256 of "<timestamp>.<body>")."""
    signed_content = timestamp.encode() + b"." + body
    return hmac.new(secret.encode(), signed_content, hashlib.sha256).hexdigest()


def _svix_signature(body: bytes, secret: str, msg_id: str, timestamp: str) -> str:
    """Compute a Svix v1 signature header for *body* using *secret*."""
    key = (
        base64.b64decode(secret.removeprefix("whsec_"))
        if secret.startswith("whsec_")
        else secret.encode()
    )
    signed = msg_id.encode() + b"." + timestamp.encode() + b"." + body
    digest = hmac.new(key, signed, hashlib.sha256).digest()
    return "v1," + base64.b64encode(digest).decode()


# ===================================================================
# Signature validation
# ===================================================================


class TestValidateSignature:
    """Tests for WebhookAdapter._validate_signature."""

    def test_validate_github_signature_valid(self):
        """Valid X-Hub-Signature-256 is accepted."""
        adapter = _make_adapter()
        body = b'{"action": "opened"}'
        secret = "webhook-secret-42"
        sig = _github_signature(body, secret)
        req = _mock_request(headers={"X-Hub-Signature-256": sig})
        assert adapter._validate_signature(req, body, secret) is True

    def test_validate_github_signature_invalid(self):
        """Wrong X-Hub-Signature-256 is rejected."""
        adapter = _make_adapter()
        body = b'{"action": "opened"}'
        secret = "webhook-secret-42"
        req = _mock_request(headers={"X-Hub-Signature-256": "sha256=deadbeef"})
        assert adapter._validate_signature(req, body, secret) is False

    def test_validate_gitlab_token(self):
        """GitLab plain-token match via X-Gitlab-Token."""
        adapter = _make_adapter()
        secret = "gl-token-value"
        req = _mock_request(headers={"X-Gitlab-Token": secret})
        assert adapter._validate_signature(req, b"{}", secret) is True

    def test_validate_gitlab_token_wrong(self):
        """Wrong X-Gitlab-Token is rejected."""
        adapter = _make_adapter()
        req = _mock_request(headers={"X-Gitlab-Token": "wrong"})
        assert adapter._validate_signature(req, b"{}", "correct") is False

    def test_validate_no_signature_with_secret_rejects(self):
        """Secret configured but no recognised signature header → reject."""
        adapter = _make_adapter()
        req = _mock_request(headers={})  # no sig headers at all
        assert adapter._validate_signature(req, b"{}", "my-secret") is False

    def test_validate_no_secret_allows_all(self):
        """When the secret is empty/falsy, the validator is never even called
        by the handler (secret check is 'if secret and secret != _INSECURE...').
        Verify that an empty secret isn't accidentally passed to the validator."""
        # This tests the semantics: empty secret means skip validation entirely.
        # The handler code does: if secret and secret != _INSECURE_NO_AUTH: validate
        # So with an empty secret, _validate_signature is never reached.
        # We just verify the code path is correct by constructing an adapter
        # with no secret and confirming the route config resolves to "".
        adapter = _make_adapter(
            routes={"test": {"prompt": "hello"}},
            secret="",
        )
        # The route has no secret, global secret is empty
        route_secret = adapter._routes["test"].get("secret", adapter._global_secret)
        assert not route_secret  # empty → validation is skipped in handler

    def test_validate_generic_signature_valid(self):
        """Valid X-Webhook-Signature (generic HMAC-SHA256 hex) is accepted."""
        adapter = _make_adapter()
        body = b'{"event": "push"}'
        secret = "generic-secret"
        sig = _generic_signature(body, secret)
        req = _mock_request(headers={"X-Webhook-Signature": sig})
        assert adapter._validate_signature(req, body, secret) is True

    def test_validate_generic_v2_signature_valid(self):
        """Valid X-Webhook-Signature-V2 (timestamp-bound) is accepted."""
        adapter = _make_adapter()
        body = b'{"event": "push"}'
        secret = "generic-secret"
        timestamp = str(int(time.time()))
        sig = _generic_v2_signature(body, secret, timestamp)
        req = _mock_request(headers={
            "X-Webhook-Signature-V2": sig,
            "X-Webhook-Timestamp": timestamp,
        })
        assert adapter._validate_signature(req, body, secret) is True

    def test_validate_generic_v2_old_timestamp_rejects(self):
        """A V2 signature outside the replay window is rejected even though
        the HMAC itself would otherwise be valid for that (stale) timestamp
        — this is the actual replay-protection guarantee: an attacker who
        captured (body, signature, timestamp) once cannot resubmit it after
        the window closes."""
        adapter = _make_adapter()
        body = b'{"event": "push"}'
        secret = "generic-secret"
        timestamp = str(int(time.time()) - 301)
        sig = _generic_v2_signature(body, secret, timestamp)
        req = _mock_request(headers={
            "X-Webhook-Signature-V2": sig,
            "X-Webhook-Timestamp": timestamp,
        })
        assert adapter._validate_signature(req, body, secret) is False

    def test_validate_generic_v2_wrong_timestamp_rejects(self):
        """The timestamp is cryptographically bound into the V2 signature —
        this is the actual fix for the V1 replay hole. An attacker who only
        has a captured (body, signature) pair for V1 (no timestamp binding)
        cannot forge a valid V2 signature for a fresh timestamp without the
        secret, unlike V1 where the signature covers the body alone and a
        forged/fresh timestamp would otherwise sail through unverified."""
        adapter = _make_adapter()
        body = b'{"event": "push"}'
        secret = "generic-secret"
        real_timestamp = str(int(time.time()))
        sig = _generic_v2_signature(body, secret, real_timestamp)
        forged_timestamp = str(int(time.time()) + 1)
        req = _mock_request(headers={
            "X-Webhook-Signature-V2": sig,
            "X-Webhook-Timestamp": forged_timestamp,
        })
        assert adapter._validate_signature(req, body, secret) is False

    def test_validate_generic_v2_malformed_timestamp_rejects(self):
        adapter = _make_adapter()
        body = b'{"event": "push"}'
        secret = "generic-secret"
        req = _mock_request(headers={
            "X-Webhook-Signature-V2": "deadbeef",
            "X-Webhook-Timestamp": "not-a-number",
        })
        assert adapter._validate_signature(req, body, secret) is False

    def test_validate_generic_v1_still_works_without_timestamp(self):
        """Legacy V1 (body-only) senders that never send X-Webhook-Timestamp
        must keep working — this is the backward-compatibility guarantee for
        existing integrations that predate the V2 scheme."""
        adapter = _make_adapter()
        body = b'{"event": "push"}'
        secret = "generic-secret"
        sig = _generic_signature(body, secret)
        req = _mock_request(headers={"X-Webhook-Signature": sig})
        assert adapter._validate_signature(req, body, secret) is True

    def test_validate_generic_v2_preferred_when_both_sent(self):
        """If a sender sends both V1 and V2 headers (mid-migration), V2 must
        win — a stale/wrong V1 must not be able to override a valid V2."""
        adapter = _make_adapter()
        body = b'{"event": "push"}'
        secret = "generic-secret"
        timestamp = str(int(time.time()))
        v2_sig = _generic_v2_signature(body, secret, timestamp)
        req = _mock_request(headers={
            "X-Webhook-Signature-V2": v2_sig,
            "X-Webhook-Timestamp": timestamp,
            # Deliberately wrong V1 — must be ignored since V2 is checked first.
            "X-Webhook-Signature": "0" * 64,
        })
        assert adapter._validate_signature(req, body, secret) is True

    def test_validate_generic_v2_stripped_timestamp_does_not_downgrade_to_v1(self):
        """Regression test for a downgrade attack found in review: a sender
        migrating to V2 typically sends BOTH the V1 and V2 signatures
        together (for compatibility while both ends update). If an
        attacker captures one such mixed request and replays it with the
        X-Webhook-Timestamp header stripped, the presence of
        X-Webhook-Signature-V2 must still commit to V2 validation and
        reject — it must NOT silently fall through to validating the
        still-present, still-unprotected V1 signature instead. Falling
        through would let an attacker downgrade a V2-protected request
        back into the exact replay hole V2 exists to close, just by
        deleting one header from a captured request."""
        adapter = _make_adapter()
        body = b'{"event": "push"}'
        secret = "generic-secret"
        timestamp = str(int(time.time()))
        v2_sig = _generic_v2_signature(body, secret, timestamp)
        v1_sig = _generic_signature(body, secret)
        # Simulates a captured mixed V1+V2 request replayed with the
        # timestamp header stripped — V1 signature is still valid on its
        # own, but must not be reachable via this path.
        req = _mock_request(headers={
            "X-Webhook-Signature-V2": v2_sig,
            "X-Webhook-Signature": v1_sig,
            # X-Webhook-Timestamp deliberately omitted.
        })
        assert adapter._validate_signature(req, body, secret) is False

    def test_v1_replay_attack_succeeds_demonstrating_the_hole_v2_closes(self):
        """Regression/documentation test: a captured (body, signature) V1
        pair replays successfully no matter how much time has passed,
        because the V1 signature has no timestamp binding at all. This is
        the exact vulnerability V2 fixes — it is not asserting desired
        behavior, it is pinning the known, accepted-with-warning legacy
        gap so a future change to V1's semantics doesn't silently alter it
        without a deliberate decision."""
        adapter = _make_adapter()
        body = b'{"event": "push"}'
        secret = "generic-secret"
        sig = _generic_signature(body, secret)
        original_request = _mock_request(headers={"X-Webhook-Signature": sig})
        assert adapter._validate_signature(original_request, body, secret) is True
        # "Time passes" — nothing about a V1 signature depends on time, so
        # a captured pair replayed much later still validates.
        replayed_request = _mock_request(headers={"X-Webhook-Signature": sig})
        assert adapter._validate_signature(replayed_request, body, secret) is True

    def test_validate_svix_signature_valid(self):
        """Valid Svix/AgentMail v1 signature headers are accepted."""
        adapter = _make_adapter()
        body = b'{"event_type":"message.received"}'
        secret = "whsec_" + base64.b64encode(b"agentmail-signing-secret").decode()
        msg_id = "msg_123"
        timestamp = str(int(time.time()))
        sig = _svix_signature(body, secret, msg_id, timestamp)
        req = _mock_request(
            headers={
                "svix-id": msg_id,
                "svix-timestamp": timestamp,
                "svix-signature": sig,
            }
        )
        assert adapter._validate_signature(req, body, secret) is True

    def test_validate_svix_signature_wrong_body_rejects(self):
        """Svix/AgentMail signatures are bound to the exact raw request body."""
        adapter = _make_adapter()
        signed_body = b'{"event_type":"message.received"}'
        received_body = b'{"event_type":"message.sent"}'
        secret = "whsec_" + base64.b64encode(b"agentmail-signing-secret").decode()
        msg_id = "msg_123"
        timestamp = str(int(time.time()))
        sig = _svix_signature(signed_body, secret, msg_id, timestamp)
        req = _mock_request(
            headers={
                "svix-id": msg_id,
                "svix-timestamp": timestamp,
                "svix-signature": sig,
            }
        )
        assert adapter._validate_signature(req, received_body, secret) is False

    def test_validate_svix_signature_old_timestamp_rejects(self):
        """Svix/AgentMail signatures outside the replay window are rejected."""
        adapter = _make_adapter()
        body = b'{"event_type":"message.received"}'
        secret = "whsec_" + base64.b64encode(b"agentmail-signing-secret").decode()
        msg_id = "msg_123"
        timestamp = str(int(time.time()) - 301)
        sig = _svix_signature(body, secret, msg_id, timestamp)
        req = _mock_request(
            headers={
                "svix-id": msg_id,
                "svix-timestamp": timestamp,
                "svix-signature": sig,
            }
        )
        assert adapter._validate_signature(req, body, secret) is False

    def test_validate_svix_signature_multiple_entries_accepts_matching_v1(self):
        """Svix rotation headers may contain multiple space-separated signatures."""
        adapter = _make_adapter()
        body = b'{"event_type":"message.received"}'
        secret = "whsec_" + base64.b64encode(b"agentmail-signing-secret").decode()
        msg_id = "msg_123"
        timestamp = str(int(time.time()))
        sig = _svix_signature(body, secret, msg_id, timestamp)
        req = _mock_request(
            headers={
                "svix-id": msg_id,
                "svix-timestamp": timestamp,
                "svix-signature": "v1,wrong " + sig,
            }
        )
        assert adapter._validate_signature(req, body, secret) is True

    def test_validate_svix_signature_missing_signature_rejects(self):
        """Partial Svix headers reject instead of falling through to another scheme."""
        adapter = _make_adapter()
        req = _mock_request(headers={"svix-id": "msg_123"})
        assert adapter._validate_signature(req, b"{}", "secret") is False

    def test_validate_svix_signature_unsupported_version_rejects(self):
        """Only Svix v1 signatures are accepted."""
        adapter = _make_adapter()
        body = b'{"event_type":"message.received"}'
        secret = "whsec_" + base64.b64encode(b"agentmail-signing-secret").decode()
        msg_id = "msg_123"
        timestamp = str(int(time.time()))
        sig = _svix_signature(body, secret, msg_id, timestamp).replace("v1,", "v2,")
        req = _mock_request(
            headers={
                "svix-id": msg_id,
                "svix-timestamp": timestamp,
                "svix-signature": sig,
            }
        )
        assert adapter._validate_signature(req, body, secret) is False

    def test_validate_svix_signature_invalid_whsec_rejects(self):
        """Malformed whsec_ secrets are rejected, not silently treated as raw secrets."""
        adapter = _make_adapter()
        body = b'{"event_type":"message.received"}'
        malformed_secret = "whsec_not-valid-base64!"
        msg_id = "msg_123"
        timestamp = str(int(time.time()))
        raw_sig = _svix_signature(
            body, malformed_secret.removeprefix("whsec_"), msg_id, timestamp
        )
        req = _mock_request(
            headers={
                "svix-id": msg_id,
                "svix-timestamp": timestamp,
                "svix-signature": raw_sig,
            }
        )
        assert adapter._validate_signature(req, body, malformed_secret) is False

    def test_validate_svix_signature_raw_secret_valid(self):
        """Raw shared secrets are accepted for Svix-style senders without whsec_ secrets."""
        adapter = _make_adapter()
        body = b'{"event_type":"message.received"}'
        secret = "raw-agentmail-secret"
        msg_id = "msg_123"
        timestamp = str(int(time.time()))
        sig = _svix_signature(body, secret, msg_id, timestamp)
        req = _mock_request(
            headers={
                "svix-id": msg_id,
                "svix-timestamp": timestamp,
                "svix-signature": sig,
            }
        )
        assert adapter._validate_signature(req, body, secret) is True


# ===================================================================
# Prompt rendering
# ===================================================================


class TestRenderPrompt:
    """Tests for WebhookAdapter._render_prompt."""

    def test_render_prompt_dot_notation(self):
        """Dot-notation {pull_request.title} resolves nested keys."""
        adapter = _make_adapter()
        payload = {"pull_request": {"title": "Fix bug", "number": 42}}
        result = adapter._render_prompt(
            "PR #{pull_request.number}: {pull_request.title}",
            payload,
            "pull_request",
            "github",
        )
        assert result == "PR #42: Fix bug"

    def test_render_prompt_missing_key_preserved(self):
        """{nonexistent} is left as-is when key doesn't exist in payload."""
        adapter = _make_adapter()
        result = adapter._render_prompt(
            "Hello {nonexistent}!",
            {"action": "opened"},
            "push",
            "test",
        )
        assert "{nonexistent}" in result

    def test_render_prompt_no_template_dumps_json(self):
        """Empty template → JSON dump fallback with event/route context."""
        adapter = _make_adapter()
        payload = {"key": "value"}
        result = adapter._render_prompt("", payload, "push", "my-route")
        assert "push" in result
        assert "my-route" in result
        assert "key" in result


# ===================================================================
# Delivery extra rendering
# ===================================================================


class TestRenderDeliveryExtra:
    def test_render_delivery_extra_templates(self):
        """String values in deliver_extra are rendered with payload data."""
        adapter = _make_adapter()
        extra = {"repo": "{repository.full_name}", "pr_number": "{number}", "static": 42}
        payload = {"repository": {"full_name": "org/repo"}, "number": 7}
        result = adapter._render_delivery_extra(extra, payload)
        assert result["repo"] == "org/repo"
        assert result["pr_number"] == "7"
        assert result["static"] == 42  # non-string left as-is


# ===================================================================
# Event filtering
# ===================================================================


class TestEventFilter:
    """Tests for event type filtering in _handle_webhook."""

    @pytest.mark.asyncio
    async def test_event_filter_accepts_matching(self):
        """Matching event type passes through."""
        routes = {
            "gh": {
                "secret": _INSECURE_NO_AUTH,
                "events": ["pull_request"],
                "prompt": "PR: {action}",
            }
        }
        adapter = _make_adapter(routes=routes)
        # Stub handle_message to avoid running the agent
        adapter.handle_message = AsyncMock()

        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.post(
                "/webhooks/gh",
                json={"action": "opened"},
                headers={"X-GitHub-Event": "pull_request"},
            )
            assert resp.status == 202

    @pytest.mark.asyncio
    async def test_event_filter_rejects_non_matching(self):
        """Non-matching event type returns 200 with status=ignored."""
        routes = {
            "gh": {
                "secret": _INSECURE_NO_AUTH,
                "events": ["pull_request"],
                "prompt": "test",
            }
        }
        adapter = _make_adapter(routes=routes)

        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.post(
                "/webhooks/gh",
                json={"action": "opened"},
                headers={"X-GitHub-Event": "push"},
            )
            assert resp.status == 200
            data = await resp.json()
            assert data["status"] == "ignored"

    @pytest.mark.asyncio
    async def test_event_filter_empty_allows_all(self):
        """No events list → accept any event type."""
        routes = {
            "all": {
                "secret": _INSECURE_NO_AUTH,
                "prompt": "got it",
            }
        }
        adapter = _make_adapter(routes=routes)
        adapter.handle_message = AsyncMock()

        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.post(
                "/webhooks/all",
                json={"action": "any"},
                headers={"X-GitHub-Event": "whatever"},
            )
            assert resp.status == 202

    @pytest.mark.asyncio
    async def test_event_filter_accepts_payload_type_field(self):
        """Svix-style payloads often use a top-level `type` event field."""
        routes = {
            "svix": {
                "secret": _INSECURE_NO_AUTH,
                "events": ["message.received"],
                "prompt": "got it",
            }
        }
        adapter = _make_adapter(routes=routes)
        adapter.handle_message = AsyncMock()

        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.post(
                "/webhooks/svix",
                json={"type": "message.received"},
            )
            assert resp.status == 202


# ===================================================================
# Payload filters
# ===================================================================


class TestPayloadFilters:
    """Tests for route-level payload filters in _handle_webhook."""

    @pytest.mark.asyncio
    async def test_filter_rejects_before_agent_dispatch(self):
        """A non-matching filter returns ignored and never starts the agent."""
        routes = {
            "todoist": {
                "secret": _INSECURE_NO_AUTH,
                "filters": [{"field": "payload.label", "equals": "urgent"}],
                "prompt": "Task: {payload.content}",
            }
        }
        adapter = _make_adapter(routes=routes)
        adapter.handle_message = AsyncMock()

        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.post(
                "/webhooks/todoist",
                json={"payload": {"label": "later", "content": "Buy milk"}},
                headers={"X-GitHub-Delivery": "filter-skip-1"},
            )
            assert resp.status == 200
            data = await resp.json()
            assert data == {
                "status": "ignored",
                "reason": "filter",
                "route": "todoist",
            }

        adapter.handle_message.assert_not_called()
        assert "filter-skip-1" not in adapter._seen_deliveries

    @pytest.mark.asyncio
    async def test_filter_accepts_nested_any_and_in_file(self, tmp_path, monkeypatch):
        """Nested any groups can match dynamic watchlists under HERMES_HOME."""
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        watchlist = tmp_path / "data" / "watchlist.json"
        watchlist.parent.mkdir()
        watchlist.write_text(json.dumps(["chat-1", "chat-2"]), encoding="utf-8")
        routes = {
            "waha": {
                "secret": _INSECURE_NO_AUTH,
                "filters": [
                    {"field": "payload.fromMe", "equals": False},
                    {
                        "any": [
                            {
                                "field": "payload.chatId",
                                "in_file": "~/.hermes/data/watchlist.json",
                            },
                            {
                                "field": "payload.id.remote",
                                "in_file": "~/.hermes/data/watchlist.json",
                            },
                        ]
                    },
                ],
                "prompt": "Message from {payload.chatId}: {payload.body}",
            }
        }
        adapter = _make_adapter(routes=routes)
        captured = []

        async def _capture(event):
            captured.append(event)

        adapter.handle_message = _capture

        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.post(
                "/webhooks/waha",
                json={
                    "payload": {
                        "fromMe": False,
                        "chatId": "chat-2",
                        "body": "hello",
                    }
                },
                headers={"X-GitHub-Delivery": "filter-match-1"},
            )
            assert resp.status == 202

        await asyncio.sleep(0.05)
        assert len(captured) == 1
        assert captured[0].text == "Message from chat-2: hello"

    @pytest.mark.asyncio
    async def test_filter_applies_to_deliver_only_before_delivery(self):
        """Filtered direct-delivery routes skip target delivery too."""
        routes = {
            "alerts": {
                "secret": _INSECURE_NO_AUTH,
                "deliver": "telegram",
                "deliver_only": True,
                "deliver_extra": {"chat_id": "123"},
                "filters": [{"field": "severity", "in": ["critical"]}],
                "prompt": "Alert: {message}",
            }
        }
        adapter = _make_adapter(routes=routes)
        mock_target = AsyncMock()
        mock_target.send = AsyncMock(return_value=SendResult(success=True))
        mock_runner = MagicMock()
        mock_runner.adapters = {Platform.TELEGRAM: mock_target}
        adapter.gateway_runner = mock_runner

        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.post(
                "/webhooks/alerts",
                json={"severity": "info", "message": "noise"},
                headers={"X-GitHub-Delivery": "filter-direct-1"},
            )
            assert resp.status == 200
            assert (await resp.json())["reason"] == "filter"

        mock_target.send.assert_not_called()

    @pytest.mark.asyncio
    async def test_script_transforms_payload_before_prompt_rendering(self, tmp_path, monkeypatch):
        """A script can replace the payload used by prompt templates."""
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        scripts = tmp_path / "scripts"
        scripts.mkdir()
        script = scripts / "todoist_filter.py"
        script.write_text(
            "import json, sys\n"
            "payload = json.load(sys.stdin)\n"
            "payload['body'] = payload['task']['content'].upper()\n"
            "print(json.dumps(payload))\n",
            encoding="utf-8",
        )
        routes = {
            "todoist": {
                "secret": _INSECURE_NO_AUTH,
                "script": "todoist_filter.py",
                "prompt": "Task: {body}",
            }
        }
        adapter = _make_adapter(routes=routes)
        captured = []

        async def _capture(event):
            captured.append(event)

        adapter.handle_message = _capture

        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.post(
                "/webhooks/todoist",
                json={"task": {"content": "pay bills"}},
                headers={"X-GitHub-Delivery": "script-transform-1"},
            )
            assert resp.status == 202

        await asyncio.sleep(0.05)
        assert captured[0].text == "Task: PAY BILLS"
        assert captured[0].raw_message["body"] == "PAY BILLS"

    @pytest.mark.asyncio
    async def test_script_tilde_hermes_path_resolves_to_active_profile_home(self, tmp_path, monkeypatch):
        """~/.hermes/scripts paths must resolve through HERMES_HOME for profiles."""
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        scripts = tmp_path / "scripts"
        scripts.mkdir()
        (scripts / "todoist_filter.py").write_text(
            "import json, sys\n"
            "payload = json.load(sys.stdin)\n"
            "payload['body'] = 'profile-safe'\n"
            "print(json.dumps(payload))\n",
            encoding="utf-8",
        )
        routes = {
            "todoist": {
                "secret": _INSECURE_NO_AUTH,
                "script": "~/.hermes/scripts/todoist_filter.py",
                "prompt": "Task: {body}",
            }
        }
        adapter = _make_adapter(routes=routes)
        captured = []

        async def _capture(event):
            captured.append(event)

        adapter.handle_message = _capture

        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.post(
                "/webhooks/todoist",
                json={"task": {"content": "pay bills"}},
                headers={"X-GitHub-Delivery": "script-profile-path-1"},
            )
            assert resp.status == 202

        await asyncio.sleep(0.05)
        assert captured[0].text == "Task: profile-safe"

    @pytest.mark.asyncio
    async def test_script_silent_stdout_ignores_without_idempotency_hit(self, tmp_path, monkeypatch):
        """Empty or [SILENT] script stdout filters the webhook out."""
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        scripts = tmp_path / "scripts"
        scripts.mkdir()
        (scripts / "skip.py").write_text("print('[SILENT]')\n", encoding="utf-8")
        routes = {
            "todoist": {
                "secret": _INSECURE_NO_AUTH,
                "script": "skip.py",
                "prompt": "Task: {body}",
            }
        }
        adapter = _make_adapter(routes=routes)
        adapter.handle_message = AsyncMock()

        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.post(
                "/webhooks/todoist",
                json={"body": "ignore me"},
                headers={"X-GitHub-Delivery": "script-silent-1"},
            )
            assert resp.status == 200
            data = await resp.json()
            assert data == {
                "status": "ignored",
                "reason": "script",
                "route": "todoist",
            }

        adapter.handle_message.assert_not_called()
        assert "script-silent-1" not in adapter._seen_deliveries

    @pytest.mark.asyncio
    async def test_script_nonzero_exit_ignores_webhook(self, tmp_path, monkeypatch):
        """A script can fail closed by exiting nonzero."""
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        scripts = tmp_path / "scripts"
        scripts.mkdir()
        (scripts / "skip.py").write_text(
            "import sys\nsys.exit(2)\n",
            encoding="utf-8",
        )
        routes = {
            "todoist": {
                "secret": _INSECURE_NO_AUTH,
                "script": "skip.py",
                "prompt": "Task: {body}",
            }
        }
        adapter = _make_adapter(routes=routes)
        adapter.handle_message = AsyncMock()

        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.post(
                "/webhooks/todoist",
                json={"body": "ignore me"},
                headers={"X-GitHub-Delivery": "script-nonzero-1"},
            )
            assert resp.status == 200
            data = await resp.json()
            assert data["status"] == "ignored"
            assert data["reason"] == "script"

        adapter.handle_message.assert_not_called()
        assert "script-nonzero-1" not in adapter._seen_deliveries


# ===================================================================
# HTTP handling
# ===================================================================


class TestHTTPHandling:

    @pytest.mark.asyncio
    async def test_unknown_route_returns_404(self):
        """POST to an unknown route returns 404."""
        adapter = _make_adapter(routes={"real": {"secret": _INSECURE_NO_AUTH, "prompt": "x"}})
        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.post("/webhooks/nonexistent", json={"a": 1})
            assert resp.status == 404

    @pytest.mark.asyncio
    async def test_webhook_handler_returns_202(self):
        """Valid request returns 202 Accepted."""
        routes = {"test": {"secret": _INSECURE_NO_AUTH, "prompt": "hi"}}
        adapter = _make_adapter(routes=routes)
        adapter.handle_message = AsyncMock()

        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.post("/webhooks/test", json={"data": "value"})
            assert resp.status == 202
            data = await resp.json()
            assert data["status"] == "accepted"
            assert data["route"] == "test"

    @pytest.mark.asyncio
    async def test_route_without_secret_rejects_unsigned_request(self):
        """Missing HMAC secret must fail closed even if connect() was bypassed."""
        routes = {"test": {"prompt": "hi"}}
        adapter = _make_adapter(routes=routes, secret="")
        adapter.handle_message = AsyncMock()

        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.post("/webhooks/test", json={"data": "value"})
            assert resp.status == 403
            data = await resp.json()
            assert data["error"] == "Webhook route is missing an HMAC secret"

        adapter.handle_message.assert_not_called()

    @pytest.mark.asyncio
    async def test_health_endpoint(self):
        """GET /health returns 200 with status=ok."""
        adapter = _make_adapter()
        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.get("/health")
            assert resp.status == 200
            data = await resp.json()
            assert data["status"] == "ok"
            assert data["platform"] == "webhook"

    @pytest.mark.asyncio
    async def test_connect_starts_server(self):
        """connect() starts the HTTP listener and marks adapter as connected."""
        routes = {"r1": {"secret": _INSECURE_NO_AUTH, "prompt": "x"}}
        adapter = _make_adapter(routes=routes, host="127.0.0.1", port=0)
        # Use port 0 — the OS picks a free port, but aiohttp requires a real bind.
        # We just test that the method completes and marks connected.
        # Need to mock TCPSite to avoid actual binding.
        with patch("gateway.platforms.webhook.web.AppRunner") as MockRunner, \
             patch("gateway.platforms.webhook.web.TCPSite") as MockSite:
            mock_runner_inst = AsyncMock()
            MockRunner.return_value = mock_runner_inst
            mock_site_inst = AsyncMock()
            MockSite.return_value = mock_site_inst

            result = await adapter.connect()
            assert result is True
            assert adapter.is_connected
            mock_runner_inst.setup.assert_awaited_once()
            mock_site_inst.start.assert_awaited_once()

        await adapter.disconnect()

    @pytest.mark.asyncio
    async def test_disconnect_cleans_up(self):
        """disconnect() stops the server and marks adapter disconnected."""
        adapter = _make_adapter()
        # Simulate a runner that was previously set up
        mock_runner = AsyncMock()
        adapter._runner = mock_runner
        adapter._running = True

        await adapter.disconnect()
        mock_runner.cleanup.assert_awaited_once()
        assert adapter._runner is None
        assert not adapter.is_connected


# ===================================================================
# Idempotency
# ===================================================================


class TestIdempotency:

    @pytest.mark.asyncio
    async def test_duplicate_delivery_id_returns_200(self):
        """Second request with same delivery ID returns 200 duplicate."""
        routes = {"idem": {"secret": _INSECURE_NO_AUTH, "prompt": "test"}}
        adapter = _make_adapter(routes=routes)
        adapter.handle_message = AsyncMock()

        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            headers = {"X-GitHub-Delivery": "delivery-123"}
            resp1 = await cli.post("/webhooks/idem", json={"a": 1}, headers=headers)
            assert resp1.status == 202

            resp2 = await cli.post("/webhooks/idem", json={"a": 1}, headers=headers)
            assert resp2.status == 200
            data = await resp2.json()
            assert data["status"] == "duplicate"

    @pytest.mark.asyncio
    async def test_expired_delivery_id_allows_reprocess(self):
        """After TTL expires, the same delivery ID is accepted again."""
        routes = {"idem": {"secret": _INSECURE_NO_AUTH, "prompt": "test"}}
        adapter = _make_adapter(routes=routes)
        adapter._idempotency_ttl = 1  # 1 second TTL for test speed
        adapter.handle_message = AsyncMock()

        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            headers = {"X-GitHub-Delivery": "delivery-456"}

            resp1 = await cli.post("/webhooks/idem", json={"x": 1}, headers=headers)
            assert resp1.status == 202

            # Backdate the cache entry so it appears expired
            adapter._seen_deliveries["delivery-456"] = time.time() - 3700

            resp2 = await cli.post("/webhooks/idem", json={"x": 1}, headers=headers)
            assert resp2.status == 202  # re-accepted

    @pytest.mark.asyncio
    async def test_svix_id_used_as_delivery_id_for_deduplication(self):
        """Svix retries reuse svix-id, so use it as the delivery ID when present."""
        routes = {"idem": {"secret": _INSECURE_NO_AUTH, "prompt": "test"}}
        adapter = _make_adapter(routes=routes)
        adapter.handle_message = AsyncMock()

        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            headers = {"svix-id": "msg_duplicate"}
            resp1 = await cli.post("/webhooks/idem", json={"a": 1}, headers=headers)
            assert resp1.status == 202

            resp2 = await cli.post("/webhooks/idem", json={"a": 1}, headers=headers)
            assert resp2.status == 200
            data = await resp2.json()
            assert data["status"] == "duplicate"
            assert data["delivery_id"] == "msg_duplicate"


# ===================================================================
# Rate limiting
# ===================================================================


class TestRateLimiting:

    @pytest.mark.asyncio
    async def test_rate_limit_rejects_excess(self):
        """Exceeding the rate limit returns 429."""
        routes = {"limited": {"secret": _INSECURE_NO_AUTH, "prompt": "test"}}
        adapter = _make_adapter(routes=routes, rate_limit=2)
        adapter.handle_message = AsyncMock()

        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            # Two requests within limit
            for i in range(2):
                resp = await cli.post(
                    "/webhooks/limited",
                    json={"n": i},
                    headers={"X-GitHub-Delivery": f"d-{i}"},
                )
                assert resp.status == 202, f"Request {i} should be accepted"

            # Third request should be rate-limited
            resp = await cli.post(
                "/webhooks/limited",
                json={"n": 99},
                headers={"X-GitHub-Delivery": "d-99"},
            )
            assert resp.status == 429

    @pytest.mark.asyncio
    async def test_rate_limit_window_resets(self):
        """After the 60-second window passes, requests are allowed again."""
        routes = {"limited": {"secret": _INSECURE_NO_AUTH, "prompt": "test"}}
        adapter = _make_adapter(routes=routes, rate_limit=1)
        adapter.handle_message = AsyncMock()

        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.post(
                "/webhooks/limited",
                json={"n": 1},
                headers={"X-GitHub-Delivery": "d-a"},
            )
            assert resp.status == 202

            # Backdate all rate-limit timestamps to > 60 seconds ago
            adapter._rate_counts["limited"] = deque([time.time() - 120])

            resp = await cli.post(
                "/webhooks/limited",
                json={"n": 2},
                headers={"X-GitHub-Delivery": "d-b"},
            )
            assert resp.status == 202  # allowed again

    def test_rate_limit_prunes_incrementally_from_left(self):
        """Expired rate-limit entries are pruned without rebuilding the window."""
        adapter = _make_adapter(rate_limit=2)
        adapter._rate_counts["limited"] = deque([100.0, 220.0])

        assert adapter._record_rate_limit_hit("limited", 221.0) is True

        window = adapter._rate_counts["limited"]
        assert list(window) == [220.0, 221.0]

    def test_seen_delivery_ttl_is_checked_per_delivery_without_full_prune(self):
        """Expired delivery IDs can reprocess even when stale siblings remain."""
        adapter = _make_adapter(rate_limit=1)
        adapter._idempotency_ttl = 60
        adapter._seen_deliveries = {
            "expired-target": 100.0,
            "expired-sibling": 101.0,
            "fresh-sibling": 155.0,
        }

        now = 200.0
        assert adapter._record_delivery_id("expired-target", now) is True

        assert adapter._seen_deliveries["expired-target"] == now
        assert "expired-sibling" in adapter._seen_deliveries
        assert "fresh-sibling" in adapter._seen_deliveries


# ===================================================================
# Body size limit
# ===================================================================


class TestBodySize:

    @pytest.mark.asyncio
    async def test_oversized_payload_rejected(self):
        """Content-Length > max_body_bytes returns 413."""
        routes = {"big": {"secret": _INSECURE_NO_AUTH, "prompt": "test"}}
        adapter = _make_adapter(routes=routes, max_body_bytes=100)

        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            large_payload = {"data": "x" * 200}
            resp = await cli.post(
                "/webhooks/big",
                json=large_payload,
                headers={"Content-Length": "999999"},
            )
            assert resp.status == 413

    @pytest.mark.asyncio
    async def test_chunked_oversized_payload_rejected(self):
        """Chunked request bodies (no Content-Length) over the limit return 413."""
        routes = {"big": {"secret": _INSECURE_NO_AUTH, "prompt": "test"}}
        adapter = _make_adapter(routes=routes, max_body_bytes=100)
        adapter.handle_message = AsyncMock()

        async def _chunked_body():
            payload = json.dumps({"data": "x" * 500}).encode("utf-8")
            for i in range(0, len(payload), 64):
                yield payload[i : i + 64]
                await asyncio.sleep(0)

        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.post(
                "/webhooks/big",
                data=_chunked_body(),
                headers={"Content-Type": "application/json"},
            )
            assert resp.status == 413
            adapter.handle_message.assert_not_awaited()


# ===================================================================
# INSECURE_NO_AUTH
# ===================================================================


class TestInsecureNoAuth:

    @pytest.mark.asyncio
    async def test_insecure_no_auth_skips_validation(self):
        """Setting secret to _INSECURE_NO_AUTH bypasses signature check."""
        routes = {"open": {"secret": _INSECURE_NO_AUTH, "prompt": "hello"}}
        adapter = _make_adapter(routes=routes)
        adapter.handle_message = AsyncMock()

        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            # No signature header at all — should still be accepted
            resp = await cli.post("/webhooks/open", json={"test": True})
            assert resp.status == 202


# ===================================================================
# Session isolation
# ===================================================================


class TestSessionIsolation:

    @pytest.mark.asyncio
    async def test_concurrent_webhooks_get_independent_sessions(self):
        """Two events on the same route produce different session keys."""
        routes = {"ci": {"secret": _INSECURE_NO_AUTH, "prompt": "build"}}
        adapter = _make_adapter(routes=routes)

        captured_events = []

        async def _capture(event):
            captured_events.append(event)

        adapter.handle_message = _capture

        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            resp1 = await cli.post(
                "/webhooks/ci",
                json={"ref": "main"},
                headers={"X-GitHub-Delivery": "aaa-111"},
            )
            assert resp1.status == 202

            resp2 = await cli.post(
                "/webhooks/ci",
                json={"ref": "dev"},
                headers={"X-GitHub-Delivery": "bbb-222"},
            )
            assert resp2.status == 202

        # Wait for the async tasks to be created
        await asyncio.sleep(0.05)

        assert len(captured_events) == 2
        ids = {ev.source.chat_id for ev in captured_events}
        assert len(ids) == 2, "Each delivery must have a unique session chat_id"


# ===================================================================
# Delivery info cleanup
# ===================================================================


class TestDeliveryCleanup:

    @pytest.mark.asyncio
    async def test_delivery_info_survives_multiple_sends(self):
        """send() must NOT pop delivery_info.

        Interim status messages (fallback notifications, context-pressure
        warnings, etc.) flow through the same send() path as the final
        response.  If the entry were popped on the first send, the final
        response would silently downgrade to the ``log`` deliver type.
        Regression test for that bug.
        """
        adapter = _make_adapter()
        chat_id = "webhook:test:d-xyz"
        adapter._delivery_info[chat_id] = {
            "deliver": "log",
            "deliver_extra": {},
        }
        adapter._delivery_info_created[chat_id] = time.time()

        # First send (e.g. an interim status message)
        result1 = await adapter.send(chat_id, "Status: switching to fallback")
        assert result1.success is True
        # Entry must still be present so the final send can read it
        assert chat_id in adapter._delivery_info

        # Second send (the final agent response)
        result2 = await adapter.send(chat_id, "Final agent response")
        assert result2.success is True
        assert chat_id in adapter._delivery_info

    @pytest.mark.asyncio
    async def test_delivery_info_pruned_via_ttl(self):
        """Stale delivery_info entries are dropped on the next POST."""
        adapter = _make_adapter()
        adapter._idempotency_ttl = 60  # short TTL for the test
        now = time.time()

        # Stale entry — older than TTL
        adapter._delivery_info["webhook:test:old"] = {"deliver": "log"}
        adapter._delivery_info_created["webhook:test:old"] = now - 120

        # Fresh entry — should survive
        adapter._delivery_info["webhook:test:new"] = {"deliver": "log"}
        adapter._delivery_info_created["webhook:test:new"] = now - 5

        adapter._prune_delivery_info(now)

        assert "webhook:test:old" not in adapter._delivery_info
        assert "webhook:test:old" not in adapter._delivery_info_created
        assert "webhook:test:new" in adapter._delivery_info
        assert "webhook:test:new" in adapter._delivery_info_created

    @pytest.mark.asyncio
    async def test_delivery_info_prune_uses_ordered_incremental_queue(self):
        """Delivery-info TTL pruning stops at the first fresh queued entry."""
        adapter = _make_adapter()
        adapter._idempotency_ttl = 60
        now = 1000.0
        for key, created_at in (
            ("webhook:test:old", now - 120),
            ("webhook:test:new", now - 5),
            ("webhook:test:newer", now),
        ):
            adapter._delivery_info[key] = {"deliver": "log"}
            adapter._delivery_info_created[key] = created_at
            adapter._delivery_info_order.append((created_at, key))

        adapter._prune_delivery_info(now)

        assert "webhook:test:old" not in adapter._delivery_info
        assert "webhook:test:new" in adapter._delivery_info
        assert "webhook:test:newer" in adapter._delivery_info
        assert list(adapter._delivery_info_order) == [
            (now - 5, "webhook:test:new"),
            (now, "webhook:test:newer"),
        ]


# ===================================================================
# check_webhook_requirements
# ===================================================================


class TestCheckRequirements:
    def test_returns_true_when_aiohttp_available(self):
        assert check_webhook_requirements() is True

    @patch("gateway.platforms.webhook.AIOHTTP_AVAILABLE", False)
    def test_returns_false_without_aiohttp(self):
        assert check_webhook_requirements() is False


# ===================================================================
# __raw__ template token
# ===================================================================


class TestRawTemplateToken:
    """Tests for the {__raw__} special token in _render_prompt."""

    def test_raw_resolves_to_full_json_payload(self):
        """{__raw__} in a template dumps the entire payload as JSON."""
        adapter = _make_adapter()
        payload = {"action": "opened", "number": 42}
        result = adapter._render_prompt(
            "Payload: {__raw__}", payload, "push", "test"
        )
        expected_json = json.dumps(payload, indent=2)
        assert result == f"Payload: {expected_json}"

    def test_raw_truncated_at_4000_chars(self):
        """{__raw__} output is truncated at 4000 characters for large payloads."""
        adapter = _make_adapter()
        # Build a payload whose JSON repr exceeds 4000 chars
        payload = {"data": "x" * 5000}
        result = adapter._render_prompt("{__raw__}", payload, "push", "test")
        assert len(result) <= 4000

    def test_raw_mixed_with_other_variables(self):
        """{__raw__} can be mixed with regular template variables."""
        adapter = _make_adapter()
        payload = {"action": "closed", "number": 7}
        result = adapter._render_prompt(
            "Action={action} Raw={__raw__}", payload, "push", "test"
        )
        assert result.startswith("Action=closed Raw=")
        assert '"action": "closed"' in result
        assert '"number": 7' in result


# ===================================================================
# Cross-platform delivery thread_id passthrough
# ===================================================================


class TestDeliverCrossPlatformThreadId:
    """Tests for thread_id passthrough in _deliver_cross_platform."""

    def _setup_adapter_with_mock_target(self):
        """Set up a webhook adapter with a mocked gateway_runner and target adapter."""
        adapter = _make_adapter()
        mock_target = AsyncMock()
        mock_target.send = AsyncMock(return_value=SendResult(success=True))

        mock_runner = MagicMock()
        mock_runner.adapters = {Platform("telegram"): mock_target}
        mock_runner.config.get_home_channel.return_value = None

        adapter.gateway_runner = mock_runner
        return adapter, mock_target

    @pytest.mark.asyncio
    async def test_thread_id_passed_as_metadata(self):
        """thread_id from deliver_extra is passed as metadata to adapter.send()."""
        adapter, mock_target = self._setup_adapter_with_mock_target()
        delivery = {
            "deliver_extra": {
                "chat_id": "12345",
                "thread_id": "999",
            }
        }
        await adapter._deliver_cross_platform("telegram", "hello", delivery)
        mock_target.send.assert_awaited_once_with(
            "12345", "hello", metadata={"thread_id": "999"}
        )

    @pytest.mark.asyncio
    async def test_message_thread_id_passed_as_thread_id(self):
        """message_thread_id from deliver_extra is mapped to thread_id in metadata."""
        adapter, mock_target = self._setup_adapter_with_mock_target()
        delivery = {
            "deliver_extra": {
                "chat_id": "12345",
                "message_thread_id": "888",
            }
        }
        await adapter._deliver_cross_platform("telegram", "hello", delivery)
        mock_target.send.assert_awaited_once_with(
            "12345", "hello", metadata={"thread_id": "888"}
        )

    @pytest.mark.asyncio
    async def test_no_thread_id_sends_no_metadata(self):
        """When no thread_id is present, metadata is None."""
        adapter, mock_target = self._setup_adapter_with_mock_target()
        delivery = {
            "deliver_extra": {
                "chat_id": "12345",
            }
        }
        await adapter._deliver_cross_platform("telegram", "hello", delivery)
        mock_target.send.assert_awaited_once_with(
            "12345", "hello", metadata=None
        )


class TestInsecureNoAuthSafetyRail:
    """connect() refuses to start when INSECURE_NO_AUTH is combined with a
    non-loopback bind. Guards against accidentally exposing an unauthenticated
    webhook endpoint on a public interface."""

    @pytest.mark.asyncio
    async def test_connect_rejects_insecure_no_auth_on_public_bind(self):
        """INSECURE_NO_AUTH + 0.0.0.0 is refused before the server starts."""
        routes = {"r1": {"secret": _INSECURE_NO_AUTH, "prompt": "x"}}
        adapter = _make_adapter(routes=routes, host="0.0.0.0", port=0)
        with pytest.raises(ValueError, match="INSECURE_NO_AUTH"):
            await adapter.connect()

    @pytest.mark.asyncio
    async def test_connect_rejects_insecure_no_auth_on_lan_ip(self):
        """A LAN IP is treated as public."""
        routes = {"r1": {"secret": _INSECURE_NO_AUTH, "prompt": "x"}}
        adapter = _make_adapter(routes=routes, host="192.168.1.50", port=0)
        with pytest.raises(ValueError, match="non-loopback"):
            await adapter.connect()

    @pytest.mark.asyncio
    async def test_connect_rejects_insecure_no_auth_on_empty_host(self):
        """Empty host is conservatively treated as non-loopback."""
        routes = {"r1": {"secret": _INSECURE_NO_AUTH, "prompt": "x"}}
        adapter = _make_adapter(routes=routes, host="", port=0)
        with pytest.raises(ValueError, match="INSECURE_NO_AUTH"):
            await adapter.connect()

    @pytest.mark.parametrize(
        "host",
        ["127.0.0.1", "localhost"],
    )
    @pytest.mark.asyncio
    async def test_connect_allows_insecure_no_auth_on_loopback(self, host):
        """Recognised loopback hosts are permitted with INSECURE_NO_AUTH."""
        routes = {"r1": {"secret": _INSECURE_NO_AUTH, "prompt": "x"}}
        adapter = _make_adapter(routes=routes, host=host, port=0)
        try:
            with patch.object(adapter, "_reload_dynamic_routes"):
                result = await adapter.connect()
            assert result is True
        finally:
            await adapter.disconnect()

    @pytest.mark.parametrize(
        "host",
        ["127.0.0.1", "localhost", "Localhost", "::1", "ip6-localhost", "ip6-loopback"],
    )
    def test_is_loopback_host_accepts(self, host):
        """_is_loopback_host covers all documented loopback spellings."""
        from gateway.platforms.webhook import _is_loopback_host
        assert _is_loopback_host(host) is True

    @pytest.mark.parametrize(
        "host",
        ["0.0.0.0", "192.168.1.5", "10.0.0.1", "example.com", "", None],
    )
    def test_is_loopback_host_rejects(self, host):
        """_is_loopback_host treats public/LAN/empty as non-loopback."""
        from gateway.platforms.webhook import _is_loopback_host
        assert _is_loopback_host(host) is False

    @pytest.mark.asyncio
    async def test_connect_allows_real_secret_on_public_bind(self):
        """A real HMAC secret bound to 0.0.0.0 is the normal production case."""
        routes = {"r1": {"secret": "real-secret-abc123", "prompt": "x"}}
        adapter = _make_adapter(routes=routes, host="0.0.0.0", port=0)
        try:
            with patch.object(adapter, "_reload_dynamic_routes"):
                result = await adapter.connect()
            assert result is True
        finally:
            await adapter.disconnect()
