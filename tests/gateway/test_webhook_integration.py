"""Integration tests for the generic webhook platform adapter.

These tests exercise end-to-end flows through the webhook adapter:
1. GitHub PR webhook → agent MessageEvent created
2. Skills config injects skill content into the prompt
3. Cross-platform delivery routes to a mock Telegram adapter
4. GitHub comment delivery invokes ``gh`` CLI (mocked subprocess)
"""

import asyncio
import hashlib
import hmac
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from gateway.config import (
    GatewayConfig,
    HomeChannel,
    Platform,
    PlatformConfig,
)
from gateway.platforms.base import MessageEvent, MessageType, SendResult
from gateway.platforms.webhook import WebhookAdapter, _INSECURE_NO_AUTH


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_adapter(routes, **extra_kw) -> WebhookAdapter:
    """Create a WebhookAdapter with the given routes."""
    extra = {"host": "0.0.0.0", "port": 0, "routes": routes}
    extra.update(extra_kw)
    config = PlatformConfig(enabled=True, extra=extra)
    return WebhookAdapter(config)


def _create_app(adapter: WebhookAdapter) -> web.Application:
    """Build the aiohttp Application from the adapter."""
    app = web.Application()
    app.router.add_get("/health", adapter._handle_health)
    app.router.add_post("/webhooks/{route_name}", adapter._handle_webhook)
    return app


def _github_signature(body: bytes, secret: str) -> str:
    """Compute X-Hub-Signature-256 for *body* using *secret*."""
    return "sha256=" + hmac.new(
        secret.encode(), body, hashlib.sha256
    ).hexdigest()


# A realistic GitHub pull_request event payload (trimmed)
GITHUB_PR_PAYLOAD = {
    "action": "opened",
    "number": 42,
    "pull_request": {
        "title": "Add webhook adapter",
        "body": "This PR adds a generic webhook platform adapter.",
        "html_url": "https://github.com/org/repo/pull/42",
        "user": {"login": "contributor"},
        "head": {"ref": "feature/webhooks"},
        "base": {"ref": "main"},
    },
    "repository": {
        "full_name": "org/repo",
        "html_url": "https://github.com/org/repo",
    },
    "sender": {"login": "contributor"},
}


# ===================================================================
# Test 1: GitHub PR webhook triggers agent
# ===================================================================

class TestGitHubPRWebhook:

    @pytest.mark.asyncio
    async def test_github_pr_webhook_triggers_agent(self):
        """POST with a realistic GitHub PR payload should:
        1. Return 202 Accepted
        2. Call handle_message with a MessageEvent
        3. The event text contains the rendered prompt
        4. The event source has chat_type 'webhook'
        """
        secret = "gh-webhook-test-secret"
        routes = {
            "github-pr": {
                "secret": secret,
                "events": ["pull_request"],
                "prompt": (
                    "Review PR #{number} by {sender.login}: "
                    "{pull_request.title}\n\n{pull_request.body}"
                ),
                "deliver": "log",
            }
        }
        adapter = _make_adapter(routes)

        captured_events: list[MessageEvent] = []

        async def _capture(event: MessageEvent):
            captured_events.append(event)

        adapter.handle_message = _capture

        app = _create_app(adapter)
        body = json.dumps(GITHUB_PR_PAYLOAD).encode()
        sig = _github_signature(body, secret)

        async with TestClient(TestServer(app)) as cli:
            resp = await cli.post(
                "/webhooks/github-pr",
                data=body,
                headers={
                    "Content-Type": "application/json",
                    "X-GitHub-Event": "pull_request",
                    "X-Hub-Signature-256": sig,
                    "X-GitHub-Delivery": "gh-delivery-001",
                },
            )
            assert resp.status == 202
            data = await resp.json()
            assert data["status"] == "accepted"
            assert data["route"] == "github-pr"
            assert data["event"] == "pull_request"
            assert data["delivery_id"] == "gh-delivery-001"

        # Let the asyncio.create_task fire
        await asyncio.sleep(0.05)

        assert len(captured_events) == 1
        event = captured_events[0]
        assert "Review PR #42 by contributor" in event.text
        assert "Add webhook adapter" in event.text
        assert event.source.chat_type == "webhook"
        assert event.source.platform == Platform.WEBHOOK
        assert "github-pr" in event.source.chat_id
        assert event.message_id == "gh-delivery-001"


# ===================================================================
# Test 2: Skills injected into prompt
# ===================================================================

class TestSkillsInjection:

    @pytest.mark.asyncio
    async def test_skills_injected_into_prompt(self):
        """When a route has skills: [code-review], the adapter should
        call build_skill_invocation_message() and use its output as the
        prompt instead of the raw template render."""
        routes = {
            "pr-review": {
                "secret": _INSECURE_NO_AUTH,
                "events": ["pull_request"],
                "prompt": "Review this PR: {pull_request.title}",
                "skills": ["code-review"],
            }
        }
        adapter = _make_adapter(routes)

        captured_events: list[MessageEvent] = []

        async def _capture(event: MessageEvent):
            captured_events.append(event)

        adapter.handle_message = _capture

        skill_content = (
            "You are a code reviewer. Review the following:\n"
            "Review this PR: Add webhook adapter"
        )

        # The imports are lazy (inside the handler), so patch the source module
        with patch(
            "agent.skill_commands.build_skill_invocation_message",
            return_value=skill_content,
        ) as mock_build, patch(
            "agent.skill_commands.get_skill_commands",
            return_value={"/code-review": {"name": "code-review"}},
        ):
            app = _create_app(adapter)
            async with TestClient(TestServer(app)) as cli:
                resp = await cli.post(
                    "/webhooks/pr-review",
                    json=GITHUB_PR_PAYLOAD,
                    headers={
                        "X-GitHub-Event": "pull_request",
                        "X-GitHub-Delivery": "skill-test-001",
                    },
                )
                assert resp.status == 202

            await asyncio.sleep(0.05)

            assert len(captured_events) == 1
            event = captured_events[0]
            # The prompt should be the skill content, not the raw template
            assert "You are a code reviewer" in event.text
            mock_build.assert_called_once()


# ===================================================================
# Test 3: Cross-platform delivery (webhook → Telegram)
# ===================================================================

class TestCrossPlatformDelivery:

    @pytest.mark.asyncio
    async def test_cross_platform_delivery(self):
        """When deliver='telegram', the response is routed to the
        Telegram adapter via gateway_runner.adapters."""
        routes = {
            "alerts": {
                "secret": _INSECURE_NO_AUTH,
                "prompt": "Alert: {message}",
                "deliver": "telegram",
                "deliver_extra": {"chat_id": "12345"},
            }
        }
        adapter = _make_adapter(routes)
        adapter.handle_message = AsyncMock()

        # Set up a mock gateway runner with a mock Telegram adapter
        mock_tg_adapter = AsyncMock()
        mock_tg_adapter.send = AsyncMock(return_value=SendResult(success=True))

        mock_runner = MagicMock()
        mock_runner.adapters = {Platform.TELEGRAM: mock_tg_adapter}
        mock_runner.config = GatewayConfig(
            platforms={Platform.TELEGRAM: PlatformConfig(enabled=True, token="fake")}
        )
        adapter.gateway_runner = mock_runner

        # First, simulate a webhook POST to set up delivery_info
        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.post(
                "/webhooks/alerts",
                json={"message": "Server is on fire!"},
                headers={"X-GitHub-Delivery": "alert-001"},
            )
            assert resp.status == 202

        # The adapter should have stored delivery info
        chat_id = "webhook:alerts:alert-001"
        assert chat_id in adapter._delivery_info

        # Now call send() as if the agent has finished
        result = await adapter.send(chat_id, "I've acknowledged the alert.")

        assert result.success is True
        mock_tg_adapter.send.assert_awaited_once_with(
            "12345", "I've acknowledged the alert.", metadata=None
        )
        # Delivery info is retained after send() so interim status messages
        # don't strand the final response (TTL-based cleanup happens on POST).
        assert chat_id in adapter._delivery_info


# ===================================================================
# Test 4: GitHub comment delivery via gh CLI
# ===================================================================

class TestGitHubCommentDelivery:

    @pytest.mark.asyncio
    async def test_github_comment_delivery(self):
        """When deliver='github_comment', the adapter invokes
        ``gh pr comment`` via subprocess.run (mocked)."""
        routes = {
            "pr-bot": {
                "secret": _INSECURE_NO_AUTH,
                "prompt": "Review: {pull_request.title}",
                "deliver": "github_comment",
                "deliver_extra": {
                    "repo": "{repository.full_name}",
                    "pr_number": "{number}",
                },
            }
        }
        adapter = _make_adapter(routes)
        adapter.handle_message = AsyncMock()

        # POST a webhook to set up delivery info
        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.post(
                "/webhooks/pr-bot",
                json=GITHUB_PR_PAYLOAD,
                headers={
                    "X-GitHub-Event": "pull_request",
                    "X-GitHub-Delivery": "gh-comment-001",
                },
            )
            assert resp.status == 202

        chat_id = "webhook:pr-bot:gh-comment-001"
        assert chat_id in adapter._delivery_info

        # Verify deliver_extra was rendered with payload data
        delivery = adapter._delivery_info[chat_id]
        assert delivery["deliver_extra"]["repo"] == "org/repo"
        assert delivery["deliver_extra"]["pr_number"] == "42"

        # Mock subprocess.run and call send()
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "Comment posted"
        mock_result.stderr = ""

        with patch(
            "gateway.platforms.webhook.subprocess.run",
            return_value=mock_result,
        ) as mock_run:
            result = await adapter.send(
                chat_id, "LGTM! The code looks great."
            )

        assert result.success is True
        mock_run.assert_called_once_with(
            [
                "gh", "pr", "comment", "42",
                "--repo", "org/repo",
                "--body", "LGTM! The code looks great.",
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        # Delivery info is retained after send() so interim status messages
        # don't strand the final response (TTL-based cleanup happens on POST).
        assert chat_id in adapter._delivery_info


# ===================================================================
# Test 5: Alertmanager groupKey-based delivery_id dedup
# ===================================================================

ALERTMANAGER_PAYLOAD = {
    "receiver": "hermes",
    "status": "firing",
    "alerts": [
        {
            "status": "firing",
            "labels": {
                "alertname": "HighRequestLatency",
                "service": "api",
                "severity": "warning",
            },
            "annotations": {"summary": "API latency spiked"},
            "startsAt": "2026-01-01T00:00:00Z",
            "endsAt": "0001-01-01T00:00:00Z",
            "generatorURL": "",
            "fingerprint": "abc123",
        }
    ],
    "groupLabels": {"alertname": "HighRequestLatency"},
    "commonLabels": {
        "alertname": "HighRequestLatency",
        "service": "api",
        "severity": "warning",
    },
    "commonAnnotations": {"summary": "API latency spiked"},
    "externalURL": "",
    "version": "4",
    # The key field this test exercises: stable per-group identifier.
    "groupKey": '{}/{}:{alertname="HighRequestLatency"}',
    "truncatedAlerts": 0,
}


import re


class TestAlertmanagerGroupKeyDedup:

    @pytest.mark.asyncio
    async def test_groupkey_derived_delivery_id_dedups_retries(self):
        """When the payload has a `groupKey` field, an immediate identical
        re-POST must collapse to a single delivery_id (derived from the route
        name and the normalized payload) so the idempotency loop catches the
        second call as a duplicate. Without this, Alertmanager deliveries
        (which omit X-GitHub-Delivery / svix-id / X-Request-ID) re-trigger the
        agent every time. The near-identical-but-drifting repeat_interval case
        is covered by test_repeat_interval_endsat_drift_still_dedups."""
        routes = {
            "alertmanager": {
                "prompt": "Alert: {commonLabels.alertname}",
                "secret": _INSECURE_NO_AUTH,
                "deliver": "log",
            }
        }
        adapter = _make_adapter(routes)
        adapter.handle_message = AsyncMock()

        app = _create_app(adapter)
        body = json.dumps(ALERTMANAGER_PAYLOAD).encode()

        async with TestClient(TestServer(app)) as cli:
            first = await cli.post(
                "/webhooks/alertmanager",
                data=body,
                headers={"Content-Type": "application/json"},
            )
            assert first.status == 202
            data1 = await first.json()
            assert data1["status"] == "accepted"
            assert re.fullmatch(
                r"alertmanager:[0-9a-f]{16}", data1["delivery_id"]
            ), f"unexpected delivery_id: {data1['delivery_id']}"

            # Same payload, same groupKey -> must dedup.
            second = await cli.post(
                "/webhooks/alertmanager",
                data=body,
                headers={"Content-Type": "application/json"},
            )
            assert second.status == 200
            data2 = await second.json()
            assert data2["status"] == "duplicate"
            assert data2["delivery_id"] == data1["delivery_id"]

    @pytest.mark.asyncio
    async def test_explicit_delivery_header_wins_over_groupkey(self):
        """Senders that DO supply X-GitHub-Delivery / svix-id /
        X-Request-ID continue to win - the payload-based derivation is a
        fallback, not an override. Two POSTs with the same groupKey but
        different X-Request-IDs are NOT deduped."""
        routes = {
            "alertmanager": {
                "prompt": "Alert: {commonLabels.alertname}",
                "secret": _INSECURE_NO_AUTH,
                "deliver": "log",
            }
        }
        adapter = _make_adapter(routes)
        adapter.handle_message = AsyncMock()

        app = _create_app(adapter)
        body = json.dumps(ALERTMANAGER_PAYLOAD).encode()

        async with TestClient(TestServer(app)) as cli:
            first = await cli.post(
                "/webhooks/alertmanager",
                data=body,
                headers={
                    "Content-Type": "application/json",
                    "X-Request-ID": "explicit-id-001",
                },
            )
            assert first.status == 202
            data1 = await first.json()
            assert data1["delivery_id"] == "explicit-id-001"

            second = await cli.post(
                "/webhooks/alertmanager",
                data=body,
                headers={
                    "Content-Type": "application/json",
                    "X-Request-ID": "explicit-id-002",
                },
            )
            assert second.status == 202
            data2 = await second.json()
            assert data2["delivery_id"] == "explicit-id-002"

    @pytest.mark.asyncio
    async def test_different_routes_get_distinct_delivery_ids(self):
        """Same groupKey on two different routes must produce different
        delivery_ids so a duplicate on route A never silently masks a
        legitimate delivery on route B."""
        routes = {
            "alertmanager-a": {
                "prompt": "A: {commonLabels.alertname}",
                "secret": _INSECURE_NO_AUTH,
                "deliver": "log",
            },
            "alertmanager-b": {
                "prompt": "B: {commonLabels.alertname}",
                "secret": _INSECURE_NO_AUTH,
                "deliver": "log",
            },
        }
        adapter = _make_adapter(routes)
        adapter.handle_message = AsyncMock()

        app = _create_app(adapter)
        body = json.dumps(ALERTMANAGER_PAYLOAD).encode()

        async with TestClient(TestServer(app)) as cli:
            a = await cli.post(
                "/webhooks/alertmanager-a",
                data=body,
                headers={"Content-Type": "application/json"},
            )
            b = await cli.post(
                "/webhooks/alertmanager-b",
                data=body,
                headers={"Content-Type": "application/json"},
            )
            assert a.status == 202
            assert b.status == 202
            id_a = (await a.json())["delivery_id"]
            id_b = (await b.json())["delivery_id"]
            assert id_a != id_b
            assert id_a.startswith("alertmanager:")
            assert id_b.startswith("alertmanager:")

    @pytest.mark.parametrize(
        "group_key",
        [
            pytest.param(None, id="missing"),
            pytest.param("", id="empty-string"),
            pytest.param("   ", id="whitespace-only"),
            pytest.param("\t\n", id="tab-newline"),
            pytest.param(123, id="integer"),
            pytest.param(["foo"], id="list"),
        ],
    )
    @pytest.mark.asyncio
    async def test_invalid_groupkey_falls_through_to_timestamp(self, group_key):
        """Non-string or effectively-empty groupKey must fall through to
        the timestamp fallback (a fresh unique ID per request), never to
        a stable hash. Otherwise an absent groupKey would silently dedup
        every retry of every alert across the deployment."""
        routes = {
            "alertmanager": {
                "prompt": "Alert: x",
                "secret": _INSECURE_NO_AUTH,
                "deliver": "log",
            }
        }
        adapter = _make_adapter(routes)
        adapter.handle_message = AsyncMock()

        payload = dict(ALERTMANAGER_PAYLOAD)
        if group_key is None:
            payload.pop("groupKey", None)
        else:
            payload["groupKey"] = group_key

        app = _create_app(adapter)
        body = json.dumps(payload).encode()

        async with TestClient(TestServer(app)) as cli:
            resp = await cli.post(
                "/webhooks/alertmanager",
                data=body,
                headers={"Content-Type": "application/json"},
            )
            assert resp.status == 202
            delivery_id = (await resp.json())["delivery_id"]
            # Timestamp fallback is digits-only; the hash path starts with
            # "alertmanager:". Either way it must NOT be the hash path.
            assert not delivery_id.startswith("alertmanager:")
            assert delivery_id.isdigit()

    @pytest.mark.asyncio
    async def test_status_transition_same_groupkey_not_deduped(self):
        """A later notification that shares the groupKey but carries a real
        state change (here: the `resolved` notification) must NOT be swallowed
        as a duplicate. Alertmanager reuses one groupKey for a group's whole
        lifecycle, so a firing->resolved transition still has to reach the
        agent. Regression guard against keying idempotency on groupKey alone.
        Covers the status-change axis only; the endsAt-drift axis is covered by
        test_repeat_interval_endsat_drift_still_dedups."""
        routes = {
            "alertmanager": {
                "prompt": "Alert: {commonLabels.alertname}",
                "secret": _INSECURE_NO_AUTH,
                "deliver": "log",
            }
        }
        adapter = _make_adapter(routes)
        adapter.handle_message = AsyncMock()

        app = _create_app(adapter)
        firing = json.dumps(ALERTMANAGER_PAYLOAD).encode()

        # Same groupKey, but the group has now resolved: status flips and the
        # alert gets an endsAt. Deep-copy via a JSON round-trip so the nested
        # alert dict is not shared with ALERTMANAGER_PAYLOAD.
        resolved_payload = json.loads(json.dumps(ALERTMANAGER_PAYLOAD))
        resolved_payload["status"] = "resolved"
        resolved_payload["alerts"][0]["status"] = "resolved"
        resolved_payload["alerts"][0]["endsAt"] = "2026-01-01T00:05:00Z"
        resolved = json.dumps(resolved_payload).encode()

        async with TestClient(TestServer(app)) as cli:
            first = await cli.post(
                "/webhooks/alertmanager",
                data=firing,
                headers={"Content-Type": "application/json"},
            )
            assert first.status == 202
            id_firing = (await first.json())["delivery_id"]

            second = await cli.post(
                "/webhooks/alertmanager",
                data=resolved,
                headers={"Content-Type": "application/json"},
            )
            # Must be processed, not deduped: different content -> different id.
            assert second.status == 202
            data2 = await second.json()
            assert data2["status"] == "accepted"
            assert data2["delivery_id"] != id_firing
            assert data2["delivery_id"].startswith("alertmanager:")

    @pytest.mark.asyncio
    async def test_repeat_interval_endsat_drift_still_dedups(self):
        """A repeat_interval re-send of an UNCHANGED firing group must still
        dedup even though its body is not byte-identical: Prometheus refreshes
        `alerts[].endsAt` on every evaluation cycle for an active alert, so the
        end time drifts between notifications without any material change.
        endsAt is normalized out for firing alerts, so the two still collapse to
        one delivery. Without that normalization this is exactly the
        're-trigger the agent every notification interval' bug the derivation
        exists to prevent."""
        routes = {
            "alertmanager": {
                "prompt": "Alert: {commonLabels.alertname}",
                "secret": _INSECURE_NO_AUTH,
                "deliver": "log",
            }
        }
        adapter = _make_adapter(routes)
        adapter.handle_message = AsyncMock()

        app = _create_app(adapter)
        first_body = json.dumps(ALERTMANAGER_PAYLOAD).encode()

        # Same firing group, later evaluation cycle: only endsAt moved forward.
        drifted = json.loads(json.dumps(ALERTMANAGER_PAYLOAD))
        drifted["alerts"][0]["endsAt"] = "2026-01-01T00:10:00Z"
        drifted_body = json.dumps(drifted).encode()

        async with TestClient(TestServer(app)) as cli:
            first = await cli.post(
                "/webhooks/alertmanager",
                data=first_body,
                headers={"Content-Type": "application/json"},
            )
            assert first.status == 202
            id1 = (await first.json())["delivery_id"]

            second = await cli.post(
                "/webhooks/alertmanager",
                data=drifted_body,
                headers={"Content-Type": "application/json"},
            )
            assert second.status == 200
            data2 = await second.json()
            assert data2["status"] == "duplicate"
            assert data2["delivery_id"] == id1

    @pytest.mark.asyncio
    async def test_alert_array_order_does_not_affect_dedup(self):
        """Two notifications with the same alert *set* in a different array
        order are the same delivery: alerts are ordered by fingerprint before
        hashing, so upstream ordering differences never cause a spurious
        non-dedup."""
        routes = {
            "alertmanager": {
                "prompt": "Alert: {commonLabels.alertname}",
                "secret": _INSECURE_NO_AUTH,
                "deliver": "log",
            }
        }
        adapter = _make_adapter(routes)
        adapter.handle_message = AsyncMock()

        app = _create_app(adapter)

        two_alerts = json.loads(json.dumps(ALERTMANAGER_PAYLOAD))
        first_alert = two_alerts["alerts"][0]
        second_alert = json.loads(json.dumps(first_alert))
        second_alert["fingerprint"] = "def456"
        second_alert["labels"]["service"] = "worker"
        two_alerts["alerts"] = [first_alert, second_alert]
        body_ab = json.dumps(two_alerts).encode()

        reordered = json.loads(json.dumps(two_alerts))
        reordered["alerts"] = [second_alert, first_alert]
        body_ba = json.dumps(reordered).encode()

        async with TestClient(TestServer(app)) as cli:
            first = await cli.post(
                "/webhooks/alertmanager",
                data=body_ab,
                headers={"Content-Type": "application/json"},
            )
            assert first.status == 202
            id1 = (await first.json())["delivery_id"]

            second = await cli.post(
                "/webhooks/alertmanager",
                data=body_ba,
                headers={"Content-Type": "application/json"},
            )
            assert second.status == 200
            data2 = await second.json()
            assert data2["status"] == "duplicate"
            assert data2["delivery_id"] == id1
