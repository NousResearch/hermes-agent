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
from unittest.mock import AsyncMock, MagicMock, call, patch

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from gateway.config import (
    GatewayConfig,
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


async def _drain_background_tasks(
    adapter: WebhookAdapter, timeout: float = 5.0
) -> None:
    deadline = asyncio.get_event_loop().time() + timeout
    while adapter._background_tasks and asyncio.get_event_loop().time() < deadline:
        await asyncio.sleep(0.02)
    await asyncio.sleep(0.05)


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
        chat_id = "webhook:alerts:delivery:alert-001"
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

    @pytest.mark.asyncio
    async def test_overlapping_persistent_turns_keep_per_delivery_targets(self):
        """A later turn cannot replace an in-flight turn's rendered target."""
        routes = {
            "support": {
                "secret": _INSECURE_NO_AUTH,
                "session_key": "{tenant_id}:{account_id}:{conversation_id}",
                "prompt": "{message}",
                "deliver": "telegram",
                "deliver_extra": {"chat_id": "{target}"},
            }
        }
        adapter = _make_adapter(routes)
        mock_tg_adapter = AsyncMock()
        mock_tg_adapter.send = AsyncMock(return_value=SendResult(success=True))

        class _Runner:
            adapters = {Platform.TELEGRAM: mock_tg_adapter}
            config = GatewayConfig(
                platforms={
                    Platform.TELEGRAM: PlatformConfig(enabled=True, token="fake")
                }
            )

            @staticmethod
            def _profile_name_for_source(source):
                return None

        adapter.gateway_runner = _Runner()
        first_started = asyncio.Event()
        release_first = asyncio.Event()
        seen_chat_ids = []

        async def _message_handler(event: MessageEvent):
            seen_chat_ids.append(event.source.chat_id)
            if event.message_id == "delivery-1":
                first_started.set()
                await release_first.wait()
            # Background work spawned while a turn is active inherits that
            # turn's ContextVar. It must therefore retain this delivery's
            # rendered destination rather than falling back to the persistent
            # session chat id (which deliberately is not a delivery-info key).
            await asyncio.create_task(
                adapter.send(event.source.chat_id, f"status-{event.message_id}")
            )
            return f"reply-{event.message_id}"

        adapter._message_handler = _message_handler
        app = _create_app(adapter)
        common = {
            "tenant_id": "tenant-7",
            "account_id": "account-3",
            "conversation_id": "conversation-9",
        }

        async with TestClient(TestServer(app)) as cli:
            first = await cli.post(
                "/webhooks/support",
                json={**common, "message": "first", "target": "chat-a"},
                headers={"X-GitHub-Delivery": "delivery-1"},
            )
            assert first.status == 202
            await asyncio.wait_for(first_started.wait(), timeout=2)

            second = await cli.post(
                "/webhooks/support",
                json={**common, "message": "second", "target": "chat-b"},
                headers={"X-GitHub-Delivery": "delivery-2"},
            )
            assert second.status == 202
            release_first.set()
            await _drain_background_tasks(adapter)

        assert seen_chat_ids == [
            "webhook:support:session:tenant-7:account-3:conversation-9",
            "webhook:support:session:tenant-7:account-3:conversation-9",
        ]
        assert "delivery-1" in adapter._delivery_info
        assert "delivery-2" in adapter._delivery_info
        assert mock_tg_adapter.send.await_args_list == [
            call("chat-a", "status-delivery-1", metadata=None),
            call("chat-a", "reply-delivery-1", metadata=None),
            call("chat-b", "status-delivery-2", metadata=None),
            call("chat-b", "reply-delivery-2", metadata=None),
        ]

    @pytest.mark.asyncio
    async def test_processing_completion_restores_prior_delivery_context(self):
        """A completed webhook turn cannot leak its delivery target in-task."""
        adapter = _make_adapter({})
        prior = adapter._active_delivery_id.set("prior-delivery")
        event = MessageEvent(
            text="test",
            message_type=MessageType.TEXT,
            source=adapter.build_source(
                chat_id="webhook:alerts:delivery:new-delivery",
                chat_name="webhook/alerts",
                chat_type="webhook",
                user_id="webhook:alerts",
                user_name="alerts",
            ),
            message_id="new-delivery",
            metadata={"webhook_persistent_session": True},
        )

        await adapter.on_processing_start(event)
        assert adapter._active_delivery_id.get() == "new-delivery"
        await adapter.on_processing_complete(event, outcome=None)
        assert adapter._active_delivery_id.get() == "prior-delivery"
        adapter._active_delivery_id.reset(prior)

    @pytest.mark.asyncio
    async def test_persistent_key_and_fallback_delivery_use_disjoint_session_ids(self):
        """A fallback delivery ID cannot close or reuse a persistent conversation."""
        routes = {
            "support": {
                "secret": _INSECURE_NO_AUTH,
                "session_key": "{conversation_id}",
                "prompt": "{message}",
                "deliver": "log",
            }
        }
        adapter = _make_adapter(routes)
        adapter.handle_message = AsyncMock()
        app = _create_app(adapter)

        async with TestClient(TestServer(app)) as cli:
            persistent = await cli.post(
                "/webhooks/support",
                json={"conversation_id": "abc", "message": "persistent turn"},
                headers={"X-GitHub-Delivery": "persistent-delivery"},
            )
            fallback = await cli.post(
                "/webhooks/support",
                json={"message": "fallback turn"},
                headers={"X-GitHub-Delivery": "abc"},
            )
            assert persistent.status == fallback.status == 202
            await _drain_background_tasks(adapter)

        seen_chat_ids = [
            call.args[0].source.chat_id for call in adapter.handle_message.await_args_list
        ]
        assert seen_chat_ids == [
            "webhook:support:session:abc",
            "webhook:support:delivery:abc",
        ]


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

        chat_id = "webhook:pr-bot:delivery:gh-comment-001"
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
            encoding="utf-8",
            errors="replace",
            timeout=30,
        )
        # Delivery info is retained after send() so interim status messages
        # don't strand the final response (TTL-based cleanup happens on POST).
        assert chat_id in adapter._delivery_info
