"""Linear platform adapter.

Receives Linear webhook events (issue assignments, comment @mentions, state
changes), routes them through an event queue, and gives the agent tools to
manage issues and respond via Linear comments.

Architecture follows the openclaw-linear plugin design:
  Webhook POST → HMAC verify → event router → work queue → agent dispatch
  Agent response → Linear comment via `linear` CLI or GraphQL API

Configuration in config.yaml:
  platforms:
    linear:
      enabled: true
      extra:
        webhook_secret: "your-hmac-secret"
        api_key: "lin_api_..."          # Linear API key
        agent_user_id: "uuid"           # Linear user ID mapped to this agent
        team_ids: ["AI", "ENG"]         # Optional: filter to specific teams
        host: "0.0.0.0"                 # Webhook listener host
        port: 8645                      # Webhook listener port
        debounce_seconds: 30            # Batch window for events

Environment variables (override config):
  LINEAR_WEBHOOK_SECRET   — HMAC webhook signing secret (required)
  LINEAR_API_KEY          — Linear API key for posting comments (required)
  LINEAR_AGENT_USER_ID    — Linear user UUID that represents the agent
  LINEAR_TEAM_IDS         — Comma-separated team keys to filter
  LINEAR_WEBHOOK_HOST     — Webhook listener host (default: 0.0.0.0)
  LINEAR_WEBHOOK_PORT     — Webhook listener port (default: 8645)
"""

import asyncio
import hashlib
import hmac
import json
import logging
import os
import re
import ssl
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

try:
    from aiohttp import web

    AIOHTTP_AVAILABLE = True
except ImportError:
    AIOHTTP_AVAILABLE = False
    web = None  # type: ignore[assignment]

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import (
    BasePlatformAdapter,
    MessageEvent,
    MessageType,
    SendResult,
)

logger = logging.getLogger(__name__)

DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 8645
MAX_BODY_BYTES = 1024 * 1024  # 1 MB
DEDUP_TTL_SECONDS = 600  # 10 minutes
DEDUP_MAX_SIZE = 10_000
DEFAULT_DEBOUNCE_SECONDS = 30
MAX_COMMENT_LENGTH = 10_000  # Linear comment character limit

# Linear's webhook source IPs (https://linear.app/developers/webhooks)
# Linear notes: "We may occasionally update this list to add new IP addresses."
LINEAR_WEBHOOK_IPS = frozenset({
    "35.231.147.226",
    "35.243.134.228",
    "34.140.253.14",
    "34.38.87.206",
    "34.134.222.122",
    "35.222.25.142",
})


def check_linear_requirements() -> bool:
    """Check if Linear adapter dependencies are available."""
    if not AIOHTTP_AVAILABLE:
        return False
    # Need either LINEAR_API_KEY or linear CLI for posting comments
    api_key = os.getenv("LINEAR_API_KEY", "")
    webhook_secret = os.getenv("LINEAR_WEBHOOK_SECRET", "")
    return bool(api_key and webhook_secret)


def _verify_signature(body: str, signature: str, secret: str) -> bool:
    """Verify Linear webhook HMAC-SHA256 signature."""
    expected = hmac.new(
        secret.encode("utf-8"), body.encode("utf-8"), hashlib.sha256
    ).hexdigest()
    if len(expected) != len(signature):
        return False
    return hmac.compare_digest(expected, signature)


def _extract_mentioned_user_ids(
    body: str, body_data: Any, agent_user_id: str
) -> List[str]:
    """Extract @mentioned user IDs from a Linear comment body.

    Linear stores mentions in two ways:
    1. In bodyData (ProseMirror JSON) as mention nodes with attrs.id
    2. In the body text as @displayName (fallback, less reliable)

    We check if the agent's user ID is mentioned.
    """
    mentioned = set()

    # Method 1: Parse ProseMirror bodyData for mention nodes
    if body_data and isinstance(body_data, dict):
        _walk_prosemirror_mentions(body_data, mentioned)

    # Method 2: If agent_user_id was found, return it
    if agent_user_id in mentioned:
        return [agent_user_id]

    return list(mentioned)


def _walk_prosemirror_mentions(node: dict, mentioned: Set[str]) -> None:
    """Recursively walk ProseMirror document to find mention nodes."""
    if node.get("type") == "mention":
        attrs = node.get("attrs", {})
        user_id = attrs.get("id") or attrs.get("userId")
        if user_id:
            mentioned.add(str(user_id))

    for child in node.get("content", []):
        if isinstance(child, dict):
            _walk_prosemirror_mentions(child, mentioned)


class LinearAdapter(BasePlatformAdapter):
    """Linear webhook platform adapter.

    Receives webhooks, filters events, and dispatches to the agent.
    Agent responds by posting comments back to Linear issues.
    """

    MAX_MESSAGE_LENGTH = MAX_COMMENT_LENGTH

    def __init__(self, config: PlatformConfig):
        super().__init__(config, Platform.LINEAR)
        extra = config.extra or {}

        self._host: str = extra.get("host", os.getenv("LINEAR_WEBHOOK_HOST", DEFAULT_HOST))
        self._port: int = int(extra.get("port", os.getenv("LINEAR_WEBHOOK_PORT", str(DEFAULT_PORT))))
        self._webhook_secret: str = extra.get(
            "webhook_secret", os.getenv("LINEAR_WEBHOOK_SECRET", "")
        )
        self._api_key: str = extra.get(
            "api_key", os.getenv("LINEAR_API_KEY", "")
        )
        self._agent_user_id: str = extra.get(
            "agent_user_id", os.getenv("LINEAR_AGENT_USER_ID", "")
        )

        # Team filter
        team_ids_str = extra.get("team_ids", os.getenv("LINEAR_TEAM_IDS", ""))
        if isinstance(team_ids_str, list):
            self._team_ids: List[str] = team_ids_str
        elif isinstance(team_ids_str, str) and team_ids_str:
            self._team_ids = [t.strip() for t in team_ids_str.split(",") if t.strip()]
        else:
            self._team_ids = []

        self._debounce_seconds: float = float(
            extra.get("debounce_seconds", DEFAULT_DEBOUNCE_SECONDS)
        )

        # IP allowlist — enabled by default, can be disabled for testing
        enforce_ip = extra.get(
            "enforce_ip_allowlist",
            os.getenv("LINEAR_ENFORCE_IP_ALLOWLIST", "true"),
        )
        self._enforce_ip_allowlist: bool = str(enforce_ip).lower() not in ("false", "0", "no")

        # TLS — serve HTTPS directly (no reverse proxy needed)
        self._tls_cert: str = extra.get(
            "tls_cert", os.getenv("LINEAR_TLS_CERT", "")
        )
        self._tls_key: str = extra.get(
            "tls_key", os.getenv("LINEAR_TLS_KEY", "")
        )

        # State
        self._runner = None
        self._http_session: Optional[Any] = None  # aiohttp.ClientSession — lazy init
        self._processed_deliveries: Dict[str, float] = {}
        self._debounce_buffer: Dict[str, List[dict]] = {}  # agent_key -> events
        self._debounce_tasks: Dict[str, asyncio.Task] = {}

    async def connect(self) -> bool:
        """Start the webhook HTTP server."""
        if not self._webhook_secret:
            logger.error("[linear] No webhook secret configured — cannot start")
            return False
        if not self._api_key:
            logger.error("[linear] No API key configured — cannot post comments")
            return False

        if web is None:
            logger.error("[linear] aiohttp not available")
            return False

        app = web.Application()
        app.router.add_get("/health", self._handle_health)
        app.router.add_post("/hooks/linear", self._handle_webhook)

        self._runner = web.AppRunner(app)
        await self._runner.setup()

        # Set up TLS if cert and key are provided
        ssl_context = None
        if self._tls_cert and self._tls_key:
            ssl_context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
            try:
                ssl_context.load_cert_chain(self._tls_cert, self._tls_key)
            except Exception as exc:
                logger.error("[linear] Failed to load TLS cert/key: %s", exc)
                await self._runner.cleanup()
                self._runner = None
                return False

        try:
            site = web.TCPSite(
                self._runner, self._host, self._port, ssl_context=ssl_context
            )
            await site.start()
        except OSError as exc:
            logger.error("[linear] Port %d in use: %s", self._port, exc)
            await self._runner.cleanup()
            self._runner = None
            return False

        proto = "https" if ssl_context else "http"
        self._running = True
        logger.info(
            "[linear] Webhook server listening on %s://%s:%d/hooks/linear",
            proto,
            self._host,
            self._port,
        )
        return True

    async def disconnect(self) -> None:
        """Stop the webhook server."""
        self._running = False
        # Cancel pending debounce tasks
        for task in self._debounce_tasks.values():
            task.cancel()
        self._debounce_tasks.clear()
        self._debounce_buffer.clear()

        if self._http_session and not self._http_session.closed:
            await self._http_session.close()
            self._http_session = None

        if self._runner:
            await self._runner.cleanup()
            self._runner = None

    async def send(
        self, chat_id: str, text: str, **kwargs
    ) -> SendResult:
        """Send a message as a Linear comment on an issue.

        chat_id is the issue identifier (e.g., 'AI-123').
        """
        if not text or not text.strip():
            return SendResult(success=True)

        # Truncate if needed
        if len(text) > MAX_COMMENT_LENGTH:
            text = text[:MAX_COMMENT_LENGTH - 100] + "\n\n…(truncated)"

        try:
            result = await self._post_comment(chat_id, text)
            return result
        except Exception as exc:
            logger.error("[linear] Failed to post comment to %s: %s", chat_id, exc)
            return SendResult(success=False, error=str(exc))

    async def send_typing(self, chat_id: str) -> None:
        """No typing indicator in Linear."""
        pass

    async def send_image(
        self, chat_id: str, image_url: str, caption: Optional[str] = None
    ) -> SendResult:
        """Send an image as a comment with markdown image link."""
        text = f"![image]({image_url})"
        if caption:
            text = f"{caption}\n\n{text}"
        return await self.send(chat_id, text)

    async def get_chat_info(self, chat_id: str) -> Dict[str, Any]:
        """Get info about a Linear issue (used as chat)."""
        return {
            "name": f"Linear Issue {chat_id}",
            "type": "issue",
            "chat_id": chat_id,
        }

    # ── Webhook Handlers ────────────────────────────────────────────

    async def _handle_health(self, request: web.Request) -> web.Response:
        """Health check endpoint."""
        return web.Response(text="OK")

    async def _handle_webhook(self, request: web.Request) -> web.Response:
        """Handle incoming Linear webhook POST."""
        if request.method != "POST":
            return web.Response(status=405, text="Method Not Allowed")

        # IP allowlist check — reject requests not from Linear's servers
        if self._enforce_ip_allowlist:
            # Support X-Forwarded-For for reverse proxies (Caddy, nginx, etc.)
            forwarded_for = request.headers.get("X-Forwarded-For", "")
            if forwarded_for:
                # X-Forwarded-For: client, proxy1, proxy2 — take the leftmost
                source_ip = forwarded_for.split(",")[0].strip()
            else:
                source_ip = request.remote or ""
            if source_ip not in LINEAR_WEBHOOK_IPS:
                logger.warning(
                    "[linear] Rejected webhook from non-Linear IP: %s", source_ip
                )
                return web.Response(status=403, text="Forbidden")

        # Check content length
        content_length = request.content_length or 0
        if content_length > MAX_BODY_BYTES:
            return web.Response(status=413, text="Payload Too Large")

        # Read body
        try:
            raw_body = await request.text()
        except Exception:
            return web.Response(status=500, text="Error reading body")

        if len(raw_body) > MAX_BODY_BYTES:
            return web.Response(status=413, text="Payload Too Large")

        # Verify HMAC signature
        signature = request.headers.get("Linear-Signature", "")
        if not signature or not _verify_signature(raw_body, signature, self._webhook_secret):
            logger.warning("[linear] Invalid webhook signature")
            return web.Response(status=400, text="Invalid signature")

        # Parse payload
        try:
            payload = json.loads(raw_body)
        except json.JSONDecodeError:
            logger.error("[linear] Failed to parse webhook JSON")
            return web.Response(status=400, text="Invalid JSON")

        # Dedup by delivery ID
        delivery_id = request.headers.get("Linear-Delivery", "")
        if delivery_id:
            self._prune_deliveries()
            if delivery_id in self._processed_deliveries:
                logger.info("[linear] Duplicate delivery skipped: %s", delivery_id)
                return web.Response(status=200, text="OK")
            self._processed_deliveries[delivery_id] = time.time()

        # Check webhook timestamp to guard against replay attacks (within 60s)
        webhook_ts = payload.get("webhookTimestamp")
        if webhook_ts is not None:
            try:
                ts_ms = int(webhook_ts)
                if abs(time.time() * 1000 - ts_ms) > 60_000:
                    logger.warning("[linear] Rejected stale webhook (timestamp drift > 60s)")
                    return web.Response(status=401, text="Stale webhook")
            except (ValueError, TypeError):
                pass  # If timestamp is malformed, skip check

        # Always return 200 to prevent Linear retry storms
        # Process event asynchronously
        action = payload.get("action", "")
        event_type = payload.get("type", "")
        data = payload.get("data", payload)
        updated_from = payload.get("updatedFrom", {})

        logger.info(
            "[linear] Webhook: %s %s (id=%s)",
            action,
            event_type,
            data.get("id", "unknown"),
        )

        # Route the event
        asyncio.create_task(
            self._route_event(action, event_type, data, updated_from)
        )

        return web.Response(status=200, text="OK")

    def _prune_deliveries(self) -> None:
        """Remove expired delivery IDs from dedup cache."""
        now = time.time()
        expired = [
            k for k, ts in self._processed_deliveries.items()
            if now - ts > DEDUP_TTL_SECONDS
        ]
        for k in expired:
            del self._processed_deliveries[k]
        # Cap size
        if len(self._processed_deliveries) > DEDUP_MAX_SIZE:
            oldest = sorted(self._processed_deliveries.items(), key=lambda x: x[1])
            for k, _ in oldest[: len(self._processed_deliveries) - DEDUP_MAX_SIZE]:
                del self._processed_deliveries[k]

    # ── Event Routing ───────────────────────────────────────────────

    async def _route_event(
        self,
        action: str,
        event_type: str,
        data: dict,
        updated_from: dict,
    ) -> None:
        """Route a webhook event to the appropriate handler."""
        try:
            # Apply team filter
            if self._team_ids:
                team_id = data.get("teamId", "")
                team = data.get("team", {})
                team_key = team.get("key", "") if isinstance(team, dict) else ""
                if not any(t in (team_id, team_key) for t in self._team_ids):
                    logger.debug("[linear] Event filtered by team: %s/%s", team_id, team_key)
                    return

            if event_type == "Comment" and action in ("create", "update"):
                await self._handle_comment_event(data)
            elif event_type == "Issue" and action == "update":
                await self._handle_issue_update(data, updated_from)
            elif event_type == "Issue" and action == "create":
                await self._handle_issue_create(data)

        except Exception:
            logger.exception("[linear] Error routing event")

    async def _handle_comment_event(self, data: dict) -> None:
        """Handle a comment create/update event — check for @mentions."""
        body = data.get("body", "")
        if not body or not body.strip():
            return

        # Skip self-comments (prevent loops)
        user = data.get("user", {})
        comment_user_id = (
            user.get("id") if isinstance(user, dict) else data.get("userId")
        )
        if comment_user_id and comment_user_id == self._agent_user_id:
            logger.debug("[linear] Skipping self-comment")
            return

        # Check for @mention of the agent
        body_data = data.get("bodyData")
        if self._agent_user_id:
            mentioned = _extract_mentioned_user_ids(body, body_data, self._agent_user_id)
            if self._agent_user_id not in mentioned:
                # Also check for "@hermes" or "@Hermes" text mention as fallback
                if not re.search(r"@hermes\b", body, re.IGNORECASE):
                    return
        else:
            # No agent user ID configured — only respond to @hermes text mentions
            if not re.search(r"@hermes\b", body, re.IGNORECASE):
                return

        # Build context from the issue
        issue = data.get("issue", {})
        issue_id = ""
        issue_identifier = ""
        issue_title = ""
        if isinstance(issue, dict):
            issue_id = issue.get("id", "")
            issue_identifier = issue.get("identifier", "")
            issue_title = issue.get("title", "")

        if not issue_identifier:
            issue_identifier = str(data.get("issueId", "unknown"))

        # Build the message for the agent
        commenter_name = user.get("name", "Unknown") if isinstance(user, dict) else "Unknown"
        comment_id = data.get("id", "")

        message_text = (
            f"[Linear Comment on {issue_identifier}: {issue_title}]\n"
            f"From: {commenter_name}\n\n"
            f"{body}"
        )

        # Create a MessageEvent and dispatch
        source = self.build_source(
            chat_id=issue_identifier,
            chat_name=f"{issue_identifier}: {issue_title}",
            chat_type="issue",
            user_id=str(comment_user_id) if comment_user_id else None,
            user_name=commenter_name,
        )

        event = MessageEvent(
            text=message_text,
            message_type=MessageType.TEXT,
            source=source,
            message_id=comment_id,
            timestamp=datetime.now(),
        )

        logger.info(
            "[linear] @mention from %s on %s — dispatching to agent",
            commenter_name,
            issue_identifier,
        )

        await self.handle_message(event)

    async def _handle_issue_update(self, data: dict, updated_from: dict) -> None:
        """Handle issue update events (assignment changes, state changes)."""
        if not self._agent_user_id:
            return

        assignee = data.get("assignee", {})
        assignee_id = assignee.get("id", "") if isinstance(assignee, dict) else data.get("assigneeId", "")

        # Check if issue was assigned to the agent
        old_assignee_id = updated_from.get("assigneeId", "")
        if assignee_id == self._agent_user_id and old_assignee_id != self._agent_user_id:
            identifier = data.get("identifier", data.get("id", "unknown"))
            title = data.get("title", "")
            priority = data.get("priority", 0)
            state = data.get("state", {})
            state_name = state.get("name", "") if isinstance(state, dict) else ""

            message_text = (
                f"[Linear Issue Assigned: {identifier}]\n"
                f"Title: {title}\n"
                f"Priority: {priority}\n"
                f"State: {state_name}\n\n"
                f"You have been assigned this issue. Use the linear CLI to view details and work on it."
            )

            source = self.build_source(
                chat_id=identifier,
                chat_name=f"{identifier}: {title}",
                chat_type="issue",
            )

            event = MessageEvent(
                text=message_text,
                message_type=MessageType.TEXT,
                source=source,
                timestamp=datetime.now(),
            )

            logger.info("[linear] Issue %s assigned to agent — dispatching", identifier)
            await self.handle_message(event)

    async def _handle_issue_create(self, data: dict) -> None:
        """Handle new issue creation — notify if assigned to agent."""
        if not self._agent_user_id:
            return

        assignee = data.get("assignee", {})
        assignee_id = assignee.get("id", "") if isinstance(assignee, dict) else data.get("assigneeId", "")

        if assignee_id != self._agent_user_id:
            return

        identifier = data.get("identifier", data.get("id", "unknown"))
        title = data.get("title", "")
        description = data.get("description", "")
        priority = data.get("priority", 0)

        message_text = (
            f"[New Linear Issue Assigned: {identifier}]\n"
            f"Title: {title}\n"
            f"Priority: {priority}\n"
        )
        if description:
            # Truncate long descriptions
            desc_preview = description[:1000]
            if len(description) > 1000:
                desc_preview += "…"
            message_text += f"Description:\n{desc_preview}\n"

        message_text += "\nReview and begin work on this issue."

        source = self.build_source(
            chat_id=identifier,
            chat_name=f"{identifier}: {title}",
            chat_type="issue",
        )

        event = MessageEvent(
            text=message_text,
            message_type=MessageType.TEXT,
            source=source,
            timestamp=datetime.now(),
        )

        logger.info("[linear] New issue %s assigned to agent — dispatching", identifier)
        await self.handle_message(event)

    # ── Comment Posting ─────────────────────────────────────────────

    async def _post_comment(self, issue_identifier: str, body: str) -> SendResult:
        """Post a comment on a Linear issue.

        Uses the Linear GraphQL API directly to post comments.
        Falls back to the `linear` CLI if available.
        """
        # Try GraphQL API first (more reliable, no CLI dependency)
        if self._api_key:
            return await self._post_comment_graphql(issue_identifier, body)

        # Fallback: use linear CLI
        return await self._post_comment_cli(issue_identifier, body)

    async def _get_http_session(self):
        """Get or create a persistent aiohttp ClientSession."""
        import aiohttp
        if self._http_session is None or self._http_session.closed:
            self._http_session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=15),
            )
        return self._http_session

    async def _graphql(self, query: str, variables: Optional[dict] = None) -> dict:
        """Make a GraphQL request to the Linear API using aiohttp."""
        payload = {"query": query}
        if variables:
            payload["variables"] = variables

        headers = {
            "Authorization": self._api_key,
            "Content-Type": "application/json",
        }

        session = await self._get_http_session()
        async with session.post(
            "https://api.linear.app/graphql",
            json=payload,
            headers=headers,
        ) as resp:
            if resp.status != 200:
                text = await resp.text()
                raise RuntimeError(f"Linear API HTTP {resp.status}: {text}")
            return await resp.json()

    async def _post_comment_graphql(self, issue_identifier: str, body: str) -> SendResult:
        """Post a comment via Linear GraphQL API."""
        # First resolve the issue UUID from identifier
        try:
            result = await self._graphql(
                'query($id: String!) { issue(id: $id) { id identifier } }',
                {"id": issue_identifier},
            )
        except Exception as exc:
            logger.error("[linear] Failed to resolve issue %s: %s", issue_identifier, exc)
            return SendResult(success=False, error=f"Failed to resolve issue: {exc}")

        issue_data = result.get("data", {}).get("issue")
        if not issue_data:
            errors = result.get("errors", [])
            err_msg = errors[0]["message"] if errors else "Issue not found"
            return SendResult(success=False, error=err_msg)

        issue_uuid = issue_data["id"]

        # Now post the comment
        try:
            result = await self._graphql(
                'mutation($input: CommentCreateInput!) { '
                'commentCreate(input: $input) { '
                'success comment { id body } } }',
                {"input": {"issueId": issue_uuid, "body": body}},
            )
        except Exception as exc:
            logger.error("[linear] GraphQL comment failed: %s", exc)
            return SendResult(success=False, error=str(exc))

        create_data = result.get("data", {}).get("commentCreate", {})
        if create_data.get("success"):
            comment_id = create_data.get("comment", {}).get("id", "")
            logger.info("[linear] Posted comment on %s (id=%s)", issue_identifier, comment_id)
            return SendResult(success=True, message_id=comment_id)

        errors = result.get("errors", [])
        err_msg = errors[0]["message"] if errors else "Unknown error"
        return SendResult(success=False, error=err_msg)

    async def _post_comment_cli(self, issue_identifier: str, body: str) -> SendResult:
        """Post a comment via the linear CLI (fallback)."""
        loop = asyncio.get_event_loop()

        def _do_cli():
            proc = subprocess.run(
                ["linear", "issue", "comment", "create", issue_identifier],
                input=body,
                capture_output=True,
                text=True,
                timeout=30,
            )
            return proc

        try:
            proc = await loop.run_in_executor(None, _do_cli)
            if proc.returncode == 0:
                logger.info("[linear] Posted comment on %s via CLI", issue_identifier)
                return SendResult(success=True)
            else:
                logger.error("[linear] CLI comment failed: %s", proc.stderr)
                return SendResult(success=False, error=proc.stderr)
        except Exception as exc:
            logger.error("[linear] CLI comment failed: %s", exc)
            return SendResult(success=False, error=str(exc))
