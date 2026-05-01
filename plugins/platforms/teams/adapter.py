"""
Microsoft Teams platform adapter for Hermes Agent.

Uses the microsoft-teams-apps SDK for authentication and activity processing.
Runs an aiohttp webhook server to receive messages from Teams.
Proactive messaging (send, typing) uses the SDK's App.send() method.

Requires:
    pip install microsoft-teams-apps aiohttp
    TEAMS_CLIENT_ID, TEAMS_CLIENT_SECRET, and TEAMS_TENANT_ID env vars

Configuration in config.yaml:
    platforms:
      teams:
        enabled: true
        extra:
          client_id: "your-client-id"      # or TEAMS_CLIENT_ID env var
          client_secret: "your-secret"      # or TEAMS_CLIENT_SECRET env var
          tenant_id: "your-tenant-id"       # or TEAMS_TENANT_ID env var
          port: 3978                        # or TEAMS_PORT env var
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from typing import Any, Dict, Optional

try:
    from aiohttp import web

    AIOHTTP_AVAILABLE = True
except ImportError:
    AIOHTTP_AVAILABLE = False
    web = None  # type: ignore[assignment]

try:
    from microsoft_teams.apps import App, ActivityContext
    from microsoft_teams.common.http.client import ClientOptions
    from microsoft_teams.api import MessageActivity, ConversationReference
    from microsoft_teams.api.activities.typing import TypingActivityInput
    from microsoft_teams.api.activities.invoke.adaptive_card import AdaptiveCardInvokeActivity
    from microsoft_teams.api.models.adaptive_card import (
        AdaptiveCardActionCardResponse,
        AdaptiveCardActionMessageResponse,
    )
    from microsoft_teams.api.models.invoke_response import InvokeResponse, AdaptiveCardInvokeResponse
    from microsoft_teams.apps.http.adapter import (
        HttpMethod,
        HttpRequest,
        HttpResponse,
        HttpRouteHandler,
    )
    from microsoft_teams.cards import AdaptiveCard, ExecuteAction, TextBlock

    TEAMS_SDK_AVAILABLE = True
except ImportError:
    TEAMS_SDK_AVAILABLE = False
    ClientOptions = None  # type: ignore[assignment,misc]
    App = None  # type: ignore[assignment,misc]
    ActivityContext = None  # type: ignore[assignment,misc]
    MessageActivity = None  # type: ignore[assignment,misc]
    ConversationReference = None  # type: ignore[assignment,misc]
    TypingActivityInput = None  # type: ignore[assignment,misc]
    AdaptiveCardInvokeActivity = None  # type: ignore[assignment,misc]
    AdaptiveCardActionCardResponse = None  # type: ignore[assignment,misc]
    AdaptiveCardActionMessageResponse = None  # type: ignore[assignment,misc]
    AdaptiveCardInvokeResponse = None  # type: ignore[assignment,misc,union-attr]
    InvokeResponse = None  # type: ignore[assignment,misc]
    HttpMethod = str  # type: ignore[assignment,misc]
    HttpRequest = None  # type: ignore[assignment,misc]
    HttpResponse = None  # type: ignore[assignment,misc]
    HttpRouteHandler = None  # type: ignore[assignment,misc]
    AdaptiveCard = None  # type: ignore[assignment,misc]
    ExecuteAction = None  # type: ignore[assignment,misc]
    TextBlock = None  # type: ignore[assignment,misc]

from gateway.config import Platform, PlatformConfig
from gateway.platforms.helpers import MessageDeduplicator
from gateway.platforms.base import (
    BasePlatformAdapter,
    MessageEvent,
    MessageType,
    SendResult,
    cache_image_from_url,
)

logger = logging.getLogger(__name__)

_DEFAULT_PORT = 3978
_WEBHOOK_PATH = "/api/messages"


class _AiohttpBridgeAdapter:
    """HttpServerAdapter that bridges the Teams SDK into an aiohttp server.

    Without a custom adapter, ``App()`` unconditionally imports fastapi/uvicorn
    and allocates a ``FastAPI()`` instance.  This bridge captures the SDK's
    route registrations and wires them into our own aiohttp ``Application``.
    """

    def __init__(self, aiohttp_app: "web.Application"):
        self._aiohttp_app = aiohttp_app

    def register_route(self, method: "HttpMethod", path: str, handler: "HttpRouteHandler") -> None:
        """Register an SDK route handler as an aiohttp route."""

        async def _aiohttp_handler(request: "web.Request") -> "web.Response":
            body = await request.json()
            headers = dict(request.headers)
            result: "HttpResponse" = await handler(HttpRequest(body=body, headers=headers))
            status = result.get("status", 200)
            resp_body = result.get("body")
            if resp_body is not None:
                return web.Response(
                    status=status,
                    body=json.dumps(resp_body),
                    content_type="application/json",
                )
            return web.Response(status=status)

        self._aiohttp_app.router.add_route(method, path, _aiohttp_handler)

    def serve_static(self, path: str, directory: str) -> None:
        pass

    async def start(self, port: int) -> None:
        raise NotImplementedError("aiohttp server is managed by the adapter")

    async def stop(self) -> None:
        pass


def check_requirements() -> bool:
    """Return True when all Teams dependencies and credentials are present."""
    return TEAMS_SDK_AVAILABLE and AIOHTTP_AVAILABLE


def validate_config(config) -> bool:
    """Return True when the config has the minimum required credentials."""
    extra = getattr(config, "extra", {}) or {}
    client_id = os.getenv("TEAMS_CLIENT_ID") or extra.get("client_id", "")
    client_secret = os.getenv("TEAMS_CLIENT_SECRET") or extra.get("client_secret", "")
    tenant_id = os.getenv("TEAMS_TENANT_ID") or extra.get("tenant_id", "")
    return bool(client_id and client_secret and tenant_id)


def is_connected(config) -> bool:
    """Check whether Teams is configured (env or config.yaml)."""
    return validate_config(config)


# Keep the old name as an alias so existing test imports don't break.
check_teams_requirements = check_requirements


class TeamsAdapter(BasePlatformAdapter):
    """Microsoft Teams adapter using the microsoft-teams-apps SDK."""

    MAX_MESSAGE_LENGTH = 28000  # Teams text message limit (~28 KB)

    def __init__(self, config: PlatformConfig):
        super().__init__(config, Platform("teams"))
        extra = config.extra or {}
        self._client_id = extra.get("client_id") or os.getenv("TEAMS_CLIENT_ID", "")
        self._client_secret = extra.get("client_secret") or os.getenv("TEAMS_CLIENT_SECRET", "")
        self._tenant_id = extra.get("tenant_id") or os.getenv("TEAMS_TENANT_ID", "")
        self._port = int(extra.get("port") or os.getenv("TEAMS_PORT", str(_DEFAULT_PORT)))
        self._app: Optional["App"] = None
        self._runner: Optional["web.AppRunner"] = None
        self._dedup = MessageDeduplicator(max_size=1000)
        # Maps chat_id → ConversationReference captured from incoming messages.
        # Used to send cards with the correct conversation type (personal/group/channel).
        self._conv_refs: Dict[str, Any] = {}

    async def connect(self) -> bool:
        if not TEAMS_SDK_AVAILABLE:
            self._set_fatal_error(
                "MISSING_SDK",
                "microsoft-teams-apps not installed. Run: pip install microsoft-teams-apps",
                retryable=False,
            )
            return False

        if not AIOHTTP_AVAILABLE:
            self._set_fatal_error(
                "MISSING_SDK",
                "aiohttp not installed. Run: pip install aiohttp",
                retryable=False,
            )
            return False

        if not self._client_id or not self._client_secret or not self._tenant_id:
            self._set_fatal_error(
                "MISSING_CREDENTIALS",
                "TEAMS_CLIENT_ID, TEAMS_CLIENT_SECRET, and TEAMS_TENANT_ID are all required",
                retryable=False,
            )
            return False

        try:
            # Set up aiohttp app first — the bridge adapter wires SDK routes into it
            aiohttp_app = web.Application()
            aiohttp_app.router.add_get("/health", lambda _: web.Response(text="ok"))

            self._app = App(
                client_id=self._client_id,
                client_secret=self._client_secret,
                tenant_id=self._tenant_id,
                http_server_adapter=_AiohttpBridgeAdapter(aiohttp_app),
                client=ClientOptions(headers={"User-Agent": "Hermes"}),
            )

            # Register message handler before initialize()
            @self._app.on_message
            async def _handle_message(ctx: ActivityContext[MessageActivity]):
                await self._on_message(ctx)

            @self._app.on_card_action
            async def _handle_card_action(
                ctx: ActivityContext[AdaptiveCardInvokeActivity],
            ) -> InvokeResponse[AdaptiveCardActionMessageResponse]:
                return await self._on_card_action(ctx)

            # initialize() calls register_route() on the bridge, which adds
            # POST /api/messages to aiohttp_app automatically
            await self._app.initialize()

            self._runner = web.AppRunner(aiohttp_app)
            await self._runner.setup()
            site = web.TCPSite(self._runner, "0.0.0.0", self._port)
            await site.start()

            self._running = True
            self._mark_connected()
            logger.info(
                "[teams] Webhook server listening on 0.0.0.0:%d%s",
                self._port,
                _WEBHOOK_PATH,
            )
            return True

        except Exception as e:
            self._set_fatal_error(
                "CONNECT_FAILED",
                f"Teams connection failed: {e}",
                retryable=True,
            )
            logger.error("[teams] Failed to connect: %s", e)
            return False

    async def disconnect(self) -> None:
        self._running = False
        if self._runner:
            await self._runner.cleanup()
            self._runner = None
        self._app = None
        self._mark_disconnected()
        logger.info("[teams] Disconnected")

    async def _on_message(self, ctx: ActivityContext[MessageActivity]) -> None:
        """Process an incoming Teams message and dispatch to the gateway."""
        activity = ctx.activity

        # Self-message filter
        bot_id = self._app.id if self._app else None
        if bot_id and getattr(activity.from_, "id", None) == bot_id:
            return

        # Deduplication
        msg_id = getattr(activity, "id", None)
        if msg_id and self._dedup.is_duplicate(msg_id):
            return

        # Cache the conversation reference for proactive sends (approval cards, etc.)
        conv_id = getattr(activity.conversation, "id", None)
        if conv_id:
            self._conv_refs[conv_id] = ctx.conversation_ref

        # Extract text — strip bot @mentions
        text = ""
        if hasattr(activity, "text") and activity.text:
            text = activity.text
        # Strip <at>BotName</at> HTML tags that Teams prepends for @mentions
        if "<at>" in text:
            import re
            text = re.sub(r"<at>[^<]*</at>\s*", "", text).strip()

        # Determine chat type from conversation
        conv = activity.conversation
        conv_type = getattr(conv, "conversation_type", None) or ""
        if conv_type == "personal":
            chat_type = "dm"
        elif conv_type == "groupChat":
            chat_type = "group"
        elif conv_type == "channel":
            chat_type = "channel"
        else:
            chat_type = "dm"

        # Build source
        from_account = activity.from_
        user_id = getattr(from_account, "aad_object_id", None) or getattr(from_account, "id", "")
        user_name = getattr(from_account, "name", None) or ""

        source = self.build_source(
            chat_id=conv.id,
            chat_name=getattr(conv, "name", None) or "",
            chat_type=chat_type,
            user_id=str(user_id),
            user_name=user_name,
            guild_id=getattr(conv, "tenant_id", None) or self._tenant_id,
        )

        # Handle image attachments
        media_urls = []
        media_types = []
        for att in getattr(activity, "attachments", None) or []:
            content_url = getattr(att, "content_url", None)
            content_type = getattr(att, "content_type", None) or ""
            if content_url and content_type.startswith("image/"):
                try:
                    cached = await cache_image_from_url(content_url)
                    if cached:
                        media_urls.append(cached)
                        media_types.append(content_type)
                except Exception as e:
                    logger.warning("[teams] Failed to cache image attachment: %s", e)

        msg_type = MessageType.PHOTO if media_urls else MessageType.TEXT

        event = MessageEvent(
            text=text,
            source=source,
            message_type=msg_type,
            media_urls=media_urls,
            media_types=media_types,
            message_id=msg_id,
        )
        await self.handle_message(event)

    async def _send_card(self, chat_id: str, card: "AdaptiveCard") -> "Any":
        """Send an AdaptiveCard, using a stored ConversationReference when available."""
        from microsoft_teams.api import MessageActivityInput

        conv_ref = self._conv_refs.get(chat_id)
        if conv_ref and self._app:
            activity = MessageActivityInput().add_card(card)
            return await self._app.activity_sender.send(activity, conv_ref)
        elif self._app:
            return await self._app.send(chat_id, card)
        return None

    async def _on_card_action(
        self, ctx: "ActivityContext[AdaptiveCardInvokeActivity]"
    ) -> "InvokeResponse[AdaptiveCardActionMessageResponse]":
        """Handle an Adaptive Card Action.Execute button click."""
        from tools.approval import resolve_gateway_approval, has_blocking_approval

        action = ctx.activity.value.action
        data = action.data or {}
        hermes_action = data.get("hermes_action", "")
        session_key = data.get("session_key", "")

        if not hermes_action or not session_key:
            return InvokeResponse(
                status=200,
                body=AdaptiveCardActionMessageResponse(value="Unknown action."),
            )

        # Only authorized users may click approval buttons.
        allowed_csv = os.getenv("TEAMS_ALLOWED_USERS", "").strip()
        if allowed_csv:
            from_account = ctx.activity.from_
            clicker_id = getattr(from_account, "aad_object_id", None) or getattr(from_account, "id", "")
            allowed_ids = {uid.strip() for uid in allowed_csv.split(",") if uid.strip()}
            if "*" not in allowed_ids and clicker_id not in allowed_ids:
                logger.warning("[teams] Unauthorized card action by %s — ignoring", clicker_id)
                return InvokeResponse(
                    status=200,
                    body=AdaptiveCardActionMessageResponse(value="⛔ Not authorized."),
                )

        choice_map = {
            "approve_once": "once",
            "approve_session": "session",
            "approve_always": "always",
            "deny": "deny",
        }
        choice = choice_map.get(hermes_action)
        if not choice:
            return InvokeResponse(
                status=200,
                body=AdaptiveCardActionMessageResponse(value="Unknown action."),
            )

        if not has_blocking_approval(session_key):
            return InvokeResponse(
                status=200,
                body=AdaptiveCardActionCardResponse(
                    value=AdaptiveCard()
                    .with_version("1.4")
                    .with_body([TextBlock(text="⚠️ Approval already resolved or expired.", wrap=True)])
                ),
            )

        resolve_gateway_approval(session_key, choice)

        label_map = {
            "once": "✅ Allowed (once)",
            "session": "✅ Allowed (session)",
            "always": "✅ Always allowed",
            "deny": "❌ Denied",
        }
        cmd = data.get("cmd", "")
        desc = data.get("desc", "")
        body = []
        if cmd:
            body.append(TextBlock(text="⚠️ Command Approval Required", wrap=True, weight="Bolder"))
            body.append(TextBlock(text=f"```\n{cmd}\n```", wrap=True))
        if desc:
            body.append(TextBlock(text=f"Reason: {desc}", wrap=True, isSubtle=True))
        body.append(TextBlock(text=label_map[choice], wrap=True, weight="Bolder"))

        return InvokeResponse(
            status=200,
            body=AdaptiveCardActionCardResponse(
                value=AdaptiveCard().with_version("1.4").with_body(body)
            ),
        )

    async def send_exec_approval(
        self,
        chat_id: str,
        command: str,
        session_key: str,
        description: str = "dangerous command",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SendResult:
        """Send an Adaptive Card approval prompt with Allow/Deny buttons."""
        if not self._app:
            return SendResult(success=False, error="Teams app not initialized")

        cmd_preview = command[:2000] + "..." if len(command) > 2000 else command
        # Truncated for button data payload — just enough to reconstruct the card body.
        btn_data_base = {
            "session_key": session_key,
            "cmd": command[:200] + "..." if len(command) > 200 else command,
            "desc": description,
        }

        card = (
            AdaptiveCard()
            .with_version("1.4")
            .with_body([
                TextBlock(text="⚠️ Command Approval Required", wrap=True, weight="Bolder"),
                TextBlock(text=f"```\n{cmd_preview}\n```", wrap=True),
                TextBlock(text=f"Reason: {description}", wrap=True, isSubtle=True),
            ])
            .with_actions([
                ExecuteAction(
                    title="Allow Once",
                    verb="hermes_approve",
                    data={**btn_data_base, "hermes_action": "approve_once"},
                    style="positive",
                ),
                ExecuteAction(
                    title="Allow Session",
                    verb="hermes_approve",
                    data={**btn_data_base, "hermes_action": "approve_session"},
                ),
                ExecuteAction(
                    title="Always Allow",
                    verb="hermes_approve",
                    data={**btn_data_base, "hermes_action": "approve_always"},
                ),
                ExecuteAction(
                    title="Deny",
                    verb="hermes_approve",
                    data={**btn_data_base, "hermes_action": "deny"},
                    style="destructive",
                ),
            ])
        )

        try:
            result = await self._send_card(chat_id, card)
            message_id = getattr(result, "id", None) if result else None
            return SendResult(success=True, message_id=message_id)
        except Exception as e:
            logger.error("[teams] send_exec_approval failed: %s", e, exc_info=True)
            return SendResult(success=False, error=str(e), retryable=True)

    async def send(
        self,
        chat_id: str,
        content: str,
        reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SendResult:
        if not self._app:
            return SendResult(success=False, error="Teams app not initialized")

        formatted = self.format_message(content)
        chunks = self.truncate_message(formatted)
        last_message_id = None

        for chunk in chunks:
            try:
                result = await self._app.send(chat_id, chunk)
                last_message_id = getattr(result, "id", None)
            except Exception as e:
                return SendResult(success=False, error=str(e), retryable=True)

        return SendResult(success=True, message_id=last_message_id)

    async def send_typing(self, chat_id: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        if not self._app:
            return
        try:
            await self._app.send(chat_id, TypingActivityInput())
        except Exception:
            pass

    async def send_image(
        self,
        chat_id: str,
        image_url: str,
        caption: Optional[str] = None,
        reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SendResult:
        if not self._app:
            return SendResult(success=False, error="Teams app not initialized")

        try:
            import base64
            import mimetypes
            from microsoft_teams.api import Attachment, MessageActivityInput

            if image_url.startswith("http://") or image_url.startswith("https://"):
                content_url = image_url
                mime_type = "image/png"
            else:
                # Local path — encode as base64 data URI
                path = image_url.removeprefix("file://")
                mime_type = mimetypes.guess_type(path)[0] or "image/png"
                with open(path, "rb") as f:
                    content_url = f"data:{mime_type};base64,{base64.b64encode(f.read()).decode()}"

            attachment = Attachment(content_type=mime_type, content_url=content_url)
            activity = MessageActivityInput().add_attachments(attachment)
            if caption:
                activity = activity.add_text(caption)

            conv_ref = self._conv_refs.get(chat_id)
            if conv_ref:
                result = await self._app.activity_sender.send(activity, conv_ref)
            else:
                result = await self._app.send(chat_id, activity)

            return SendResult(success=True, message_id=getattr(result, "id", None))
        except Exception as e:
            logger.error("[teams] send_image failed: %s", e, exc_info=True)
            return SendResult(success=False, error=str(e), retryable=True)

    async def send_image_file(
        self,
        chat_id: str,
        image_path: str,
        caption: Optional[str] = None,
        reply_to: Optional[str] = None,
        **kwargs,
    ) -> SendResult:
        return await self.send_image(
            chat_id=chat_id,
            image_url=image_path,
            caption=caption,
            reply_to=reply_to,
        )

    async def get_chat_info(self, chat_id: str) -> dict:
        return {"name": chat_id, "type": "unknown", "chat_id": chat_id}


# ── Interactive setup ─────────────────────────────────────────────────────────

def interactive_setup() -> None:
    """Guide the user through Teams setup using the Teams CLI."""
    from hermes_cli.config import (
        get_env_value,
        save_env_value,
    )
    from hermes_cli.cli_output import (
        prompt,
        prompt_yes_no,
        print_info,
        print_success,
        print_warning,
    )

    existing_id = get_env_value("TEAMS_CLIENT_ID")
    if existing_id:
        print_info(f"Teams: already configured (app ID: {existing_id})")
        if not prompt_yes_no("Reconfigure Teams?", False):
            return

    print_info("You'll need the Teams CLI. If you haven't already:")
    print_info("  npm install -g @microsoft/teams.cli@preview")
    print_info("  teams login")
    print()
    print_info("Then expose port 3978 publicly (devtunnel / ngrok / cloudflared),")
    print_info("and create your bot:")
    print_info("  teams app create --name \"Hermes\" --endpoint \"https://<tunnel>/api/messages\"")
    print()
    print_info("The CLI will print CLIENT_ID, CLIENT_SECRET, and TENANT_ID. Paste them below.")
    print()

    client_id = prompt("Client ID", default=existing_id or "")
    if not client_id:
        print_warning("Client ID is required — skipping Teams setup")
        return
    save_env_value("TEAMS_CLIENT_ID", client_id.strip())

    client_secret = prompt("Client secret", default=get_env_value("TEAMS_CLIENT_SECRET") or "", password=True)
    if not client_secret:
        print_warning("Client secret is required — skipping Teams setup")
        return
    save_env_value("TEAMS_CLIENT_SECRET", client_secret.strip())

    tenant_id = prompt("Tenant ID", default=get_env_value("TEAMS_TENANT_ID") or "")
    if not tenant_id:
        print_warning("Tenant ID is required — skipping Teams setup")
        return
    save_env_value("TEAMS_TENANT_ID", tenant_id.strip())

    print()
    print_info("To find your AAD object ID for the allowlist: teams status --verbose")
    if prompt_yes_no("Restrict access to specific users? (recommended)", True):
        allowed = prompt(
            "Allowed AAD object IDs (comma-separated)",
            default=get_env_value("TEAMS_ALLOWED_USERS") or "",
        )
        if allowed:
            save_env_value("TEAMS_ALLOWED_USERS", allowed.replace(" ", ""))
            print_success("Allowlist configured")
        else:
            save_env_value("TEAMS_ALLOWED_USERS", "")
    else:
        save_env_value("TEAMS_ALLOW_ALL_USERS", "true")
        print_warning("⚠️  Open access — anyone who can message the bot can command it.")

    print()
    print_success("Teams configuration saved to ~/.hermes/.env")
    print_info("Install the app in Teams:  teams app install --id <teamsAppId>")
    print_info("Restart the gateway:       hermes gateway restart")


# ── Plugin entry point ────────────────────────────────────────────────────────

def register(ctx) -> None:
    """Plugin entry point — called by the Hermes plugin system."""
    ctx.register_platform(
        name="teams",
        label="Microsoft Teams",
        adapter_factory=lambda cfg: TeamsAdapter(cfg),
        check_fn=check_requirements,
        validate_config=validate_config,
        is_connected=is_connected,
        required_env=["TEAMS_CLIENT_ID", "TEAMS_CLIENT_SECRET", "TEAMS_TENANT_ID"],
        install_hint="pip install microsoft-teams-apps aiohttp",
        setup_fn=interactive_setup,
        # Auth env vars for _is_user_authorized() integration
        allowed_users_env="TEAMS_ALLOWED_USERS",
        allow_all_env="TEAMS_ALLOW_ALL_USERS",
        # Teams supports up to ~28 KB per message
        max_message_length=28000,
        # Display
        emoji="💼",
        allow_update_command=True,
        # LLM guidance
        platform_hint=(
            "You are chatting via Microsoft Teams. Teams renders a subset of "
            "markdown — bold (**text**), italic (*text*), and inline code "
            "(`code`) work, but complex tables or raw HTML do not. Keep "
            "responses clear and professional."
        ),
    )
