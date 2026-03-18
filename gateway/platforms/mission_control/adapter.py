"""Mission Control webhook platform adapter.

Receives webhooks from Mission Control on port 8888.
Handles task creation, updates, and agent status changes.

Environment Variables:
    MC_WEBHOOK_PORT: Port to listen on (default: 8888)
    MC_WEBHOOK_SECRET: Shared secret for HMAC verification
    MC_WEBHOOK_PATH: URL path (default: /webhooks/mc)
    MC_DB_PATH: SQLite database path
    MC_AGENT_NAME: Agent name for auto-accept (default: hermes-cli)
    MC_AUTO_ACCEPT: Auto-accept assignments (default: true)
"""

import json
import logging
import os
from pathlib import Path
from typing import Optional, Any, Dict

# Maximum request body size (1MB) to prevent memory exhaustion
MAX_REQUEST_BODY_SIZE = 1024 * 1024

try:
    from aiohttp import web
    AIOHTTP_AVAILABLE = True
except ImportError:
    AIOHTTP_AVAILABLE = False
    web = None  # type: ignore

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import (
    BasePlatformAdapter,
    MessageEvent,
    SendResult,
)

from .database import MissionControlDatabase
from .signature import verify_signature
from .notifications import CLINotifier
from .task_manager import TaskManager

logger = logging.getLogger(__name__)

# Default configuration
DEFAULT_PORT = 8888
DEFAULT_PATH = "/webhooks/mc"
DEFAULT_AGENT_NAME = "hermes-cli"


def check_mc_requirements() -> bool:
    """Check if Mission Control adapter dependencies are available."""
    if not AIOHTTP_AVAILABLE:
        logger.error("[mc] aiohttp not installed. Run: pip install aiohttp")
        return False
    return True


class MissionControlAdapter(BasePlatformAdapter):
    """
    Mission Control webhook receiver adapter.
    
    Listens for HTTP POST webhooks from MC on a configurable port.
    Handles task lifecycle events and agent status updates.
    """
    
    def __init__(self, config: PlatformConfig):
        super().__init__(config, Platform.MISSION_CONTROL)
        
        # Configuration from environment
        self._port = int(os.getenv("MC_WEBHOOK_PORT", str(DEFAULT_PORT)))
        self._path = os.getenv("MC_WEBHOOK_PATH", DEFAULT_PATH)
        self._secret = os.getenv("MC_WEBHOOK_SECRET", "")
        self._agent_name = os.getenv("MC_AGENT_NAME", DEFAULT_AGENT_NAME)
        self._auto_accept = os.getenv("MC_AUTO_ACCEPT", "true").lower() == "true"
        
        # Database path
        default_db = Path.home() / ".hermes" / "mission_control.db"
        self._db_path = Path(os.getenv("MC_DB_PATH", str(default_db)))
        
        # Components (initialized in connect)
        self._db: Optional[MissionControlDatabase] = None
        self._notifier: Optional[CLINotifier] = None
        self._task_manager: Optional[TaskManager] = None
        self._app: Optional["web.Application"] = None
        self._runner: Optional["web.AppRunner"] = None
        self._site: Optional["web.TCPSite"] = None
        
    async def connect(self) -> bool:
        """Start the webhook HTTP server."""
        if not AIOHTTP_AVAILABLE:
            logger.error("[mc] aiohttp required but not installed")
            return False
            
        # Initialize components
        self._db = MissionControlDatabase(self._db_path)
        self._notifier = CLINotifier()
        self._task_manager = TaskManager(
            self._db,
            self._notifier,
            self._agent_name,
            self._auto_accept
        )
        
        # Create aiohttp app
        self._app = web.Application()
        self._app.router.add_post(self._path, self._handle_webhook)
        self._app.router.add_get("/health", self._handle_health)
        
        # Start server
        self._runner = web.AppRunner(self._app)
        await self._runner.setup()
        
        # Bind to all interfaces to allow external webhooks
        host = os.getenv("MC_WEBHOOK_HOST", "0.0.0.0")
        self._site = web.TCPSite(self._runner, host, self._port)
        await self._site.start()
        
        self._running = True
        
        logger.info(
            "[mc] Mission Control webhook server listening on %s:%d%s",
            host, self._port, self._path
        )
        logger.info("[mc] Agent name: %s, Auto-accept: %s", 
                   self._agent_name, self._auto_accept)
        
        return True
        
    async def disconnect(self) -> None:
        """Stop the webhook server."""
        self._running = False
        
        if self._site:
            await self._site.stop()
        if self._runner:
            await self._runner.cleanup()
            
        logger.info("[mc] Mission Control webhook server stopped")
        
    async def send(self, message: str, chat_id: Optional[str] = None,
                   **kwargs) -> SendResult:
        """
        Send message - not typically used for MC adapter
        (MC is inbound-only, but we can log to CLI).
        """
        if self._notifier:
            self._notifier.send(message)
        return SendResult(success=True, message_id=None)
        
    async def _handle_health(self, request: "web.Request") -> "web.Response":
        """Health check endpoint."""
        return web.json_response({
            "status": "ok",
            "platform": "mission_control",
            "running": self._running
        })
        
    async def _handle_webhook(self, request: "web.Request") -> "web.Response":
        """
        Handle incoming webhook from Mission Control.
        
        Verifies signature, checks idempotency, routes to handler.
        """
        # Check request body size to prevent memory exhaustion
        if request.content_length and request.content_length > MAX_REQUEST_BODY_SIZE:
            logger.warning("[mc] Request body too large: %d bytes", request.content_length)
            return web.Response(status=413, text="Payload too large")
        
        # Read raw body for signature verification
        body = await request.read()
        
        # Verify signature
        signature = request.headers.get("X-MC-Signature", "")
        if not verify_signature(body, signature, self._secret):
            logger.warning("[mc] Invalid signature from %s", request.remote)
            return web.Response(status=401, text="Invalid signature")
            
        # Parse JSON
        try:
            payload = json.loads(body)
        except json.JSONDecodeError as e:
            logger.error("[mc] Invalid JSON: %s", e)
            return web.Response(status=400, text="Invalid JSON")
            
        # Extract event info
        event_type = payload.get("event", "")
        timestamp = payload.get("timestamp", 0)
        data = payload.get("data", {})
        
        # Generate event ID for idempotency
        event_id = self._task_manager.generate_event_id(event_type, timestamp, data)
        
        # Check for duplicate
        if self._task_manager.is_duplicate(event_id):
            logger.debug("[mc] Duplicate event: %s", event_id)
            return web.Response(status=200, text="Already processed")
            
        # Log receipt
        self._db.log_webhook_delivery(event_id, event_type, 
                                       self._hash_payload(body))
        
        logger.info("[mc] Received %s", event_type)
        
        # Route to handler
        try:
            handled = await self._route_event(event_type, data)
            
            if handled:
                self._db.mark_event_processed(event_id)
                return web.Response(status=200, text="OK")
            else:
                logger.warning("[mc] Event not handled: %s", event_type)
                self._db.mark_event_processed(event_id, "Not handled")
                return web.Response(status=200, text="Not handled")
                
        except Exception as e:
            logger.exception("[mc] Error handling event: %s", e)
            self._db.mark_event_processed(event_id, str(e))
            # Return 200 to prevent MC retries for unrecoverable errors
            return web.Response(status=200, text="Error logged")
            
    async def _route_event(self, event_type: str, data: Dict[str, Any]) -> bool:
        """Route event to appropriate handler."""
        if not self._task_manager:
            return False
            
        handlers = {
            # Task events (both activity.* and direct task.* formats)
            "activity.task_created": self._task_manager.handle_task_created,
            "activity.task_updated": self._task_manager.handle_task_updated,
            "activity.task_status_changed": self._task_manager.handle_task_status_changed,
            "activity.task_deleted": self._handle_task_deleted,
            "task.created": self._task_manager.handle_task_created,
            "task.updated": self._task_manager.handle_task_updated,
            "task.status_changed": self._task_manager.handle_task_status_changed,
            "task.deleted": self._handle_task_deleted,
            # Agent events
            "agent.status_change": self._task_manager.handle_agent_status_changed,
            "agent.error": self._task_manager.handle_agent_status_changed,
            "agent.updated": self._task_manager.handle_agent_status_changed,
            "agent.created": self._task_manager.handle_agent_status_changed,
            "agent.deleted": self._handle_agent_deleted,
            "agent.synced": self._task_manager.handle_agent_status_changed,
            "agent.status_changed": self._task_manager.handle_agent_status_changed,
            # Chat events
            "chat.message": self._handle_chat_message,
            "chat.message.deleted": self._handle_chat_message_deleted,
            # Notification events
            "notification.created": self._handle_notification_created,
            "notification.read": self._handle_notification_read,
            # Activity events
            "activity.created": self._handle_activity_created,
            # Security/Audit events
            "audit.security": self._handle_security_event,
            "security.event": self._handle_security_event,
            # Connection events
            "connection.created": self._handle_connection_event,
            "connection.disconnected": self._handle_connection_event,
            # GitHub events
            "github.synced": self._handle_github_synced,
            # Test events
            "test.ping": self._handle_test_ping,
        }
        
        handler = handlers.get(event_type)
        if handler:
            return handler(data)
        else:
            # Unknown event - log but don't fail
            return self._task_manager.handle_unknown_event(event_type, data)
            
    def _handle_task_deleted(self, data: Dict[str, Any]) -> bool:
        """Handle task deletion."""
        task_id = data.get("id")
        if task_id:
            logger.info("[mc] Task %d deleted", task_id)
            # Optionally remove from DB or mark as deleted
        return True

    def _handle_agent_deleted(self, data: Dict[str, Any]) -> bool:
        """Handle agent deletion."""
        agent_id = data.get("id") or data.get("agent_id")
        agent_name = data.get("name") or data.get("agent_name", "unknown")
        if agent_id:
            logger.info("[mc] Agent %s (ID: %s) deleted", agent_name, agent_id)
        return True

    def _handle_chat_message(self, data: Dict[str, Any]) -> bool:
        """Handle chat message."""
        message_id = data.get("id")
        content = data.get("content", "")
        sender = data.get("sender", "unknown")
        channel = data.get("channel", "default")
        logger.info("[mc] Chat message in %s from %s: %s", channel, sender, content[:50])
        # Could route to CLI notifications or create a session
        return True

    def _handle_chat_message_deleted(self, data: Dict[str, Any]) -> bool:
        """Handle chat message deletion."""
        message_id = data.get("id")
        if message_id:
            logger.info("[mc] Chat message %s deleted", message_id)
        return True

    def _handle_notification_created(self, data: Dict[str, Any]) -> bool:
        """Handle notification creation."""
        notification_id = data.get("id")
        title = data.get("title", "Notification")
        message = data.get("message", "")
        level = data.get("level", "info")
        logger.info("[mc] Notification [%s]: %s - %s", level, title, message[:50])
        if self._notifier:
            self._notifier.notify(f"📢 {title}: {message}")
        return True

    def _handle_notification_read(self, data: Dict[str, Any]) -> bool:
        """Handle notification read."""
        notification_id = data.get("id")
        if notification_id:
            logger.info("[mc] Notification %s marked as read", notification_id)
        return True

    def _handle_activity_created(self, data: Dict[str, Any]) -> bool:
        """Handle activity log entry."""
        activity_type = data.get("type", "unknown")
        description = data.get("description", "")
        user = data.get("user", "system")
        logger.info("[mc] Activity: %s by %s - %s", activity_type, user, description[:50])
        return True

    def _handle_security_event(self, data: Dict[str, Any]) -> bool:
        """Handle security/audit event."""
        event_type = data.get("type") or data.get("event_type", "unknown")
        severity = data.get("severity", "info")
        description = data.get("description", "")
        logger.warning("[mc] Security [%s]: %s - %s", severity, event_type, description[:50])
        if self._notifier and severity in ("high", "critical", "error"):
            self._notifier.notify(f"🚨 Security Alert [{severity}]: {event_type}")
        return True

    def _handle_connection_event(self, data: Dict[str, Any]) -> bool:
        """Handle connection created/disconnected events."""
        event_type = data.get("type", "unknown")
        connection_id = data.get("id") or data.get("connection_id")
        agent_name = data.get("agent_name", "unknown")
        if event_type == "connection.created":
            logger.info("[mc] Connection established: %s (agent: %s)", connection_id, agent_name)
        else:
            logger.info("[mc] Connection disconnected: %s (agent: %s)", connection_id, agent_name)
        return True

    def _handle_github_synced(self, data: Dict[str, Any]) -> bool:
        """Handle GitHub sync event."""
        repo = data.get("repository") or data.get("repo", "unknown")
        sync_type = data.get("sync_type", "unknown")
        commit_count = data.get("commit_count", 0)
        logger.info("[mc] GitHub sync: %s (%s) - %d commits", repo, sync_type, commit_count)
        return True

    def _handle_test_ping(self, data: Dict[str, Any]) -> bool:
        """Handle test ping event."""
        logger.info("[mc] Test ping received")
        return True

    def _hash_payload(self, body: bytes) -> str:
        """Generate hash of payload for logging."""
        import hashlib
        return hashlib.sha256(body).hexdigest()[:16]
        
    async def get_chat_info(self, chat_id: str) -> Dict[str, Any]:
        """Return chat info - MC is inbound-only, return generic info."""
        return {"name": chat_id, "type": "system"}