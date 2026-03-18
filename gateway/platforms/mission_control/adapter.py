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

import asyncio
import json
import logging
import os
from pathlib import Path
from typing import Optional, Any, Dict

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
        
        # Bind to localhost if MC is on same host, else 0.0.0.0
        host = "127.0.0.1"  # MC on same machine
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
            "activity.task_created": self._task_manager.handle_task_created,
            "activity.task_updated": self._task_manager.handle_task_updated,
            "activity.task_status_changed": self._task_manager.handle_task_status_changed,
            "activity.task_deleted": self._handle_task_deleted,
            "agent.status_change": self._task_manager.handle_agent_status_changed,
            "agent.error": self._task_manager.handle_agent_status_changed,
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
        
    def _hash_payload(self, body: bytes) -> str:
        """Generate hash of payload for logging."""
        import hashlib
        return hashlib.sha256(body).hexdigest()[:16]
        
    async def get_chat_info(self, chat_id: str) -> Dict[str, Any]:
        """Return chat info - MC is inbound-only, return generic info."""
        return {"name": chat_id, "type": "system"}