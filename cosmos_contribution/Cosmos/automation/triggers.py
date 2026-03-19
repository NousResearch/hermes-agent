"""
cosmos Trigger System

"When something happens, something else happens! It's like dominoes, but with code!"

Event-driven triggers for automation workflows.
"""

import asyncio
import json
from datetime import datetime
from typing import Dict, List, Any, Optional, Callable, Coroutine
from dataclasses import dataclass, field
from enum import Enum
import uuid
from pathlib import Path
from loguru import logger

try:
    from aiohttp import web
    HAS_AIOHTTP = True
except ImportError:
    HAS_AIOHTTP = False

try:
    from watchdog.observers import Observer
    from watchdog.events import FileSystemEventHandler
    HAS_WATCHDOG = True
except ImportError:
    HAS_WATCHDOG = False


class TriggerType(Enum):
    """Types of triggers."""
    WEBHOOK = "webhook"
    SCHEDULE = "schedule"
    FILE_CHANGE = "file_change"
    EVENT = "event"
    EMAIL = "email"
    MANUAL = "manual"


class TriggerStatus(Enum):
    """Trigger status."""
    ACTIVE = "active"
    INACTIVE = "inactive"
    ERROR = "error"


@dataclass
class Trigger:
    """A trigger definition."""
    id: str
    name: str
    type: TriggerType
    config: Dict[str, Any]
    callback: Optional[Callable] = None
    callback_name: str = ""
    status: TriggerStatus = TriggerStatus.INACTIVE
    fire_count: int = 0
    last_fired: Optional[datetime] = None
    last_error: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "type": self.type.value,
            "config": self.config,
            "callback_name": self.callback_name,
            "status": self.status.value,
            "fire_count": self.fire_count,
            "last_fired": self.last_fired.isoformat() if self.last_fired else None,
            "last_error": self.last_error,
            "created_at": self.created_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Trigger":
        return cls(
            id=data["id"],
            name=data["name"],
            type=TriggerType(data["type"]),
            config=data["config"],
            callback_name=data.get("callback_name", ""),
            status=TriggerStatus(data.get("status", "inactive")),
            fire_count=data.get("fire_count", 0),
            last_fired=datetime.fromisoformat(data["last_fired"]) if data.get("last_fired") else None,
            last_error=data.get("last_error"),
            created_at=datetime.fromisoformat(data["created_at"]) if "created_at" in data else datetime.utcnow(),
        )


@dataclass
class WebhookTrigger(Trigger):
    """Webhook trigger configuration."""

    def __init__(
        self,
        name: str,
        path: str,
        method: str = "POST",
        secret: str = None,
        callback: Callable = None,
        callback_name: str = "",
    ):
        super().__init__(
            id=str(uuid.uuid4()),
            name=name,
            type=TriggerType.WEBHOOK,
            config={
                "path": path,
                "method": method,
                "secret": secret,
            },
            callback=callback,
            callback_name=callback_name,
        )


@dataclass
class ScheduleTrigger(Trigger):
    """Schedule-based trigger."""

    def __init__(
        self,
        name: str,
        cron: str = None,
        interval_seconds: int = None,
        callback: Callable = None,
        callback_name: str = "",
    ):
        config = {}
        if cron:
            config["cron"] = cron
        if interval_seconds:
            config["interval"] = interval_seconds

        super().__init__(
            id=str(uuid.uuid4()),
            name=name,
            type=TriggerType.SCHEDULE,
            config=config,
            callback=callback,
            callback_name=callback_name,
        )


@dataclass
class EventTrigger(Trigger):
    """Event-based trigger."""

    def __init__(
        self,
        name: str,
        event_type: str,
        filters: Dict[str, Any] = None,
        callback: Callable = None,
        callback_name: str = "",
    ):
        super().__init__(
            id=str(uuid.uuid4()),
            name=name,
            type=TriggerType.EVENT,
            config={
                "event_type": event_type,
                "filters": filters or {},
            },
            callback=callback,
            callback_name=callback_name,
        )


class FileChangeHandler(FileSystemEventHandler):
    """Handle file system events."""

    def __init__(self, callback: Callable, patterns: List[str] = None):
        self.callback = callback
        self.patterns = patterns or ["*"]

    def _should_handle(self, path: str) -> bool:
        import fnmatch
        return any(fnmatch.fnmatch(path, p) for p in self.patterns)

    def on_created(self, event):
        if not event.is_directory and self._should_handle(event.src_path):
            asyncio.create_task(self.callback({
                "type": "created",
                "path": event.src_path,
            }))

    def on_modified(self, event):
        if not event.is_directory and self._should_handle(event.src_path):
            asyncio.create_task(self.callback({
                "type": "modified",
                "path": event.src_path,
            }))

    def on_deleted(self, event):
        if not event.is_directory and self._should_handle(event.src_path):
            asyncio.create_task(self.callback({
                "type": "deleted",
                "path": event.src_path,
            }))


class TriggerManager:
    """
    Manages all triggers for cosmos automation.

    Features:
    - Webhook endpoints
    - File system watching
    - Event subscriptions
    - Schedule triggers (via TaskScheduler)
    """

    def __init__(
        self,
        storage_path: Path = None,
        webhook_port: int = 8090,
    ):
        self.storage_path = storage_path or Path("./data/triggers")
        self.storage_path.mkdir(parents=True, exist_ok=True)

        self.webhook_port = webhook_port
        self.triggers: Dict[str, Trigger] = {}
        self.callbacks: Dict[str, Callable] = {}

        # Webhook server
        self._webhook_app: Optional[web.Application] = None
        self._webhook_runner: Optional[web.AppRunner] = None

        # File watchers
        self._file_observers: Dict[str, Observer] = {}

        # Event subscriptions
        self._event_handlers: Dict[str, List[Trigger]] = {}

        self._load_triggers()

    def _load_triggers(self):
        """Load triggers from storage."""
        triggers_file = self.storage_path / "triggers.json"
        if triggers_file.exists():
            try:
                with open(triggers_file) as f:
                    data = json.load(f)
                    for trigger_data in data.get("triggers", []):
                        trigger = Trigger.from_dict(trigger_data)
                        self.triggers[trigger.id] = trigger
                logger.info(f"Loaded {len(self.triggers)} triggers")
            except Exception as e:
                logger.error(f"Failed to load triggers: {e}")

    def _save_triggers(self):
        """Save triggers to storage."""
        triggers_file = self.storage_path / "triggers.json"
        try:
            with open(triggers_file, "w") as f:
                json.dump({
                    "triggers": [t.to_dict() for t in self.triggers.values()],
                    "updated_at": datetime.utcnow().isoformat(),
                }, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save triggers: {e}")

    # =========================================================================
    # CALLBACK MANAGEMENT
    # =========================================================================

    def register_callback(self, name: str, callback: Callable):
        """Register a callback function."""
        self.callbacks[name] = callback

    def unregister_callback(self, name: str):
        """Unregister a callback."""
        if name in self.callbacks:
            del self.callbacks[name]

    # =========================================================================
    # TRIGGER MANAGEMENT
    # =========================================================================

    def create_webhook_trigger(
        self,
        name: str,
        path: str,
        callback_name: str,
        method: str = "POST",
        secret: str = None,
    ) -> Trigger:
        """Create a webhook trigger."""
        trigger = WebhookTrigger(
            name=name,
            path=path,
            method=method,
            secret=secret,
            callback_name=callback_name,
        )
        self.triggers[trigger.id] = trigger
        self._save_triggers()
        logger.info(f"Created webhook trigger: {name} at {path}")
        return trigger

    def create_schedule_trigger(
        self,
        name: str,
        callback_name: str,
        cron: str = None,
        interval_seconds: int = None,
    ) -> Trigger:
        """Create a schedule trigger."""
        trigger = ScheduleTrigger(
            name=name,
            cron=cron,
            interval_seconds=interval_seconds,
            callback_name=callback_name,
        )
        self.triggers[trigger.id] = trigger
        self._save_triggers()
        logger.info(f"Created schedule trigger: {name}")
        return trigger

    def create_event_trigger(
        self,
        name: str,
        event_type: str,
        callback_name: str,
        filters: Dict[str, Any] = None,
    ) -> Trigger:
        """Create an event trigger."""
        trigger = EventTrigger(
            name=name,
            event_type=event_type,
            filters=filters,
            callback_name=callback_name,
        )
        self.triggers[trigger.id] = trigger

        # Register for events
        if event_type not in self._event_handlers:
            self._event_handlers[event_type] = []
        self._event_handlers[event_type].append(trigger)

        self._save_triggers()
        logger.info(f"Created event trigger: {name} for {event_type}")
        return trigger

    def create_file_trigger(
        self,
        name: str,
        path: str,
        callback_name: str,
        patterns: List[str] = None,
        recursive: bool = True,
    ) -> Trigger:
        """Create a file change trigger."""
        trigger = Trigger(
            id=str(uuid.uuid4()),
            name=name,
            type=TriggerType.FILE_CHANGE,
            config={
                "path": path,
                "patterns": patterns or ["*"],
                "recursive": recursive,
            },
            callback_name=callback_name,
        )
        self.triggers[trigger.id] = trigger
        self._save_triggers()
        logger.info(f"Created file trigger: {name} watching {path}")
        return trigger

    def get_trigger(self, trigger_id: str) -> Optional[Trigger]:
        """Get a trigger by ID."""
        return self.triggers.get(trigger_id)

    def list_triggers(
        self,
        trigger_type: TriggerType = None,
        status: TriggerStatus = None,
    ) -> List[Trigger]:
        """List all triggers."""
        triggers = list(self.triggers.values())

        if trigger_type:
            triggers = [t for t in triggers if t.type == trigger_type]

        if status:
            triggers = [t for t in triggers if t.status == status]

        return triggers

    def delete_trigger(self, trigger_id: str) -> bool:
        """Delete a trigger."""
        trigger = self.triggers.pop(trigger_id, None)
        if trigger:
            # Clean up event handlers
            for handlers in self._event_handlers.values():
                if trigger in handlers:
                    handlers.remove(trigger)

            # Stop file watcher if applicable
            if trigger_id in self._file_observers:
                self._file_observers[trigger_id].stop()
                del self._file_observers[trigger_id]

            self._save_triggers()
            logger.info(f"Deleted trigger: {trigger.name}")
            return True
        return False

    # =========================================================================
    # WEBHOOK SERVER
    # =========================================================================

    async def start_webhook_server(self):
        """Start the webhook server."""
        if not HAS_AIOHTTP:
            logger.warning("aiohttp not installed, webhook server unavailable")
            return

        self._webhook_app = web.Application()
        self._webhook_app.router.add_route("*", "/{path:.*}", self._handle_webhook)

        self._webhook_runner = web.AppRunner(self._webhook_app)
        await self._webhook_runner.setup()

        site = web.TCPSite(self._webhook_runner, "0.0.0.0", self.webhook_port)
        await site.start()

        logger.info(f"Webhook server started on port {self.webhook_port}")

    async def stop_webhook_server(self):
        """Stop the webhook server."""
        if self._webhook_runner:
            await self._webhook_runner.cleanup()
            logger.info("Webhook server stopped")

    async def _handle_webhook(self, request: web.Request) -> web.Response:
        """Handle incoming webhook requests."""
        path = "/" + request.match_info.get("path", "")
        method = request.method

        # Find matching trigger
        matching_trigger = None
        for trigger in self.triggers.values():
            if trigger.type != TriggerType.WEBHOOK:
                continue
            if trigger.config.get("path") == path:
                if trigger.config.get("method", "POST") == method:
                    matching_trigger = trigger
                    break

        if not matching_trigger:
            return web.Response(status=404, text="Webhook not found")

        # Verify secret if configured
        secret = matching_trigger.config.get("secret")
        if secret:
            request_secret = request.headers.get("X-Webhook-Secret")
            if request_secret != secret:
                return web.Response(status=401, text="Invalid secret")

        # Parse body
        try:
            if request.content_type == "application/json":
                body = await request.json()
            else:
                body = await request.text()
        except Exception:
            body = None

        # Fire trigger
        await self._fire_trigger(matching_trigger, {
            "method": method,
            "path": path,
            "headers": dict(request.headers),
            "body": body,
            "query": dict(request.query),
        })

        return web.Response(status=200, text="OK")

    # =========================================================================
    # FILE WATCHING
    # =========================================================================

    def start_file_watcher(self, trigger_id: str):
        """Start watching files for a trigger."""
        if not HAS_WATCHDOG:
            logger.warning("watchdog not installed, file watching unavailable")
            return

        trigger = self.triggers.get(trigger_id)
        if not trigger or trigger.type != TriggerType.FILE_CHANGE:
            return

        path = trigger.config.get("path")
        patterns = trigger.config.get("patterns", ["*"])
        recursive = trigger.config.get("recursive", True)

        async def file_callback(event_data):
            await self._fire_trigger(trigger, event_data)

        handler = FileChangeHandler(file_callback, patterns)
        observer = Observer()
        observer.schedule(handler, path, recursive=recursive)
        observer.start()

        self._file_observers[trigger_id] = observer
        trigger.status = TriggerStatus.ACTIVE
        logger.info(f"Started file watcher for trigger: {trigger.name}")

    def stop_file_watcher(self, trigger_id: str):
        """Stop watching files for a trigger."""
        observer = self._file_observers.pop(trigger_id, None)
        if observer:
            observer.stop()
            observer.join()

        trigger = self.triggers.get(trigger_id)
        if trigger:
            trigger.status = TriggerStatus.INACTIVE

    # =========================================================================
    # EVENT HANDLING
    # =========================================================================

    async def emit_event(self, event_type: str, data: Dict[str, Any]):
        """Emit an event to trigger event-based triggers."""
        handlers = self._event_handlers.get(event_type, [])

        for trigger in handlers:
            # Check filters
            filters = trigger.config.get("filters", {})
            if self._match_filters(data, filters):
                await self._fire_trigger(trigger, {
                    "event_type": event_type,
                    "data": data,
                })

    def _match_filters(self, data: Dict[str, Any], filters: Dict[str, Any]) -> bool:
        """Check if data matches filters."""
        for key, value in filters.items():
            if key not in data:
                return False
            if data[key] != value:
                return False
        return True

    # =========================================================================
    # TRIGGER EXECUTION
    # =========================================================================

    async def _fire_trigger(self, trigger: Trigger, data: Dict[str, Any]):
        """Fire a trigger."""
        logger.debug(f"Firing trigger: {trigger.name}")

        trigger.fire_count += 1
        trigger.last_fired = datetime.utcnow()

        try:
            # Get callback
            callback = trigger.callback or self.callbacks.get(trigger.callback_name)
            if not callback:
                raise ValueError(f"Callback not found: {trigger.callback_name}")

            # Execute callback
            if asyncio.iscoroutinefunction(callback):
                await callback(trigger, data)
            else:
                callback(trigger, data)

            trigger.last_error = None

        except Exception as e:
            trigger.last_error = str(e)
            trigger.status = TriggerStatus.ERROR
            logger.error(f"Trigger error: {trigger.name} - {e}")

        self._save_triggers()

    # =========================================================================
    # LIFECYCLE
    # =========================================================================

    async def start(self):
        """Start all trigger systems."""
        # Start webhook server
        await self.start_webhook_server()

        # Start file watchers
        for trigger in self.triggers.values():
            if trigger.type == TriggerType.FILE_CHANGE:
                self.start_file_watcher(trigger.id)

        # Mark webhook triggers as active
        for trigger in self.triggers.values():
            if trigger.type == TriggerType.WEBHOOK:
                trigger.status = TriggerStatus.ACTIVE

        logger.info("Trigger manager started")

    async def stop(self):
        """Stop all trigger systems."""
        # Stop webhook server
        await self.stop_webhook_server()

        # Stop file watchers
        for trigger_id in list(self._file_observers.keys()):
            self.stop_file_watcher(trigger_id)

        # Mark all as inactive
        for trigger in self.triggers.values():
            trigger.status = TriggerStatus.INACTIVE

        logger.info("Trigger manager stopped")


# Singleton instance
trigger_manager = TriggerManager()
