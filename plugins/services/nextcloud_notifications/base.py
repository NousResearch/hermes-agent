"""Shared base class + event dataclass for Nextcloud background services.

Lives inside the plugin so the service module is self-contained and
doesn't depend on a sibling ``gateway/services/`` directory in the core
tree. Mirrors the canonical ``BaseService`` shape so future Hermes
service plugins can adopt the same interface.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class ServiceEvent:
    """Event surfaced by a background service for routing/agent dispatch."""

    service: str
    notification_id: int
    app: str
    object_type: str
    object_id: str
    subject: str
    message: str
    link: str
    sender: str
    timestamp: str
    action: str  # "react" (forward to agent) or "silent" (record only)
    raw: dict = field(default_factory=dict)


class BaseService(ABC):
    """Background service that observes an external source and routes events.

    The plugin-registered ``service_factory`` is called with
    ``(config_dict, gateway_runner)`` — subclasses MUST accept both. The
    factory contract is enforced by
    ``gateway.service_registry.service_registry.create_service``.
    """

    name: str = "base"

    def __init__(self, config: dict, gateway_runner: Optional[Any] = None):
        self.config = config
        self.gateway_runner = gateway_runner

    @abstractmethod
    async def start(self) -> bool:
        """Start background tasks. Returns True on success."""

    @abstractmethod
    async def stop(self) -> None:
        """Clean shutdown — cancel tasks, close connections."""
