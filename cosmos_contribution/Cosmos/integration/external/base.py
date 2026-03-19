"""
cosmos External Integration Interface.

"I can wire anything into anything! I'm the Professor!"

This module defines the standard interface for connecting cosmos to 3rd party apps.
It uses the Nexus to emit events from these apps into the swarm.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from enum import Enum

from Cosmos.core.nexus import nexus, Signal, SignalType

class ConnectionStatus(Enum):
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    ERROR = "error"

@dataclass
class IntegrationConfig:
    name: str
    api_key: Optional[str] = None
    enabled: bool = True
    poll_interval: float = 60.0 # Seconds

class ExternalProvider(ABC):
    """
    Abstract base class for all external app integrations.
    """
    def __init__(self, config: IntegrationConfig):
        self.config = config
        self.status = ConnectionStatus.DISCONNECTED

    @abstractmethod
    async def connect(self) -> bool:
        """Authenticate and establish connection."""
        pass

    @abstractmethod
    async def sync(self):
        """Poll for updates (if webhook not available)."""
        pass
        
    @abstractmethod
    async def execute_action(self, action: str, params: Dict[str, Any]) -> Any:
        """Perform an action on the external service."""
        pass

    async def emit_event(self, event_type: str, payload: Dict[str, Any]):
        """Helper to inject external events into the Nexus."""
        await nexus.emit(
            SignalType.EXTERNAL_ALERT,
            {
                "provider": self.config.name,
                "event": event_type,
                "data": payload
            },
            source=f"ext_{self.config.name}"
        )
