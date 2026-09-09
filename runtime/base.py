"""
Abstract Runtime Base Interface.
"""

from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Dict, Optional


class RuntimeType(str, Enum):
    LOCAL = "local"
    SSH = "ssh"
    CONTAINER = "container"
    SANDBOX = "sandbox"
    REMOTE = "remote"


class Runtime(ABC):
    """Abstract interface for execution runtimes."""

    def __init__(self, runtime_id: str, name: str, runtime_type: RuntimeType) -> None:
        self.runtime_id = runtime_id
        self.name = name
        self.runtime_type = runtime_type
        self.connected = False

    @abstractmethod
    def connect(self) -> bool:
        pass

    @abstractmethod
    def disconnect(self) -> bool:
        pass

    @abstractmethod
    def is_connected(self) -> bool:
        pass

    @abstractmethod
    def execute_command(self, command: str, cwd: Optional[str] = None) -> Dict[str, Any]:
        pass

    @abstractmethod
    def get_capabilities(self) -> Dict[str, Any]:
        pass

    def get_status(self) -> Dict[str, Any]:
        return {
            "runtime_id": self.runtime_id,
            "name": self.name,
            "type": self.runtime_type.value,
            "connected": self.is_connected(),
        }
