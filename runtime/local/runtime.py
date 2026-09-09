"""
Local Runtime Implementation.
"""

from typing import Any, Dict, Optional
from runtime.base import Runtime, RuntimeType
from hermes_platform.factory import get_platform_adapter


class LocalRuntime(Runtime):
    def __init__(self, runtime_id: str = "local_default", name: str = "Local System") -> None:
        super().__init__(runtime_id, name, RuntimeType.LOCAL)
        self.adapter = get_platform_adapter()

    def connect(self) -> bool:
        self.connected = True
        return True

    def disconnect(self) -> bool:
        self.connected = False
        return True

    def is_connected(self) -> bool:
        return True

    def execute_command(self, command: str, cwd: Optional[str] = None) -> Dict[str, Any]:
        return self.adapter.terminal.execute(command, cwd=cwd)

    def get_capabilities(self) -> Dict[str, Any]:
        return {
            "platform": self.adapter.platform_type.value,
            "filesystem": True,
            "terminal": True,
            "browser": True,
            "processes": True,
        }
