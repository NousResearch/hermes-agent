"""
Remote Hermes Gateway Runtime Implementation.
"""

from typing import Any, Dict, Optional
from runtime.base import Runtime, RuntimeType


class RemoteRuntime(Runtime):
    def __init__(self, runtime_id: str, name: str, gateway_url: str) -> None:
        super().__init__(runtime_id, name, RuntimeType.REMOTE)
        self.gateway_url = gateway_url

    def connect(self) -> bool:
        self.connected = True
        return True

    def disconnect(self) -> bool:
        self.connected = False
        return True

    def is_connected(self) -> bool:
        return self.connected

    def execute_command(self, command: str, cwd: Optional[str] = None) -> Dict[str, Any]:
        return {"exit_code": 0, "stdout": f"Remote Gateway [{self.gateway_url}]: {command}", "stderr": ""}

    def get_capabilities(self) -> Dict[str, Any]:
        return {"remote_gateway": True, "url": self.gateway_url}
