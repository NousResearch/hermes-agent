"""
Sandbox Runtime Implementation (Modal/Vercel/Daytona).
"""

from typing import Any, Dict, Optional
from runtime.base import Runtime, RuntimeType


class SandboxRuntime(Runtime):
    def __init__(self, runtime_id: str, name: str, provider: str = "modal") -> None:
        super().__init__(runtime_id, name, RuntimeType.SANDBOX)
        self.provider = provider

    def connect(self) -> bool:
        self.connected = True
        return True

    def disconnect(self) -> bool:
        self.connected = False
        return True

    def is_connected(self) -> bool:
        return self.connected

    def execute_command(self, command: str, cwd: Optional[str] = None) -> Dict[str, Any]:
        return {"exit_code": 0, "stdout": f"Sandbox [{self.provider}]: {command}", "stderr": ""}

    def get_capabilities(self) -> Dict[str, Any]:
        return {"sandbox": True, "provider": self.provider}
