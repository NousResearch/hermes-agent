"""
SSH Runtime Implementation.
"""

from typing import Any, Dict, Optional
from runtime.base import Runtime, RuntimeType


class SSHRuntime(Runtime):
    def __init__(self, runtime_id: str, name: str, host: str, port: int = 22, user: str = "root") -> None:
        super().__init__(runtime_id, name, RuntimeType.SSH)
        self.host = host
        self.port = port
        self.user = user

    def connect(self) -> bool:
        self.connected = True
        return True

    def disconnect(self) -> bool:
        self.connected = False
        return True

    def is_connected(self) -> bool:
        return self.connected

    def execute_command(self, command: str, cwd: Optional[str] = None) -> Dict[str, Any]:
        return {"exit_code": 0, "stdout": f"SSH [{self.user}@{self.host}]: {command}", "stderr": ""}

    def get_capabilities(self) -> Dict[str, Any]:
        return {"remote": True, "host": self.host, "user": self.user}
